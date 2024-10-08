import time
from tqdm import tqdm
from ..utils.misc import *
from ..utils.eval import *
from .admm import *

def standard_train(configs, cepoch, model, data_loader, criterion, optimizer, scheduler, ADMM=None, masks=None, comm=False):

    batch_acc    = AverageMeter()
    batch_loss   = AverageMeter()
    batch_comm   = AverageMeter()
    evalHelper   = EvalHelper(configs['data_code'])
    
    if comm:
        partition = configs['partition']
    
    if ADMM is not None: 
        admm_initialization(configs, ADMM=ADMM, model=model)
        
    start_time = time.time()
    n_data = configs['batch_size'] * len(data_loader)
    pbar = tqdm(enumerate(data_loader), total=n_data/configs['batch_size'], ncols=150)
    
    for batch_idx, batch in pbar:
           
        data   = ()
        for piece in batch[:-1]:
            data += (piece.float().to(configs['device']),)
        target = batch[-1].to(configs['device'])
        total_loss = 0
        comm_loss = 0
        comp_loss = 0

        data = (torch.cat(data, dim=1),)
        
        optimizer.zero_grad()
        
        if configs['mix_up']:
            data, target_a, target_b, lam = mixup_data(*data, y=target, alpha=configs['alpha'])
        # print('data:', data)
        
        # print(len(data))
        # print(data[0].shape)
        output = model(*data)
        # print('output:', output)
        
        if configs['mix_up']:
            loss = mixup_criterion(criterion, output, target_a, target_b, lam, configs['smooth'])
        else:
            loss = criterion(output, target, smooth=configs['smooth'])
            # loss = criterion(output, target.unsqueeze(1).float())
        # print('xentropy_loss:', loss)
        total_loss += (loss * configs['xentropy_weight'])
        # print('total_loss:', total_loss)
        
        if ADMM is not None:
            z_u_update(configs, ADMM, model, cepoch, batch_idx)  # update Z and U variables
            prev_loss, admm_loss, total_loss = append_admm_loss(ADMM, model, total_loss)  # append admm losses
            
        if comm:
            for (name, W) in model.named_parameters():
                if name in ADMM.prune_ratios:
                    comm_cost = torch.abs(W) * configs['comm_costs'][name]
                    '''
                    v1: abs(W)*comm_cost
                    '''
                    comm_cost = comm_cost.view(comm_cost.size(0), -1).sum()
                    if configs['comm_outsize']:
                        comm_loss += comm_cost*partition[name]['outsize']
                    else:
                        comm_loss += comm_cost
                    
                    '''
                    #v2: further constraint on max(abs(W)*comm_cost)
                    '''
                    '''
                    comm_cost = comm_cost.reshape(W.shape[0], W.shape[1], -1).sum(-1)
                    for i in range(partition[name]['num']):
                        for j in range(partition[name]['num']):
                            if i==j: continue
                            cost_interp = comm_cost[partition[name]['filter_id'][i][:,None],
                                                    partition[name]['channel_id'][j]].sum()
                            #comm_loss += cost_interp*partition[name]['outsize']) #p_{count}
                            comm_loss = max(comm_loss,cost_interp*partition[name]['outsize']) # p_{max}
                            
                            #comp_loss = max(comp_loss, cost_interp*partition[name]['outsize'])
                    '''
                    '''
                    computation cost:
                    for i in range(partition[name]['num']):
                        comp_loss = max(comp_loss, torch.abs(W).view(W.size(0), -1)[partition[name]['filter_id'][i],:].sum())
                    '''
            total_loss += configs['lambda_comm'] * comm_loss + configs['lambda_comp'] * comp_loss
            # print('total_loss:', total_loss)
        
        total_loss.backward() # Back Propagation
        
        # For masked training
        if masks is not None:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks and W.grad is not None:
                        W.grad *= masks[name]
                        
        optimizer.step()
        
        # adjust learning rate
        if ADMM is not None:
            admm_adjust_learning_rate(optimizer, cepoch, configs)
        else:
            scheduler.step()

        acc1 = evalHelper.call(output, target)
        batch_loss.update(loss.item(), target.size(0))
        batch_comm.update(comm_loss.item() if comm_loss else comm_loss, target.size(0))
        batch_acc.update(acc1[0].item(), target.size(0))

        
        # # # preparation log information and print progress # # #
        msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] Loss:{loss:.4f} CommLoss:{commloss:.4f} Acc:{acc:.4f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*configs['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*configs['batch_size']/n_data), 
                        loss = batch_loss.avg,
                        commloss = batch_comm.avg,
                        acc  = batch_acc.avg,
                    )

        pbar.set_description(msg)
    #print('Training time per epoch is {:.2f}s.'.format(time.time()-start_time))

    
def distill_train(configs, cepoch, teacher, student, data_loader, optimizer, scheduler):

    batch_acc    = AverageMeter()
    distillloss   = AverageMeter()
    filtloss   = AverageMeter()
    
    n_data = configs['batch_size'] * len(data_loader)
    
    start_time = time.time()
    pbar = tqdm(enumerate(data_loader), total=n_data/configs['batch_size'], ncols=150)
    for batch_idx, (data, target) in pbar:
           
        data   = data.to(configs['device'])
        target = target.to(configs['device'])
        
        optimizer.zero_grad()
        
        t_output, t_filt = teacher(data)
        s_output, s_filt = student(data)
        distill_loss = distillation(s_output, t_output, target, configs['distill_temp'], configs['distill_alpha'],)
        filt_loss = sum([actTransfer_loss(x, y) for x, y in zip([s_filt], [t_filt])])
        loss = distill_loss + configs['lambda_f']*filt_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc1 = accuracy(s_output, target, topk=(1,))
        distillloss.update(distill_loss.item(), data.size(0))
        filtloss.update(filt_loss.item(), data.size(0))
        batch_acc.update(acc1[0].item(), data.size(0))

        # # # preparation log information and print progress # # #
        msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] DistillLoss:{distillloss:.4f} FiltLoss:{filtloss:.4f} Acc:{acc:.4f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*configs['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*configs['batch_size']/n_data),
                        distillloss = distillloss.avg,
                        filtloss = filtloss.avg,
                        acc  = batch_acc.avg,
                    )

        pbar.set_description(msg)
    #print('Training time per epoch is {:.2f}s.'.format(time.time()-start_time))
    
