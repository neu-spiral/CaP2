from .. import *
from . import *
from .engine import *
from os import environ

def get_args():
    """ args from input
    """
    parser = argparse.ArgumentParser(description='Model Partition')
    
    parser.add_argument('-cfg', '--config',default=environ.get("config"),type=str, help='config input path')
    parser.add_argument('-tt', '--training_type', default='',type=str, help='training types [hsicprune|backprop]')
    parser.add_argument('-bs', '--batch_size', default=0,type=int, help='minibatch size')
    parser.add_argument('-op', '--optimizer', default='', type=str, help='optimizer')
    parser.add_argument('-lr', '--learning_rate', default=0,type=float, help='learning rate')
    parser.add_argument('-ep', '--epochs', default=-1,type=int, help='number of training epochs')
    parser.add_argument('-sd', '--seed', default=0,type=int, help='random seed for the trial')
    parser.add_argument('-dc', '--data_code', default='',type=str, help='name of the working dataset [mnist|cifar10|cifar100]')
    parser.add_argument('-m', '--model', default='',type=str, help='model architecture')
    parser.add_argument('-mf', '--model_file', default='',type=str, help='filename for saved model file')
    parser.add_argument('-nc', '--num_classes', default=0, type=int, help='number of classes')
    parser.add_argument('--device', type=str, default='',help='CUDA training')

    ### arguments for pruning
    parser.add_argument('-cp', '--create_partition', default=False, type=bool, help='Create partition mapping and save to yaml file')
    parser.add_argument('-st', '--sparsity_type', default='', type=str, help='for pruning')
    parser.add_argument('-pr', '--prune_ratio', default=1, type=float, help='pruning ratio')
    parser.add_argument('--admm', action='store_true', help='prune by admm')
    parser.add_argument('--admm_epochs', default=300, type=int, help='number of interval epochs to update admm (default: 1)')
    parser.add_argument('--rho', default=0, type=float, help='admm learning rate (default: 1)')
    parser.add_argument('--multi_rho', action='store_true', help='It works better to make rho monotonically increasing')
    parser.add_argument('-ree', '--retrain_ep', default=-1, type=int, help='training epoch of the masked retrain (default: -lr)')
    parser.add_argument('-relr', '--retrain_lr', default=0, type=float, help='learning rate of the masked retrain')
    parser.add_argument('-rebs', '--retrain_bs', default=0, type=int, help='batch size of the masked retrain (default: -bs)')
    parser.add_argument('-relx', '--retrain_lx', default=0, type=float, help='lx of the masked retrain (default: -bs)')
    parser.add_argument('-rely', '--retrain_ly', default=0, type=float, help='ly of the masked retrain (default: -bs)')
    parser.add_argument('-reopt', '--retrain_opt', default='',type=str, help='retraining optimizer')
    parser.add_argument('-rett', '--retraining_type', default='',type=str, help='retraining types [hsictprune|backprop]')
    parser.add_argument('-slmo', '--save_last_model_only', action='store_true', help='save last model only')
    parser.add_argument('-ldm', '--load_dense_model', default=False, type=bool, help='use pre-trained model')
    parser.add_argument('-ldmf', '--load_dense_model_file', default='', type=str, help='filename of the pre-trained model file')
    parser.add_argument('-lpm', '--load_pruned_model', default=False, type=bool, help='use pruned model')
    parser.add_argument('-lpmf', '--load_pruned_model_file', default='', type=str, help='filename of the pruned model file')
    
    # control the weight of xentropy and hsic, if not specified in yaml file
    parser.add_argument('-xw', '--xentropy_weight', default=0,type=float, help='how much weight to put on xentropy wrt hsic')
    
    
    ### Tricks for cifar10 but not used:
    parser.add_argument('--lr_scheduler', type=str, default='', help='define lr scheduler')
    parser.add_argument('--warmup', action='store_true', default=False, help='warm-up scheduler')
    parser.add_argument('--warmup_lr', type=float, default=0.0001, metavar='M', help='warmup-lr, smaller than original lr')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='M', help='number of epochs for lr warmup')
    parser.add_argument('--mixup', action='store_true', default=False, help='ce mixup')
    parser.add_argument('--alpha', type=float, default=0.0, metavar='M', help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
    parser.add_argument('--smooth', action='store_true', default=False, help='lable smooth')
    parser.add_argument('--smooth_eps', type=float, default=0.0, metavar='M', help='smoothing rate [0.0, 1.0], set to 0.0 to disable')
    
    # arguments for distillation
    parser.add_argument('--distill_model', type=str, default='', help='where to load pretrained distillation model')
    parser.add_argument('--distill_loss', type=str, default='kl', help='type of distillation loss')
    parser.add_argument('--distill_temp', default=30, type=float, help='temperature for kl')
    parser.add_argument('--distill_alpha', default=1, type=float, help='weight for distillation loss')
   
    # partition
    # parser.add_argument('-pfl','--par-first-layer', action='store_true', default=False, help='lable smooth')
    parser.add_argument('-pfl','--par_first_layer', default=False, help='lable smooth')
    parser.add_argument('-np', '--num_partition', default='', type=str, help='number of partition')
    parser.add_argument('-lt', '--layer_type', default='', type=str, help='regular/masked')
    parser.add_argument('-bt', '--bn_type', default='', type=str, help='regular/masked')
    parser.add_argument('--num_students', type=int, default=0, help='number of students')
    parser.add_argument('--filter_sizes', metavar='N', default='', help ="Ex, for 2 students --filter_sizes 64,64")
    parser.add_argument('-lcm','--lambda_comm', default=0, type=float, help='the coefficient of the comm objective')  
    parser.add_argument('-lcp','--lambda_comp', default=0, type=float, help='the coefficient of the comp objective')
    # parser.add_argument('-co','--comm_outsize', action='store_true', default=False, help='consider output size')
    parser.add_argument('-co','--comm_outsize', default=False, help='consider output size')
    args = parser.parse_args()

    return args
    
def main():

    args = get_args()
    config_dict = load_yaml(args.config)
    
    # gpu device
    if args.device:
    # if not 'device' in config_dict and args.device:
        config_dict['device'] = args.device
    
    # partition
    # if args.par_first_layer:
    if 'par_first_layer' not in config_dict and args.par_first_layer:
        config_dict['par_first_layer'] = args.par_first_layer
    # if args.comm_outsize:
    if 'comm_outsize' not in config_dict and args.comm_outsize:
        config_dict['comm_outsize'] = args.comm_outsize
    if args.num_partition:
    # if 'num_partition' not in config_dict and args.num_partition:
        config_dict['num_partition'] = args.num_partition
    # if args.layer_type:
    if 'layer_type' not in config_dict and args.layer_type:
        config_dict['layer_type'] = args.layer_type
    # if args.bn_type:
    if 'bn_type' not in config_dict and args.bn_type:
        config_dict['bn_type'] = args.bn_type
    if args.lambda_comm:
    # if 'lambda_comm' not in config_dict and args.lambda_comm:
        config_dict['lambda_comm'] = args.lambda_comm
    # if args.lambda_comp:
    if 'lambda_comp' not in config_dict and args.lambda_comp:
        config_dict['lambda_comp'] = args.lambda_comp
    # distillation
    # if args.distill_model:
    if 'distill_model' not in config_dict and args.distill_model:
        config_dict['distill_model'] = args.distill_model
    # if args.distill_loss:
    if 'distill_loss' not in config_dict and args.distill_loss:
        config_dict['distill_loss'] = args.distill_loss
    # if args.distill_temp:
    if 'distill_temp' not in config_dict and args.distill_temp:
        config_dict['distill_temp'] = args.distill_temp
    # if args.distill_alpha:
    if 'distill_alpha' not in config_dict and args.distill_alpha:
        config_dict['distill_alpha'] = args.distill_alpha

    # if args.optimizer:
    if 'optimizer' not in config_dict and args.optimizer:
        config_dict['optimizer'] = args.optimizer
    # if args.seed:
    if 'seed' not in config_dict and args.seed:
        config_dict['seed'] = args.seed
    # if args.learning_rate:
    if 'learning_rate' not in config_dict and args.learning_rate:
        config_dict['learning_rate'] = args.learning_rate
    # if args.model_file:
    if 'model_file' not in config_dict and args.model_file:
        config_dict['model_file'] = args.model_file
    # if args.load_model:
    if 'load_dense_model' not in config_dict and args.load_dense_model:
        config_dict['load_dense_model'] = args.load_dense_model
    if 'load_dense_model_file' not in config_dict and args.load_dense_model_file:
        config_dict['load_dense_model_file'] = args.load_dense_model_file
    if 'load_pruned_model' not in config_dict and args.load_pruned_model:
        config_dict['load_pruned_model'] = args.load_pruned_model
    if 'load_pruned_model_file' not in config_dict and args.load_pruned_model_file:
        config_dict['load_pruned_model_file'] = args.load_pruned_model_file
    # if args.batch_size:
    if 'batch_size' not in config_dict and args.batch_size:
        config_dict['batch_size'] = args.batch_size
    # if args.epochs >= 0:
    if 'epochs' not in config_dict and args.epochs >= 0:
        config_dict['epochs'] = args.epochs
    # if args.data_code:
    if 'data_code' not in config_dict and args.data_code:
        config_dict['data_code'] = args.data_code
    # if args.model:
    if 'model' not in config_dict and args.model:
        config_dict['model'] = args.model
    # if args.num_classes:
    if 'num_classes' not in config_dict and args.num_classes:
        config_dict['num_classes'] = args.num_classes
   
    # admm prune
    # if args.admm or 'admm' not in config_dict:
    if 'admm' not in config_dict and args.admm:
        config_dict['admm'] = args.admm
    # if args.admm_epochs:
    if 'admm_epochs' not in config_dict and args.admm_epochs:
        config_dict['admm_epochs'] = args.admm_epochs
    if 'create_partition' not in config_dict and args.create_partition:
        config_dict['create_partition'] = args.create_partition
    # if args.sparsity_type:
    if 'sparsity_type' not in config_dict and args.sparsity_type:
        config_dict['sparsity_type'] = args.sparsity_type
    if args.prune_ratio or args.prune_ratio == 0:
    # if 'prune_ratio' not in config_dict and args.prune_ratio:
        config_dict['prune_ratio'] = args.prune_ratio
    # if args.rho:
    if 'rho' not in config_dict and args.rho:
        config_dict['rho'] = args.rho
    # if args.multi_rho:
    if 'multi_rho' not in config_dict and args.multi_rho:
        config_dict['multi_rho'] = args.multi_rho

    # finetune
    # if args.retrain_lr:
    if 'retrain_lr' not in config_dict and args.retrain_lr:
        config_dict['retrain_lr'] = args.retrain_lr
    # if args.retrain_bs:
    if 'retrain_bs' not in config_dict and args.retrain_bs:
        config_dict['retrain_bs'] = args.retrain_bs
    # if args.retrain_ep >= 0:
    if 'retrain_ep' not in config_dict and args.retrain_ep >= 0:
        config_dict['retrain_ep'] = args.retrain_ep
    # if args.retrain_lx:
    if 'retrain_lx' not in config_dict and args.retrain_lx:
        config_dict['retrain_lx'] = args.retrain_lx
    # if args.retrain_ly:
    if 'retrain_ly' not in config_dict and args.retrain_ly:
        config_dict['retrain_ly'] = args.retrain_ly
    # if args.retrain_opt:
    if 'retrain_opt' not in config_dict and args.retrain_opt:
        config_dict['retrain_opt'] = args.retrain_opt
    # if args.retraining_type:
    if 'retraining_type' not in config_dict and args.retraining_type:
        config_dict['retraining_type'] = args.retraining_type

    ## comparison with light model train from scratch
    # if args.save_last_model_only:
    if 'save_last_model_only' not in config_dict and args.save_last_model_only:
        config_dict['save_last_model_only'] = args.save_last_model_only
    
    # if args.xentropy_weight:
    if 'xentropy_weight' not in config_dict and args.xentropy_weight:
        config_dict['xentropy_weight'] = args.xentropy_weight
        
    # tricks:
    if 'lr_scheduler' not in config_dict and args.lr_scheduler:
        config_dict['lr_scheduler'] = args.lr_scheduler
    # if args.lr_scheduler:
    #     config_dict['lr_scheduler'] = 'cosine'
    # if args.warmup or 'warmup' not in config_dict:
    if 'warmup' not in config_dict and args.warmup or 'warmup' not in config_dict:
        config_dict['warmup'] = args.warmup
    if 'warmup_lr' not in config_dict and args.warmup_lr:
         config_dict['warmup_lr'] = args.warmup_lr 
    if 'warmup_epochs' not in config_dict and args.warmup_epochs:
        config_dict['warmup_epochs'] = args.warmup_epochs 
    if 'mix_up' not in config_dict and args.mixup:
        config_dict['mix_up'] = False 
    if 'alpha' not in config_dict and args.alpha:
        config_dict['alpha'] = 0 
    if 'smooth' not in config_dict and args.smooth:
        config_dict['smooth'] = False 
    if 'smooth_eps' not in config_dict and args.smooth_eps:
        config_dict['smooth_eps'] = 0 

    config_dict['load_pruned_model_file'] = f"{config_dict['data_code']}-{config_dict['model']}-{config_dict['sparsity_type']}-np{config_dict['num_partition']}-pr{config_dict['prune_ratio']}-lcm{config_dict['lambda_comm']}.pt"
    config_dict['partition_path'] = f"config/{config_dict['model']}-np{config_dict['num_partition']}.yaml"
    config_dict['load_dense_model_file'] = f"{config_dict['data_code']}-{config_dict['model']}.pt"
    
    for key, val in config_dict.items():
        print(key, ': ', val)
        
    return config_dict
   
    
if __name__ == "__main__":
    configs = main()
    print(configs)
    mop = MoP(configs)
    if not configs['create_partition'] and not configs['load_pruned_model']:
        mop.prune()
        mop.finetune()
    elif configs['load_pruned_model']:
        # Run accuracy test
        # Run timing test
        # pruned model file is: configs['load_pruned_model_file']
        model = get_model_from_code(configs).to(configs['device'])
        state_dict = torch.load(get_model_path_split("{}".format(configs["load_pruned_model_file"])), map_location=configs['device'])
        model = load_state_dict(model, 
                                        state_dict['model_state_dict'] if 'model_state_dict' in state_dict 
                                        else state_dict['state_dict'] if 'state_dict' in state_dict else state_dict,)

        
           



    # mop.pruneMask()
    # mop.finetuneWeight()