'''
    Load RESNET model
    Vertically split model 
    Execute split model
    See block diagram : [TODO: add ref]

    Functions :
        1. Get list of layers/modules from model
        2. Get module i from model
        3. Conv splitting
        4. BN splitting
        5. Linear splitting
        6. Get indexes of start of residual block, start of residual connection, and final layer in residual block
        7. Get indexs of relu and avg pooling layers
        8. Get input machines 
        9. Get output machines

    TODO:
        - add bias split to conv layer 
        - test 6
        - add 8,9
        - add main
'''

from source.core.engine import MoP
from source.core import run_partition as run_p
from os import environ
from source.utils.dataset import *
from source.utils.misc import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from source.models import resnet

import torch.nn.functional as F

import numpy as np

from source.utils import io
from source.utils import testers
from source.core import engine
import json
import itertools

from torchsummary import summary

import time


def get_list_of_modules(model):
    ''' Takes in a model object and returns a list of module names in order they will be executed'''
    return [module[0] for i, module in enumerate(model.named_modules())]

def get_module(model, imodule):
    ''' Takes in a model object and returns the module and module name at index imodule'''
    name, module = next((x for i,x in enumerate(model.named_modules()) if i==imodule)) 

    return name, module

def split_conv_layer(module_full, input_channels):
    '''Takes a full convolution module and splits it based on input_channels

        Inputs:
            - module_full (torch convolutional layer)
            - input_channels (tensor): tensor listing the input channel for the split module/layer 
    
    '''
    N_in = len(input_channels)
    split_layer = nn.Conv2d(N_in,
                    module_full.weight.shape[0], # TODO does this need to be an int? (currently tensor)
                    kernel_size= module_full.kernel_size,
                    stride=module_full.stride,
                    padding=module_full.padding, 
                    bias=False) # TODO: add bias during input collecting step on next layer 

    # write parameters to split layer 
    split_layer.weight = torch.nn.Parameter(module_full.weight.index_select(1, input_channels))

    # TODO: add support for splitting bias

    return split_layer

def split_bn_layer(module_full, input_channels):
    '''Takes a full batch normal module and splits it based on input_channels

        Inputs:
            - module_full (torch bn layer)
            - input_channels (tensor): tensor listing the input channel for the split module/layer 
    
    '''
    N_in = len(input_channels)
    split_layer = nn.BatchNorm2d(N_in, 
                    split_layer.eps,
                    momentum=split_layer.momentum, 
                    affine=split_layer.affine, 
                    track_running_stats=split_layer.track_running_stats)

    # write parameters to split layer 
    split_layer.weight = torch.nn.Parameter(split_layer.weight.index_select(0, input_channels))
    split_layer.running_mean = torch.nn.Parameter(split_layer.running_mean.index_select(0, input_channels))
    split_layer.running_var = torch.nn.Parameter(split_layer.running_var.index_select(0, input_channels))

    if not split_layer.bias == None:
            split_layer.bias = torch.nn.Parameter(split_layer.bias.index_select(0, input_channels))

def split_linear_layer(module_full, input_channels):
    '''Takes a full linear module and splits it based on input_channels

        Inputs:
            - module_full (torch Linear layer)
            - input_channels (tensor): tensor listing the input channel for the split module/layer 
    
    '''
    N_in = len(input_channels)
    split_layer = nn.Linear(N_in, 
                    split_layer.weight.shape[0])

    # write parameters to split layer 
    split_layer.weight = torch.nn.Parameter(split_layer.weight.index_select(1, input_channels))

    # TODO: double check bias is applied correctly
    if not split_layer.bias == None:
            split_layer.bias = split_layer.bias

def get_residual_block_indexes(model):
    ''' get indexes of important points in model execution for residual blocks/shortcut connections.
        Specifically: residual block, start of residual connection, and final layer in residual block
    '''

    residual_block_start = []
    residual_connection_start = []
    residual_block_end = []
    layer_names = get_list_of_modules(model)
    
    for imodule,pair in enumerate(model.named_modules()):
        curr_module = pair[1]
        name = pair[0]
        if type(curr_module) == resnet.BasicBlock:
                if hasattr(curr_module, 'shortcut') and (len(curr_module.shortcut) > 0):
                    residual_block_start += [imodule+1]
        elif len(name) >= 8 and name[-8:] == 'shortcut':
            N_modules = len(curr_module)
            if N_modules > 0:
                residual_connection_start += [imodule+1]
                residual_block_end += [imodule+N_modules]
    
    return residual_block_start, residual_connection_start, residual_block_end

def get_functional_layers(model_name):
    ''' get layers that directly preceed relu and avg pooling
        layers called out below are expected to be: conv, bn, or linear    
    '''
     
    if model_name == 'resnet18':
        relu_layers = [2,7,8,13,14,20,24,28,29,35,39,43,44,50,54,58,59] 
        avg_pool_layers = [59]
        return relu_layers,avg_pool_layers

def main():
    # setup config
    dataset='cifar10'
    environ["config"] = f"config/{dataset}.yaml"

    configs = run_p.main()

    configs["device"] = "cpu"
    configs['load_model'] = "cifar10-resnet18-kernel-npv2-pr0.75-lcm0.001.pt"
    configs["num_partition"] = '4' #'resnet18-v2.yaml'

    # load full model
    model = get_model_from_code(configs).to(configs['device']) # grabs model architecture from ./source/models/escnet.py

    # get parameters for split model execution
    relu_layers,avg_pool_layers = get_functional_layers(configs['model'])
    residual_block_start, residual_connection_start, residual_block_end = get_residual_block_indexes(model)
    layer_names = get_list_of_modules(model)
    N_layers = len(layer_names)
    N_machines = int(configs["num_partition"])

    # init structures for comms and residual I/O storage
    residual_input = {} # use this to keep track of inputs stored in machine memory for residule layers
    add_bias = False # add bias for previous conv layer 

    # make inference 
    with torch.no_grad():
        # iterate through layers 1 module at a time 
        for imodule in range(N_layers): # 16 <=> layer_1 block 

                if imodule in [0]:
                        continue

                # initialize output for ilayer
                #output = np.empty((N_machines, N_machines), dtype=torch.Tensor) # square list indexed as: output[destination/RX machine][origin/TX machine]
                # TODO: find a better datastructure for this 
                output = [None]*N_machines
                output = [output[:] for i in range(N_machines)]
                
                send_module_outputs = True

                print(f'Executing module {imodule}: {layer_names[imodule]}')

                # iterate through each machine (done in parallel later)
                for imach in range(N_machines):
                        print(f'\tExecuting on machine {imach}')
                        
                        add_residual = False

                        # combine inputs from machines
                        curr_input = False 
                        rx_count = 0
                        for i in range(N_machines):
                                if not input[imach][i] == None:
                                        if not torch.is_tensor(curr_input):
                                                curr_input = input[imach][i] # initialize curr_input with first input tensor 
                                        else:
                                                curr_input += input[imach][i]
                                        rx_count += 1
                        if add_bias:
                                # TODO: check if this works (this is not required for resnet18 because no bias on conv layers)
                                dummy, prev_module = next((x for i,x in enumerate(model.named_modules()) if i==imodule-1))
                                bias = prev_module.bias 
                                curr_input += bias/rx_count

                        # skip this machine+module if there is no input to compute 
                        if not torch.is_tensor(curr_input):
                                print('\t\t-No input sent to this machine. Skipping module')
                                continue

                        # debug
                        print(f'\t\t received input channels {get_nonzero_channels(curr_input)}')

                        # get the current module
                        # TODO: this is very bad for latency. Only load module if you have to 
                        curr_name, curr_module = next((x for i,x in enumerate(model.named_modules()) if i==imodule)) 

                        # update communication I/O for this layer  
                        # TODO: revist this implementation
                        split_param_name = curr_name + '.weight'
                        if split_param_name in split_layer_names:

                                # skip if machine doesnt expect input
                                if len(configs['partition'][split_param_name]['channel_id'][imach]) == 0:
                                        print(f'\t\t-No input assigned to this machine. Skipping...')
                                        continue
                                
                                # TODO: reconsider implementation 
                                # What input channels does this machine compute?
                                input_channels = torch.tensor(configs['partition'][split_param_name]['channel_id'][imach],
                                        device=torch.device(configs['device']))
                                N_in = len(input_channels) # TODO: is this used?

                                # Where to send output (map of output channels to different machines)
                                output_channel_map = configs['partition'][split_param_name]['filter_id']
                        elif type(curr_module) == nn.BatchNorm2d:
                                # TODO: address the following assumptions:
                                #       - assume all BN layers have C_in divisable by N_machines
                                #       - assume C_in are evenly split in sequential order WARNING THIS WILL BREAK WHEN WE START TO DO ASSIGN WEIGHTS TO DIFF MACHINES
                                N_Cin = curr_module.num_features
                                Cin_per_machine = N_Cin/N_machines
                                if Cin_per_machine % 1 > 0:
                                        print('ERROR: UNEXPECTED NUMBER OF I/O FOR BATCH NORMAL MODULE {imodule}')
                                Cin_per_machine = int(Cin_per_machine)
                                input_channels = np.arange(Cin_per_machine) + imach*Cin_per_machine
                                output_channel_map = [None]*N_machines
                                for i in range(N_machines):
                                        if i == imach:
                                                output_channel_map[i] = input_channels
                                        else:
                                                output_channel_map[i] = np.array([])
                                input_channels = torch.tensor(input_channels, device=torch.device(configs['device']))

                        # reduce computation-- make vertically split layer 
                        # TODO: generalize this to more than conv layers 
                        if type(curr_module) == nn.Conv2d:
                                split_layer = nn.Conv2d(N_in,
                                                curr_module.weight.shape[0], # TODO does this need to be an int? (currently tensor)
                                                kernel_size= curr_module.kernel_size,
                                                stride=curr_module.stride,
                                                padding=curr_module.padding, 
                                                bias=False) # TODO: add bias during input collecting step on next layer 

                                # write parameters to split layer 
                                split_layer.weight = torch.nn.Parameter(curr_module.weight.index_select(1, input_channels))

                                # TODO: add support for splitting bias

                                # if this is the start of the residual layer
                                # 1. store output from main stream model path
                                # 2. grab stored tensor from beginning of block
                                if 'shortcut' in curr_name:
                                        residual_input[str(imach)]['block_out'] = curr_input
                                        curr_input = residual_input[str(imach)]['block_in']

                        elif type(curr_module) == nn.BatchNorm2d:
                                split_layer = nn.BatchNorm2d(N_in, 
                                                curr_module.eps,
                                                momentum=curr_module.momentum, 
                                                affine=curr_module.affine, 
                                                track_running_stats=curr_module.track_running_stats)

                                # write parameters to split layer 
                                split_layer.weight = torch.nn.Parameter(curr_module.weight.index_select(0, input_channels))
                                split_layer.running_mean = torch.nn.Parameter(curr_module.running_mean.index_select(0, input_channels))
                                split_layer.running_var = torch.nn.Parameter(curr_module.running_var.index_select(0, input_channels))

                                if not curr_module.bias == None:
                                        split_layer.bias = torch.nn.Parameter(curr_module.bias.index_select(0, input_channels))
                                

                                # TODO: revise implementation to only compute necessary C_in to C_out 
                                # assume mach-Cout map from previous conv layer can be used as inputs for this bn layer
                                #input_channels = output_channel_map[imach]

                        elif type(curr_module) == nn.Linear:
                                # TODO: assumes there is a bias 
                                split_layer = nn.Linear(N_in, 
                                                curr_module.weight.shape[0])

                                # write parameters to split layer 
                                split_layer.weight = torch.nn.Parameter(curr_module.weight.index_select(1, input_channels))

                                # TODO: double check bias is applied correctly
                                if not curr_module.bias == None:
                                        split_layer.bias = curr_module.bias

                                # prep for linear layer
                                # TODO: assumes this always happens before linear layer 
                                # bn takes one in channel C_in_i and produces one out channel C_out_j. No communication is needed. 
                                curr_input = F.avg_pool2d(curr_input, 4)
                                curr_input = curr_input.view(curr_input.size(0), -1)

                        elif type(curr_module) == resnet.BasicBlock:
                                # save input for later 
                                residual_input[str(imach)] = {}
                                residual_input[str(imach)]['block_in'] = curr_input
                                print('\t\t-Saving input for later...')
                                send_module_outputs = False
                                continue
                        else:
                                print(f'\t\t-Skipping module {type(curr_module).__name__}')
                                send_module_outputs = False
                                continue
                        
                        # make sure layer is in eval mode
                        # TODO: if you set model.eval() can we skip this, also only required for bn layers? Maybe 
                        split_layer.eval()

                        # eval split
                        out_tensor = split_layer(curr_input.index_select(1, input_channels))
                        if type(curr_module) == nn.BatchNorm2d:
                                tmp_out_tensor = torch.zeros(curr_input.shape)
                                tmp_out_tensor[:,input_channels.numpy(),:,:] = out_tensor
                                out_tensor = tmp_out_tensor

                        # debug
                        nonzero_out_tensor = torch.unique(torch.nonzero(out_tensor, as_tuple=True)[1])


                        # check if this is residual layer
                        if add_residual:
                                print('\t\t-adding residual')
                                out_tensor += residual_input[str(imach)]['block_out']

                                # erase stored 
                                residual_input[str(imach)] = {}

                        # apply ReLU after batch layers
                        if imodule in relu_modules:
                                print('\t\t-Applying ReLU')
                                out_tensor = F.relu(out_tensor)

                        # look at which C_out need to be computed and sent
                        #nonzero_Cout = torch.unique(torch.nonzero(split_layer.weight, as_tuple=True)[0]) # find nonzero dimensions in output channels
                        nonzero_Cout = get_nonzero_channels(out_tensor)

                        # communicate
                        out_channel_array = torch.arange(out_tensor.shape[1])
                        for rx_mach in range(N_machines):
                                # only add to output if communication is necessary 

                                # Get output channels for current rx machine? TODO: consider removing, this just maps C_out's to machine
                                output_channels = torch.tensor(output_channel_map[rx_mach],
                                        device=torch.device(configs['device']))

                                # TODO: is there a faster way to do this? Consider putting larger array 1st... just not sure which one that'd be
                                nonzero_out_channels = nonzero_Cout[torch.isin(nonzero_Cout, output_channels)]
                                if nonzero_out_channels.nelement() > 0:
                                        communication_mask = torch.isin(out_channel_array, nonzero_out_channels)

                                        # TODO: this is inefficient, redo. Probbably need to send a tensor and some info what output channels are being sent
                                        tmp_out = torch.zeros(out_tensor.shape) 
                                        tmp_out[:,communication_mask,:,:] = out_tensor[:,communication_mask,:,:]
                                        output[rx_mach][imach] = tmp_out

                                        # debug
                                        print(f'\t\t sending C_out {nonzero_out_channels} to machine {rx_mach}')

                # send to next layer  
                if send_module_outputs:      
                        input = output
                print(f'Finished execution of layer {imodule}')
                print()

        # collect outputs -- assumes ends with Linear layer. Not sure how generalizable this is
        # if loop stops on module that doesnt calculate anything use input struct 
        if send_module_outputs:
                tmp_output = output
        else:
                tmp_output = input 
        need_to_init  = True
        for rx_mach in range(N_machines):
                for tx_mach in range(N_machines):
                        if not tmp_output[rx_mach][tx_mach] == None:
                                if need_to_init:
                                        final_output = tmp_output[rx_mach][tx_mach]
                                        need_to_init = False
                                else:
                                        # TODO: += causes assignment issues, switched to x = x+y which might be more more inefficent memory wise ... 
                                        final_output = final_output + tmp_output[rx_mach][tx_mach] 
                                        nz_channels = get_nonzero_channels(final_output)
                                        #print(f'({rx_mach},{tx_mach}) {nz_channels}')

        print()

if __name__ == "__main__":
        main()