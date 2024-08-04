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

from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

def get_list_of_modules_named(model):
    ''' Takes in a model object and returns a list of module names in order they will be executed'''
    return [module[0] for i, module in enumerate(model.named_modules())]

def get_module(model, imodule):
    ''' Takes in a model object and returns the module and module name at index imodule'''
    name, module = next((x for i,x in enumerate(model.named_modules()) if i==imodule)) 

    return name, module

def execute_layer(model, input, layer):
    '''  get the horizontal output from model at layer using input '''

    layer = get_current_module(model, layer)

    layer.eval()
    with torch.no_grad():
        return layer(input) 

def replace_all_tensors(input, val):
    ''' takes in a nested list of tensors and replaces the non-zero values of those tensors. For debugging split layer '''

    for out in input:
        for el in input:
            if torch.is_tensor(el):
                el[not (el == 0)] = val
    
    return input

def get_current_module(model, imodule):
    '''  
    gets current module using fx indexing
    - assumes imodule is not functional like relu, view, etc. and is in field of model object 
    '''

    layer_names_fx =  get_graph_node_names(model)[1]
    layer_name = layer_names_fx[imodule]

    if layer_name == 'x':
        return model 
    else:
        tmp = model
        layer_name_split = layer_name.split('.')
        for lname in layer_name_split:
            if lname.isdigit():
                tmp = tmp[int(lname)]
            else:
                tmp = getattr(tmp,lname)
        return tmp
                         

def get_layer_output(model, input_tensor, imodule):
    ''' 
        splits model horizontally and computes output up to layer = ilayer
    '''
    
    # map requested layer to layer in fx 
    eval_names = get_graph_node_names(model)[1]

    print(f'Grabbing module {eval_names[imodule]} from fx list') # debug
    get_layers = {
         eval_names[imodule] : f'layer_{imodule}'
         }
    
    # get intermediate outputs 
    extractor_model = create_feature_extractor(model,return_nodes = get_layers)
    with torch.no_grad():
        extractor_model.eval()
        intermediate_out = extractor_model(input_tensor)
        return intermediate_out[f'layer_{imodule}']

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
                    module_full.eps,
                    momentum=module_full.momentum, 
                    affine=module_full.affine, 
                    track_running_stats=module_full.track_running_stats)

    # write parameters to split layer 
    split_layer.weight = torch.nn.Parameter(module_full.weight.index_select(0, input_channels))
    split_layer.running_mean = torch.nn.Parameter(module_full.running_mean.index_select(0, input_channels))
    split_layer.running_var = torch.nn.Parameter(module_full.running_var.index_select(0, input_channels))

    if not split_layer.bias == None:
            split_layer.bias = torch.nn.Parameter(module_full.bias.index_select(0, input_channels))
    return split_layer

def split_linear_layer(module_full, input_channels):
    '''Takes a full linear module and splits it based on input_channels

        Inputs:
            - module_full (torch Linear layer)
            - input_channels (tensor): tensor listing the input channel for the split module/layer 
    
    '''
    N_in = len(input_channels)
    split_layer = nn.Linear(N_in, 
        module_full.weight.shape[0], 
        bias=False)

    # write parameters to split layer 
    split_layer.weight = torch.nn.Parameter(module_full.weight.index_select(1, input_channels))

    # TODO: current implementation assumes Linear is final layer and bias is handled separately 
    #if not split_layer.bias == None:
    #       split_layer.bias = module_full.bias
    
    return split_layer

def get_residual_block_indexes(model):
    ''' get indexes of important points in model execution for residual blocks/shortcut connections.
        Specifically: residual block, start of residual connection, and final layer in residual block
    '''

    residual_block_start = np.array([])
    residual_connection_start = np.array([])
    residual_block_end = np.array([])
    layer_names = get_graph_node_names(model)[1]
    
    block_num = '-1'
    imodule = 0
    for name in layer_names:

        # unpack layer name
        tmp = name.split('.')
        if len(tmp) == 1:
           # assume belongs to no large layer block e.g. start of the model 
           tmp_layer_type = tmp[0]
           tmp_block_num = '-1'
        else:
            # unpack 
            tmp_large_layer = tmp[0]
            tmp_block_num = tmp[1]
            tmp_layer_type = tmp[2]
        
        # detect when a new block is entered and save the index 
        if not (tmp_block_num == block_num):
            block_num = tmp_block_num
            residual_block_start = np.append(residual_block_start, imodule)

        # detect first shortcut layer
        if 'shortcut.0' in name:
            residual_connection_start = np.append(residual_connection_start, imodule)

        # detect residual summing 
        if '.add' in name:
            residual_block_end = np.append(residual_block_end, imodule)

        imodule += 1
    return residual_block_start, residual_connection_start, residual_block_end

def get_nonzero_channels(atensor, dim=1):
    return torch.unique(torch.nonzero(atensor, as_tuple=True)[dim]) 

def  compare_tensors(t1, t2, dim=1, rshape=(1,64,-1)):
    diff = torch.abs(t1-t2)

    max_diff_pin_dim = torch.max(diff.reshape(rshape), dim)
    return max_diff_pin_dim[0]

def get_output_at_each_layer(model, input_tensor):
    ''' gets the true model output for all layers in the model '''

    layer_names_fx = get_graph_node_names(model)[1]
    total_layers_fx = len(layer_names_fx)

    # get list of intermediate outputs 
    get_horz_out = {}
    for aname in layer_names_fx:
        get_horz_out[aname] = aname

    extractor_model = create_feature_extractor(model,return_nodes = get_horz_out)
    with torch.no_grad():
        extractor_model.eval()
        horz_output = extractor_model(input_tensor)
    
    size_LUT = {}
    index = 0
    for out_name in horz_output:
        if torch.is_tensor(horz_output[out_name]):
            size_LUT[out_name] = horz_output[out_name].shape
        else:
            size_LUT[out_name] =  [None]
        index += 1

    return horz_output, size_LUT

def compare_outputs(full_output, horz_output):
    ''' 
        Used for debugging output differences b/w full and split models 
    '''
    diff_output = torch.abs(horz_output - full_output)

    N_batch = horz_output.shape[0]

    print('Max diff:')
    max_diff= torch.max(torch.reshape(diff_output, (N_batch, -1)), dim=1)[0]
    print(max_diff)
    #plt.hist(diff_output.reshape((-1,)))
    #plt.show()

    max_by_Cout = torch.max(torch.abs(diff_output.reshape((1,full_output.shape[1],-1))), dim=2)

    print()
    print(max_by_Cout[0])
    print(get_nonzero_channels(max_by_Cout[0]))


    # get C_out with zero and non-zero diff
    nonzero_Cout = get_nonzero_channels(horz_output)
    failing_Cout = nonzero_Cout[torch.isin(nonzero_Cout, get_nonzero_channels(max_by_Cout[0]))]
    passing_Cout = nonzero_Cout[torch.isin(nonzero_Cout, get_nonzero_channels(max_by_Cout[0])) == False]
    print() 
    print(f'failing Cout = {failing_Cout}  (len = {len(failing_Cout)})')
    print(f'passing Cout = {passing_Cout}  (len = {len(passing_Cout)})')

    return max_diff.item(), max_by_Cout

def combine_inputs(input_struct, num_machines, imach):
    ''' helper to gather input tensors and add them  '''
    # combine inputs from machines
    curr_input = False 
    rx_count = 0
    for i in range(num_machines):
        if not input_struct[imach][i] == None:
            if not torch.is_tensor(curr_input):
                curr_input = input_struct[imach][i] # initialize curr_input with first input tensor 
            else:
                curr_input = curr_input + input_struct[imach][i]
                rx_count += 1
    return curr_input

def combine_all_inputs(input_struct, num_machines):
    ''' helper to gather input tensors and add them  '''
    # combine inputs from machines
    combined_input = False 

    for i in range(num_machines):
        for j in range(num_machines):
            if not input_struct[j][i] == None:
                if not torch.is_tensor(combined_input):
                    combined_input = input_struct[j][i] # initialize curr_input with first input tensor 
                else:
                    combined_input = combined_input + input_struct[j][i]
                    
    return combined_input

def main():
    print()

if __name__ == "__main__":
    main()