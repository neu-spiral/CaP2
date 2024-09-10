from source.utils import io

from ..utils.misc import get_model_from_code
from ..utils.split_network import *

import torch
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

import functools
import operator

class SplitManager:
    """
    Each thread runs a SplitManager object to handle I/O for it's portion of the vertically split model 
    TODO:
    - load only necessary split layers (either on CPU or GPU) for this thread before executing anything. Replace "get_current_module" on line ~112 and remove layer splitting 
    - change implementation to handle and use smallest tensor possible no matter bn or conv (currently bn output with be a tensor with dim=1 be larger than the required C_out for the split layer)
    - find another way to create size_LUT and remove input_tensor and get_output_at_each_layer from constructor 
    """

    def __init__(self, configs, machine, N_machines, input_tensor, debug=False):
        
        torch.manual_seed(configs['seed'])
        self.configs = configs
        self.model_file = configs['model_file']

        # misc
        self.machine = machine
        self.N_machines = N_machines # number of total machines executing 
        self.debug = debug 
        if 'dtype' in configs:
            if configs['dtype'] == 'float64':
                self.dtype = torch.float64
            elif configs['dtype'] == 'float32':
                self.dtype = torch.float32
            else:
                print('Unsupported dtype')
        else:
            print('Warning found no dtype field in config')
        self.output_channel_map = None

        # load model TODO: only load what is necessary for this thread
        model = get_model_from_code(configs).to(configs['device']) # grabs model architecture from ./source/models/escnet.py
        state_dict = torch.load(io.get_model_path_split("{}".format(configs["load_model"])), map_location=configs['device'])
        self.model = io.load_state_dict(model, 
                    state_dict['model_state_dict'] if 'model_state_dict' in state_dict 
                    else state_dict['state_dict'] if 'state_dict' in state_dict else state_dict,)
        
        model = model.type(self.dtype)
        self.model.eval()

        # residual connections/block related
        self.layer_names_fx =  get_graph_node_names(model)[1]
        self.total_layers_fx = len(self.layer_names_fx)
        self.residual_input = {}
        self.residual_block_start, self.residual_connection_start, self.residual_block_end = get_residual_block_indexes(self.model)

        # model logic
        self.split_module_names = list(self.configs['partition'].keys())

        # debugging/ verification
        self.horz_output, self.size_LUT = get_output_at_each_layer(model, input_tensor) # TODO: find a way to get size_LUT without executing on every layer
    
    def execute_split_layer(self, curr_input, imodule):
        '''   
            This function implements machine v executing layer l and returns the split output for communication 

            return output_tensor, do_comms
                output_tensor - (tensor) split output from layer=imodule from input curr_input, 
                                -1 if curr_input is empty, machine is not required to compute 
                                output for this layer, or if unrecognized layer type
                do_comms - (bool) whether or not the machine needs to communcate output 
        '''

        with torch.no_grad():

            # skip this machine+module if there is no input to compute 
            # print('curr input shape:', len(curr_input))
            if not torch.is_tensor(curr_input):
                # if not torch.is_tensor(curr_input[0]):
                if 'bn' in self.layer_names_fx[imodule]:
                    print('\t\t-No input received but bn still needs to produce output.')
                    curr_input = torch.zeros(self.size_LUT[self.layer_names_fx[imodule]], dtype=self.dtype,  device=torch.device(self.configs['device']))
                else:
                    print('\t\t-No input sent to this machine. Skipping module')
                    return -1, False
            
            # debug
            if self.debug:
                print(f'\t\t received input channels {get_nonzero_channels(curr_input)}')
            
            # non-comms operations 
            if 'relu' in self.layer_names_fx[imodule]:
                # just relu no comm necessary 
                print('\t\t-Applying ReLU')
                return F.relu(curr_input), False
            
            elif 'dropout' in self.layer_names_fx[imodule]:
                # just dropout no comm necessary 
                print('\t\t-Applying Dropout')
                return F.dropout(curr_input, p=0.3, training=True), False
                
            elif 'add' in self.layer_names_fx[imodule]:
                # residual layer. No comm necessary 
                if self.machine in self.residual_input:
                    print('\t\t-adding residual')
                    # print(f'self.residual_input: {self.residual_input}')
                    if 'block_out' in self.residual_input[self.machine]:
                        curr_input = curr_input + self.residual_input[self.machine]['block_out']
                    elif 'block_in' in self.residual_input[self.machine]:
                        curr_input = curr_input + self.residual_input[self.machine]['block_in']
                        print('\t\t-assuming shortcut had no layers')
                else:
                    print(f'\t\t-assuming this machine did not rx any input at the beginning of this block. No residual found')

                # erase stored 
                self.residual_input[self.machine] = {}

                return curr_input, False
            
            # elif 'avg_pool1' in self.layer_names_fx[imodule]:
            #     print('\t\t-average pooling')
            #     kern = self.model.avg_pool1.kernel_size
            #     return F.avg_pool2d(curr_input, kern), False
            
            # elif 'avg_pool2' in self.layer_names_fx[imodule]:
            #     print('\t\t-average pooling')
            #     kern = self.model.avg_pool2.kernel_size
            #     return F.avg_pool2d(curr_input, kern), False
            
            elif 'avg_pool' in self.layer_names_fx[imodule]:
                print('\t\t-average pooling')
                layer_path = self.layer_names_fx[imodule]
                modules = layer_path.split('.')
                layer_name = functools.reduce(getattr, modules, self.model)
                kern = layer_name.kernel_size
                return F.avg_pool2d(curr_input, kern), False
            
            # elif 'max_pool1' in self.layer_names_fx[imodule]:
            #     print(self.layer_names_fx[imodule])
            #     print('\t\t-max pooling')
            #     kern = self.model.self.layer_names_fx[imodule].kernel_size
            #     return F.max_pool2d(curr_input, kern), False
            
            # elif 'max_pool2' in self.layer_names_fx[imodule]:
            #     print('\t\t-max pooling')
            #     kern = self.model.max_pool2.kernel_size
            #     return F.max_pool2d(curr_input, kern), False
            
            elif 'max_pool' in self.layer_names_fx[imodule]:
                print('\t\t-max pooling')
                layer_path = self.layer_names_fx[imodule]
                modules = layer_path.split('.')
                layer_name = functools.reduce(getattr, modules, self.model)
                kern = layer_name.kernel_size
                return F.max_pool2d(curr_input, kern), False
            
            # elif 'pool2' in self.layer_names_fx[imodule]:
            #     print('\t\t-average pooling')
            #     return F.avg_pool2d(curr_input, 2), False
            
            # elif 'pool' in self.layer_names_fx[imodule]:
            #     print('\t\t-max pooling')
            #     return F.max_pool2d(curr_input, 2), False
            
            elif 'size' in self.layer_names_fx[imodule]:
                print('\t\t-skipping')
                return curr_input, False
            
            elif 'view' in self.layer_names_fx[imodule]:
                print('\t\t-reshaping (view)')
                return curr_input.view(curr_input.size(0), -1), False
            
            # elif 'getitem' in self.layer_names_fx[imodule]:
            #     print('\t\t-getitem')
            #     return curr_input[0], False
            
            # elif 'getitem_1' in self.layer_names_fx[imodule]:
            #     print('\t\t-getitem_1')
            #     return curr_input[1], False
            
            # elif 'getitem_2' in self.layer_names_fx[imodule]:
            #     print('\t\t-getitem_2')
            #     return curr_input[2], False
            
            # elif 'getitem_3' in self.layer_names_fx[imodule]:
            #     print('\t\t-getitem_3')
            #     return curr_input[3], False
            
            # elif 'getitem_4' in self.layer_names_fx[imodule]:
            #     print('\t\t-getitem_4')
            #     return curr_input[4], False
            
            # elif 'cat' in self.layer_names_fx[imodule]:
            #     print('\t\t-concatenating')
            #     return torch.cat(curr_input, 1), False
                
            elif 'x' == self.layer_names_fx[imodule]:
            # elif 'x' == self.layer_names_fx[imodule] or '_x' == self.layer_names_fx[imodule] or 'getitem' == self.layer_names_fx[imodule] or 'getitem_1' == self.layer_names_fx[imodule] or 'getitem_2' == self.layer_names_fx[imodule] or 'getitem_3' == self.layer_names_fx[imodule] or 'getitem_4' == self.layer_names_fx[imodule] or 'cat' == self.layer_names_fx[imodule]:
                # do nothing if model input
                print('\t\t-model input layer.. skipping')
                return curr_input, False
            
            # swap out io for residual connection
            if imodule in self.residual_block_start:
                # save input for later 
                self.residual_input[self.machine] = {}
                self.residual_input[self.machine]['block_in'] = curr_input.detach().clone()
                print('\t\t-Saving input for later...')
            elif imodule in self.residual_connection_start:
                # swap tensors
                self.residual_input[self.machine]['block_out'] = curr_input
                curr_input = self.residual_input[self.machine]['block_in'] 
                print('\t\t-Saving current input. Swapping for input saved from start of block')

            # get the current module
            # TODO: ideally we save the split layers beforehand and have them preloaded ready to be called
            curr_layer = get_current_module(self.model, imodule)

            # update communication I/O for this layer  
            # TODO: prep this before running execution and give this it's own method
            split_param_name = self.layer_names_fx[imodule] + '.weight'
            # print(f'split_param_name: {split_param_name}')
            # print(f'self.split_module_names: {self.split_module_names}')
            if type(curr_layer) == nn.Linear and imodule == self.total_layers_fx-1:
                # if final layer output all goes to machine 0 
                # TODO: find better way to handle this. Also will we encounter Linear layers not at the end of the model
                N_Cin = curr_layer.in_features
                Cin_per_machine = N_Cin/self.N_machines
                if Cin_per_machine % 1 > 0:
                        print('ERROR: UNEXPECTED NUMBER OF I/O FOR LINEAR MODULE {imodule}')
                Cin_per_machine = int(Cin_per_machine)
                input_channels = np.arange(Cin_per_machine) + self.machine*Cin_per_machine
                N_Cout = curr_layer.out_features 
                self.output_channel_map = [None]*self.N_machines
                for i in range(self.N_machines):
                        if i == 0:
                            # send all outputs to machine 0 for final layer
                            # TODO: revisit this implementation choice
                            self.output_channel_map[i] = np.arange(N_Cout) 
                        else:
                            self.output_channel_map[i] = np.array([])
                input_channels = torch.tensor(input_channels, device=torch.device(self.configs['device']))
            elif split_param_name in self.split_module_names:
                # skip if machine doesnt expect input
                if len(self.configs['partition'][split_param_name]['channel_id'][self.machine]) == 0:
                        print(f'\t\t-WARNING: No input assigned to this machine (but it was sent input?). Skipping...')
                        return -1, False

                # TODO: reconsider implementation 
                # What input channels does this machine compute?
                input_channels = torch.tensor(self.configs['partition'][split_param_name]['channel_id'][self.machine],
                        device=torch.device(self.configs['device']))
                N_in = len(input_channels) # TODO: is this used?

                # Where to send output (map of output channels to different machines)
                self.output_channel_map = self.configs['partition'][split_param_name]['filter_id']
            else:
                # for batch normal, and functional passes through the code
                # TODO: address the following assumptions:
                #       - assume all BN layers have C_in divisable by self.N_machines
                #       - assume C_in are evenly split in sequential order WARNING THIS WILL BREAK WHEN WE START TO DO ASSIGN WEIGHTS TO DIFF MACHINES
                N_Cin = curr_layer.num_features
                Cin_per_machine = N_Cin/self.N_machines
                if Cin_per_machine % 1 > 0:
                        print('ERROR: UNEXPECTED NUMBER OF I/O FOR BATCH NORMAL MODULE {imodule}')
                Cin_per_machine = int(Cin_per_machine)
                input_channels = np.arange(Cin_per_machine) + self.machine*Cin_per_machine
                self.output_channel_map = [None]*self.N_machines
                for i in range(self.N_machines):
                        if i == self.machine:
                                self.output_channel_map[i] = input_channels
                        else:
                                self.output_channel_map[i] = np.array([])
                input_channels = torch.tensor(input_channels, device=torch.device(self.configs['device']))

            # make vertically split layer. TODO: remove this and replace curr_layer to be split_layer when first made 
            # print(f'current layer type: {type(curr_layer)}')
            if type(curr_layer) == nn.Conv2d:
                print(f'\t\t-Splitting conv layer {imodule}')
                split_layer = split_conv_layer(curr_layer, input_channels)
            elif type(curr_layer) == nn.BatchNorm2d:
                print(f'\t\t-Splitting batch norm layer {imodule}')
                split_layer = split_bn_layer(curr_layer, input_channels)
            elif type(curr_layer) == nn.Linear:
                print(f'\t\t-Splitting linear layer {imodule}')
                split_layer = split_linear_layer(curr_layer, input_channels)
            else:
                print(f'\t\t-Skipping module {type(curr_layer).__name__}')
                return -1, False
            
            # eval split
            split_layer.eval()
            out_tensor = split_layer(curr_input.index_select(1, input_channels))
            if type(curr_layer) == nn.BatchNorm2d:
                # place bn output in lager tensor to maintain standardized output size
                # TODO: change implementation to handle and use smallest tensor possible no matter bn or conv
                tmp_out_tensor = torch.zeros(curr_input.shape, dtype=self.dtype)
                tmp_out_tensor[:,input_channels.numpy(),:,:] = out_tensor
                out_tensor = tmp_out_tensor
                
            return out_tensor, True
            
