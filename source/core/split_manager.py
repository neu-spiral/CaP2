from source.utils import io

from ..utils.misc import get_model_from_code
from ..utils.split_network import *

import torch
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

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
        self.device = configs['device']

        # load model TODO: only load what is necessary for this thread
        model = get_model_from_code(configs).to(configs['device']) # grabs model architecture from ./source/models/escnet.py
        state_dict = torch.load(io.get_model_path("{}".format(configs["load_model"])), map_location=configs['device'])
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
        self.current_layer = self.get_machine_starting_layer() # layer that needs to be executed (current_tensor is always the input to this layer), corresponds to index in self.layer_names_fx
        # TODO: add logic for detecting layer a machine should start at 

        # debugging/ verification
        self.horz_output, self.size_LUT = get_output_at_each_layer(model, input_tensor) # TODO: find a way to get size_LUT without executing on every layer

        self.current_tensor = torch.zeros(self.size_LUT[self.layer_names_fx[self.current_layer]], dtype=self.dtype, device=self.device) # keep track of tensor I/O on this node. This is the size of the full tensor input for layer = current_layer


    def get_machine_starting_layer(self):
        ''' 
            determine where to start calucation from
        '''

        for i in range(self.total_layers_fx):
            layer_name = self.layer_names_fx[i] + '.weight'

            if layer_name in self.split_module_names:
                if len(self.configs['partition'][layer_name]['channel_id'][self.machine]) > 0:
                    return i

        return -1

    def execute_layers_until_comms(self):
        ''' 
            Executes split, starting at the current layer, until reaching a layer that requires input for communication
        '''
        
        split_layer_output, do_comm = self.execute_split_layer(self.current_tensor, self.current_layer) # execute split layer
        while not do_comm:
            self.update_current_tensor(split_layer_output)
            split_layer_output, do_comm = self.execute_split_layer(self.current_tensor, self.current_layer) # execute split layer

        return split_layer_output

    def update_current_tensor(self, out_tensor):
        ''' 
            Updates the localy stored current_tensor with out_tensor and increments current_layer
        '''

        # debug
        #nonzero_out_tensor = torch.unique(torch.nonzero(out_tensor, as_tuple=True)[1])

        # prep mask for this node
        output_channels = torch.tensor(self.output_channel_map[self.machine],
                device=self.device)

        # TODO: this is inefficient, redo. Probbably need to send a tensor and some info what output channels are being sent 
        tmp = torch.zeros(out_tensor.size, device=self.device, dtype=self.dtype)
        if self.current_layer == self.total_layers_fx-1:
                tmp = tmp + out_tensor[:,output_channels]
        else:
                tmp = tmp + out_tensor[:,output_channels,:,:]

        # debug
        print(f'\t\t Updating local tensor: C_out {output_channels} on machine {self.machine}')
        self.current_tensor = tmp
        self.current_layer += 1 # update layer to execute  


    def prep_output(self, out_tensor):
        ''' 
            Take output tensor from the split layer output and prepare outputs for sending.

            Input:
                out_tensor - (tensor) output from a split layer of dimension batch size, output channels (equal to the full output size for this layer)
                convolution height and width = [# batch, Cout, H, W]

            Output:
                all_output (list of dicts)
                    layer - (int) the layer this output was generated from 
                    is_empty - (bool) indicates this layer produces nothing for the (tnesor and Cin_map are absent from dict in this case)
                    node - (int) the network node this input was generated from 
                    node_to - (int) node that this output should be sent to 
                    tensor - (tensor) the input tensor with dimesnions batch size, input channels (for this layer
                            which may be a subset of the total Cin this node expects) convolution height 
                            and width = [# batch, Cin', H, W]
                    Cin_map - (1 x Cin' list) maps Cin' dimension to dimension in Cin of full input to this layer
        '''
        print(f'\t\t Output tensor shape : {out_tensor.shape}')

        # debug
        #nonzero_out_tensor = torch.unique(torch.nonzero(out_tensor, as_tuple=True)[1])

        # look at which C_out need to be computed and sent
        #nonzero_Cout = torch.unique(torch.nonzero(split_layer.weight, as_tuple=True)[0]) # find nonzero dimensions in output channels
        nonzero_Cout = get_nonzero_channels(out_tensor)

        # prep communications by populating output
        out_channel_array = torch.arange(out_tensor.shape[1]) 
        all_output = []
        for rx_mach in range(self.num_machines): # TODO: change implementation to only consider parent nodes

            # skip sending to local node
            if rx_mach == self.machine:
                continue
            
            # init output dict
            output_element = {}
            output_element['layer'] = self.current_layer
            output_element['node'] = self.machine
            output_element['node_to'] = rx_mach

            # Get output channels for current rx machine? TODO: consider removing, this just maps C_out's to machine
            #output_channels = torch.tensor(configs['partition'][][rx_mach],
            #        device=torch.device(configs['device']))
            output_channels = torch.tensor(self.output_channel_map[rx_mach],
                    device=self.device)

            # TODO: is there a faster way to do this? Consider putting larger array 1st... just not sure which one that'd be
            nonzero_out_channels = nonzero_Cout[torch.isin(nonzero_Cout, output_channels)]
            if nonzero_out_channels.nelement() > 0:
                    output_element['is_empty'] =False

                    communication_mask = torch.isin(out_channel_array, nonzero_out_channels)
                    output_element['Cin_map'] = communication_mask.tolist()

                    # TODO: this is inefficient, redo. Probbably need to send a tensor and some info what output channels are being sent 
                    if self.current_layer == self.total_layers_fx-1:
                            output_element[str(rx_mach)]['tensor'] = out_tensor[:,communication_mask]
                    else:
                            output_element[str(rx_mach)]['tensor'] = out_tensor[:,communication_mask,:,:]

                    # debug
                    print(f'\t\t Prepping to send C_out {nonzero_out_channels} to machine {rx_mach}')
            else:
                output_element['is_empty'] =True

            all_output.append(output_element)


    def process_input(self, collected_data):
        '''
            Adds collected input from different nodes to the local current tensor  

            Input:
                collected_data - (list) list of dicts with keys:
                    layer - (int) the layer this input was generated from 
                    is_empty - (bool) indicates this layer receives nothing from previous 
                    node - (int) the network node this input was generated from 
                    tensor - (tensor) the input tensor with dimesnions batch size, input channels (for this layer
                             which may be a subset of the total Cin this node expects) convolution height 
                             and width = [# batch, Cin', H, W]
                    Cin_map - (1 x Cin' list) maps Cin' dimension to dimension in Cin of full input to this 
            
            Output:
                None
        '''

        with torch.no_grad():
            for data_dict in collected_data:
                if data_dict['layer'] == self.current_layer -1 and not data_dict['is_empty']:
                    
                    # get input channels
                    input_channels = torch.tensor(data_dict['Cin'],
                            device=self.device)

                    # add to current tensor 
                    # TODO: experiment with adding up on CPU first then sending to GPU vs sending over and over to GPU
                    self.current_tensor[:, input_channels,] = data_dict['tensor'].to(self.device) + self.current_tensor.index_select(1, input_channels)

    def is_comm_layer(self, layer):
        ''' 
            Determine if communication is necessary after the execution of layer
        '''
        if self.layer_names_fx[layer] in ['relu', 'add', 'avg_pool2d', 'size', 'view', 'x']:
            return False
        else:
            return True

    def add_bias_to_linear(self):
        ''' 
            Required for runnning after linear layer. 
            Assumes this node/machine has access to all bias weights
            TODO: find better implementation for this.
        '''
        return self.current_tensor + get_current_module(self.model, self.current_layer).bias


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
            if not torch.is_tensor(curr_input):
                if 'bn' in self.layer_names_fx[imodule]:
                    print('\t\t-No input received but bn still needs to produce output.')
                    curr_input = torch.zeros(self.size_LUT[self.layer_names_fx[imodule]], dtype=self.dtype,  device=self.device)
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
                
            elif 'add' in self.layer_names_fx[imodule]:
                # residual layer. No comm necessary 
                if self.machine in self.residual_input:
                    print('\t\t-adding residual')
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
            
            elif 'avg_pool2d' in self.layer_names_fx[imodule]:
                print('\t\t-average pooling')
                return F.avg_pool2d(curr_input, 4), False
            
            elif 'size' in self.layer_names_fx[imodule]:
                print('\t\t-skipping')
                return curr_input, False
            
            elif 'view' in self.layer_names_fx[imodule]:
                print('\t\t-reshaping (view)')
                return curr_input.view(curr_input.size(0), -1), False
                
            elif 'x' == self.layer_names_fx[imodule]:
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
            if split_param_name in self.split_module_names:
                # skip if machine doesnt expect input
                if len(self.configs['partition'][split_param_name]['channel_id'][self.machine]) == 0:
                        print(f'\t\t-WARNING: No input assigned to this machine (but it was sent input?). Skipping...')
                        return -1, False

                # TODO: reconsider implementation 
                # What input channels does this machine compute?
                input_channels = torch.tensor(self.configs['partition'][split_param_name]['channel_id'][self.machine],
                        device=self.device)
                N_in = len(input_channels) # TODO: is this used?

                # Where to send output (map of output channels to different machines)
                self.output_channel_map = self.configs['partition'][split_param_name]['filter_id']
            elif type(curr_layer) == nn.Linear and imodule == self.total_layers_fx-1:
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
                input_channels = torch.tensor(input_channels, device=self.device)
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
                input_channels = torch.tensor(input_channels, device=self.device)

            # make vertically split layer. TODO: remove this and replace curr_layer to be split_layer when first made 
            if type(curr_layer) == nn.Conv2d:
                split_layer = split_conv_layer(curr_layer, input_channels)
            elif type(curr_layer) == nn.BatchNorm2d:
                split_layer = split_bn_layer(curr_layer, input_channels)
            elif type(curr_layer) == nn.Linear:
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
            
