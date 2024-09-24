from source.utils import io

from ..utils.misc import get_model_from_code
from ..utils import split_network
from ..utils.calculate_flops import calflops

import torch
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import numpy as np
import logging
import os
import sys

# Logger initialization
logger = logging.getLogger(__name__)

class SplitManager:
    """
    Each thread runs a SplitManager object to handle I/O for its portion of the vertically split model 
    TODO:
    - load only necessary split layers (either on CPU or GPU) for this thread before executing anything. Replace "get_current_module" on line ~112 and remove layer splitting 
    - change implementation to handle and use smallest tensor possible no matter bn or conv (currently bn output with be a tensor with dim=1 be larger than the required C_out for the split layer)
    - find another way to create layer_output_size_LUT and remove input_tensor and get_output_at_each_layer from constructor 
    - assign machines the final layer they have to compute 
    """

    def __init__(self, configs, machine, N_machines, input_tensor, final_node, debug=False):
        
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
                logger.error('Unsupported dtype')
        else:
            logger.warning('No dtype field found in config')
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
        self.add_extra_partition_logic(final_node) # TODO: expand this to cover all models 
        self.layer_names_fx.append('FINAL_MODEL_OUTPUT')
        self.total_layers_fx = len(self.layer_names_fx)
        self.residual_input = {}
        self.residual_block_start, self.residual_connection_start, self.residual_block_end = split_network.get_residual_block_indexes(self.model)

        # model logic
        self.split_module_names = list(self.configs['partition'].keys())
        self.starting_layer = self.get_machine_starting_layer()
        self.current_layer =  self.starting_layer # layer that needs to be executed (current_tensor is always the input to this layer), corresponds to index in self.layer_names_fx
        # TODO: add logic for detecting layer a machine should start at 
        self.final_layer = self.get_machine_ending_layer()
        self.expected_comms = self.init_expected_comms()

        # debugging/ verification
        # layer_output_size_LUT 
        self.horz_output, self.layer_output_size_LUT = split_network.get_output_at_each_layer(model, input_tensor) # TODO: find a way to get layer_output_size_LUT without executing on every layer

        self.current_tensor = torch.zeros(self.get_full_layer_input_size(), dtype=self.dtype, device=self.device) # keep track of tensor I/O on this node. This is the size of the full tensor input for layer = current_layer

        if 'split_layers_path' in self.configs:
            split_layers_path = self.configs['split_layers_path']
            self.split_layers = self.load_split_layers(split_layers_path)
        else:
            logger.warning('No split layers found to load from. Relying on loading from full model (slow execution)')
            self.split_layers = {}
        
    def add_extra_partition_logic(self, final_node):
        '''
            Extra splitting logic for final output of model 
            
            Input:
                configs - (dict) dictionary with configuration settings
                num_machines - (int) number of network nodes
                layer_names_fx - (list) list of layer naes
                N_cout_linear - (int) number of output channels for final linear layer
                final_node - (int) network node meant to execute the final linear layer output of the resnet model
        '''

        model_name = self.configs['model']
        if model_name == 'resnet18':
            N_cout_linear = 10

            # add logic for final layer TODO: add this in automatically somewhere
            linear_map = SplitManager.get_io_for_linear(self.configs, self.layer_names_fx, self.N_machines, final_node, N_cout_linear)
            self.configs['partition']['linear.weight'] = linear_map
            self.configs['partition']['FINAL_MODEL_OUTPUT.weight'] = { 'channel_id' : [np.array([])]*self.N_machines }
            self.configs['partition']['FINAL_MODEL_OUTPUT.weight']['channel_id'][final_node] = np.arange(N_cout_linear) # send all outputs to machine final_node
        else:
            logger.warning(f'Did not add any logic for final execution of {model_name} model')

    def update_horz_output(self, input_tensor):
        '''
            Updates the horizontal output used for checking that split model output matches full model output
        '''
        self.horz_output, self.layer_output_size_LUT = split_network.get_output_at_each_layer(self.model, input_tensor) # TODO: find a way to get layer_output_size_LUT without executing on every layer


    def get_full_layer_input_size(self):
        '''
            Get size of tensor input to layer (full model input size)
        '''
        return self.layer_output_size_LUT[self.layer_names_fx[self.current_layer-1]]

    def get_current_layer_name(self):
        layer = min(self.current_layer, self.final_layer) # handle edge case: current layer cannot surpass final layer 
        return self.layer_names_fx[layer]

    def get_layer_name(self, layer):
        return self.layer_names_fx[layer]

    def is_done(self):
        if self.current_layer > self.final_layer:
            return 1
        else:
            return 0

    @staticmethod
    def combine_all_dict_inputs(input_size, input_list, layer, device='cpu', dtype=torch.float32):
        ''' helper to gather all input tensors in input from the same layer and add them '''
        # combine inputs from machines
        combined_input = False 

        # init zeros tensor for full input
        input_tensor = torch.zeros(input_size, device=device, dtype=dtype)

        # iterate through all input element
        for input_el in input_list:

            # add if for this layer 
            if input_el['layer'] == layer and not input_el['is_empty']:
                input_channels = input_el['Cin']
                if not torch.is_tensor(input_tensor):
                    input_tensor[:,input_channels,] = input_el['tensor'] # initialize curr_input with first input tensor 
                else:
                    input_tensor[:,input_channels,] = input_tensor[:,input_channels,] + input_el['tensor']
                
        return input_tensor

    def get_machine_starting_layer(self):
        ''' 
            determine where to start calculation from

            The first split layer is either:
                1. The first conv layer with non-zero configs['partition']['channel_id'][self.machine], or
                2. The layer immediately after the first conv layer with non-zero configs['partition']['filter_id'][self.machine]

            1 is a split conv layer, 2 is assumed to be a bn layer
        '''

        for i in range(self.total_layers_fx):
            layer_name = self.layer_names_fx[i] + '.weight'

            if layer_name in self.split_module_names:
                if len(self.configs['partition'][layer_name]['channel_id'][self.machine]) > 0:
                    return i
                if len(self.configs['partition'][layer_name]['filter_id'][self.machine]) > 0:
                    return i+1

        return -1

    def get_machine_ending_layer(self):
        ''' 
            determine where to end calculation
        '''

        layer = -1
        for i in range(self.starting_layer, self.total_layers_fx):
            layer_name = self.layer_names_fx[i] + '.weight'

            if layer_name in self.split_module_names:
                if len(self.configs['partition'][layer_name]['channel_id'][self.machine]) > 0:
                    layer = i

        return layer

    def execute_layers_until_comms(self):
        ''' 
            Executes split, starting at the current layer, until reaching a layer that requires input for communication
        '''
        
        do_comm = False
        while not do_comm:
            split_layer_output, do_comm = self.execute_split_layer(self.current_tensor, self.current_layer) # execute split layer
            logger.info(f'Executed layer={self.current_layer} {self.get_current_layer_name()}')
            self.init_current_tensor(split_layer_output)

            if self.debug and not do_comm:
                self.check_current_tensor()

        return split_layer_output

    def init_current_tensor(self, out_tensor):
        ''' 
            Initialize the locally stored current_tensor with out_tensor for the current layer and 
            increments current_layer 
            TODO: assumes out_tensor is the size of the full layer output
        '''

        # debug
        #nonzero_out_tensor = torch.unique(torch.nonzero(out_tensor, as_tuple=True)[1])

        if self.current_layer == self.total_layers_fx-1:
            self.current_tensor = out_tensor
            self.current_layer += 1 # update layer to execute

            # debug
            curr_layer_name = self.layer_names_fx[-1]
            logger.debug(f'FINAL TENSOR FOR: #{self.current_layer}-{curr_layer_name}; Shape={list(self.current_tensor.shape)}')
            logger.debug(f'SPLIT OUTPUT: {self.current_tensor}')
            logger.debug(f'FULL OUTPUT: {self.horz_output[self.layer_names_fx[-2]]}')
        else:
            # prep mask for this node
            out_channel_array = torch.arange(out_tensor.shape[1]) # all indexes of full output
            output_channels = torch.tensor(self.output_channel_map[self.machine],
                    device=self.device) # output channels assigned to this machine
            output_channels_mask = torch.isin(out_channel_array, output_channels)

            # TODO: this is inefficient, redo. Probably need to send a tensor and some info what output channels are being sent 
            tmp = torch.zeros(out_tensor.shape, device=self.device, dtype=self.dtype)
            tmp[:,output_channels_mask] = tmp[:,output_channels_mask] + out_tensor[:,output_channels_mask,]

            self.current_tensor = tmp
            self.current_layer += 1 # update layer to execute

            # debug
            curr_layer_name = self.get_current_layer_name()
            #logger.debug(f'INITIALIZING CURRENT TENSOR FOR: #{self.current_layer}-{curr_layer_name}; Shape={list(tmp.shape)}; C_in={output_channels.numpy()}')  


    def get_last_partition(self):
        '''
            get the C_out from the previous conv layer 
        '''
        
        ilayer = self.current_layer-1 # dont start on current layer 

        # handle edge case at end of model execution
        if self.current_layer == self.total_layers_fx:
            ilayer -= 1
        while ilayer > 0:
            layer_name = self.get_layer_name(ilayer)+'.weight'
            if layer_name in self.configs['partition']:
                return self.configs['partition'][layer_name]['filter_id'][self.machine]
            ilayer -= 1

    @staticmethod
    def compare_helper(split_output, truth_output):
        indent_str = '\t'*2

        diff_output = torch.abs(split_output - truth_output)

        N_batch = split_output.shape[0]

        print_str = ''
        print_str = print_str + indent_str + 'Max diff:'
        max_diff= torch.max(torch.reshape(diff_output, (N_batch, -1)), dim=1)[0]
        print_str = print_str + indent_str + str(max_diff)
        #plt.hist(diff_output.reshape((-1,)))
        #plt.show()

        max_by_Cout = torch.max(torch.abs(diff_output.reshape((1,truth_output.shape[1],-1))), dim=2)

        print_str = print_str + '\n'
        print_str = print_str + indent_str + str(max_by_Cout[0])
        print_str = print_str + indent_str + str(split_network.get_nonzero_channels(max_by_Cout[0]))


        # get C_out with zero and non-zero diff
        nonzero_Cout = split_network.get_nonzero_channels(split_output)
        failing_Cout = nonzero_Cout[torch.isin(nonzero_Cout, split_network.get_nonzero_channels(max_by_Cout[0]))]
        passing_Cout = nonzero_Cout[torch.isin(nonzero_Cout, split_network.get_nonzero_channels(max_by_Cout[0])) == False]
        print_str = print_str + '\n'
        print_str = print_str + indent_str + f'failing Cout = {failing_Cout}  (len = {len(failing_Cout)})'
        print_str = print_str + indent_str + f'passing Cout = {passing_Cout}  (len = {len(passing_Cout)})'

        return max_diff, max_by_Cout, print_str

        
    def check_current_tensor(self, limit=1e-4):
        '''
            Checks if current tensor is correct
        '''

        input_channels = self.get_last_partition()
        if self.current_layer == self.total_layers_fx:
            output_layer_name = self.layer_names_fx[self.current_layer-2]
        else:
            output_layer_name = self.layer_names_fx[self.current_layer-1]

        # handle check for final layer
        if self.current_layer == self.total_layers_fx:
            is_final_layer = True
        else:
            is_final_layer = False

        truth_output = self.horz_output[output_layer_name]
        
        if torch.is_tensor(truth_output):
            #logger.debug(f'Checking output from {output_layer_name} C_in {input_channels}: ')
            max_diff, max_by_Cout, print_str = SplitManager.compare_helper(self.current_tensor[:,input_channels,], truth_output[:,input_channels,])
            #logger.debug('\n\n')

            if max_diff > limit or is_final_layer:
                logger.error('ERROR:')
                logger.error(print_str)
                return 0
            else:
                return 1 
        else:
            # handles "size" layer that doesn't output anything (e.g. truth_output = 1)
            #logger.debug(f'Skipping check for output {output_layer_name} (no tensor found for ref.) ')
            return 1

    def expected_comms_for_layer(self, network_node, layer):
        '''
            Determine which nodes this network_node needs to wait on for the execution of layer. Expects to be called 
            on layers directly after convolutional layers.

            Input:
                network_node - 
                layer - (int) layer that needs to be computed 

            Output:
                tx_nodes - (np.array) array of network nodes that this network_node expects comms from 
        '''

        # get last conv/split layer
        cout_map, cin_map, conv_layer = SplitManager.get_last_split_maps(self.configs, self.layer_names_fx, layer)

        # handle leaf nodes
        if layer == 1:
            # leaf inputs come from network node -1
            # only expect input if network_node is assigned input channels for 1st layer 
            if len(cin_map[network_node]) > 0:
                tx_nodes = np.array([-1])
            else:
                tx_nodes = np.array([])
        else:
            output_channels_for_machine = cout_map[network_node]
            curr_layer = split_network.get_current_module(self.model, conv_layer)
            corresponding_input_channels = torch.unique(torch.nonzero(curr_layer.weight[output_channels_for_machine,], as_tuple=True)[1])

            tx_nodes = np.array([])
            for i in range(self.N_machines):
                if torch.any(torch.isin(corresponding_input_channels, torch.tensor(cin_map[i]))):
                    tx_nodes = np.append(tx_nodes, i)

        # remove receiving from your own network_node 
        tx_nodes = np.delete(tx_nodes, tx_nodes == network_node)

        return tx_nodes


    def enough_comms_received(self, collected_data):
        ''' 
            Looks at an array of dicts received in a buffer and determines if there is enough in the array for 
            the current layer to be executed. Should be done before prep_output. TODO: generalize this for 
            different network topologies

            Input:
                collected_data - (array of data dicts) data sent to client collected in queue

            Output:
                2 == received too many inputs
                1 == received enough input
                0 == received too few inputs 
        '''

        rx_from_nodes = np.array([])
        for data in collected_data:
            if data['layer'] == self.current_layer-1:
                rx_from_nodes = np.append(rx_from_nodes, data['node'])
        
        rx_from_nodes = np.sort(rx_from_nodes)

        expected_comms = self.get_expected_comms()

        if len(expected_comms) == 0:
            logger.info(f'No comms needed to start layer {self.current_layer}')
            return 1
        elif len(rx_from_nodes) < len(expected_comms):
            logger.info(f'Too few inputs for layer={self.current_layer}')
            logger.debug(f'\t\t Collected inputs from nodes: {rx_from_nodes}')
            logger.debug(f'\t\t Need inputs from nodes: {expected_comms}')
            return 0
        elif len(rx_from_nodes) > len(expected_comms):
            logger.warning(f'Too many inputs for layer={self.current_layer}')
            return 2
        elif np.all(rx_from_nodes == expected_comms):
            return 1
        else:
            logger.warning(f'Received unexpected inputs for layer={self.current_layer}')
            return 1

    def prep_output(self, out_tensor):
        ''' 
            Take output tensor from the split layer output and prepare outputs for sending.

            Input:
                out_tensor - (tensor) output from a split layer of dimension batch size, output channels (equal to the full output size for this layer)
                convolution height and width = [# batch, Cout, H, W]

            Output:
                all_output (list of dicts)
                    layer - (int) the layer this output was generated from 
                    is_empty - (bool) indicates this layer produces nothing for the (tensor and Cin are absent from dict in this case)
                    node - (int) the network node this input was generated from 
                    node_to - (int) node that this output should be sent to 
                    tensor - (tensor) the input tensor with dimensions batch size, input channels (for this layer
                            which may be a subset of the total Cin this node expects) convolution height 
                            and width = [# batch, Cin', H, W]
                    Cin - (1 x Cin' list) maps Cin' dimension to dimension in Cin of full input to this layer

        '''
        #logger.debug(f'\t\tOutput tensor shape : {out_tensor.shape}')

        # debug
        #nonzero_out_tensor = torch.unique(torch.nonzero(out_tensor, as_tuple=True)[1])

        # look at which C_out need to be computed and sent
        #nonzero_Cout = torch.unique(torch.nonzero(split_layer.weight, as_tuple=True)[0]) # find nonzero dimensions in output channels
        nonzero_Cout = split_network.get_nonzero_channels(out_tensor) 
        # TODO: update this along with execute_split_model to use arbitrary out_tensor size. Current inplementation assumes size is equal to full layer output

        # prep communications by populating output
        out_channel_array = torch.arange(out_tensor.shape[1]) 
        all_output = []
        for rx_mach in range(self.N_machines): # TODO: change implementation to only consider parent nodes
            
            # only send message if rx_mach expects communication from this machine
            if not (self.machine in self.expected_comms_for_layer(rx_mach, self.current_layer)):
                continue

            # init output dict
            output_element = {}
            output_element['layer'] = self.current_layer-1 # TODO: this assumes init_current_tensor is called after execute_split_layer
            output_element['node'] = self.machine
            output_element['node_to'] = rx_mach

            # Get output channels for current rx machine? TODO: consider removing, this just maps C_out's to machine
            output_channels = torch.tensor(self.output_channel_map[rx_mach], device=self.device)

            # TODO: is there a faster way to do this? Consider putting larger array 1st... just not sure which one that'd be
            nonzero_out_channels = nonzero_Cout[torch.isin(nonzero_Cout, output_channels)]
            if nonzero_out_channels.nelement() > 0:
                output_element['is_empty'] =False

                communication_out_mask = out_channel_array[torch.isin(out_channel_array, nonzero_out_channels)]
                communication_out_channels = out_channel_array[communication_out_mask]
                output_element['Cin'] = communication_out_mask.tolist()

                # TODO: this is inefficient, redo. Probbably need to send a tensor and some info what output channels are being sent 
                output_element['tensor'] = out_tensor[:,communication_out_mask,]

                curr_layer_name = self.get_current_layer_name()
                #logger.debug(f'Machine={self.machine}, Layer to execute = {self.current_layer}:{curr_layer_name}.')
                #logger.debug(f'Prepping to send C_out {communication_out_channels} to machine {rx_mach}')
            else:
                output_element['is_empty'] =True

            all_output.append(output_element)
        return all_output


    def process_input(self, collected_data):
        '''
            Adds collected input from different nodes to the local current tensor  
            Assumes self.current_tensor is the correct size for self.current_layer in model
            TODO: Implementation assumes calculation proceeds after the requisite amount of communication was received
            TODO: assumes N_machines is always the requisite # of communications

            Input:
                collected_data - (list) list of dicts with keys:
                    layer - (int) the layer this input was generated from 
                    is_empty - (bool) indicates this layer receives nothing from previous 
                    node - (int) the network node this input was generated from 
                    tensor - (tensor) the input tensor with dimensions batch size, input channels (for this layer
                             which may be a subset of the total Cin this node expects) convolution height 
                             and width = [# batch, Cin', H, W]
                    Cin - (1 x Cin' list) maps Cin' dimension to dimension in Cin of full input to this 
            
            Output:
                success = 
                     2 == received more than enough
                     1 == received exact amount  
                     0 == did not get enough comms
        '''

        # check that enough data is present
        enough_inputs = self.enough_comms_received(collected_data)

        if enough_inputs > 0:

            # TODO: update this to be dynamic
            if self.current_layer == 1:
                N_expected_comms = 1
            else:
                N_expected_comms = self.N_machines-1 

            count = 0
            with torch.no_grad():
                for data_dict in collected_data:
                    if  data_dict['layer'] == self.current_layer -1:
                        
                        count += 1 # count each communication sent to this node

                        if not data_dict['is_empty']:
                            # get input channels
                            input_channels = torch.tensor(data_dict['Cin'], device=self.device)

                            # add to current tensor 
                            # TODO: experiment with adding up on CPU first then sending to GPU vs sending over and over to GPU
                            self.current_tensor[:, input_channels,] = data_dict['tensor'].to(self.device) + self.current_tensor.index_select(1, input_channels)

        return enough_inputs


    def is_comm_layer(self, layer):
        ''' 
            Determine if communication is necessary after the execution of layer
            TODO: unused, remove
        '''
        if self.layer_names_fx[layer] in ['relu', 'add', 'avg_pool2d', 'size', 'view', 'x']:
            return False
        else:
            return True

    def add_bias_to_linear(self):
        ''' 
            Required for running after linear layer. 
            Assumes this node/machine has access to all bias weights
            TODO: find better implementation for this.
        '''
        module = split_network.get_current_module(self.model, self.current_layer-1)
        return self.current_tensor + module.bias

    def get_input_size(self, layer):
        return self.layer_output_size_LUT[self.layer_names_fx[layer]]

    @staticmethod
    def get_last_split_maps(configs, layer_names_fx, ilayer):
            '''
                get the entire C_out and C_in map from the last conv layer 

                Output:
                    cout_map - (list of np arrays) output channel assignment indexed by machine and elements in array correspond with output channels
                    cin_map - (list of np arrays) input channel assignment indexed by machine and elements in array correspond with input channels
                    layer - (int) layer number of the last convolutional layer (last layer that needed communication)
                    OR
                    -1 if reached the beginning of model and no conv layer was found 
            '''
            if ilayer <= 1:
                # if given layer 1, start at layer 1 TODO: does this make sense for use cases? 
                ilayer = 1
            else:
                ilayer = ilayer-1 # dont start on current layer 

            while ilayer > 0:
                layer_name = layer_names_fx[ilayer]+'.weight'
                if layer_name in configs['partition']:
                    return configs['partition'][layer_name]['filter_id'], configs['partition'][layer_name]['channel_id'], ilayer
                ilayer -= 1
            
            return -1
    
    @staticmethod
    def get_io_for_linear(configs, layer_names_fx, N_machines, final_node, N_Cout):
        '''
            Linear layer output is split across machines but no data structure exists to coordinate how it is communicated and to whom
            Creates a dictionary to determine where input and output goes

            Input:
                configs - model configuration 
                layer_names_fx - list of layer names
                N_machines - total number of network nodes
                final_node - final node 
                N_Cout - number of output channels for the entire model 

            Output:
                partition_map = {
                    channel_id : list of len # total machines. Each element is an np array of the C_in channels associated with that machine==index,
                    feature_id : directs all output channels to final_node
                }

        '''

        cout_map = []
        for i in range(N_machines):
            if i == final_node:
                cout_map.append(np.arange(N_Cout))
            else:
                cout_map.append(np.array([]))

        partition_map = {
            "channel_id" :  SplitManager.get_last_split_maps(configs, layer_names_fx, len(layer_names_fx)-1)[0],
            "filter_id" : cout_map
        }
        
        return partition_map


    def execute_split_layer(self, curr_input, imodule):
        '''   
            This function implements machine v executing layer l and returns the split output for communication 

            return output_tensor, do_comms
                output_tensor - (tensor) split output from layer=imodule from input curr_input, 
                                -1 if curr_input is empty, machine is not required to compute 
                                output for this layer, or if unrecognized layer type. The output 
                                tensor size is equal to the full model output TODO: update to be 
                                arbitrary output size. Add output output_channels - (tensor) 
                                to map output tensor dim=1 to filters in the full output 
                do_comms - (bool) whether or not the machine needs to communcate output 
        '''

        layer_name = self.layer_names_fx[imodule]
        with torch.no_grad():

            # skip this machine+module if there is no input to compute 
            if not torch.is_tensor(curr_input):
                if 'bn' in layer_name:
                    logger.info('No input received but bn still needs to produce output.')
                    curr_input = torch.zeros(self.get_input_size(imodule), dtype=self.dtype,  device=self.device)
                else:
                    logger.info('No input sent to this machine. Skipping module')
                    return -1, False
            
            #logger.debug(f'Current input channels {split_network.get_nonzero_channels(curr_input)}')
            
            # non-comms operations 
            if imodule == self.total_layers_fx-1:
                return self.final_output_routine(curr_input)
            elif 'relu' in layer_name:
                #logger.info('Applying ReLU')
                return F.relu(curr_input), False
                
            elif 'add' in layer_name:
                # residual layer. No comm necessary 
                if self.machine in self.residual_input:
                    logger.info('Adding residual')
                    if 'block_out' in self.residual_input[self.machine]:
                        curr_input = curr_input + self.residual_input[self.machine]['block_out']
                    elif 'block_in' in self.residual_input[self.machine]:
                        curr_input = curr_input + self.residual_input[self.machine]['block_in']
                        logger.info('Assuming shortcut had no layers')
                else:
                    logger.info('Assuming this machine did not receive any input at the beginning of this block. No residual found')

                # erase stored 
                self.residual_input[self.machine] = {}

                return curr_input, False
            
            elif 'avg_pool2d' in layer_name:
                #logger.info('Average pooling')
                return F.avg_pool2d(curr_input, 4), False
            
            elif 'size' in layer_name:
                #logger.info('Skipping')
                return curr_input, False
            
            elif 'view' in layer_name:
                #logger.info('Reshaping (view)')
                return curr_input.view(curr_input.size(0), -1), False
                
            elif 'x' == layer_name:
                #logger.info('Model input layer.. skipping')
                return curr_input, False
            
            # swap out io for residual connection
            if imodule in self.residual_block_start:
                # save input for later 
                self.residual_input[self.machine] = {}
                self.residual_input[self.machine]['block_in'] = curr_input.detach().clone()
                logger.info('Saving input for later...')
            elif imodule in self.residual_connection_start:
                # swap tensors
                self.residual_input[self.machine]['block_out'] = curr_input
                curr_input = self.residual_input[self.machine]['block_in'] 
                logger.info('Saving current input. Swapping for input saved from start of block')

            # get the current module
            # TODO: support other splitting splitting options for ADMM and implementaiton B
            # TODO: remove loading from full model entirely. Keep full model checking optional 
            input_channels, self.output_channel_map, do_comm = self.get_channels(imodule)
            if len(self.split_layers) == 0:
                # load from full model if split layers is empty 
                curr_layer = split_network.get_current_module(self.model, imodule)
                # make vertically split layer. TODO: remove this and replace curr_layer to be split_layer when first made 
                if type(curr_layer) == torch.nn.Conv2d:
                    split_layer = split_network.split_conv_layer(curr_layer, input_channels)
                elif type(curr_layer) == torch.nn.BatchNorm2d:
                    split_layer = split_network.split_bn_layer(curr_layer, input_channels)
                elif type(curr_layer) == torch.nn.Linear:
                    split_layer = split_network.split_linear_layer(curr_layer, input_channels)
                else:
                    logger.info(f'Skipping module {type(curr_layer).__name__}')
                    return -1, False
            else:
                # grab from pre loaded split layers
                split_layer = self.split_layers[layer_name]
            
            # eval split
            split_layer.eval()
            out_tensor = split_layer(curr_input.index_select(1, input_channels))
            if type(split_layer) == torch.nn.BatchNorm2d:
                # place bn output in lager tensor to maintain standardized output size
                # TODO: change implementation to handle and use smallest tensor possible no matter bn or conv
                tmp_out_tensor = torch.zeros(curr_input.shape, dtype=self.dtype)
                tmp_out_tensor[:,input_channels.numpy(),:,:] = out_tensor
                out_tensor = tmp_out_tensor
            
            if self.debug:
                # calculate FLOPS
                curr_input_slice = curr_input.index_select(1, input_channels)

                num_FLOPS, num_params = calflops.calflops(split_layer, (curr_input_slice,), do_print=False) # TODO: sanity check this makes sense 
                logger.debug(f'FLOPS for {layer_name} layer={imodule} FLOPS={num_FLOPS} parameters={num_params}')

            exec_layer_name = self.get_layer_name(imodule)
            nonzero_output_channels = split_network.get_nonzero_channels(out_tensor)
            #logger.debug(f'EXECUTED: #{imodule}-{exec_layer_name}')  
            #logger.debug(f'SPLIT OUTPUT: Shape={list(out_tensor.shape)}; C_out={nonzero_output_channels.numpy()}')  
            return out_tensor, do_comm

    
    def get_channels(self, imodule):
        '''
            Get the input channels and output channel map for layer=imodule
        '''
        # update communication I/O for this layer  
        # TODO: prep this before running execution and give this it's own method
        split_param_name = self.layer_names_fx[imodule] + '.weight'
        if split_param_name in self.split_module_names:
            # skip if machine doesn't expect input
            if len(self.configs['partition'][split_param_name]['channel_id'][self.machine]) == 0:
                logger.warning(f'No input assigned to this machine (but it was sent input?). Skipping...')
                return -1, False

            # TODO: reconsider implementation 
            # What input channels does this machine compute?
            input_channels = torch.tensor(self.configs['partition'][split_param_name]['channel_id'][self.machine],
                    device=self.device)
            N_in = len(input_channels) # TODO: is this used?

            # Where to send output (map of output channels to different machines)
            output_channel_map = self.configs['partition'][split_param_name]['filter_id']

            do_comm = True
        else:
            # for batch normal, and functional passes through the code
            # TODO: address the following assumptions:
            #       - assume all BN layers have C_in divisable by self.N_machines
            #       - assume C_in are evenly split in sequential order WARNING THIS WILL BREAK WHEN WE START TO DO ASSIGN WEIGHTS TO DIFF MACHINES
            N_Cin = self.layer_output_size_LUT[self.layer_names_fx[imodule-1]][1]
            Cin_per_machine = N_Cin/self.N_machines
            if Cin_per_machine % 1 > 0:
                logger.error(f'Unexpected number of I/O for Batch Normal Module {imodule}')
            Cin_per_machine = int(Cin_per_machine)
            input_channels = np.arange(Cin_per_machine) + self.machine*Cin_per_machine
            output_channel_map = [None]*self.N_machines
            for i in range(self.N_machines):
                if i == self.machine:
                    output_channel_map[i] = input_channels
                else:
                    output_channel_map[i] = np.array([])
            input_channels = torch.tensor(input_channels, device=self.device)

            do_comm = False # no comm after BN layer
        
        return input_channels, output_channel_map, do_comm

    def init_expected_comms(self):
        '''
            Initializes data structure for machine to see what inputs it 
            expects 
        '''
        expected_comms = {}
        for layer in range(self.starting_layer, self.final_layer+1):
            expected_comms_layer = self.expected_comms_for_layer(self.machine, layer)
            layer_name = self.get_layer_name(layer)
            expected_comms[layer_name] = expected_comms_layer
        
        return expected_comms


    def get_expected_comms(self):
        '''
            Get an array of the nodes this machine expects to receive from 
        '''
        layer_name = self.get_current_layer_name()
        return self.expected_comms[layer_name]

    def final_output_routine(self, input_tensor):
        '''
            Handle final layer output (assumed linear layer)
        '''

        logger.info('FINISHED MODEL EXECUTION')

        vertical_output = self.add_bias_to_linear()
        
        return vertical_output, True


    def save_split_layers(self, output_path):
        '''
            Saves each split layer for this machine to a file 
        '''

        for layer in range(self.starting_layer, self.final_layer):
            
            curr_layer_name = self.layer_names_fx[layer]

            # is final layer
            is_final_layer = self.total_layers_fx-1 == layer
            is_operation = any([ el in curr_layer_name for el in ['relu', 'add', 'avg_pool2d', 'size', 'view', 'x'] ])
            if not (is_final_layer or is_operation):
                # get layer from full module
                curr_layer = split_network.get_current_module(self.model, layer)
                input_channels = self.get_channels(layer)[0]

                # prep split version
                if type(curr_layer) == torch.nn.Conv2d:
                    split_layer = split_network.split_conv_layer(curr_layer, input_channels)
                elif type(curr_layer) == torch.nn.BatchNorm2d:
                    split_layer = split_network.split_bn_layer(curr_layer, input_channels)
                elif type(curr_layer) == torch.nn.Linear:
                    split_layer = split_network.split_linear_layer(curr_layer, input_channels)
                else:
                    print(f'Skipping module {type(curr_layer).__name__}')
                
                # save
                #curr_module_name = curr_layer_name.replace('.', '-')
                curr_machine_path = os.path.join(output_path,f'machine-{self.machine}')
                fpath = os.path.join(curr_machine_path, f'{curr_layer_name}.pth')
                torch.save(split_layer, fpath)


    def load_split_layers(self, split_layer_dir):
        '''
            Load each split layer assigned to this machine
        '''

        split_layers = {} # init dictionary 

        for layer in range(self.starting_layer, self.final_layer+1):
            curr_layer_name = self.layer_names_fx[layer]
            
            is_final_layer = self.total_layers_fx-1 == layer
            is_operation = any([ el in curr_layer_name for el in ['relu', 'add', 'avg_pool2d', 'size', 'view', 'x'] ])

            # assign to  struct if split layer
            if not (is_final_layer or is_operation):
                layer_path = os.path.join(split_layer_dir, f'{curr_layer_name}.pth')
                split_layers[curr_layer_name] = torch.load(layer_path, map_location=self.device).eval()

        return split_layers


            
        