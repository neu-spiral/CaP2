import argparse 
import json 
import queue
import threading
import torch
import time
import sys
from os import environ
import os, sys

import torch.types

try:
    from source.core import split_manager, run_partition, engine
    from source.utils import misc, split_network
    from source.SplitModelNetworking import node
except:
    sys.path.append(os.path.join(os.path.dirname(__name__), "..\source"))
    print(os.getcwd())
    print(sys.path)
    from source.core import split_manager, run_partition, engine
    from source.utils import misc, split_network
    from source.SplitModelNetworking import node    


def config_setup(num_nodes, model_file_path):
    '''  
        Construct the  config for how the model should be split 
        TODO: avoid having to load model here, should only be loaded once in split manager
    '''

    # setup config
    sys.argv = [sys.argv[0]]
    dataset='cifar10' # TODO: automatically determine from inputs  
    environ["config"] = f"config/{dataset}.yaml"
    configs = run_partition.main()

    configs["device"] = "cpu"
    configs['load_model'] = model_file_path
    configs["num_partition"] = str(num_nodes)
    configs['dtype'] = 'float32'

    # load model 
    model = misc.get_model_from_code(configs).to(configs['device']) 

    # populate config
    input_var = engine.get_input_from_code(configs)
    configs = engine.partition_generator(configs, model) # Config partitions and prune_ratio
    configs['partition'] = engine.featuremap_summary(model, configs['partition'], input_var) # Compute output size of each layer

    return configs

def get_input_tensor(collected_data):
    '''
        Searches for input tensor in collected_data
    '''

    for data in collected_data:
        if isinstance(data, dict) and data['node'] == -1:
            return data['tensor']
    
    return -1

def main():
    ''' 
        TODO: cases to handle:
        - FC network, some connections are more important than others, continue even if nodes dont send anything
        - network nodes that do not need input will start and send right away. Consider adding a delay to ensure sure recipients arent missing communication
    
    '''

    parser = argparse.ArgumentParser(description="Server application with configurable settings.")
    parser.add_argument('ip_map_file', type=str, help='Path to ip map JSON file')
    parser.add_argument('network_graph_file', type=str, help='Path to network graph JSON file')
    parser.add_argument('node', type=int, help='Node in the network to launch server for')
    parser.add_argument('model_file', type=str, help='File path to model')
    args = parser.parse_args()

    # figure out who this machine/network node receives from 
    required_clients, connection_type, num_nodes, final_node = node.load_config(args.ip_map_file, args.network_graph_file, args.node)
    print(f"Required Clients: {required_clients}")

    # make split manager for executing split execution 
    configs = config_setup(num_nodes, args.model_file)
    input_tensor = torch.rand(1, 3, 32, 32, device=torch.device(configs['device']))
    if 'dtype' in configs:
        if configs['dtype'] == 'float64':
            input_tensor = input_tensor.type(torch.float64)
        elif configs['dtype'] == 'float32':
            input_tensor = input_tensor.type(torch.float32)
        else:
            print('Unsupported dtype')
    else:
        print('Warning found no dtype field in config')

    imach = args.node # get network node ID number
    model_manager = split_manager.SplitManager(configs, imach, num_nodes, input_tensor, final_node, debug=True)

    # open ip map 
    with open(args.ip_map_file, 'r') as file:
        ip_map = json.load(file)
    node_servers = ip_map[str(args.node)]

    client_data_queue = queue.Queue()  # Create a shared queue

    # Start servers on this node
    server_threads = []
    index = 0
    for a_server in node_servers:
        server_threads += [threading.Thread(target=node.server, args=(a_server['ip'], a_server['port'], required_clients, client_data_queue))]
        server_threads[index].start()
        index += 1

    collected_data = [] # server fills this up 

    try:
        while True:

            # shut down server if machine is finished 
            if model_manager.is_done():
                    print(f'\tMachine {model_manager.machine} has finished calculations. Shutting down...\n')
                    for iserver in range(len(server_threads)):
                        server_threads[iserver].join()  # Wait for the server thread to finish
                    return True # end execution
            
            # updates local tensor if enough input is present 
            enough_input = model_manager.process_input(collected_data) 

            # check if update was made 
            if enough_input:
                # grab input tensor for debugging and final check 
                # TODO: this implementation needs to be changed to accomidate escnet where full input is multiple tensors, also doesnt work if final node does not receive model input 
                if model_manager.current_layer == 1:
                    input_tensor = get_input_tensor(collected_data)
                    if torch.is_tensor(input_tensor):
                        model_manager.update_horz_output(input_tensor)
                        print(f'Updating input tensor')
                    else:
                        print(f'Could not find input tensor')

                # execute split layers
                output_tensor = model_manager.execute_layers_until_comms()

                # always send output unless on final layer
                if not model_manager.current_layer == model_manager.total_layers_fx:
                    # prep output
                    processed_output = model_manager.prep_output(output_tensor) # prepare communication
                        
                    # send data to correct node in network 
                    node.send_to_nodes(processed_output, ip_map, connection_type)

                    # remove data from the queue that was processed already 
                    collected_data = [el for el in collected_data if el['layer'] != model_manager.current_layer-2]
            else:
                # continue waiting
                collected_data = collected_data + node.collect_data_from_server(client_data_queue, 1, model_manager.current_layer)

            # Optionally, add a sleep interval to avoid high CPU usage
            time.sleep(5)
    except KeyboardInterrupt:
        print("Shutting down...")
        for iserver in range(len(server_threads)):
            server_threads[iserver].join()  # Wait for the server thread to finish

if __name__ == "__main__":
    main()