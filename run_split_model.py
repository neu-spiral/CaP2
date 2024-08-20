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

def final_routine(model_manager, input_tensor):
    '''
        Handle final layer output
    '''

    print('FINISHED MODEL EXECUTION')

    if model_manager.machine == 0: # TODO: replace this with logic for root node. ATM assumes machine 0 is sent final outputs
        vertical_output = model_manager.add_bias_to_linear()

        with torch.no_grad():
            model_manager.model.eval()
            full_output = model_manager.model(input_tensor)
        
        split_network.compare_outputs(full_output, vertical_output)


def main():
    parser = argparse.ArgumentParser(description="Server application with configurable settings.")
    parser.add_argument('ip_map_file', type=str, help='Path to ip map JSON file')
    parser.add_argument('network_graph_file', type=str, help='Path to network graph JSON file')
    parser.add_argument('node', type=int, help='Node in the network to launch server for')
    parser.add_argument('model_file', type=str, help='File path to model')
    args = parser.parse_args()

    # figure out who this machine/network node receives from 
    required_clients, servers, num_nodes = node.load_config(args.ip_map_file, args.network_graph_file, args.node)
    print(f"Required Clients: {required_clients}")

    # make split manager for executing split execution 
    configs = config_setup(num_nodes, args.model_file)
    dtype= torch.types
    input_tensor = engine.get_input_from_code(configs)[0]
    if 'dtype' in configs:
        if configs['dtype'] == 'float64':
            input_tensor.type(torch.float64)
        elif configs['dtype'] == 'float32':
            input_tensor.type(torch.float32)
        else:
            print('Unsupported dtype')
    else:
        print('Warning found no dtype field in config')

    model_manager = split_manager.SplitManager(configs, args.node, num_nodes, input_tensor)

    # open ip map 
    with open(args.ip_map_file, 'r') as file:
        ip_map = json.load(file)
    server_ip = ip_map[str(args.node)]['ip']
    server_port = ip_map[str(args.node)]['port']

    client_data_queue = queue.Queue()  # Create a shared queue

    # Start the server in a separate thread
    server_thread = threading.Thread(target=node.server, args=(server_ip, server_port, required_clients, client_data_queue))
    server_thread.start()
    
    # for keeping track of layer outputs being received out of order TODO: change this implementation
    update_count = 0 
    leftover_collected_data = []

    # for final comparison
    input_tensor = []

    first_pass = True # used to handle  leaf nodes for non-DAG topology

    try:
        while True:
            if first_pass:
                collected_data = node.collect_data_from_server(client_data_queue, 1, model_manager.current_layer, collected_for_layer=update_count)
            else:
                first_pass = False
                collected_data = node.collect_data_from_server(client_data_queue, 1, model_manager.current_layer, collected_for_layer=update_count)


            if collected_data:
                collected_data.append(leftover_collected_data)

                # grab input tensor for debugging and final check
                if model_manager.current_layer == 1:
                    input_tensor = collected_data[0]['tensor']
                    print(f'Grabbing input tensor for later')

                # process collected data 
                model_manager.process_input(collected_data) # update local tensor with inputs 

                if model_manager.current_layer == model_manager.total_layers_fx+1:
                    final_routine(model_manager)
                    server_thread.join()  # Wait for the server thread to finish
                    return True # end execution
                else:
                    # execute split layers
                    output_tensor = model_manager.execute_layers_until_comms()

                    # prep output
                    processed_output = model_manager.prep_output(output_tensor) # prepare communication
                    
                    # update local tensor and increment layer
                    model_manager.update_current_tensor(output_tensor)
                    
                    # send data to correct node in network 
                    node.send_to_nodes(processed_output, ip_map)

                    # update networking logic
                    required_clients_tmp = required_clients # after receive leaf node inputs update to typical model TODO: not required for DAGs 
                    leftover_collected_data = [el for el in collected_data if el['layer'] != model_manager.current_layer-1]
                    update_count = 0
                    for el in leftover_collected_data:
                        if el['layer'] == model_manager.current_layer:
                            update_count += 1

            # Optionally, add a sleep interval to avoid high CPU usage
            time.sleep(5)
    except KeyboardInterrupt:
        print("Shutting down...")
        server_thread.join()  # Wait for the server thread to finish


if __name__ == "__main__":
    main()