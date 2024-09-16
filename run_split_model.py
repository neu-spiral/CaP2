import argparse 
import json 
import queue
import threading
import torch
import time
import sys
from os import environ
import os, sys
import logging

import torch.types

# Logger initialization
logger = logging.getLogger(__name__)

# Set the overall logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.DEBUG)

# Create handlers: one for file and one for console
file_handler = logging.FileHandler('logfile.log')
console_handler = logging.StreamHandler()

# Set the logging level for each handler (DEBUG, INFO, WARNING, ERROR, CRITICAL)
file_handler.setLevel(logging.DEBUG)
console_handler.setLevel(logging.INFO)

# Create formatters and add them to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

try:
    from source.core import split_manager, run_partition, engine
    from source.utils import misc, split_network
    from source.SplitModelNetworking import node
except:
    sys.path.append(os.path.join(os.path.dirname(__name__), "..\source"))
    logging.warning(f"Current working directory: {os.getcwd()}")
    logging.warning(f"Updated sys.path: {sys.path}")
    from source.core import split_manager, run_partition, engine
    from source.utils import misc, split_network
    from source.SplitModelNetworking import node    

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
    parser.add_argument('model_file', type=str, help='File model file name e.g. cifar10-resnet18-kernel-npv0-pr0.75-lcm0.001.pt')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
    parser.add_argument('debug', choices=['True', 'False'], default='True', help='Check each output tensor of split model')
    args = parser.parse_args()

    # figure out who this machine/network node receives from 
    required_clients, connection_type, num_nodes, final_node = node.load_config(args.ip_map_file, args.network_graph_file, args.node)
    logging.info(f"Required Clients: {required_clients}")

    # make split manager for executing split execution 
    # TODO: generalize 
    configs = split_network.config_setup_resnet(num_nodes, args.model_file)
    # file path for split layers
    configs['split_layers_path'] = os.path.join('assets','models',f'vsplit-{args.model_file[:-3]}', f'machine-{args.node}')

    input_tensor = torch.rand(1, 3, 32, 32, device=torch.device(configs['device']))
    if 'dtype' in configs:
        if configs['dtype'] == 'float64':
            input_tensor = input_tensor.type(torch.float64)
        elif configs['dtype'] == 'float32':
            input_tensor = input_tensor.type(torch.float32)
        else:
            logging.error('Unsupported dtype')
    else:
        logging.warning('No dtype field found in config')
    

    model_manager = split_manager.SplitManager(configs, args.node, num_nodes, input_tensor, final_node, args.debug == 'True')

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
                    logging.info(f'Machine {model_manager.machine} has finished calculations. Shutting down...')
                    for iserver in range(len(server_threads)):
                        server_threads[iserver].join()  # Wait for the server thread to finish
                    return True # end execution

            
            # updates local tensor if enough input is present 
            enough_input = model_manager.process_input(collected_data) 

            # check if update was made 
            if enough_input:
                # grab input tensor for debugging and final check 
                # TODO: this implementation needs to be changed to accommodate escnet where full input is multiple tensors, also doesn't work if final node does not receive model input 
                if model_manager.current_layer == 1:
                    input_tensor = get_input_tensor(collected_data)
                    if torch.is_tensor(input_tensor):
                        model_manager.update_horz_output(input_tensor)
                        logging.info('Updating input tensor')
                    else:
                        logging.warning('Could not find input tensor')

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
        logging.info("Shutting down...")
        for iserver in range(len(server_threads)):
            server_threads[iserver].join()  # Wait for the server thread to finish

if __name__ == "__main__":
    main()
