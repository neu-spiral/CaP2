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
logger = logging.getLogger() # get root

# Set the overall logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.DEBUG)

try:
    from source.core import split_manager, run_partition, engine
    from source.utils import misc, split_network
    from source.SplitModelNetworking import node
    from source.utils import dataset
except:
    sys.path.append(os.path.join(os.path.dirname(__name__), "..\source"))
    logger.warning(f"Current working directory: {os.getcwd()}")
    logger.warning(f"Updated sys.path: {sys.path}")
    from source.core import split_manager, run_partition, engine
    from source.utils import misc, split_network
    from source.SplitModelNetworking import node    
    from source.utils import dataset

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
    parser.add_argument('log_dir_name', type=str, help='Directory name that log outputs are saved to i.e. CaP/logs/[logdir name]')
    parser.add_argument('-b', '--batch_size', type=int, help='Expected batch size of inputs', default=16)
    parser.add_argument('-d', '--device', type=str, help='Computation device e.g. cpu, cuda:0, etc.', default='cpu')
    parser.add_argument('-p', '--precision', type=str, help='Computational precision', default='float32')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
    parser.add_argument('debug', choices=['True', 'False'], default='True', help='Check each output tensor of split model')
    args = parser.parse_args()

    device= args.device
    dtype = args.precision
    batch_size = args.batch_size # TODO: run split model should be able to handle arbirary batch size 

    machine_number = args.node
    model_name = args.model_file.split('-')[1]
    split_manager_debug = args.debug == 'True'

    ## START LOGGER CONFIGURATION

    # setup save directory for logs 
    log_path = os.path.join('logs', args.log_dir_name)   
    os.makedirs(log_path, exist_ok=True) 
    if split_manager_debug:
        log_file_path = os.path.join( log_path, f'node{machine_number}_{model_name}_debug.log')
    else:
        log_file_path = os.path.join( log_path, f'node{machine_number}_{model_name}.log')

    # Create handlers: one for file and one for console
    file_handler = logging.FileHandler(log_file_path)
    console_handler = logging.StreamHandler(sys.stdout)

    # Set the logging level for each handler (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    ## END LOGGER CONFIGURATION

    # figure out who this machine/network node receives from 
    required_clients, connection_type, num_nodes, final_node = node.load_config(args.ip_map_file, args.network_graph_file, args.node)
    logger.info(f'Starting server for node {args.node}')
    logger.info(f"Required Clients: {required_clients}")

    # make split manager for executing split execution 
    configs = split_network.config_setup(num_nodes, args.model_file, device, dtype)

    # file path for split layers
    configs['split_layers_path'] = os.path.join('assets','models','perm', f'vsplit-{args.model_file[:-3]}', f'machine-{args.node}')

    # TODO: load from dataset
    input_size = misc.get_input_dim(configs, batch_size)[0] # TODO: handle different input sizes for esc and flashnet?
    tensor = misc.get_rand_tensor(input_size, configs['device'], configs['dtype'])

    model_manager = split_manager.SplitManager(configs, args.node, num_nodes, final_node, tensor, split_manager_debug)

    # open ip map 
    with open(args.ip_map_file, 'r') as file:
        ip_map = json.load(file)
    node_servers = ip_map[str(args.node)]

    client_data_queue = queue.Queue()  # Create a shared queue

    # Start servers on this node
    server_threads = []
    index = 0
    for a_server in node_servers:
        server_ip = a_server['ip']
        server_port = a_server['port']
        logger.info(f'Node {args.node} starting server on {server_ip}:{server_port}')
        server_threads += [threading.Thread(target=node.server, args=(server_ip, server_port, required_clients, client_data_queue))]
        server_threads[index].start()
        index += 1

    collected_data = [] # server fills this up 

    first_input_received = False
    process_input_time = 0
    try:
        while True:

            # shut down server if machine is finished 
            if model_manager.is_done():
                logger.info(f'Machine {model_manager.machine} has finished calculations. Shutting down...')
                total_runtime = (time.perf_counter() - run_split_model_start)
                logger.info(f"Total runtime={total_runtime}s")
                for iserver in range(len(server_threads)):
                    server_threads[iserver].join()  # Wait for the server thread to finish
                return True # end execution

             # check that enough data is present
            enough_input = model_manager.enough_comms_received(collected_data)

            # check if update was made 
            if enough_input > 0:

                # start counting model exectuion time when enough input is received for first layer
                # do not count idle time waiting for input
                if not first_input_received:
                    logger.debug('Starting timer for total runtime')
                    run_split_model_start = time.perf_counter()
                    idle_time = 0
                    first_input_received = True
                else:
                    idle_time = (time.perf_counter() - idle_time_start)*1e3
                logger.debug(f'Idle time={idle_time}ms for layer={model_manager.current_layer-1}') # PLOT THIS

                # grab input tensor for debugging and final check 
                # TODO: this implementation needs to be changed to accommodate escnet where full input is multiple tensors, also doesn't work if final node does not receive model input 
                if model_manager.current_layer == 1 and model_manager.debug == True:
                    input_tensor = get_input_tensor(collected_data)
                    if torch.is_tensor(input_tensor):
                        model_manager.update_horz_output(input_tensor)
                    else:
                        logger.warning('Could not find input tensor')

                # execute split layers

                # start timers
                execute_layers_start = time.perf_counter()
                start_process_input = time.perf_counter()

                # sum received data
                model_manager.process_input(collected_data) 
                process_input_time = (time.perf_counter() - start_process_input)*1e3

                # execute split model 
                output_tensor = model_manager.execute_layers_until_comms()

                # log timing 
                execute_layers_time = (time.perf_counter() - execute_layers_start)*1e3
                prev_layer_name = model_manager.get_layer_name(model_manager.current_layer-1)
                logger.debug(f'Executed to {prev_layer_name} layer={model_manager.current_layer-1} in time={execute_layers_time}ms process input time={process_input_time}ms') # PLOT THIS

                # TODO: add timing here
                # always send output unless on final layer
                if not model_manager.current_layer == model_manager.total_layers_fx:
                    # start sending state 
                    
                    # prep output 
                    start_prep_out_start = time.perf_counter()
                    send_nodes_start = time.perf_counter()
                    processed_output = model_manager.prep_output(output_tensor) # prepare communication. TODO: this probably takes awhile??
                    prep_out_time = (time.perf_counter() - start_prep_out_start)*1e3
                    layer_sent = model_manager.current_layer-1
                    logger.debug(f'Prep output layer={layer_sent} time={prep_out_time}ms')

                    # send data to correct node in network 
                    logger.debug('Send to nodes start')
                    node.send_to_nodes(processed_output, ip_map, connection_type)
                    send_nodes_time = (time.perf_counter() - send_nodes_start)*1e3
                    logger.debug(f'Sent layer={layer_sent} to nodes in time={send_nodes_time}ms')

                # start idle timer and reset process input timer
                idle_time_start = time.perf_counter()
                
                # clean out queue TODO: compare methods 
                #collected_data = [el for el in collected_data if el['layer'] != model_manager.current_layer-2]
                collected_data = list(filter(lambda el: el['layer'] != model_manager.current_layer-2, collected_data))

            else:
                # continue waiting
                collected_data = collected_data + node.collect_data_from_server(client_data_queue, 1, model_manager.current_layer)

            # Optionally, add a sleep interval to avoid high CPU usage
            #time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        for iserver in range(len(server_threads)):
            server_threads[iserver].join()  # Wait for the server thread to finish

if __name__ == "__main__":
    main()
