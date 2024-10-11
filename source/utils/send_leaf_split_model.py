import json
import argparse
import torch 

from source.SplitModelNetworking import leaf
from source.core import engine
from source.utils import misc

def prep_data(tensor):
    '''
        Prepare input tensors to model 
    '''
    return {
        'node' : -1,
        'layer' : 0,
        'tensor' : tensor, 
        'Cin' : list(range(tensor.size(1))), 
        'is_empty': False
    }


def main():
    parser = argparse.ArgumentParser(description="Leaf node data collector and sender.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    parser.add_argument('-b', '--batch_size', type=int, help='Expected batch size of inputs', default=16)
    parser.add_argument('-ds', '--dataset', type=str, choices=['cifar10', 'cifar100', 'esc'], help='Dataset used for input', default='cifar10')
    parser.add_argument('-p', '--precision', type=str, help='Computational precision', default='float32')
    args = parser.parse_args()

    batch_size =  args.batch_size

    # Load configuration
    with open(args.config_file, "r") as f:
        config = json.load(f)

    servers = [(srv['ip'], srv['port']) for srv in config['servers']]
    send_data = [srv['data'] == 'True' for srv in config['servers']]

    # make input tensor
    configs = {'data_code': args.dataset}
    input_size = misc.get_input_dim(configs, batch_size)[0] # TODO: handle different input sizes for esc and flashnet?
    tensor = misc.get_rand_tensor(input_size, 'cpu', args.precision)
    
    # 3 send each server different data 
    for iserver in range(len(servers)):

        if send_data[iserver]:
           processed_data = prep_data(tensor)
        else:
            processed_data = {"start": 1, 'layer':0, 'node':-1}

        print(f'Sending data to {servers[iserver]}')
        leaf.send_to_servers(processed_data,[servers[iserver]])     

if __name__ == "__main__":
    main()