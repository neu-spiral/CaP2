import json
import argparse
import torch 

from source.SplitModelNetworking import leaf
from source.core import engine

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
    args = parser.parse_args()

    batch_size =  args.batch_size

    # Load configuration
    with open(args.config_file, "r") as f:
        config = json.load(f)

    servers = [(srv['ip'], srv['port']) for srv in config['servers']]
    send_data = [srv['data'] == 'True' for srv in config['servers']]

    tensor = torch.rand((batch_size,3,32,32)) # single image from cifar 

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