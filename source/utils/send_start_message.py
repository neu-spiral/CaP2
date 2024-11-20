import json
import argparse
import torch 

from source.SplitModelNetworking import leaf

def main():
    parser = argparse.ArgumentParser(description="Leaf node data collector and sender.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config_file, "r") as f:
        config = json.load(f)

    servers = [(srv['ip'], srv['port']) for srv in config['servers']]

    # send each server it is time to start execution 
    start_message = {"start": 1, 'layer':0, 'node':-1}
    for iserver in range(len(servers)):

        print(f'Sending starting message to {servers[iserver]}')
        leaf.send_to_servers(start_message,[servers[iserver]])     

if __name__ == "__main__":
    main()