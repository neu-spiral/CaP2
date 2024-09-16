import argparse 
import torch
import time
import os

from source.utils import split_network
from source.core import split_manager

def make_split_model_dirs(model_name, num_machines):
    '''
        Makes directory to save models to and subdirectories for each machine e.g.
        CaP/assets/models/[model name]/machine-[machine #]

        Input:
            model_name -- (str) name of the model
            num_machines -- (int) determines model splitting

        Output:
            folder_path -- (str) path directory with saved split models e.g. 'assets/models/[model name]'
    '''

    print(f'Number of machines = {num_machines}')

    # make dir name 
    model_name = model_name.replace('.pt', '')
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    folder_name = 'vsplit-{}-{}'.format(model_name,time_stamp)

    # make folder 
    folder_path = os.path.join(os.getcwd(), 'assets', 'models',folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i in range(num_machines):
        machine_path = os.path.join(folder_path,f'machine-{i}')
        if not os.path.exists(machine_path):
            os.makedirs(machine_path)

    print(f'save path = {folder_path}')

    return folder_path

def main():
    ''' 
        TODO: generalize
    '''

    parser = argparse.ArgumentParser(description="Server application with configurable settings.")
    parser.add_argument('num_nodes', type=int, help='Number of nodes in the network. This determines the model splitting.')
    parser.add_argument('model_file', type=str, help='File path to model')
    args = parser.parse_args()

    # make split manager for executing split execution 
    configs = split_network.config_setup_resnet(args.num_nodes, args.model_file) # TODO: generalize
    input_tensor = torch.rand(1, 3, 32, 32, device=torch.device(configs['device']))
    if 'dtype' in configs:
        if configs['dtype'] == 'float64':
            input_tensor = input_tensor.type(torch.float64)
        elif configs['dtype'] == 'float32':
            input_tensor = input_tensor.type(torch.float32)

    folder_path = make_split_model_dirs(args.model_file, args.num_nodes)
    configs_copy = configs

    split_managers = [split_manager.SplitManager]*args.num_nodes
    for i in range(args.num_nodes):
        split_managers[i] = split_manager.SplitManager(configs_copy, i, args.num_nodes, input_tensor, i, debug=True)

        split_managers[i].save_split_layers(folder_path)


if __name__ == "__main__":
    main()
