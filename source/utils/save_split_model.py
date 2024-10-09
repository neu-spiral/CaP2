import argparse 
import torch
import time
import os

from source.utils import split_network, misc
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
    #time_stamp = time.strftime("%Y%m%d-%H%M%S")
    folder_name = 'vsplit-{}'.format(model_name)

    # make folder 
    folder_path = os.path.join(os.getcwd(), 'assets', 'models', 'perm',folder_name)
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
    parser.add_argument('model_file', type=str, help='file name of model e.g. cifar100-resnet101-kernel-np4-pr0.5-lcm0.0001.pt')
    parser.add_argument('-d', '--device', type=str, help='Computation device e.g. cpu, cuda:0, etc.', default='cpu')
    parser.add_argument('-p', '--precision', type=str, help='Computational precision', default='float32')
    args = parser.parse_args()

    # make split manager for executing split execution 
    configs = split_network.config_setup(args.num_nodes, args.model_file, args.device, args.precision) # TODO: generalize

    folder_path = make_split_model_dirs(args.model_file, args.num_nodes)
    configs_copy = configs

    # TODO: load from dataset
    input_sizes = misc.get_input_dim(configs, 1)
    input_tensor = misc.get_rand_tensor(input_sizes[0], configs['device'], configs['dtype'])

    split_managers = [split_manager.SplitManager]*args.num_nodes
    for i in range(args.num_nodes):
        split_managers[i] = split_manager.SplitManager(configs_copy, i, args.num_nodes, i, input_tensor, debug=True)

        split_managers[i].save_split_layers(folder_path)


if __name__ == "__main__":
    main()
