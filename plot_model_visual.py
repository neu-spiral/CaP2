import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import yaml

from source.utils.io import load_state_dict, get_model_path_split, get_fig_path
from source.utils.misc import get_model_from_code, get_input_from_code
from source.utils.testers import plot_layer
from source.utils.masks import partition_generator, featuremap_summary




def main():

    path = os.path.join('assets', 'models', 'perm')
    # model_name = 'esc-EscFusion-kernel-np4-pr0.5-lcm100.pt'
    model_name = 'esc-EscFusion-kernel-np4-pr0.85-lcm100.pt'

    model_path = os.path.join(path, model_name)

    path_configs = os.path.join('config', 'esc.yaml')

    with open(path_configs, 'r') as stream:
        try:
            configs = yaml.load(stream, yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    
    configs['partition_path'] = os.path.join('config', 'EscFusion-np4.yaml')

    model = get_model_from_code(configs)

    state_dict = torch.load(get_model_path_split("{}".format(model_name)), map_location='cuda:1')
    model = load_state_dict(model, 
                                    state_dict['model_state_dict'] if 'model_state_dict' in state_dict 
                                    else state_dict['state_dict'] if 'state_dict' in state_dict else state_dict,)
    
    input_var = get_input_from_code(configs)

    # Config partitions and prune_ratio
    configs = partition_generator(configs, model)
        
    # Compute output size of each layer
    configs['partition'] = featuremap_summary(model, configs['partition'], input_var)

    plot_layer(model, configs['partition'], layer_id=(5,),
               savepath=get_fig_path("{}".format('.'.join(model_name))))
    


if __name__ == '__main__':
    main()