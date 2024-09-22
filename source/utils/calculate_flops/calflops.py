from __future__ import print_function
import torchvision.models as models
from collections import OrderedDict
import torch
import argparse
import os
import sys
import yaml

sys.path.append('./source/utils/calculate_flops/')
from ptflops.flops_counter import get_model_complexity_info
from thop import profile


def calflops(model, inputs, prune_ratios=[], do_print=True): 
    
    
    if not prune_ratios:
        prune_ratios = OrderedDict()
        with torch.no_grad():
            for name, W in (model.named_parameters()):
                prune_ratios[name] = 0
    
    model.train(False)
    model.eval()

    # # Apply pruning ratios to the model
    # with torch.no_grad():
    #     for name, param in model.named_parameters():
    #         if name in prune_ratios:
    #             param.data.mul_(1 - prune_ratios[name])

    macs, params = profile(model, inputs=inputs, rate = prune_ratios)
    # macs, params = profile(model, inputs=inputs)

    # # Reset parameters to their original values
    # with torch.no_grad():
    #     for name, param in model.named_parameters():
    #         if name in prune_ratios:
    #             param.data.div_(1 - prune_ratios[name])

    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs * 2/1000000000)) # GMACs
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params/1000000)) # M
    if do_print:
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs/1000000000)) # G
        print('{:<30}  {:<8}'.format('Number of parameters: ', params/1000000)) # M

    #flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
    #print(flops, params)

    return macs, params