#!/bin/bash

cuda_device=2

# Dataset and model selections, should be selected in these combinations given:

# dataset=cifar10
# model=resnet18

# dataset=cifar100
# model=wrn28

# dataset=esc
# model=escnet

dataset=flash
model=flashnet


# Put this below as a parameter in order to avoid pre-training, and use a teacher model. If not, pre-training will be performed:
# -lm ${teacher} \





# Prune ratios and num_partitions, should be selected in these combinations given:
yaml_version=0 # meaning no yaml file is used, num_partitions=2,4,8 is used

# prune_ratio=0.5
# num_partitions=2

prune_ratio=0.75
num_partitions=4

# prune_ratio=0.875
# num_partitions=8




##### NO LONGER FILE VERSIONS, USES INTEGER INSTEAD #####
# # For yaml file selection, parameter is '-np config/${model}-$2.yaml':

# yaml_version=1
# prune_ratio=0.75

# num_partitions=config/${model}-v${yaml_version}.yaml



teacher=${dataset}-${model}.pt
prune_finetune() {
    st=$5
    save_name=${dataset}-${model}-$5-np${num_partitions}-pr$4-lcm$3
    # save_name=${dataset}-${model}-$5-np$2-vers${version}-pr$4-lcm$3-lm${teacher}
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -mf ${save_name}.pt \
        --device $1 \
        -np ${num_partitions} \
        -st ${st} \
        -pfl -lcm $3 -pr $4 -co \
        -lr 0.01 \
        -ep 1 \
        -ree 1 \
        -relr 0.001 \
        >logs/${save_name}.out
}

prune_finetune "cuda:${cuda_device}" v${yaml_version} 0.001 ${prune_ratio} kernel

