#!/bin/bash

cuda_device=2
yaml_version=0 # meaning no yaml file is used, num_partitions=2,4,8 is used


# Prune ratios and num_partitions, should be selected in these combinations given:

# prune_ratio=0.5
# num_partitions=2

prune_ratio=0.75
num_partitions=4

# prune_ratio=0.875
# num_partitions=8



# For yaml file selection, not used anymore. Old parameter was '-np config/${model}-$2.yaml':

# yaml_version=1
# prune_ratio=0.5

# yaml_version=2
# prune_ratio=0.75

# yaml_version=3
# prune_ratio=0.875




# Dataset and model selections, should be selected in these combinations given:

# dataset=cifar10
# model=resnet18

# dataset=cifar100
# model=wrn28

# dataset=esc
# model=escnet

dataset=flash
model=flashnet


# Select teacher model with respect to the dataset and model:

# teacher=cifar10-resnet18-kernel-npv0.pt
# teacher=cifar100-wrn28-kernel-npv0.pt
# teacher=esc-escnet-kernel-npv0.pt
teacher=flash-flashnet-kernel-npv0.pt


# Put this below as a parameter in order to avoid pre-training, and use a teacher model. If not, pre-training will be performed:
# -lm ${teacher} \


prune_finetune() {
    st=$5
    save_name=${dataset}-${model}-$5-np$2-pr$4-lcm$3
    # save_name=${dataset}-${model}-$5-np$2-vers${version}-pr$4-lcm$3-lm${teacher}
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -mf ${save_name}.pt \
        --device $1 \
        -np ${num_partitions} \
        -st ${st} \
        -pfl -lcm $3 -pr $4 -co \
        -lr 0.01 \
        -ep 300 \
        -ree 100 \
        -relr 0.001 \
        >logs/${save_name}.out
}

prune_finetune "cuda:${cuda_device}" v${yaml_version} 0.001 ${prune_ratio} kernel

