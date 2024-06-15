#!/bin/bash

dataset=cifar100
model=wrn28

#teacher=cifar10-resnet18-teacher.pt
#teacher=cifar10-resnet18-kernel-npv2-pr0.pt # prune ratio was 75%
#teacher=cifar10-resnet18-kernel-npv2-pr0.75-lcm0.001.pt

# where is mommentum set? paper says 0.9
# where is weight decay set? 10^-4

# ep == adam epochs when unspecified. See "engine.py > MoP.prune() > misc.py > set_optimizer()"


prune_finetune() {
    st=$5
    save_name=${dataset}-${model}-$5-np$2-pr$4-lcm$3
    python -m source.core.run_partition -cfg config/${dataset}.yaml \
        -mf ${save_name}.pt  \
        --device $1 -np config/wrn28-$2.yaml -st ${st} -pfl -lcm $3 -pr $4 -co \
        -lr 0.01 -ep 300 -ree 150 -relr 0.001 \
        >logs/${save_name}.out
}

prune_finetune "cuda:0" v1 0.001 0.50 kernel
