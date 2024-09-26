#!/bin/bash

# Dataset and model selections, should be selected in these combinations given:

# dataset=cifar10
# model=resnet18

dataset=cifar100
model=resnet101

# dataset=esc
# model=EscFusion

# dataset=flash
# model=InfoFusionThree

teacher=${dataset}-${model}.pt
# Put this below as a parameter in order to avoid pre-training, and use a teacher model. If not, pre-training will be performed:
# -lm ${teacher} \

prune_finetune() {
    log_name=${dataset}-${model}-$(date +"%Y-%m-%d_%H:%M:%S")
    python -m source.core.run_partition \
           -cfg config/${dataset}.yaml \
           >logs/${log_name}.out
}

prune_finetune