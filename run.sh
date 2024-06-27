#!/bin/bash

cuda_device=1
yaml_version=0 # meaning no yaml file is used, num_partitions=2,4,8 is used

# prune_ratio=0.5
# num_partitions=2
prune_ratio=0.75
num_partitions=4
# prune_ratio=0.875
# num_partitions=8


# yaml_version=1
# prune_ratio=0.5

# yaml_version=2
# prune_ratio=0.75

# yaml_version=3
# prune_ratio=0.875






# dataset=cifar10
# model=resnet18

# dataset=cifar100
# model=wrn28

# dataset=esc
# model=escnet

dataset=flash
model=flashnet


# teacher=cifar10-resnet18-kernel-npv0.pt

# teacher=cifar100-wrn28-kernel-npv0.pt

# teacher=esc-escnet-kernel-npv0.pt

teacher=flash-flashnet-kernel-npv0.pt


# teacher=''
# 
# LOAD MODEL YAPARKEN USTTEKINI KOYMAYI UNUTMA, -lm yi koy
# -lm ${teacher} \

# -np config/${model}-$2.yaml

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


#pretrainli yaparsan -lm i commentleyip, teacher='' olmasina gerek yok btw (cunku teacher i kullanan yok -lm haric), 
#model adi = cifar10-resnet18-kernel-npv2-versx.pt, log u : cifar10-resnet18-kernel-npv2-versx-pr0.75-lcm0.001.out
#sonra pruned, ondan sonra finetuned model adi = cifar10-resnet18-kernel-npv2-versx-pr0.75-lcm0.001.pt

# lm olunca , teacher=cifar10-resnet18-teacher.pt, model adi = cifar10-resnet18-kernel-npv2-versx-pr0.875-lcm0.001.pt
# log u = cifar10-resnet18-kernel-npv2-vers0-pr0.875-lcm0.001.out




# cifar10-resnet18 in pretrain i pr0.75-lcm0.001'de yapildi, cifar10-resnet18-kernel-npv2-vers2-pr0.75-lcm0.001.out'da logu, 
# digerleri full -lm, pr0.875-lcm0.001 gibi

