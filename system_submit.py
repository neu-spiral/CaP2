import os
import re

import numpy as np
from subprocess import Popen
import pdb

import time
import numpy as np
import random

os.makedirs("logs", exist_ok=True)

prune_ratios = [0.5, 0.7, 0.75, 0.8, 0.85]

# lambda_comms = [0.000001, 0.00001, 0.0001]
lambda_comms = [0.0001]
# lambda_comms = [10, 100, 1000]

# num_partitions = ['2', '8', '16']
num_partitions = ['4']

device = 'cuda:2'

# dataset="cifar10"
# model="resnet18"

dataset="cifar100"
model="resnet101"

# dataset="esc"
# model="EscFusion"

if __name__ == "__main__":
    for prune_ratio in prune_ratios:
        for lambda_comm in lambda_comms:
            for num_partition in num_partitions:
                log_name = f"{dataset}-{model}-np{num_partition}-pr{prune_ratio}-lcm{lambda_comm}"
                file = f"--prune_ratio={prune_ratio} "\
                f"--lambda_comm={lambda_comm} "\
                f"--num_partition={num_partition} "\
                f"-cfg=config/{dataset}.yaml"

                file_full = f"python -m source.core.run_partition {file} > logs/{log_name}.out"
                print(f'Running: {file_full}')
                os.system(file_full)
                time.sleep(0.01)