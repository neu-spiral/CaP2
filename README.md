# Official Implementation of CaP @ INFOCOM 23

This codebase contains the implementation of **[Communication-Aware DNN Pruning] (INFOCOM2023)**.

## Introduction
We propose a Communication-aware Pruning (CaP) algorithm, a novel distributed inference framework for distributing DNN computations across a physical network. 
Departing from conventional pruning methods, CaP takes the physical network topology into consideration and produces DNNs that are communication-aware, designed for both accurate and fast execution over such a distributed deployment. 
Our experiments on CIFAR-10 and CIFAR-100, two deep learning benchmark datasets, show that CaP beats state of the art competitors by up to 4% w.r.t. accuracy on benchmarks. 
On experiments over real-world scenarios, it simultaneously reduces total execution time by 27%--68% at negligible performance decrease (less than 1%).
<p align="center">
<img src="./intro.png" width="850" height="400">
</p>


## Environment Setup
Please install either python 3.9.X or 3.10.X and create a virtual environment using the requirements.txt file.


## Instructions
We provide a sample bash script to run our method at 0.75 sparsity ratio on CIFAR-10.


To run CaP:

```bash
source env.sh
run-cifar10-resnet18.sh
```

## Split Network Emulation
Model inference over a network can be emulated by starting multiple threads on a local or remote machine. For windows users, it is assumed that WSL or another bash emulator is installed. The following is the procedure for making a run:

1. Train and prune the model
2. Save the model in the assets/models folder
3. Create the network setup files (see config/resnet_4_network as example):  
  i. config-leaf.json -- indicates how to reach (via ip and port) leaf nodes of network for input transmission  
  ii. ip-map.json -- indicates server (ip and port) that each node monitors for incoming connections  
  iii. network-graph.json -- defines network graph topology 
4. Update local_network/start_servers.sh and local_network/start_server_helper.bat (windows) or local_network/start_servers_linux.sh (linux) with the following:  
 i. file paths to network setup files  
 ii. the python environment activation  
 iii. terminal/bash emulator (e.g. gnome-terminal, terminator, etc.).  
 iv. model name 
5. Setup the servers (from directory ./CaP):  
~~~
  # Windows (with wsl) 
  bash local_network/start_servers.sh 

  # Linux 
  bash local_network/start_servers_linus.sh
~~~
6. Activate python environment  
7. Send inputs (WARNING only works for cifar10 inputs).:
~~~
# Windows 
python -m source.utils.send_leaf_split_model [path to config-leaf.json]

# Linux
python -m ./source/utils/send_leaf_split_model.py [path to config-leaf.json]
~~~

Ouputs will appear in the logs/[dir log out] folder specified in the start servers script. Post processing and visaulization tools are found in sandbox/plot_timing.ipynb

## Split Network Inference on Colosseum 
Example colosseum run procedure (TODO: generalize, add detail, and verify works):
1. connect to VPN via cisco
2. make reservation
3. wait until srn nodes are spun up
4. manually configure colosseum/nodes.txt
5. wait until srn nodes are running
6. open bash session in CaP/colosseum repo
7. move repo to snr nodes, start rf, and collect ip addresses:
  bash ./setup.sh nodes_test.txt cifar10-resnet18-kernel-npv2-pr0.75-lcm0.001
9. start servers for running split model:
  bash ./start_servers_colosseum.sh "./nodes_test.txt" "./ip-map.json" "./network-graph.json" "cifar10-resnet18-kernel-npv2-pr0.75-lcm0.001.pt"
10. send inputs to leaf nodes:  
  open windows terminal in CaP repo  
  conda activate cap_nb  
  source env.sh  
  python -m send_leaf_split_model colosseum/config-leaf.json  
  OR  
  gnome-terminal -- bash -c "sshpass -p ChangeMe ssh genesys-115 'cd /root/CaP && source env.sh && source ../cap-310/bin/activate && python3 -m send_leaf_split_model colosseum/config-leaf.json; bash '" &


## Cite
```
@article{jian2023cap,
  title={Communication-Aware DNN Pruning},
  author={Jian, Tong and Roy, Debashri Roy and Salehi, Batool and Soltani, Nasim and Chowdhury, Kaushik and Ioannidis, Stratis}
  journal={INFOCOM},
  year={2023}
}
```
