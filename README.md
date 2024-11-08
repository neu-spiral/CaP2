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
7. Send inputs:
~~~
# Windows 
python -m source.utils.send_start_message [path to config-leaf.json]

# Linux
python -m ./source/utils/send_start_message.py [path to config-leaf.json]
~~~

Ouputs will appear in the logs/[dir log out] folder specified in the start servers script. Post processing and visaulization tools are found in sandbox/plot_timing.ipynb

## Split Network Inference on Colosseum 
Example colosseum run procedure. This assumes full and split model files have been loaded onto the file-proxy server at /share/nas/[team name]/CaP-Models/perm beforehand (TODO: generalize, add detail, and verify works):
1. Connect to VPN via cisco
2. Make a reservation. Use the CAP-wifi-v1 container for WiFi nodes and JARVIS-server-cap1 for server nodes [TODO: make and test container for UE and base station nodes]
3. While waiting for SRN nodes to spin up, modify the bash scripts in the CaP/colosseum folder:  
  i. Manually configure colosseum/nodes.txt with the correct SRN numbers  
  ii. In prep_run.sh, update leaf node connection type [WARNING: not tested for heterogeneous networks] and rf scenario (see colosseum documentation for more details)  
  iii. In ./start_servers_colosseum.sh, select model file, batch size, and specify log output directory name  
  iv. In ./start_run.sh, uncomment/comment commands based on SRN type  
6. Open bash session in folder CaP/colosseum
7. Move repo to SRN nodes, start rf, collect ip addresses, and build json config: 
  ~~~
  bash ./prep_run.sh
  ~~~
9. Start servers on SRN nodes for split model execution (NOTE: resnet101 models take 1-2 minutes to load):
  ~~~
  bash ./start_servers_colosseum.sh
  ~~~
10. Send starting message to nodes:  
  ~~~ 
  bash ./start_run.sh [srn #]
  ~~~
11. Kill servers (can also be used to kill RF scenario, see script comments for details)
  ~~~
  bash ./kill_servers.sh
  ~~~
12. Update the log file name in CaP/colosseum/start_servers_colosseum.sh (separate run outputs) and repeat steps 9-11 for next run until finished with all runs
13. Inspect logging messages saved to colosseum's file-proxy server /share/nas/[team name] after end of reservation 

## Cite
```
@article{jian2023cap,
  title={Communication-Aware DNN Pruning},
  author={Jian, Tong and Roy, Debashri Roy and Salehi, Batool and Soltani, Nasim and Chowdhury, Kaushik and Ioannidis, Stratis}
  journal={INFOCOM},
  year={2023}
}
```
