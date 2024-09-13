#!/bin/bash

# Path to the JSON file
ip_map_file="./config/resnet_4_network/ip-map.json"
ip_map_file_win=".\config\resnet_4_network\ip-map.json"
network_graph_file=".\config\resnet_4_network\network-graph.json"
model_file="cifar10-resnet18-kernel-npv0-pr0.75-lcm0.001.pt" # this doesnt need full path, io utils handle it
debug='False' # checks each split model output when true

for key in $(jq -r 'keys[]' $ip_map_file); do

    # Spawn a new terminal and run the Python function directly
    # this implementation is using windows wsl, change the following based on platform:
    #gnome-terminal -- bash -c "

    # convert to windows path for wsl to find 
    cmd.exe /c start ".\colosseum\start_server_helper.bat" $ip_map_file_win $network_graph_file $((key)) $model_file $debug
done

