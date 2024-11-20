#!/bin/bash
#
# start local servers to emulate split network behavoir. Assumes run from top repo directory 
#

# Path to the JSON file
ip_map_file="./config/resnet_8_network/ip-map.json"
ip_map_file_win=".\config\resnet_8_network\ip-map.json"
network_graph_file=".\config\resnet_8_network\network-graph.json"

model_file="cifar10-resnet18.pt" # this doesnt need full path, io utils handle it
#model_file="esc-EscFusion-kernel-np4-pr0.5-lcm10.pt" # this doesnt need full path, io utils handle it

log_dir_name='.\local_logs\test' # saves logging messages to ./logs/[log_dir_name] 
debug='True' # checks each split model output and calculates FLOPS when true

for key in $(jq -r 'keys[]' $ip_map_file); do

    # convert to windows path for wsl to find 
    cmd.exe /c start ".\local_network\start_server_helper.bat" $ip_map_file_win $network_graph_file $((key)) $model_file "./logs/$log_dir_name" $debug
done

