#!/bin/bash
#
# start local servers to emulate split network behavoir. Assumes run from top repo directory 
#

echo $DISPLAY

# Path to the JSON file
ip_map_file="./config/resnet_4_network/ip-map.json"
network_graph_file="./config/resnet_4_network/network-graph.json"
model_file="cifar10-resnet18-kernel-npv0-pr0.75-lcm0.001.pt" # this doesnt need full path, io utils handle it
log_dir_name='full_model_load'
debug='False' # checks each split model output when true

for key in $(jq -r 'keys[]' $ip_map_file); do

    # Spawn a new terminal and run the Python function directly
    # this implementation is using windows wsl, change the following based on platform:
    #gnome-terminal -- bash -c "

    # use this if in Putty/xming ssh session 
    #gnome-terminal -- bash -c "source ../cap-env/bin/activate & python "run_split_model.py" $ip_map_file $network_graph_file $((key)) $model_file $log_dir_name $debug; bash'" &

    # use this if on local linux/OSX machine
    dbus-launch terminator -e "source ../cap-310/bin/activate && python 'run_split_model.py' $ip_map_file $network_graph_file $((key)) $model_file $log_dir_name $debug; bash" &

done

