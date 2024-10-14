#!/bin/bash
#   Start split model servers on  each snr node 
#   Should be done before ./setup.sh and ./build_routes.sh
#
#   Example:
#   ./start_servers_colosseum.sh "./nodes.txt" "./ip-map.json" "./network-graph.json" "cifar10-resnet18-kernel-npv0-pr0.75-lcm0.001.pt" "log_folder" 16
#

# use in file inputs
nodes_file="./nodes.txt" # Path to the JSON file
ip_map_file="./ip-map.json"
network_graph_file="./network-graph.json"
leaf_file="./config-leaf.json"
model_file="cifar10-resnet18-kernel-np4-pr0.5-lcm0.0001.pt" # this doesnt need full path, io utils handle it. It does need the extension in the name .pt
log_dur_name="/logs/cifar10-resnet18-kernel-np4-pr0.5-lcm0.0001"
batch_size=16

# use cmd line inputs
:'
nodes_file=$1 # Path to the JSON file
ip_map_file=$2
network_graph_file=$3
model_file=$4 # this doesnt need full path, io utils handle it. It does need the extension in the name .pt
log_dur_name="/logs/$5"
batch_size=$6
'

# Read the input file line by line
# iterate through each srn node
while IFS= read -r line; do

    #echo $line
    node_type=$(echo "$line" | cut -d'-' -f1)
    srn_number=$(echo "$line" | cut -d'-' -f2)
    node_number=$(echo "$line" | cut -d'-' -f3)
    is_leaf=$(echo "$line" | cut -d'-' -f5)

    srn_name="genesys-$srn_number"

    # get password
    case "$node_type" in
        cell)
        psswrd="scope"
        pyenv="cap-310" # TODO: configure cell container and update 
        ;;
        wifi)
        psswrd="sunflower"
        pyenv="cap-39"
        ;;
        server)
        psswrd="ChangeMe"
        pyenv="cap-310"
        ;;
    esac

    # move locally built configs to nodes
    echo "Copying JSONs to node $srn_name($node_number)"
    sshpass -p "$psswrd" scp "$ip_map_file" "$network_graph_file" "$leaf_file" "$srn_name":/root/CaP/colosseum

    # start servers on node 
    echo "Starting terminal session"
    gnome-terminal -- bash -c "sshpass -p '$psswrd' ssh '$srn_name' 'cd /root/CaP && source env.sh && source ../$pyenv/bin/activate && python3 run_split_model.py "colosseum/$ip_map_file" "colosseum/$network_graph_file" "$node_number" "$model_file" "$log_dur_name" -b $batch_size --debug "False" 2>&1 | tee output.log; bash'" &
    #terminator -- bash -c "sshpass -p '$psswrd' ssh '$srn_name' 'cd /root/CaP && source env.sh && source ../$pyenv/bin/activate && python3 run_split_model.py "colosseum/$ip_map_file" "colosseum/$network_graph_file" "$node_number" "$model_file" "$log_dur_name" -b $batch_size --debug "False" 2>&1 | tee output.log; bash'" &

    echo ""
    echo ""

done < "$nodes_file"

