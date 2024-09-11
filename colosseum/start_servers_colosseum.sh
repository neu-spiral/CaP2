#!/bin/bash
#   Start split model servers on  each snr node 
#   Should be done before ./setup.sh and ./build_routes.sh
#
#   Example:
#   ./start_servers_colosseum.sh "./nodes_test.txt" "./ip-map.json" "./network-graph.json" "cifar10-resnet18-kernel-npv0-pr0.75-lcm0.001.pt"
#
# Path to the JSON file
nodes_file=$1
ip_map_file=$2
network_graph_file=$3
model_file=$4 # this doesnt need full path, io utils handle it. It does need the extension in the name .pt

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
        ;;
        wifi)
        psswrd="sunflower"
        ;;
        server)
        psswrd="ChangeMe"
        ;;
    esac

    # move locally built configs to nodes
    echo "Copying JSONs to node $srn_name($node_number)"
    sshpass -p "$psswrd" scp "$ip_map_file" "$network_graph_file" "$srn_name":/root/CaP/colosseum

    # start servers on node 
    echo "Starting terminal session"
    gnome-terminal -- bash -c "sshpass -p '$psswrd' ssh '$srn_name' 'cd /root/CaP && source env.sh && source ../cap-310/bin/activate && python3 run_split_model.py "colosseum/$ip_map_file" "colosseum/$network_graph_file" "$node_number" "$model_file"; bash'" &

    echo ""
    echo ""

done < "$nodes_file"

