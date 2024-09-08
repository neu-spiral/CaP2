#!/bin/bash
## HOW TO RUN: 
#   bash ./build_routes_test.sh nodes_test.txt ip-map.json config-leaf.json network-graph.json
#   
#   INPUTS:
#       nodes_test.txt -- maps together: colosseum node srn, network node #, network node edges, is leaf node
#                         Expects nodes_test.txt to have a line for each network node in the network. 
#                               [node type]-[srn number]-[network node number]-[edges/nodes that this node sends to]-[bool is leaf node?]
#       ip-map.json -- filepath of ip/routing table, this will be overwritten if exists
#       config-leaf.json -- filepath list of ip/ports to send model inputs to, this will be overwritten if exists
#
#   OUTPUT:
#       ip-map.json
#       config-leaf.json

# port start
#port_start = 5000

final_node=1

# make and or clear json obejcts
echo {} >$2
echo {} >$3
echo {} >$4

# declare arrays for properties of network nodes
declare -A node_type
declare -A srn_number
declare -a network_node
declare -A edges
declare -A is_leaf

# Read the input file line by line
while IFS= read -r line; do

    #echo $line

    node=$(echo "$line" | cut -d'-' -f3)
    network_node+=($node)
    node_type["$node"]=$(echo "$line" | cut -d'-' -f1)
    srn_number["$node"]=$(echo "$line" | cut -d'-' -f2)
    is_leaf["$node"]=$(echo "$line" | cut -d'-' -f5)

    # associative arrays dont support array type values, store as string and parse later
    edges["$node"]=$(echo "$line" | cut -d'-' -f4) 

done < "$1"

# Iterate through network nodes

# Iterate through all combinations of tx and rx
for node_tx in "${network_node[@]}"; do 

    #echo "RX  : $node_rx"

    # parse string of edges
    tmp=${edges[$node_tx]}
    IFS=',' read -ra tmp_edge_array <<< "$tmp"

    python3 -m build_network_graph --network_file $4 --edges $tmp --tx_node $node_tx --tx_node_type "${node_type["$node_tx"]}" --final_node $final_node

    for node_rx in "${network_node[@]}"; do

        # skip if tx and rx are the same node
        if [ "$node_rx" != "$node_tx" ]; then
            #echo -e "\tTX : $node_tx"


            # check if a connection exists
            for e in "${tmp_edge_array[@]}"; do 
                #echo -e "\t\tEDGE: $e"
                if [[ "$e" == "$node_rx" ]]; then

                    # get ip/port 
                    ip_rx="127.0.0.1"
                    port_rx=5000

                    # pass info to python sript to build json
                    python3 -m build_ip_map_json --ip_file $2 --node_rx $node_rx --ip_rx $ip_rx --port_rx $port_rx --type_tx "${node_type["$node_tx"]}" 
                    
                fi 
            done
        fi 
    
    done 

    # if it's a leaf node add it to the leaf JSON
    leaf_connection_type='server'
    if [ "${is_leaf[$node_rx]}" -eq 1 ]; then

        # get port/ip
        ip_rx="127.0.0.1"
        port_rx=5000

        python3 -m build_leaf_json --leaf_file $3 --leaf_node $node_rx --ip $ip_rx --port $port_rx --connection_type $leaf_connection_type

    fi
done


