#!/bin/bash
## HOW TO RUN: ./setup.sh node_list.txt [model name e.g. cifar10-resnet18-kernel-npv0-pr0.75-lcm0.001]
## node_list.txt should be formatted like:
## type-srnnumber-network node
## type-srnnumber-network node (etc)
## TODO: include support for nodes using only part of the entire model, only copy over necessary models to machine instead of entire CaP-Models folder 
## Expects model name in argument to be on the file-proxy server here: /share/nas/genesys/CaP-Models
## if you are using cell, the base station is always the first node

# Initialize a flag to track if cell or wifi nodes are encountered
rf_flag=false # not needed remove

node_file="nodes.txt"
leaf_connection_type='server' # CHANGE ME

rf_scenario=51005 # select an RF scenario from the list 

# TODO: remove hardcoded params for 2nd half
final_node=1
starting_port=1009 # Starting port definition TODO: increment this? Right not this does not change
ip_map_file='ip-map.json'
leaf_file='config-leaf.json'
graph_file='network-graph.json'

# Get the number of lines in the file
num_lines=$(wc -l < "$node_file")

# Loop through each line number
for ((i=1; i<=num_lines; i++)); do
  # Read the line using sed
  line=$(sed -n "${i}p" "$node_file")

  # Split the line into three variables
  node_type=$(echo "$line" | cut -d'-' -f1)
  number=$(echo "$line" | cut -d'-' -f2)
  node=$(echo "$line" | cut -d'-' -f3)

  # Add the prefix 'genesys-' to the number portion
  prefixed_number="genesys-$number"

  echo "Configuring $prefixed_number as a $node_type node"
  sleep 1
  # Perform different configurations based on the node_type
  case "$node_type" in
    cell)
      # Configuration for cell type
      echo "Running configuration for cell type..."
      if [ "$rf_flag" = false ]; then
        echo "Starting rf scenario"
        sshpass -p "scope" ssh "$prefixed_number" "colosseumcli rf start $rf_scenario -c"
        rf_flag=true
        sleep 5
      fi
      sshpass -p "scope" scp ./jarvis.conf $prefixed_number:/root/radio_api/
      sleep 2
      sshpass -p "scope" ssh "$prefixed_number" "cd /root/radio_api && python3 scope_start.py --config-file jarvis.conf" &
      echo "Letting scope start"
      sleep 20  # Let the node start before moving on

      # Sync local repo with the remote
      echo "Syncing local repo with $prefixed_number"
      sshpass -p "scope" rsync -avz --exclude='.git' --exclude='assets/' --exclude='logs/' --exclude='local_logs/' --exclude='demo_logs/' ../../CaP/ $prefixed_number:/root/CaP/
      sleep 1

      # Copy model to network node      
      echo "Copying all models to $prefixed_number"
      sshpass -p "scope" ssh "$prefixed_number" "su srn-user -c 'cp -r -u /share/CaP-Models/perm/ /tmp/'"
      sshpass -p "scope" ssh "$prefixed_number" "mkdir -p /root/CaP/assets/models/ && cp -r -u /tmp/perm/ /root/CaP/assets/models/" &
      sleep 2
      
      ;;

    wifi)
      # Configuration for wifi type
      echo "Running configuration for wifi type..."
      if [ "$rf_flag" = false ]; then
        echo "Starting rf scenario"
        sshpass -p "sunflower" ssh -n "$prefixed_number" "colosseumcli rf start $rf_scenario -c"
        rf_flag=true
        
        wait
      fi
      sshpass -p "sunflower" ssh -n "$prefixed_number" "cd interactive_scripts && ./tap_setup.sh"
      wait 
      gnome-terminal -- bash -c "sshpass -p 'sunflower' ssh -n '$prefixed_number' 'cd interactive_scripts && ./modem_start.sh'; bash"
      #mintty -- bash -c "sshpass -p 'sunflower' ssh '$prefixed_number' 'cd interactive_scripts && ./modem_start.sh'; bash"
      wait

      # Sync local repo with the remote
      echo "Syncing local repo with $prefixed_number"
      sshpass -p "sunflower" rsync -avz --exclude='.git' --exclude='assets/' --exclude='logs/' --exclude='local_logs/' --exclude='demo_logs/'  ../../CaP/ $prefixed_number:/root/CaP/
      sleep 1

      # Copy model to network node      
      echo "Copying all models to $prefixed_number"
      sshpass -p "sunflower" ssh "$prefixed_number" "su srn-user -c 'cp -r -u /share/CaP-Models/perm/ /tmp/'"
      sshpass -p "sunflower" ssh "$prefixed_number" "mkdir -p /root/CaP/assets/models/ && cp -r -u /tmp/perm/ /root/CaP/assets/models/" &
      sleep 2

      ;;

    server)
      # Configuration for server type
      echo "Running configuration for server type..."

      # Sync local networks-for-ai directory with the remote
      echo "Syncing local repo with $prefixed_number"
      sshpass -p "ChangeMe" rsync -avz --exclude='.git' --exclude='assets/' --exclude='logs/' --exclude='local_logs/' --exclude='demo_logs/' ../../CaP/ $prefixed_number:/root/CaP/
      sleep 1

      # Copy model to network node      
      echo "Copying all models to $prefixed_number"
      sshpass -p "ChangeMe" ssh "$prefixed_number" "su srn-user -c 'cp -r -u /share/CaP-Models/perm/ /tmp/'"
      sshpass -p "ChangeMe" ssh "$prefixed_number" "mkdir -p /root/CaP/assets/models/perm/ && cp -r -u /tmp/perm/ /root/CaP/assets/models/" &
      sleep 2
      
      ;;
    
    *)
      # Handle unknown types here
      echo "Unknown type: $node_type"
      ;;
  esac

  echo ""
  echo ""

done

# Wait for background processes to complete
wait

## HOW TO RUN: 
#   bash ./build_routes_test.sh nodes.txt ip-map.json config-leaf.json network-graph.json
#   
#   INPUTS:
#       nodes.txt -- maps together: colosseum node srn, network node #, network node edges, is leaf node
#                         Expects nodes.txt to have a line for each network node in the network. 
#                               [node type]-[srn number]-[network node number]-[edges/nodes that this node sends to]-[bool is leaf node?]
#       ip-map.json -- filepath of ip/routing table, this will be overwritten if exists
#       config-leaf.json -- filepath list of ip/ports to send model inputs to, this will be overwritten if exists
#
#   OUTPUT:
#       ip-map.json
#       config-leaf.json

# make and or clear json obejcts
echo {} >$ip_map_file
echo {} >$leaf_file
echo {} >$graph_file

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

done < "$node_file"

# Iterate through network nodes

# Iterate through all combinations of tx and rx
a_port=$starting_port
for node_tx in "${network_node[@]}"; do 

    tx_node_type="${node_type["$node_tx"]}"
    tx_node_srn="genesys-${srn_number["$node_tx"]}"

    # parse string of edges
    tmp=${edges[$node_tx]}
    IFS=',' read -ra tmp_edge_array <<< "$tmp"

    echo "Adding to network graph $tx_node_srn as network node=$node_tx"
    python3 -m build_network_graph --network_file $graph_file --edges $tmp --tx_node $node_tx --tx_node_type $tx_node_type --final_node $final_node

    for node_rx in "${network_node[@]}"; do

        rx_node_type="${node_type["$node_rx"]}"
        rx_node_srn="genesys-${srn_number["$node_rx"]}"

        # skip if tx and rx are the same node
        if [ "$node_rx" != "$node_tx" ]; then
            #echo -e "\tTX : $node_tx"


            # check if a connection exists
            for e in "${tmp_edge_array[@]}"; do 
                #echo -e "\t\tEDGE: $e"
                if [[ "$e" == "$node_rx" ]]; then

                    ### BEGIN INTEGRATION

                    echo "Finding ip address for node $node_tx to node $node_rx ($rx_node_type) connection"
                    
                    # Get correct IP for next node based on its type
                    if [[ "$rx_node_type" == "server" ]]; then
                        rx_host_ip=$(sshpass -p "ChangeMe" ssh "$rx_node_srn" 'ifconfig col0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
                        echo "Setting rx_host_ip IP for node $rx_node_srn(server): $rx_host_ip"

                        # check to see if this is a UE
                        if sshpass -p "scope" ssh "$tx_node_srn" 'ifconfig tun_srsue'; then
                        tun_srsue_ip=$(sshpass -p "scope" ssh "$tx_node_srn" 'ifconfig tun_srsue' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
                        echo "Adding route to UE with: ip route add $rx_host_ip via $tun_srsue_ip"
                        sshpass -p "scope" ssh "$tx_node_srn" "ip route add $rx_host_ip via $tun_srsue_ip"
                        fi

                    elif [[ "$rx_node_type" == "wifi" && "$rx_node_type" != "wifi" ]]; then
                        rx_host_ip=$(sshpass -p "sunflower" ssh "$rx_node_srn" 'ifconfig col0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
                        echo "Setting rx_host_ip IP for node $rx_node_srn(wifi-col0): $rx_host_ip"

                    elif [[ "$rx_node_type" == "wifi" ]]; then
                        rx_host_ip=$(sshpass -p "sunflower" ssh "$rx_node_srn" 'ifconfig tap0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
                        echo "Setting rx_host_ip IP for node $rx_node_srn(wifi): $rx_host_ip"

                    elif [[ "$rx_node_type" == "cell" ]]; then
                        # Check if it is a UE
                        if sshpass -p "scope" ssh "$rx_node_srn" 'ifconfig tun_srsue'; then
                        echo "rx_host_ip is a UE"
                        rx_host_ip=$(sshpass -p "scope" ssh "$rx_node_srn" 'ifconfig tun_srsue' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
                        if [[ "$tx_node_type" == "server" ]]; then
                            echo "Adding route to cell network via colab network with: ip route add $rx_host_ip via $tx_host_ip"
                            sshpass -p "ChangeMe" ssh "$tx_node_srn" "ip route add $rx_host_ip via $tx_host_ip"
                        elif [[ "$tx_node_type" == "wifi" ]]; then
                            wifi_col0=$(sshpass -p "sunflower" ssh "$tx_node_srn" 'ifconfig col0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
                            echo "Adding route to cell network via colab network with: ip route add $rx_host_ip via $wifi_col0"
                            sshpass -p "sunflower" ssh "$tx_node_srn" "ip route add $rx_host_ip via $wifi_col0"
                        else
                            echo "unknown node type"
                        fi
                        echo "Adding route on UE to colab network via cell network with: ip route add $prev_host vi $rx_host_ip"
                        sshpass -p "scope" ssh "$rx_node_srn" "ip route add $prev_host via $rx_host_ip"

                        # Check if it is a gNB
                        elif sshpass -p "scope" ssh "$rx_node_srn" 'ifconfig srs_spgw_sgi'; then
                        echo "rx_host_ip is a gNB"
                        # Check if the current node is a cell or not
                        if [[ "$rx_node_type" == "cell" ]]; then
                            rx_host_ip=$(sshpass -p "scope" ssh "$rx_node_srn" 'ifconfig srs_spgw_sgi' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
                        else
                            rx_host_ip=$(sshpass -p "scope" ssh "$rx_node_srn" 'ifconfig col0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
                        fi
                        else
                        echo "error finding next node"
                        fi
                        echo "Setting the rx_host_ip IP for node $rx_node_srn(cell): $rx_host_ip"
                    else
                        # For other node types, replace this with logic to retrieve the IP
                        echo "rx_host_ip unknown case"
                    fi

                    case "$rx_node_type" in
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
                    echo "testing connection"
                    sshpass -p "$psswrd" ssh "$tx_node_srn" "ping $rx_host_ip -c 4"


                    ### END INTEGRATION

                    # pass info to python sript to build json
                    echo "Adding ${node_type["$node_tx"]} connection to for $node_rx : ip = '$rx_host_ip'"
                    python3 -m build_ip_map_json --ip_file $ip_map_file --node_rx $node_rx --ip_rx $rx_host_ip --port_rx $a_port --type_tx "${node_type["$node_tx"]}" 
                    
                    # use same info for leaf node if necessary
                    if [ "${is_leaf[$node_rx]}" -eq 1 ]; then
                        if [[ "${node_type["$node_rx"]}" ==  $leaf_connection_type ]]; then
                            python3 -m build_leaf_json --leaf_file $leaf_file --leaf_node $node_rx --ip $rx_host_ip --port $a_port --connection_type $leaf_connection_type
                            echo "Setting the leaf_host_ip IP for node $rx_node_srn($leaf_connection_type): $rx_host_ip"
                            is_leaf[$node_rx]=0 # indicate this has been taken care of
                        fi
                    fi
                    a_port=$((a_port+1)) # increment port

                fi 
            done
        fi 
    
    done 

    # if it's a leaf node add it to the leaf JSON
    if [ "${is_leaf[$node_tx]}" -eq 1 ]; then

        ### BEGIN INTEGRATION

        # Get correct IP for next node based on its type
        if [[ "$tx_node_type" == "server" ]]; then
            leaf_host_ip=$(sshpass -p "ChangeMe" ssh "$tx_node_srn" 'ifconfig col0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
            echo "Setting leaf_host_ip IP for node $tx_node_srn(server): $leaf_host_ip"

            # check to see if this is a UE
            if sshpass -p "scope" ssh "$tx_node_srn" 'ifconfig tun_srsue'; then
            tun_srsue_ip=$(sshpass -p "scope" ssh "$tx_node_srn" 'ifconfig tun_srsue' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
            echo "Adding route to UE with: ip route add $leaf_host_ip via $tun_srsue"
            sshpass -p "scope" ssh "$tx_node_srn" "ip route add $leaf_host_ip via $tun_srsue"
            fi

        elif [[ "$tx_node_type" == "wifi" && "$tx_node_type" != "wifi" ]]; then
            leaf_host_ip=$(sshpass -p "sunflower" ssh "$tx_node_srn" 'ifconfig col0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
            echo "Setting leaf_host_ip IP for node $tx_node_srn(wifi-col0): $leaf_host_ip"

        elif [[ "$tx_node_type" == "wifi" ]]; then
            leaf_host_ip=$(sshpass -p "sunflower" ssh "$tx_node_srn" 'ifconfig tap0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
            echo "Setting leaf_host_ip IP for node $tx_node_srn(wifi): $leaf_host_ip"

        elif [[ "$tx_node_type" == "cell" ]]; then
            # Check if it is a UE
            if sshpass -p "scope" ssh "$tx_node_srn" 'ifconfig tun_srsue'; then
            echo "leaf_host_ip is a UE"
            leaf_host_ip=$(sshpass -p "scope" ssh "$tx_node_srn" 'ifconfig tun_srsue' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
            if [[ "$tx_node_type" == "server" ]]; then
                echo "Adding route to cell network via colab network with: ip route add $leaf_host_ip via $prev_host"
                sshpass -p "ChangeMe" ssh "$tx_node_srn" "ip route add $leaf_host_ip via $prev_host"
            elif [[ "$tx_node_type" == "wifi" ]]; then
                wifi_col0=$(sshpass -p "sunflower" ssh "$tx_node_srn" 'ifconfig col0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
                echo "Adding route to cell network via colab network with: ip route add $leaf_host_ip via $wifi_col0"
                sshpass -p "sunflower" ssh "$tx_node_srn" "ip route add $leaf_host_ip via $wifi_col0"
            else
                echo "unknown node type"
            fi
            echo "Adding route on UE to colab network via cell network with: ip route add $prev_host vi $leaf_host_ip"
            sshpass -p "scope" ssh "$tx_node_srn" "ip route add $prev_host via $leaf_host_ip"

            # Check if it is a gNB
            elif sshpass -p "scope" ssh "$tx_node_srn" 'ifconfig srs_spgw_sgi'; then
            echo "leaf_host_ip is a gNB"
            # Check if the current node is a cell or not
            if [[ "$tx_node_type" == "cell" ]]; then
                leaf_host_ip=$(sshpass -p "scope" ssh "$tx_node_srn" 'ifconfig srs_spgw_sgi' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
            else
                leaf_host_ip=$(sshpass -p "scope" ssh "$tx_node_srn" 'ifconfig col0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
            fi
            else
            echo "error finding next node"
            fi
            echo "Setting the leaf_host_ip IP for node $tx_node_srn(cell): $leaf_host_ip"
        else
            # For other node types, replace this with logic to retrieve the IP
            echo "leaf_host_ip unknown case"
        fi

        case "$tx_node_type" in
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
        echo "testing connection"
        sshpass -p "$psswrd" ssh "$tx_node_srn" "ping $leaf_host_ip -c 4"

        ### END INTEGRATION

        python3 -m build_leaf_json --leaf_file $leaf_file --leaf_node $node_tx --ip $leaf_host_ip --port $a_port --connection_type $leaf_connection_type
        a_port=$((a_port+1)) # increment port
    fi


done


