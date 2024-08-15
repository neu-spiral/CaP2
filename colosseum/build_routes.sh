#!/bin/bash
## HOW TO RUN: ./setup.sh node_list.txt

# set -x
# Create an array to store individual entries
declare -a individual_entries

# Read the input file line by line
while IFS= read -r line; do
  node_type=$(echo "$line" | cut -d'-' -f1)
  number=$(echo "$line" | cut -d'-' -f2)
  layers=$(echo "$line" | cut -d'-' -f3)

  # Split the layers on comma and create individual entries
  IFS=',' read -ra layer_array <<< "$layers"
  for layer in "${layer_array[@]}"; do
    individual_entries+=("$node_type-$number-$layer")
  done
done < "$1"

# Output the individual entries
# echo "Individual entries before sorting:"
# printf '%s\n' "${individual_entries[@]}"

# Sort the individual entries by the final number
IFS=$'\n' sorted_entries=($(sort -t'-' -k3,3n <<< "${individual_entries[*]}"))
unset IFS

# Output the sorted entries
# echo "Sorted entries:"
# printf '%s\n' "${sorted_entries[@]}"

# Declare an associative array to store the sorted information
declare -A sorted_map

# Populate the associative array with sorted information
for entry in "${sorted_entries[@]}"; do
  node_type=$(echo "$entry" | cut -d'-' -f1)
  number=$(echo "$entry" | cut -d'-' -f2)
  layer=$(echo "$entry" | cut -d'-' -f3)

  sorted_map["$number-$layer"]="$node_type-$number-$layer"
done

# Get the total number of sorted entries
num_sorted_entries=${#sorted_entries[@]}
# Starting port definition
starting_port=49200
# Loop through each sorted entry
for ((i=0; i<num_sorted_entries; i++)); do
  entry="${sorted_entries[$i]}"

  # Split the entry into three variables
  node_type=$(echo "$entry" | cut -d'-' -f1)
  number=$(echo "$entry" | cut -d'-' -f2)
  layer=$(echo "$entry" | cut -d'-' -f3)

  # Add the prefix 'genesys-' to the number portion
  prefixed_number="genesys-$number"

  echo "Configuring route for $prefixed_number as a $node_type node for layer $layer"

  # Identify the previous node (circular)
  if (( i == 0 )); then
    prev_entry="${sorted_entries[$((num_sorted_entries - 1))]}"
  else
    prev_entry="${sorted_entries[$((i - 1))]}"
  fi
  prev_node=$(echo "$prev_entry" | cut -d'-' -f3)
  prev_node_type=$(echo "$prev_entry" | cut -d'-' -f1)

  # Get correct IP for previous node based on its type
  if [[ "$node_type" == "cell" ]]; then
    if sshpass -p "scope" ssh "$prefixed_number" 'ifconfig | grep -q "srs_spgw_sgi"'; then
      echo "This is a gNB, adding routing statements"
      sshpass -p "scope" ssh "$prefixed_number" 'sysctl -w net.ipv4.ip_forward=1; iptables -t nat -A POSTROUTING -j MASQUERADE; sysctl -w net.ipv4.conf.all.accept_redirects=1'
      if [[ "$prev_node_type" != "cell" ]]; then
        prev_host=$(sshpass -p "scope" ssh "$prefixed_number" 'ifconfig col0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
        echo "Setting prev_host IP for node $number (cell-col0): $prev_host"
      else
        prev_host=$(sshpass -p "scope" ssh "$prefixed_number" 'ifconfig srs_spgw_sgi' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
        echo "Setting prev_host IP for node $number (cell): $prev_host"
      fi
    else
      prev_host=$(sshpass -p "scope" ssh "$prefixed_number" 'ifconfig tun_srsue' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
      echo "Setting prev_host IP for node $number (cell): $prev_host"
    fi

  elif [[ "$node_type" == "server" ]]; then
    prev_host=$(sshpass -p "ChangeMe" ssh "$prefixed_number" 'ifconfig col0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
    echo "Setting prev_host IP for node $number (server): $prev_host"

  elif [[ "$node_type" == "wifi" ]]; then
    if [[ "$prev_node_type" != "wifi" ]]; then
      prev_host=$(sshpass -p "sunflower" ssh "$prefixed_number" 'ifconfig col0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
      echo "Setting prev_host IP for node $number (wifi-col0): $prev_host"
    else
      prev_host=$(sshpass -p "sunflower" ssh "$prefixed_number" 'ifconfig tap0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
      echo "Setting prev_host IP for node $number (wifi): $prev_host"
    fi

  else
    echo "prev_host unknown case"
  fi

  # Identify the next node (circular)
  if (( i == num_sorted_entries - 1 )); then
    next_entry="${sorted_entries[0]}"
  else
    next_entry="${sorted_entries[$((i + 1))]}"
  fi
  next_node=$(echo "$next_entry" | cut -d'-' -f3)
  next_node_type=$(echo "$next_entry" | cut -d'-' -f1)
  next_number=$(echo "$next_entry" | cut -d'-' -f2)
  prefixed_nn="genesys-$next_number"

  # Get correct IP for next node based on its type
  if [[ "$next_node_type" == "server" ]]; then
    next_host=$(sshpass -p "ChangeMe" ssh "$prefixed_nn" 'ifconfig col0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
    echo "Setting next_host IP for node $number (server): $next_host"

    # check to see if this is a UE
    if sshpass -p "scope" ssh "$prefixed_number" 'ifconfig tun_srsue'; then
      tun_srsue_ip=$(sshpass -p "scope" ssh "$prefixed_number" 'ifconfig tun_srsue' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
      echo "Adding route to UE with: ip route add $next_host via $tun_srsue"
      sshpass -p "scope" ssh "$prefixed_number" "ip route add $next_host via $tun_srsue"
    fi

  elif [[ "$next_node_type" == "wifi" && "$node_type" != "wifi" ]]; then
    next_host=$(sshpass -p "sunflower" ssh "$prefixed_nn" 'ifconfig col0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
    echo "Setting next_host IP for node $number (wifi-col0): $next_host"

  elif [[ "$next_node_type" == "wifi" ]]; then
    next_host=$(sshpass -p "sunflower" ssh "$prefixed_nn" 'ifconfig tap0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
    echo "Setting next_host IP for node $number (wifi): $next_host"

  elif [[ "$next_node_type" == "cell" ]]; then
    # Check if it is a UE
    if sshpass -p "scope" ssh "$prefixed_nn" 'ifconfig tun_srsue'; then
      echo "Next_host is a UE"
      next_host=$(sshpass -p "scope" ssh "$prefixed_nn" 'ifconfig tun_srsue' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
      if [[ "$node_type" == "server" ]]; then
        echo "Adding route to cell network via colab network with: ip route add $next_host via $prev_host"
        sshpass -p "ChangeMe" ssh "$prefixed_number" "ip route add $next_host via $prev_host"
      elif [[ "$node_type" == "wifi" ]]; then
        wifi_col0=$(sshpass -p "sunflower" ssh "$prefixed_number" 'ifconfig col0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
        echo "Adding route to cell network via colab network with: ip route add $next_host via $wifi_col0"
        sshpass -p "sunflower" ssh "$prefixed_number" "ip route add $next_host via $wifi_col0"
      else
        echo "unknown node type"
      fi
      echo "Adding route on UE to colab network via cell network with: ip route add $prev_host vi $next_host"
      sshpass -p "scope" ssh "$prefixed_nn" "ip route add $prev_host via $next_host"

    # Check if it is a gNB
    elif sshpass -p "scope" ssh "$prefixed_nn" 'ifconfig srs_spgw_sgi'; then
      echo "Next_host is a gNB"
      # Check if the current node is a cell or not
      if [[ "$node_type" == "cell" ]]; then
        next_host=$(sshpass -p "scope" ssh "$prefixed_nn" 'ifconfig srs_spgw_sgi' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
      else
        next_host=$(sshpass -p "scope" ssh "$prefixed_nn" 'ifconfig col0' | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
      fi
    else
      echo "error finding next node"
    fi
    echo "Setting the next_host IP for node $number (cell): $next_host"
  else
    # For other node types, replace this with logic to retrieve the IP
    echo "next_host unknown case"
  fi

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
  echo "testing connection"
  sshpass -p "$psswrd" ssh "$prefixed_number" "ping $next_host -c 4"

  # In the connection with a previous layer the current one acts as the server
  prev_port=$((starting_port))
  if [[ $layer == 18 ]]; then
    next_port=49200
  else
    next_port=$((starting_port+1))
  fi
  starting_port=$((starting_port+1))
  echo "Prev_port: $prev_port"
  echo "Next_port: $next_port"

  if [[ $layer == 1 ]]; then
    echo "Starting layer 1..."
    # Start the UE with layer 1
    pids=$(sshpass -p "$psswrd" ssh "$prefixed_number" "ps aux | grep \"layer $layer\" | grep -v grep | awk '{print \$2}'")
    if [[ $pids ]]; then
      echo "Processes python3 inferenceUE_testing.py are already running with PIDs: $pids. Killing them..."
      # Loop through each PID and kill it
      for pid in $pids; do
          sshpass -p "$psswrd" ssh "$prefixed_number" "kill -9 $pid"
      done
      sleep 2
    fi
    #gnome-terminal -- bash -c "sshpass -p '$psswrd' ssh '$prefixed_number' 'cd /root/networks-for-ai && python3 inferenceUE_testing.py --host_previous $prev_host --port_previous $prev_port --host_next $next_host --port_next $next_port; bash'"
    #mintty -h always -- bash -c "sshpass -p '$psswrd' ssh '$prefixed_number' 'cd /root/networks-for-ai && python3 inferenceUE_testing.py --host_previous $prev_host --port_previous $prev_port --host_next $next_host --port_next $next_port; bash'"
#  elif [[ $layer == 18 ]]; then
#    echo "Starting layer $layer..."
#    pids=$(sshpass -p "$psswrd" ssh "$prefixed_number" 'ps aux | grep "python3 inferenceNode" | grep -v grep | awk "{print \$2}"')
#    if [[ $pids ]]; then
#      echo "Processes python3 inferenceNode.py are already running with PIDs: $pids. Killing them..."
#      # Loop through each PID and kill it
#      for pid in $pids; do
#          sshpass -p "$psswrd" ssh "$prefixed_number" "kill -9 $pid"
#      done
#      sleep 2
#    fi
#    # Start the 18th node in a new terminal
#    gnome-terminal -- bash -c "sshpass -p '$psswrd' ssh '$prefixed_number' 'cd /root/networks-for-ai && python3 inferenceNode.py --host_previous $prev_host --port_previous $prev_port --host_next $next_host --port_next $next_port --layer $((layer-1)); bash'" &
  else
    echo "Starting layer $layer..."
    pids=$(sshpass -p "$psswrd" ssh "$prefixed_number" "ps aux | grep \"layer $layer\" | grep -v grep | awk '{print \$2}'")
    if [[ $pids ]]; then
      echo "Processes python3 inferenceNode.py are already running with PIDs: $pids. Killing them..."
      # Loop through each PID and kill it
      for pid in $pids; do
          sshpass -p "$psswrd" ssh "$prefixed_number" "kill -9 $pid"
      done
      sleep 2
    fi
    # Start any random node that is not the first one
    sshpass -p "$psswrd" ssh "$prefixed_number" "cd /root/networks-for-ai && python3 inferenceNode.py --host_previous $prev_host --port_previous $prev_port --host_next $next_host --port_next $next_port --layer $((layer-1))" &
  fi


  echo ""
  echo ""
done
