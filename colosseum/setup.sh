#!/bin/bash
## HOW TO RUN: ./setup.sh node_list.txt [model name e.g. cifar10-resnet18-kernel-npv0-pr0.75-lcm0.001]
## node_list.txt should be formatted like:
## type-srnnumber-network node
## type-srnnumber-network node (etc)
## TODO: include support for nodes using only part of the entire model 
## Expects model name in argument to be on the file-proxy server here: /share/nas/genesys/CaP-Models
## if you are using cell, the base station is always the first node

# Initialize a flag to track if cell or wifi nodes are encountered
rf_flag=false

# Get the number of lines in the file
num_lines=$(wc -l < "$1")

# Loop through each line number
for ((i=1; i<=num_lines; i++)); do
  # Read the line using sed
  line=$(sed -n "${i}p" "$1")

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
        sshpass -p "scope" ssh "$prefixed_number" 'colosseumcli rf start 1017 -c'
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
      sshpass -p "scope" rsync -avz --exclude='.git' --exclude='assets/' ../../CaP/ $prefixed_number:/root/CaP/
      sleep 1

      # Copy model to network node      
      echo "Copying $2.pt to $prefixed_number"
      sshpass -p "scope" ssh "$prefixed_number" "su srn-user -c 'cp /share/CaP-Models/$2.pt /tmp/'"
      sshpass -p "scope" ssh "$prefixed_number" "mkdir -p /root/CaP/assets/models && cp /tmp/$2.pt /root/CaP/assets/models/" &
      sleep 2
      
      ;;

    wifi)
      # Configuration for wifi type
      echo "Running configuration for wifi type..."
      if [ "$rf_flag" = false ]; then
        echo "Starting rf scenario"
        sshpass -p "sunflower" ssh "$prefixed_number" 'colosseumcli rf start 1017 -c'
        rf_flag=true
        sleep 10
      fi
      sshpass -p "sunflower" ssh "$prefixed_number" "cd interactive_scripts && ./tap_setup.sh"
      sleep 5
      gnome-terminal -- bash -c "sshpass -p 'sunflower' ssh '$prefixed_number' 'cd interactive_scripts && ./modem_start.sh'; bash"
      #mintty -- bash -c "sshpass -p 'sunflower' ssh '$prefixed_number' 'cd interactive_scripts && ./modem_start.sh'; bash"
      sleep 1

      # Sync local repo with the remote
      echo "Syncing local repo with $prefixed_number"
      sshpass -p "sunflower" rsync -avz --exclude='.git' --exclude='assets/'  ../../CaP/ $prefixed_number:/root/CaP/
      sleep 1

      # Copy model to network node      
      echo "Copying $2.pt to $prefixed_number"
      sshpass -p "sunflower" ssh "$prefixed_number" "su srn-user -c 'cp /share/CaP-Models/$2.pt /tmp/'"
      sshpass -p "sunflower" ssh "$prefixed_number" "mkdir -p /root/CaP/assets/models && cp /tmp/$2.pt /root/CaP/assets/models/" &
      sleep 2

      ;;

    server)
      # Configuration for server type
      echo "Running configuration for server type..."

      # Sync local networks-for-ai directory with the remote
      echo "Syncing local repo with $prefixed_number"
      sshpass -p "ChangeMe" rsync -avz --exclude='.git' --exclude='assets/' ../../CaP/ $prefixed_number:/root/CaP/
      sleep 1

      # Copy model to network node      
      echo "Copying $2.pt to $prefixed_number"
      sshpass -p "ChangeMe" ssh "$prefixed_number" "su srn-user -c 'cp /share/CaP-Models/$2.pt /tmp/'"
      sshpass -p "ChangeMe" ssh "$prefixed_number" "mkdir -p /root/CaP/assets/models && cp /tmp/$2.pt /root/CaP/assets/models/" &
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
