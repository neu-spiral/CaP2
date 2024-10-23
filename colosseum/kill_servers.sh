#!/bin/bash
#   Kill servers on srn nodes automatically
# 

# use in file inputs
nodes_file="./nodes.txt" # Path to nodes , text file

# Read the input file line by line
# iterate through each srn node
while IFS= read -r line; do

    #echo $line
    node_type=$(echo "$line" | cut -d'-' -f1)
    srn_number=$(echo "$line" | cut -d'-' -f2)
    node_number=$(echo "$line" | cut -d'-' -f3)

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
    # use ssh -n to ensure loop continues 
    # add : -e 'wifi_transceiver' to end wifi connection 
    echo "Shutting down servers on $srn_name($node_number)"
    sshpass -p "$psswrd" ssh -n "$srn_name" "
        pids=\$(ps aux | grep -e 'run_split_model' | grep -v 'grep' | awk '{print \$2}');
        if [ -n \"\$pids\" ]; then
            echo \"Killing PIDs: \$pids\";  # Optional: print PIDs to be killed
            echo \$pids | xargs kill -9;
        else
            echo 'No matching process found for run_split_model';
        fi
    " 

    # kill wifi_transceiver.py
    #|| { echo "SSH command failed for $srn_name, continuing to next element"; continue; } 
    echo ""
    wait

done < "$nodes_file"

