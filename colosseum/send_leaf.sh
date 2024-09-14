#!/bin/bash
#   Send inputs to leaf nodes. Assumes run in the ./colosseum directory 
#   TODO: update password based on snr type
#
#   Example:
#       bash ./send_leaf [srn node number]

sshpass -p ChangeMe scp config-leaf.json genesys-$1:/root/CaP/colosseum

gnome-terminal -- bash -c "sshpass -p ChangeMe ssh genesys-$1 'cd /root/CaP && source env.sh && source ../cap-310/bin/activate && python3 -m send_leaf_split_model colosseum/config-leaf.json; bash '" &