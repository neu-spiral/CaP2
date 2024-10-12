#!/bin/bash
#   Send start to all nodes in network. Assumes ran in the ./colosseum directory 
#   TODO: 
#   1. update password based on snr type
#   2. update leaf json config to have all nodes in network 
#
#   Example:
#       bash ./start_run [srn node number]

# server
#sshpass -p ChangeMe scp config-leaf.json genesys-$1:/root/CaP/colosseum
#gnome-terminal -- bash -c "sshpass -p ChangeMe ssh genesys-$1 'cd /root/CaP && source env.sh && source ../cap-310/bin/activate && python3 -m source.utils.send_start_message ./config-leaf.json; bash '" &

# wifi
sshpass -p sunflower scp config-leaf.json genesys-$1:/root/CaP/colosseum
gnome-terminal -- bash -c "sshpass -p sunflower ssh genesys-$1 'cd /root/CaP && source env.sh && source ../cap-310/bin/activate && python3 -m source.utils.send_start_message ./config-leaf.json; bash '" &