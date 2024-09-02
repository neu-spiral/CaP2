README


nodes_test.txt -- maps together: colosseum node srn, network node #, network node edges, is leaf node. Expects nodes_test.txt to have a line for each network node in the network. 
            
        [node type]-[srn number]-[network node number]-[edges/nodes that this node sends to]-[bool is leaf node]-[(optional) leaf node connection type]

build_routes_test.sh -- takes nodes_test.txt and converts it into a JSON used by SplitNetworkCommunication repo to determine where i/o goes. This JSON describes all the ip/ports network nodes have servers on/are listing on and for what connections
    {
        [rx node] : [
            {"ip" : [ip that this network node listens to], "port" : [port server listens to], "connection_from" : [type of colosseum connection e.g wifi, cell, server etc.]}, ...
        ], ...
    }

