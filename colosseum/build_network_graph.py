import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Leaf node data collector and sender.")
    parser.add_argument("--network_file", type=str, help="Path to the ip mapping config file.")
    parser.add_argument("--edges", type=str, help="Nodes that tx_node sends to e.g.  '1,2,3' ")
    parser.add_argument("--tx_node", type=str, help="ID of transmit node")
    parser.add_argument("--tx_node_type", type=str, help="Colosseum node type of transmit node")
    parser.add_argument("--final_node", type=int, help="final node in network")
    args = parser.parse_args()

    edges = args.edges.split(",")
    edges = [int(el) for el in edges]

    with open(args.network_file, "r") as f:
        network_json = json.load(f)

        # fill final node
        network_json['final_node'] = args.final_node

        # add to edges and node type 
        if "edges" in network_json:
            network_json["edges"][args.tx_node] = edges
            network_json["node_type"][args.tx_node] = args.tx_node_type
        else:
            network_json["edges"] = {
                args.tx_node  : edges
            }
            network_json["node_type"] = {
                args.tx_node : args.tx_node_type
            }
        
        # increase node count (assumes each node is called once)
        if "total_nodes" in network_json:
            network_json["total_nodes"] += 1
        else:
            network_json["total_nodes"] = 1
        
        # Serializing json
        json_serialized = json.dumps(network_json, indent=4)
        
        # Writing to sample.json
        with open(args.network_file, "w") as outfile:
            outfile.write(json_serialized)

if __name__ == "__main__":
    main()