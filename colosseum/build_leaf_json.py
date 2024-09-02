import json
import argparse

def main():
    '''
        Update leaf node json config one node at a time. Called by build_routes.sh
    
    '''
    parser = argparse.ArgumentParser(description="Leaf node data collector and sender.")
    parser.add_argument("--leaf_file", type=str, help="Path to the leaf config file to update.")
    parser.add_argument("--leaf_node", type=int, help="Network node receiving model input.")
    parser.add_argument("--ip", type=str, help="IP address for server on leaf node.")
    parser.add_argument("--port", type=int, help="Port for server on leaf node.")
    parser.add_argument("--connection_type", type=str, help="Connection type to use.")

    args = parser.parse_args()

    with open(args.leaf_file, "r") as f:
        leaf_json = json.load(f)

        new_entry = {
                        "node" : args.leaf_node ,
                        "ip" : args.ip, 
                        "port" : args.port, 
                        "connection_from" : args.connection_type
                    }

        if not "servers" in leaf_json:
            leaf_json["servers"] = [new_entry]
        else:
            # add entry 
            leaf_json["servers"].append(new_entry)
                    
    
        # Serializing json
        json_serialized = json.dumps(leaf_json, indent=4)

        # Writing to sample.json
        with open(args.leaf_file, "w") as outfile:
            outfile.write(json_serialized)

if __name__ == "__main__":
    main()