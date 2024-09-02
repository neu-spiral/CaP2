import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Leaf node data collector and sender.")
    parser.add_argument("ip_file", type=str, help="Path to the ip mapping config file.")
    parser.add_argument("node_rx", type=int, help="Network node receiving connection.")
    parser.add_argument("ip_rx", type=str, help="IP address for server on node rx.")
    parser.add_argument("port_rx", type=int, help="Port for server on node rx.")
    parser.add_argument("type_tx", type=str, help="Colosseum node type of transmitting node.")
    args = parser.parse_args()

    with open(args.ip_file, "r") as f:
        ip_json = json.load(f)
    
        if str(args.node_rx) in ip_json:
            # if connection type is already covered, skip adding this connection
            has_connection_type = False
            for el in ip_json[str(args.node_rx)]:
                if el["connection_from"] == args.type_tx:
                    has_connection_type = True
                    break
            
            if not has_connection_type:
                ip_json[str(args.node_rx)].append({"ip" : args.ip_rx, "port" : args.port_rx, "connection_from" : args.type_tx})
                
        else:
            # make new entry in json
            ip_json[str(args.node_rx)] = [{"ip" : args.ip_rx, "port" : args.port_rx, "connection_from" : args.type_tx}]
            print('here')
    
        # Serializing json
        json_serialized = json.dumps(ip_json, indent=4)
        
        # Writing to sample.json
        with open(args.ip_file, "w") as outfile:
            outfile.write(json_serialized)

if __name__ == "__main__":
    main()