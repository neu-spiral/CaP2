import pandas as pd
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import os

from source.utils import misc

def parse_log_file(log_file_path):
    ''' 
        Regular expressions to capture the timestamp and data from each log message type AND
        Initialize empty lists for storing the data

        Ouput:
            block_event_df - tracks the start of each "block" event i.e. EXECUTE a block of non-comms layers, main process IDLE time, TX, RX, PREP time spent preparing outputs for sending
            layer_event_df - tracks timestamps for when each layer in the model finishes and the FLPOS computed 
            total_runtime - array with total runtimes in seconds
    '''

    # debug... looking for empty events
    print(f'\t {log_file_path}')

    timestamp_regex = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"

    # Idle time
    idle_time_regex = re.compile(timestamp_regex + r" .*Idle time=(\d+\.?\d*)ms for layer=(\d+)")
    idle_time_data = []
    
    # Layer execution
    execute_layer_ts_regex = re.compile(timestamp_regex + r" .*Executed layer=(\d+) ([^;]*)\n")
    execute_layers_regex = re.compile(timestamp_regex + r" .*Executed to ([^;]*) layer=(\d+) in time=(\d+\.?\d*)ms process input time=(\d+\.?\d*)ms")
    execute_layer_ts_data = []
    execute_layers_data = []

    # transmit routines
    total_send_regex= re.compile(timestamp_regex + r" .*Sent layer=(\d+) to nodes in time=(\d+\.?\d*)ms")
    send_data_regex = re.compile(timestamp_regex + r" .*Sent data ACKOWLEDGED for layer=(\d+) time=(\d+\.?\d*)ms bytes=(\d+) serialize time=(\d+\.?\d*)ms encode time=(\d+\.?\d*)ms ip=([\d\.]+) port=(\d+)")
    prep_out_regex = re.compile(timestamp_regex + r" .*layer=(\d+) time=(\d+\.?\d*)ms")
    total_send_data = []
    send_data = []
    prep_out_data = []
    
    # receive routines
    received_layer_regex = re.compile(timestamp_regex + r" .*Received layer=(\d+): receive time=(\d+\.?\d*)ms, deserialize time=(\d+\.?\d*)ms bytes=(\d+)")
    received_layer_data = []

    # total runtime
    total_runtime_regex = re.compile(timestamp_regex + r" .*Total runtime=(\d+\.?\d*)s")

    # Read the log file and extract relevant data
    with open(log_file_path, 'r') as file:
        for line in file:
            # Check for idle time messages
            idle_time_match = idle_time_regex.search(line)
            if idle_time_match:
                idle_time_data.append({
                    'timestamp': idle_time_match.group(1),
                    'dur': float(idle_time_match.group(2)),
                    'layer': int(idle_time_match.group(3))
                })
                continue
            
            # Check for executed layer timestamps 
            execute_layer_ts_match = execute_layer_ts_regex.search(line)
            if execute_layer_ts_match:
                execute_layer_ts_data.append({
                    'timestamp': execute_layer_ts_match.group(1),
                    'layer': int(execute_layer_ts_match.group(2)),
                    'layer_name': str(execute_layer_ts_match.group(3))
                })
                continue

            # Check for executed layer messages (for layer blocks  vs individual)
            execute_layers_match = execute_layers_regex.search(line)
            if execute_layers_match:
                execute_layers_data.append({
                    'timestamp': execute_layers_match.group(1),
                    'layer_name': str(execute_layers_match.group(2)),
                    'layer': int(execute_layers_match.group(3)),
                    'dur': float(execute_layers_match.group(4)), 
                    'process_dur': float(execute_layers_match.group(5))
                })
                continue

            # Check for sent data ACKOWLEDGED messages
            send_data_match = send_data_regex.search(line)
            if send_data_match:
                send_data.append({
                    'timestamp': send_data_match.group(1),
                    'layer': int(send_data_match.group(2)),
                    'dur': float(send_data_match.group(3)),
                    'bytes_tx': int(send_data_match.group(4)),
                    'serialize_dur': float(send_data_match.group(5)),
                    'encode_dur_tx': float(send_data_match.group(6)),
                    'ip' : send_data_match.group(7),
                    'port' : float(send_data_match.group(8))
                })
                continue

            # Check for total time sent messages
            total_send_match = total_send_regex.search(line)
            if total_send_match:
                total_send_data.append({
                    'timestamp': total_send_match.group(1),
                    'layer': int(total_send_match.group(2)),
                    'dur': float(total_send_match.group(3))
                })
                continue
            
            # Prep out data
            prep_out_match = prep_out_regex.search(line)
            if prep_out_match:
                prep_out_data.append({
                    'timestamp': prep_out_match.group(1),
                    'layer': int(prep_out_match.group(2)),
                    'dur': float(prep_out_match.group(3))
                })

            # Check for received layer messages
            received_layer_match = received_layer_regex.search(line)
            if received_layer_match:
                received_layer_data.append({
                    'timestamp': received_layer_match.group(1),
                    'layer': int(received_layer_match.group(2)),
                    'dur': float(received_layer_match.group(3)),
                    'deserialize_time': float(received_layer_match.group(4)),
                    'bytes_rx': int(received_layer_match.group(5))
                })
            
            total_runtime_match = total_runtime_regex.search(line)
            if total_runtime_match:
                total_runtime = float(total_runtime_match.group(2))

    # get flops per layer calculated elswhere
    flops_file_path = log_file_path[:-4] + '_debug.log'
    flops_df = parse_debug_log_file(flops_file_path)

    # Convert the lists into pandas DataFrames
    idle_time_df = pd.DataFrame(idle_time_data)
    execute_layer_ts_df = pd.DataFrame(execute_layer_ts_data)
    execute_layers_df = pd.DataFrame(execute_layers_data)
    send_data_df = pd.DataFrame(send_data)
    total_send_data_df = pd.DataFrame(total_send_data)
    prep_out_df = pd.DataFrame(prep_out_data)
    received_layer_df = pd.DataFrame(received_layer_data)

    # convert timestamp strings to datetime
    time_format_str = '%Y-%m-%d %H:%M:%S,%f'
    idle_time_df['timestamp'] = pd.to_datetime(idle_time_df['timestamp'], format=time_format_str)
    execute_layer_ts_df['timestamp'] = pd.to_datetime(execute_layer_ts_df['timestamp'], format=time_format_str)
    execute_layers_df['timestamp'] = pd.to_datetime(execute_layers_df['timestamp'], format=time_format_str)
    total_send_data_df['timestamp'] = pd.to_datetime(total_send_data_df['timestamp'],format=time_format_str)
    prep_out_df['timestamp'] = pd.to_datetime(prep_out_df['timestamp'],format=time_format_str)
    received_layer_df['timestamp'] = pd.to_datetime(received_layer_df['timestamp'], format=time_format_str)
    flops_df['timestamp'] = pd.to_datetime(flops_df['timestamp'], format=time_format_str)

    # handle send data first because i might be empty 
    if len(send_data_df) > 0:
        send_data_df['timestamp'] = pd.to_datetime(send_data_df['timestamp'],format=time_format_str)

        # include send data when looking for first event timestamp 
        start_date = min(pd.concat([idle_time_df['timestamp'], execute_layer_ts_df['timestamp'],  execute_layers_df['timestamp'], send_data_df['timestamp'], total_send_data_df['timestamp'], received_layer_df['timestamp']]))

        send_data_df['time'] = (send_data_df['timestamp']-start_date).dt.total_seconds()*1e3
        
    else:
        # send df is empty, dont need to compute 

        # exclude send data when looking for first event timestamp 
        start_date = min(pd.concat([idle_time_df['timestamp'], execute_layer_ts_df['timestamp'],  execute_layers_df['timestamp'], total_send_data_df['timestamp'], received_layer_df['timestamp']]))

    # zero to 1st entry across all messages TODO: use starting model debug message as reference 
    idle_time_df['time'] = (idle_time_df['timestamp'] -start_date).dt.total_seconds()*1e3
    execute_layer_ts_df['time'] = (execute_layer_ts_df['timestamp'] - start_date).dt.total_seconds()*1e3
    execute_layers_df['time'] = (execute_layers_df['timestamp']-start_date).dt.total_seconds()*1e3
    prep_out_df['time'] = (prep_out_df['timestamp']-start_date).dt.total_seconds()*1e3
    received_layer_df['time'] = (received_layer_df['timestamp']-start_date).dt.total_seconds()*1e3
    total_send_data_df['time'] = (total_send_data_df['timestamp']-start_date).dt.total_seconds()*1e3

    # make uniform formating to prepare for merge
    # - each row is an event
    # - timestamp is the time the event starts 
    # - each event has 4 base cols: timestamp [ms], type [idle/tx/rx/prep], layer (event operates on this layer), dur [ms], 

    def adjust_df(df, type):
        if len(df) > 0:
            df['type'] = type
            df['timestamp'] = df['timestamp'] - pd.to_timedelta(df['dur'], unit='ms')
            df['time'] = df['time'] - df['dur']
        return df

    idle_time_df = adjust_df(idle_time_df, 'idle')
    execute_layers_df = adjust_df(execute_layers_df, 'execute')
    send_data_df = adjust_df(send_data_df,'send') 
    total_send_data_df = adjust_df(total_send_data_df, 'total_send')
    prep_out_df = adjust_df(prep_out_df, 'prep')
    received_layer_df = adjust_df(received_layer_df, 'receive')

    # merge
    block_event_df = pd.concat([idle_time_df, execute_layers_df, send_data_df, total_send_data_df, prep_out_df, received_layer_df], ignore_index=True, sort=False)
    block_event_df = block_event_df.sort_values(by='timestamp')
    
    # add FLOPS to execute layer timestamps
    flops_df = flops_df.drop(columns=['timestamp'])
    layer_event_df = pd.merge(execute_layer_ts_df, flops_df, on=['layer_name', 'layer'], how='outer')
    layer_event_df.sort_values(by=['timestamp'])
    layer_event_df['time'] = (layer_event_df['timestamp'] -start_date).dt.total_seconds()

    return block_event_df, layer_event_df, total_runtime


def parse_debug_log_file(log_file_path):
    ''' 
        Parses extra outputs from split manager when the debug flag is enabled.
        To get these extra outputs, a second run is required.
        Extra outputs are: FLOPS, num_parameters
        It is assumed that if the debug flag is enabled the computation will be slowed and delay results will be adversely effected 
        THEREFORE, a second run is required. 
    '''
    # Regular expressions to capture the timestamp and data from each log message type
    timestamp_regex = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"

    #2024-09-21 18:15:00,763 - source.core.split_manager - DEBUG - FLOPS for conv1 layer=1 FLOPS=589824.0 parameters=576.0
    flops_regex = re.compile(timestamp_regex + r" .*FLOPS for ([^;]*) layer=(\d+) FLOPS=(\d+\.?\d*) parameters=(\d+\.?\d*)")

    # Initialize empty lists for storing the data
    flops_data = []

    # Read the log file and extract relevant data
    with open(log_file_path, 'r') as file:
        for line in file:
            # Check for idle time messages
            flops_match = flops_regex.search(line)
            if flops_match:
                flops_data.append({
                    'timestamp': flops_match.group(1),
                    'layer_name': str(flops_match.group(2)),
                    'layer': int(flops_match.group(3)), 
                    'FLOPS': float(flops_match.group(4)),
                    'parameters': float(flops_match.group(5))
                })
                continue

    # Convert the lists into pandas DataFrames
    flops_df = pd.DataFrame(flops_data)

    # Save DataFrames to CSV (optional)
    '''
    idle_time_df.to_csv('flops_df.csv', index=False)
    '''

    # Print DataFrames to verify
    #print("FLOPS DataFrame:")
    #print(flops_df)

    return flops_df

'''
    Get data for each log file
'''
def combine_log_files(log_file_path, log_name_substr, num_nodes):
    '''
        Combines log files between nodes, and merges log files that time contiguous blocks 
    '''
    total_runtime = []
    for i in range(num_nodes):
        #print(f'\n\n Network Node {i}')
        node_log_file_path= os.path.join(log_file_path,f'node{i}_{log_name_substr}.log')
        block_event_tmp, layer_event_tmp, total_runtime_tmp = parse_log_file(node_log_file_path)

        total_runtime.append(total_runtime_tmp)

        # add node number
        block_event_tmp['node'] = i
        layer_event_tmp['node'] = i

        # add run str 
        if '\\' in log_file_path:
            run_name = log_file_path.split('\\')[-1]
        else:
            run_name = log_file_path.split('\\')[-1]
        run_str = misc.parse_filename(run_name)['run']
        block_event_tmp['run'] = run_str
        layer_event_tmp['run'] = run_str

        if i ==0:
            block_event_df = block_event_tmp
            layer_event_df = layer_event_tmp
        else:
            block_event_df = pd.concat([block_event_df, block_event_tmp])
            layer_event_df = pd.concat([layer_event_df, layer_event_tmp])

    # zero to 1st entry across all messages TODO: use starting model debug message as reference 
    start_time = min(pd.concat([block_event_df['timestamp'], layer_event_df['timestamp']]))
    block_event_df['time'] = (block_event_df['timestamp'] -start_time).dt.total_seconds()*1e3
    layer_event_df['time'] = (layer_event_df['timestamp'] - start_time).dt.total_seconds()*1e3

    block_event_df = block_event_df.sort_values(by=['time'])
    layer_event_df = layer_event_df.sort_values(by=['time'])

    #master_df[key] = master_df[key].fillna(0)

    # rearrange columns 
    if 'run' in block_event_df.columns:
        block_event_df = block_event_df[['timestamp', 'time', 'node', 'layer', 'layer_name','type', 'dur', 'process_dur', 'bytes_tx', 'serialize_dur', 'encode_dur_tx','ip', 'port', 'bytes_rx', 'deserialize_time', 'run']]
        layer_event_df = layer_event_df[['timestamp', 'time', 'node', 'layer', 'layer_name', 'FLOPS', 'parameters', 'run']]
    else:   
        block_event_df = block_event_df[['timestamp', 'time', 'node', 'layer', 'layer_name','type', 'dur', 'process_dur', 'bytes_tx', 'serialize_dur', 'encode_dur_tx','ip', 'port', 'bytes_rx', 'deserialize_time']]
        layer_event_df = layer_event_df[['timestamp', 'time', 'node', 'layer', 'layer_name', 'FLOPS', 'parameters']]

    # Save DataFrames to CSV 
    block_event_df.to_csv(os.path.join(log_file_path,'block_events.csv'), index=False)
    layer_event_df.to_csv(os.path.join(log_file_path,'layer_events.csv'), index=False)

    return block_event_df, layer_event_df, total_runtime
