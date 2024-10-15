import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import math
import matplotlib.lines as mlines
import yaml

path = os.path.join('logs', 'logs_colosseum')
# model_name = 'esc-EscFusion-kernel-np4-pr0.5-lcm100'
model_name = 'esc-EscFusion-kernel-np4-pr0.85-lcm100'
# model_name = 'cifar100-resnet101-kernel-np4-pr0.5-lcm1e-04-batch16'

path_partition_file = os.path.join('config', 'EscFusion-np4.yaml')
with open(path_partition_file, 'r') as stream:
    try:
        partition = yaml.load(stream, yaml.FullLoader)
    except yaml.YAMLError as exc:
        print(exc)

partition_list = list(partition['partition'].keys())
partition_list = partition_list + ['out']

scaling = 100
node_size = 4000
edge_width = 3

model_path = os.path.join(path, model_name)
df = pd.read_csv(os.path.join(model_path, 'block_events.csv'))

# Round up the max time to the nearest second
max_time = round((df['time'].max())/scaling) + 4
# df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create a NetworkX graph
G = nx.DiGraph()

# Define positions for the nodes
positions = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1)
}

# Add nodes to the graph
for node in positions.keys():
    G.add_node(node)

# Add edges between nodes
edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3),
         (1,0), (2,0), (3,0), (2,1), (3,1), (3,2)]
G.add_edges_from(edges)

# Initialize the figure
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
ax.axis('off')  # Hide axes
# nx.draw_networkx_nodes(G, positions, node_color='gray', node_size=node_size)
# nx.draw_networkx_edges(G, positions, edge_color='gray', arrows=False, width=edge_width, node_size=node_size)
# labels = {node: node for node in G.nodes}
labels = {node: f'SRN {node}' for node in G.nodes}
nx.draw_networkx_labels(G, positions, labels=labels, font_color='black')

network_ax = fig.add_axes([0.1, 0.3, 0.8, 0.65])  # Main plot
# bar_ax = fig.add_axes([0.1, 0.15, 0.8, 0.05])  # Progress bar

# Create legend handles for nodes and edges
node_idle = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10, label='Idle (Node)')
node_send = mlines.Line2D([], [], color='orange', marker='o', linestyle='None', markersize=10, label='Send (Node)')
node_execute = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=10, label='Execute (Node)')
edge_idle = mlines.Line2D([], [], color='gray', linewidth=2, marker='', linestyle='-', markersize=10, label='Idle (Edge)')
edge_send = mlines.Line2D([], [], color='orange', linewidth=2, marker='>', linestyle='-', markersize=10, label='Send (Edge)')
edge_execute = mlines.Line2D([], [], color='blue', linewidth=2, marker='>', linestyle='-', markersize=10, label='Execute (Edge)')

# Add legend to the plot
network_ax.legend(handles=[node_idle, node_send, node_execute, edge_idle, edge_send, edge_execute], loc='upper right')

progress_data = {node: 0 for node in positions.keys()}

done_executing = {node: False for node in positions.keys()}


def update(frame):
    print(f'Frame: {frame}')
    network_ax.clear()
    network_ax.axis('off')
    
    # Set default node and edge colors
    node_colors = ['gray'] * len(G.nodes)

    activated_edges = []
    activated_edges_color = []

    # Iterate through the dataframe to check for active processes
    for i, row in df.iterrows():
        start_time = row['time']
        duration = row['dur']
        current_node = int(row['node'])
        
        # Check if the current frame corresponds to the active time
        if start_time <= frame * scaling < start_time + duration:  # frame * 1000 to convert from frames to ms
            current_type = row['type']
            if current_type == 'send':
                target_node = int(row['port'] - 5000)  # Map port to node index
                node_colors[current_node] = 'orange'
                node_colors[target_node] = 'orange'
                activated_edges.append((current_node, target_node))
                activated_edges_color.append('orange')

            # elif current_type == 'receive':
            #     node_colors[current_node] = 'orange'

            elif current_type == 'execute':
                node_colors[current_node] = 'blue'
                done_executing[current_node] = True
                if pd.notna(row['layer_name']):
                    network_ax.text(positions[current_node][0], positions[current_node][1] + 0.2, row['layer_name'], 
                            fontsize=10, ha='center', color='black')
                    
            if done_executing[current_node]:
                progress_data[current_node] += 1
                done_executing[current_node] = False

    nx.draw_networkx_nodes(G, positions, ax=network_ax, node_color=node_colors, node_size=node_size)
    nx.draw_networkx_edges(G, positions, ax=network_ax, edgelist=[edge for edge in edges if edge not in activated_edges], edge_color='gray', arrows=False, width=edge_width, node_size=node_size)
    nx.draw_networkx_edges(G, positions, ax=network_ax, edgelist=activated_edges, edge_color=activated_edges_color, arrowstyle='-|>', arrowsize=20, width=edge_width, node_size=node_size)
    nx.draw_networkx_labels(G, positions, ax=network_ax, labels=labels, font_color='black')

    network_ax.legend(handles=[node_idle, node_send, node_execute, edge_idle, edge_send, edge_execute], loc='upper right')

    # Add progress bar text
    progress = (frame / max_time) * 100  # Calculate progress percentage
    network_ax.text(0, -2, f'Progress: {progress:.1f}%', fontsize=12, ha='center', va='center', color='black')


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=max_time, interval=1000, repeat=True)

# Save the animation
plt.axis('off')

ani.save(f'{model_name}.mp4', writer='ffmpeg', fps=3)

plt.close(fig)