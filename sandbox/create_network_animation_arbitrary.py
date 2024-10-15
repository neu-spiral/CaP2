import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import math

path = os.path.join('logs', 'logs_colosseum')
# model_name = 'esc-EscFusion-kernel-np4-pr0.5-lcm100'
model_name = 'esc-EscFusion-kernel-np4-pr0.85-lcm100'
# model_name = 'cifar100-resnet101-kernel-np4-pr0.5-lcm1e-04-batch16'

model_path = os.path.join(path, model_name)
df = pd.read_csv(os.path.join(model_path, 'block_events.csv'))

#round up the max time to the nearest second
# max_time = round((df['time'].max())/10) + 4
# df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create a NetworkX graph
G = nx.DiGraph()

# Define positions for the nodes
positions = {
    0: (1.5, 2),
    1: (1, 1.5),
    2: (1.5, 0.5),
    3: (2, 1.5)
}

# Add nodes to the graph
for node in positions.keys():
    G.add_node(node)

# Add edges between nodes
edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3),
         (1,0), (2,0), (3,0), (2,1), (3,1), (3,2)]
G.add_edges_from(edges)

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis('off')  # Hide axes
nx.draw_networkx_nodes(G, positions, node_color='gray', node_size=800)
nx.draw_networkx_edges(G, positions, edge_color='gray', arrows=False)
labels = {node: node for node in G.nodes}
nx.draw_networkx_labels(G, positions, labels=labels, font_color='black')

def update(frame):
    print(f'Frame: {frame}')
    ax.clear()
    ax.axis('off')
    
    # Set default node and edge colors
    node_colors = ['gray'] * len(G.nodes)
    edge_colors = ['gray'] * len(G.edges)

    activated_edges = []
    activated_edges_color = []

    # Extract relevant row for the current frame
    current_time = df.iloc[frame]
    current_node = current_time['node']
    # print(current_node)
    current_type = current_time['type']
    current_layer_name = current_time['layer_name']

    if current_type == 'send':
        target_node = int(current_time['port'] - 5000)  # Map port to node index
        node_colors[current_node] = 'orange'
        node_colors[target_node] = 'orange'
        # edge_idx = edges.index((current_node, target_node))
        # edge_colors[edge_idx] = 'orange'
        activated_edges.append((current_node, target_node))
        activated_edges_color.append('orange')

    # elif current_type == 'receive':
    #     node_colors[current_node] = 'orange'
    #     # node_colors[target_node] = 'orange'
    #     # edge_idx = edges.index((target_node))
    #     # edge_colors[edge_idx] = 'orange'
    #     # activated_edges.append((current_node, target_node))
    #     # activated_edges_color.append('orange')

    elif current_type == 'execute':
        node_colors[current_node] = 'blue'
        if pd.notna(current_time['layer_name']):
            ax.text(positions[current_node][0], positions[current_node][1] + 0.1, current_time['layer_name'], 
                    fontsize=10, ha='center', color='black')

    # Draw nodes, edges, and labels again with updated colors
    nx.draw_networkx_nodes(G, positions, node_color=node_colors, node_size=800)
    # Draw idle edges
    nx.draw_networkx_edges(G, positions, edgelist=[edge for edge in edges if edge not in activated_edges], edge_color='gray', arrows=False)

    # Draw activated edges
    nx.draw_networkx_edges(G, positions, edgelist=activated_edges, edge_color=activated_edges_color, arrowstyle='-|>', arrowsize=20)
    # nx.draw_networkx_edges(G, positions, edge_color=edge_colors, arrowstyle='-|>', arrowsize=20)
    nx.draw_networkx_labels(G, positions, labels=labels, font_color='black')


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(df), interval=1000, repeat=True)

# Show the animation
plt.axis('off')

ani.save(f'{model_name}.mp4', writer='ffmpeg', fps=5)

plt.close(fig)