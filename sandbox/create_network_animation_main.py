import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import math
import matplotlib.lines as mlines
import yaml
import numpy as np

path = os.path.join('logs', 'demo_logs')

# model_name = 'cifar100-resnet101-run1.1'
model_name = 'cifar100-resnet101-kernel-np4-pr0.85-lcm1e-05-run1'
path_partition_file = os.path.join('config', 'resnet101-np4.yaml')


# model_name = 'esc-EscFusion-kernel-np4-pr0.85-lcm1000-run1'
# model_name = 'esc-EscFusion-run1'
# path_partition_file = os.path.join('config', 'EscFusion-np4.yaml')


port_to_node = {49204: 0, 49200: 1, 49201: 2, 49202: 3}



with open(path_partition_file, 'r') as stream:
    try:
        partition = yaml.load(stream, yaml.FullLoader)
    except yaml.YAMLError as exc:
        print(exc)

partition_list = list(partition['partitions'].keys())
partition_list.remove('inputs')
partition_list = partition_list + ['out']

partition_list_0 = partition_list.copy()
partition_list_0.remove('conv1.weight')

# checkmark_image = plt.imread('sandbox/checkmark.png')

slowing_factor = 5

interval = 1000
scaling = 1
fps = 1000/(scaling * slowing_factor)
node_size = 5000
edge_width = 3
arrow_size = 20

model_path = os.path.join(path, model_name)
df = pd.read_csv(os.path.join(model_path, 'block_events.csv'))

# Max bytes sent
max_bytes_sent = df['bytes_tx'].max()

# Round up the max time to the nearest second
max_time = round((df['time'].max())/scaling) + int(fps) + 100
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
fig, (network_ax, bar_ax) = plt.subplots(2, 1, figsize=(10, 14), dpi=300, gridspec_kw={'height_ratios': [3, 1]})
fig.subplots_adjust(hspace=0.1)
network_ax.axis('off')
bar_ax.axis('off')
# nx.draw_networkx_nodes(G, positions, node_color='gray', node_size=node_size)
# nx.draw_networkx_edges(G, positions, edge_color='gray', arrows=False, width=edge_width, node_size=node_size)
# labels = {node: node for node in G.nodes}
labels = {node: f'Node {node}' for node in G.nodes}
# nx.draw_networkx_labels(G, positions, labels=labels, font_color='black', font_size=15, font_weight='bold')

# ax_icon = plt.axes([0, 0, 2, 2])

# network_ax = fig.add_axes([0.1, 0.3, 0.8, 0.65])  # Main plot
# network_ax = ax

# Create legend handles for nodes and edges
node_idle = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=12, label='Idle (Node)')
node_send = mlines.Line2D([], [], color='orange', marker='o', linestyle='None', markersize=12, label='Communicate (Node)')
node_execute = mlines.Line2D([], [], color='lightblue', marker='o', linestyle='None', markersize=12, label='Compute (Node)')
edge_idle = mlines.Line2D([], [], color='gray', linewidth=2, marker='', linestyle='-', markersize=12, label='Idle (Edge)')
edge_send = mlines.Line2D([], [], color='orange', linewidth=2, marker='>', linestyle='-', markersize=12, label='Communicate (Edge)')

# Add legend to the plot
# network_ax.legend(handles=[node_idle, node_send, node_execute, edge_idle, edge_send], loc='upper right', fontsize=40)

progress_data = {node: [] for node in positions.keys()}

done_executing = {node: False for node in positions.keys()}
done_executing_frame = {node: 0 for node in positions.keys()}

done_executing[4] = False
done_executing_frame[4] = 0


def update(frame):
    print(f'Frame: {frame}')
    network_ax.clear()
    network_ax.axis('off')
    
    # Set default node and edge colors
    node_colors = ['lightgray'] * len(G.nodes)

    activated_edges = []
    activated_edges_color = []

    edge_width_active = edge_width
    arrow_size_active = arrow_size

    # Iterate through the dataframe to check for active processes
    for i, row in df.iterrows():
        start_time = row['time']
        duration = row['dur']
        current_node = int(row['node'])

        if duration < 1:
            duration = 1
        
        # Check if the current frame corresponds to the active time
        if start_time <= frame * scaling < start_time + duration:  # frame * 1000 to convert from frames to ms
            current_type = row['type']
            if current_type == 'send':
                # Arrange edge_width based on the number of bytes sent
                edge_width_active = 3 + 10 * (row['bytes_tx'] / max_bytes_sent)
                arrow_size_active = 20 + 50 * (row['bytes_tx'] / max_bytes_sent)
                target_node = int(port_to_node[int(row['port'])])
                node_colors[current_node] = 'orange'
                node_colors[target_node] = 'orange'
                activated_edges.append((current_node, target_node))
                activated_edges_color.append('orange')

            # elif current_type == 'receive':
            #     node_colors[current_node] = 'orange'

            elif current_type == 'execute':
                node_colors[current_node] = 'lightblue'
                # done_executing[current_node] = True
                progress_data[current_node].append(row['layer_name'])
                # progress_data[current_node].append(partition_list_0)
                if pd.notna(row['layer_name']):
                    network_ax.text(positions[current_node][0], positions[current_node][1] + 0.2, row['layer_name'], 
                            fontsize=16, ha='center', color='black', fontweight='bold')
                    
            # elif done_executing[current_node]:
            #     progress_data[current_node] += 1
            #     done_executing[current_node] = False
    
    # remove duplicate layer names at each node
    for node in progress_data.keys():
        progress_data[node] = list(set(progress_data[node]))

    nx.draw_networkx_nodes(G, positions, ax=network_ax, node_color=node_colors, node_size=node_size, edgecolors='gray')
    nx.draw_networkx_edges(G, positions, ax=network_ax, edgelist=[edge for edge in edges if edge not in activated_edges], edge_color='gray', arrows=False, width=edge_width, node_size=node_size)
    nx.draw_networkx_edges(G, positions, ax=network_ax, edgelist=activated_edges, edge_color=activated_edges_color, arrowstyle='-|>', arrowsize=arrow_size_active, width=edge_width_active, node_size=node_size*0.75)
    nx.draw_networkx_labels(G, positions, ax=network_ax, labels=labels, font_color='black', font_size=16, font_weight='bold')

    network_ax.legend(handles=[node_idle, node_send, node_execute, edge_idle, edge_send], loc='upper right', fontsize=14)

    bar_ax.clear()
    bar_ax.set_title('Computation Progress of Model Layers', fontsize=22, fontweight='bold')
    bar_ax.set_xlim(0, 100)
    bar_ax.set_ylim(-0.5, len(positions) - 0.5)
    # bar_ax.barh(list(positions.keys()), [progress_data[node] / len(partition_list) * 100 for node in positions.keys()], color='lightblue')
    bar_ax.barh(list(positions.keys())[0], len(progress_data[0]) / len(partition_list_0) * 100, color='lightblue')
    bar_ax.barh(list(positions.keys())[1], len(progress_data[1])  / len(partition_list) * 100, color='lightblue')
    bar_ax.barh(list(positions.keys())[2], len(progress_data[2])  / len(partition_list) * 100, color='lightblue')
    bar_ax.barh(list(positions.keys())[3], len(progress_data[3])  / len(partition_list) * 100, color='lightblue')
    bar_ax.set_yticks(list(positions.keys()))
    bar_ax.set_xlabel('Progress (%)', fontsize=18)
    bar_ax.set_yticklabels([f'Node {node}' for node in positions.keys()], fontsize=18)
    bar_ax.tick_params(axis='both', which='major', labelsize=16)
    # bar_ax.set_xticklabels([f'{i}%' for i in range(0, 101, 10)], fontsize=15)
    bar_ax.invert_yaxis()

    image_width = 0.15  # Change the width of the image as needed
    image_height = 0.15  # Change the height of the image as needed
    # Add the checkmark image next to the bar
    # network_ax.imshow(checkmark_image, extent=[positions[0][0], positions[0][0] + image_width, node - image_height / 2, node + image_height / 2], aspect='auto')





    for node in progress_data.keys():
        if not done_executing[node]:
            done_executing_frame[node] += 1


    if len(progress_data[0]) >= len(partition_list_0):
        done_executing[0] = True
        network_ax.text(positions[0][0], positions[0][1] - 0.3, 'Inference Time:', fontsize=20, ha='center', color='black', fontweight='bold')
        network_ax.text(positions[0][0], positions[0][1] - 0.3 - 0.1, f'{round(done_executing_frame[0]/(fps*slowing_factor), 2)} s', fontsize=20, ha='center', color='black', fontweight='bold')
        # network_ax.text(positions[0][0] - 0.3, positions[0][1] - 0.05, '\u2713', fontsize=60, color='#009E73', ha='center', fontweight='bold')
        bar_ax.text(105, + 0.3, '\u2713', fontsize=60, color='#009E73', ha='center', fontweight='bold')
        # network_ax.imshow(checkmark_image, extent=[positions[0][0], positions[0][0] + image_width, node - image_height / 2, node + image_height / 2], aspect='auto')
        # network_ax.imshow(checkmark_image, extent=[positions[0][0] - 0.5, positions[0][0] - 0.45, positions[0][1] - 0.5, positions[0][1] - 0.45], )
        # ax_icon = plt.axes([positions[0][0], positions[0][1], 0.1, 0.1])
        # ax_icon.imshow(checkmark_image, extent=[positions[0][0], positions[0][0] + image_width, positions[0][1] - image_height / 2, positions[0][1] + image_height / 2], aspect='auto')
    if len(progress_data[1]) >= len(partition_list):
        done_executing[1] = True
        network_ax.text(positions[1][0] - 0.7, positions[1][1], 'Inference Time:', fontsize=20, ha='center', color='black', fontweight='bold')
        network_ax.text(positions[1][0] - 0.7, positions[1][1] - 0.1, f'{round(done_executing_frame[1]/(fps*slowing_factor), 2)} s', fontsize=20, ha='center', color='black', fontweight='bold')
        # network_ax.text(positions[1][0], positions[1][1] + 0.15, '\u2713', fontsize=60, color='#009E73', ha='center', fontweight='bold')
        bar_ax.text(105, 1 + 0.3, '\u2713', fontsize=60, color='#009E73', ha='center', fontweight='bold')
        # network_ax.imshow(checkmark_image, extent=[positions[1][0], positions[1][0] + image_width, node - image_height / 2, node + image_height / 2], aspect='auto')
        #  network_ax.imshow(checkmark_image, extent=[positions[1][0] - 0.5, positions[1][0] - 0.45, positions[1][1] - 0.5, positions[1][1] - 0.45])
        # ax_icon = plt.axes([positions[1][0], positions[1][1], 0.1, 0.1])
        # ax_icon.imshow(checkmark_image, extent=[positions[1][0], positions[1][0] + image_width, positions[1][1] - image_height / 2, positions[1][1] + image_height / 2], aspect='auto')
    if len(progress_data[2]) >= len(partition_list):
        done_executing[2] = True
        network_ax.text(positions[2][0], positions[2][1] - 0.3, 'Inference Time:', fontsize=20, ha='center', color='black', fontweight='bold')
        network_ax.text(positions[2][0], positions[2][1] - 0.3 - 0.1, f'{round(done_executing_frame[2]/(fps*slowing_factor), 2)} s', fontsize=20, ha='center', color='black', fontweight='bold')
        # network_ax.text(positions[2][0] + 0.3, positions[2][1] - 0.05, '\u2713', fontsize=60, color='#009E73', ha='center', fontweight='bold')
        bar_ax.text(105, 2 + 0.3, '\u2713', fontsize=60, color='#009E73', ha='center', fontweight='bold')
        # network_ax.imshow(checkmark_image, extent=[positions[2][0], positions[2][0] + image_width, node - image_height / 2, node + image_height / 2], aspect='auto')
        # network_ax.imshow(checkmark_image, extent=[positions[2][0] - 0.5, positions[2][0] - 0.45, positions[2][1] - 0.5, positions[2][1] - 0.45])
        # ax_icon = plt.axes([positions[2][0], positions[2][1], 0.1, 0.1])
        # ax_icon.imshow(checkmark_image, extent=[positions[2][0], positions[2][0] + image_width, positions[2][1] - image_height / 2, positions[2][1] + image_height / 2], aspect='auto')
    if len(progress_data[3]) >= len(partition_list):
        done_executing[3] = True
        network_ax.text(positions[3][0] + 0.7, positions[3][1], 'Inference Time:', fontsize=20, ha='center', color='black', fontweight='bold')
        network_ax.text(positions[3][0] + 0.7, positions[3][1] - 0.1, f'{round(done_executing_frame[3]/(fps*slowing_factor), 2)} s', fontsize=20, ha='center', color='black', fontweight='bold')
        # network_ax.text(positions[3][0] - 0.3, positions[3][1] - 0.05, '\u2713', fontsize=60, color='#009E73', ha='center', fontweight='bold')
        bar_ax.text(105, 3 + 0.3, '\u2713', fontsize=60, color='#009E73', ha='center', fontweight='bold')
        # network_ax.imshow(checkmark_image, extent=[positions[3][0], positions[3][0] + image_width, node - image_height / 2, node + image_height / 2], aspect='auto')
        # network_ax.imshow(checkmark_image, extent=[positions[3][0] - 0.5, positions[3][0] - 0.45, positions[3][1] - 0.5, positions[3][1] - 0.45])
        # ax_icon = plt.axes([positions[3][0], positions[3][1], 0.1, 0.1])
        # ax_icon.imshow(checkmark_image, extent=[0, 0 + image_width, 0 - image_height / 2, 0 + image_height / 2], aspect='auto')
        # bar_ax.imshow(checkmark_image, extent=[80, 80 + image_width, 0 - image_height / 2, 0 + image_height / 2], aspect='auto')
        # bar_ax.imshow(checkmark_image, extent=[90, 90 + image_width, 0 - image_height / 2, 0 + image_height / 2], aspect='auto')
        # bar_ax.imshow(checkmark_image, extent=[100, 100 + image_width*4, 0 - image_height / 2, 0 + image_height / 2], aspect='auto')
        # network_ax.imshow(checkmark_image, extent=[0, 0 + image_width, 0 - image_height / 2, 0 + image_height / 2])
    if not done_executing[4]:
        done_executing_frame[4] += 1


    # Overall Inference Time

    if len(progress_data[0]) >= len(partition_list_0) and len(progress_data[1]) >= len(partition_list) and len(progress_data[2]) >= len(partition_list) and len(progress_data[3]) >= len(partition_list):
        done_executing[4] = True
        network_ax.text(0, 1.5, 'Overall Inference Time:', fontsize=30, ha='center', color='black', fontweight='bold')
        network_ax.text(0, 1.5 -0.15, f'{round(done_executing_frame[4]/(fps*slowing_factor), 2)} s', fontsize=30, ha='center', color='black', fontweight='bold')
        # network_ax.text(0, - 0.1, '\u2713', fontsize=150, color='#009E73', ha='center', fontweight='bold')
        # put the checkmark on left and right of the text
        network_ax.text(-1.15, 1.2, '\u2713', fontsize=150, color='#009E73', ha='center', fontweight='bold')
        network_ax.text(1.15, 1.2, '\u2713', fontsize=150, color='#009E73', ha='center', fontweight='bold')

        # put the checkmark on left and right of the text
        # network_ax.imshow(checkmark_image, extent=[-1.5, -1.3, 1.5, 1.3])
        # network_ax.imshow(checkmark_image, extent=[1.3, 1.5, 1.5, 1.3])

    # print(progress_data[0])
    # print(progress_data[3])

    # for node in progress_data.keys():
    #     if len(progress_data[node]) >= len(partition_list) and not done_executing[node]:
    #         done_executing[node] = True
    #         inference_time = done_executing_frame[node] / fps
    #         # Display inference time text
    #         network_ax.text(positions[node][0], positions[node][1] - 0.3, 'Inference Time:', fontsize=20, ha='center', color='black', fontweight='bold')
    #         network_ax.text(positions[node][0], positions[node][1] - 0.4, f'{inference_time:.2f} s', fontsize=20, ha='center', color='black', fontweight='bold')
    #         # Add the checkmark image to the right of the bar

    # # Update the overall inference time display
    # if all(done_executing[node] for node in positions.keys()):
    #     done_executing[4] = True
    #     network_ax.text(0, 1.5, 'Overall Inference Time:', fontsize=30, ha='center', color='black', fontweight='bold')
    #     network_ax.text(0, 1.35, f'{done_executing_frame[4] / fps:.2f} s', fontsize=30, ha='center', color='black', fontweight='bold')



# Create the animation
ani = animation.FuncAnimation(fig, update, frames=max_time, interval=interval, repeat=False)

# Save the animation
plt.axis('off')
# plt.tight_layout()

ani.save(f'assets/figs/{model_name}_with_progress_scale{scaling}_slowing_{slowing_factor}.mp4', writer='ffmpeg', fps=fps, dpi=150)

plt.close(fig)