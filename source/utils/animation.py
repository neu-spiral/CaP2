import torch
import torch.nn as nn
import torch.fx as fx
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import time
import random

def create_base_graph():
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes representing SRNs
    G.add_nodes_from(['SRN 1', 'SRN 2', 'SRN 3', 'SRN 4'])

    # Add edges representing the connections between SRNs
    G.add_edges_from([('SRN 1', 'SRN 2'), 
                      ('SRN 2', 'SRN 3'),
                      ('SRN 2', 'SRN 4'),
                      ('SRN 3', 'SRN 4'),
                      ('SRN 4', 'SRN 1')])

    # Define custom colors for each SRN
    node_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']

    # Positions for each node
    pos = {'SRN 1': [-1, 0],
           'SRN 2': [0,  1],
           'SRN 3': [1,  0],
           'SRN 4': [0, -1]}

    # Initialize lists for active nodes and edges
    active_edges = []
    active_nodes = []

    # Draw the base graph
    draw_graph(G, pos, active_edges, active_nodes)

    return G, pos, active_edges, active_nodes

def draw_graph(G, pos, active_edges, active_nodes):
    
    # Define custom colors for each SRN
    node_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
    
    # Draw nodes with their current colors
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors='black', node_size=2000)

    # Draw all edges with default colors
    nx.draw_networkx_edges(G, pos, width=2, arrows=False)

    # Highlight active edges
    if active_edges:
        nx.draw_networkx_edges(G, pos, edgelist=active_edges, edge_color='green', style='dashed', width=8, arrows=True, arrowsize=20, node_size=1100)

    # Highlight active nodes by their active colors
    active_colors = ['blue', 'green', 'red', 'salmon']
    for node in active_nodes:
        node_index = list(G.nodes()).index(node)
        node_colors[node_index] = active_colors[node_index]

    # Draw nodes again to make active nodes appear on top
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors='black', node_size=2000)

    # Draw labels next to the nodes
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", font_family='monospace', verticalalignment='center')

    # Show the updated graph
    plt.title("SRN Network")
    plt.tight_layout()
    plt.axis('off')
    plt.show()

def toggle_edge(G, pos, edge, active_edges, active_nodes):
    if edge in active_edges:
        # If edge is already active, deactivate it
        active_edges.remove(edge)
    else:
        # If edge is not active, activate it
        active_edges.append(edge)

    # Redraw the full graph
    draw_graph(G, pos, active_edges, active_nodes)

    return active_edges

def toggle_node(G, pos, node, active_edges, active_nodes):
    if node in active_nodes:
        # If node is already active, deactivate it
        active_nodes.remove(node)
    else:
        # If node is not active, activate it
        active_nodes.append(node)

    # Redraw the full graph
    draw_graph(G, pos, active_edges, active_nodes)

    return active_nodes

def plot_loading_bar(total_layers, current_layer):
    """
    Visualizes a loading bar representing the progress through a neural network's layers.
    
    Parameters:
    - total_layers: The total number of layers in the network.
    - current_layer: The current layer being processed (1-indexed).
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 1.5))
    
    # Set axis limits
    ax.set_xlim(0, total_layers)
    ax.set_ylim(0, 1)
    
    # Disable the axes for a clean look
    ax.axis('off')

    # Define colors for the loading bar
    bar_background_color = '#EAEAEA'  # Light grey background for the loading bar
    bar_color = '#76C7C0'  # Teal color for the progress
    current_layer_color = '#FFA500'  # Orange color for the current layer highlight

    # Add background rectangle for the loading bar
    ax.add_patch(Rectangle((0, 0), total_layers, 1, color=bar_background_color, lw=0))

    # Add filled portion for the progress bar
    ax.add_patch(Rectangle((0, 0), current_layer-1, 1, color=bar_color, lw=0))

    # Add a highlight border for the current layer
    ax.add_patch(Rectangle((current_layer - 1, 0), 1, 1, fill=False, edgecolor=current_layer_color, lw=2))

    # Add progress percentage text in the middle of the bar
    progress_percentage = int(((current_layer-1) / total_layers) * 100)
    ax.text(total_layers / 2, 0.5, f'{progress_percentage}%', ha='center', va='center', fontsize=16, color='black')

    # Add markers for each layer number
    for layer in range(total_layers):
        ax.text(layer + 1, -0.25, f'{layer + 1}', ha='center', va='center', fontsize=10, color='black')
    
    # Display the plot
    plt.tight_layout()
    plt.show()
