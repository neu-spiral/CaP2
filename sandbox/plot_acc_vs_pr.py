import pandas as pd
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import os
import math

# use colorblind color palette 
plt.style.use('tableau-colorblind10')

def plot_acc_vs_pr():
    pr = [0.5, 0.7, 0.75, 0.8, 0.85]
    acc = [71.56, 71.31, 68.62, 68.55, 67.70]
    exec_time = [8.24, 6.21, 6.86, 6.07, 6.24]

    marker_sizes = [time * 100 for time in exec_time]

    plt.scatter(pr, acc, s=marker_sizes, alpha=0.5, c='blue', label='Accuracy')

    # for i, txt in enumerate(exec_time):
    #     plt.annotate(txt, (pr[i], acc[i]))

    plt.xlabel('Prune Ratio')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Prune Ratio')
    plt.legend()
    plt.grid()
    plt.savefig('acc_vs_pr.png')
    plt.close()

def plot_acc_vs_all():
    pr = [0.5, 0.7, 0.75, 0.8, 0.85]
    acc = [71.56, 71.31, 68.62, 68.55, 67.70]
    exec_time = [8.24, 6.21, 6.86, 6.07, 6.24]
        # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    # Plot accuracy on the primary y-axis
    ax1.set_xlabel('Prune Ratio')
    ax1.set_ylabel('Accuracy (%)', color='tab:blue')
    ax1.plot(pr, acc, color='tab:blue', marker='o', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a secondary y-axis for execution time
    ax2 = ax1.twinx()
    ax2.set_ylabel('Execution Time (s)', color='tab:red')
    ax2.plot(pr, exec_time, color='tab:red', marker='s', linestyle='--', label='Execution Time')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Add a title and grid
    plt.title('Accuracy and Execution Time vs. Prune Ratio')
    fig.tight_layout()
    plt.grid()

    # Show plot
    plt.show()
    plt.savefig('acc_vs_all.png')
    plt.close()

if __name__ == "__main__":
    plot_acc_vs_pr()
    plot_acc_vs_all()