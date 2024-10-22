# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import numpy as np

# # Data
# pr = [0, 0.5, 0.7, 0.75, 0.8, 0.85, 0.95]
# resnet18 = [93.83, 91.56, 91.45, 89.19, 89.71, 89.73, 85.29]
# resnet101 = [71.62, 71.56, 71.31, 68.62, 68.55, 67.70, 53.16]
# esc = [99.89, 98.89, 99.16, 98.21, 98.72, 98.49, 83.36]

# # Function to generate interpolated frames
# def interpolate_data(x, y, num_frames):
#     x_interp = np.linspace(x[0], x[-1], num_frames)
#     y_interp = np.interp(x_interp, x, y)
#     return x_interp, y_interp

# # Function to animate and save the plot
# def animate_and_save(data, color, label, marker, filename):
#     num_frames = 100  # Number of frames for a smoother animation
#     x_interp, y_interp = interpolate_data(pr, data, num_frames)

#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.set_xlim(1, 0)
#     ax.set_ylim(50, 100)

#     # increase tick size
#     ax.tick_params(axis='both', which='major', labelsize=12)

#     if label == 'ESC':
#         ax.set_ylabel('AUC (%)', fontsize=12)
#         ax.set_title(f'{label} AUC vs. Prune Ratios', fontsize=14, fontweight='bold')
#     else:
#         ax.set_ylabel('Accuracy (%)', fontsize=12)
#         ax.set_title(f'{label} Accuracy vs. Prune Ratios', fontsize=14, fontweight='bold')

#     line, = ax.plot([], [], color=color, label=label, linestyle='-')
#     markers, = ax.plot([], [], color=color, marker=marker, linestyle='None', markersize=12)
#     # ax.legend()
#     ax.grid(True)

#     # Initialization function for the animation
#     def init():
#         line.set_data([], [])
#         markers.set_data([], [])
#         return line, markers

#     # Update function for the animation
#     def update(frame):
#         # Update line data
#         line.set_data(x_interp[:frame], y_interp[:frame])

#         # Update marker data: add markers only at the actual pr points that have been reached
#         pr_reached = [x for x in pr if x <= x_interp[frame-1]]
#         data_reached = [data[i] for i in range(len(pr)) if pr[i] <= x_interp[frame-1]]
        
#         if frame > 0:  # Ensure markers are displayed only after the first frame
#             markers.set_data(pr_reached, data_reached)
#         else:
#             markers.set_data([], [])  # Clear markers on the first frame

#         return line, markers

#     # Create the animation
#     ani = FuncAnimation(fig, update, frames=num_frames + 1, init_func=init,
#                         blit=True, repeat=False, interval=300)

#     # Save the animation
#     ani.save(filename, writer='ffmpeg', fps=30)
#     plt.close(fig)

# # Saving each animation individually
# animate_and_save(resnet18, 'dimgray', 'ResNet18', 'o', 'resnet18_animation.mp4')
# animate_and_save(resnet101, 'brown', 'ResNet101', 's', 'resnet101_animation.mp4')
# animate_and_save(esc, 'purple', 'ESC', '^', 'esc_animation.mp4')




import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Data
pr = [0, 0.5, 0.7, 0.75, 0.8, 0.85, 0.95]
resnet18 = [93.83, 91.56, 91.45, 89.19, 89.71, 89.73, 85.29]
resnet101 = [71.62, 71.56, 71.31, 68.62, 68.55, 67.70, 53.16]

esc = [99.98, 95.80, 95.60, 98.21, 95.20, 94.20, 74.40]

# Function to generate interpolated frames
def interpolate_data(x, y, num_frames):
    x_interp = np.linspace(x[0], x[-1], num_frames)
    y_interp = np.interp(x_interp, x, y)
    return x_interp, y_interp

# Function to animate and save the plot
def animate_all_models():
    num_frames = 100  # Number of frames for a smoother animation
    # Interpolated data for each model
    x_interp, y_interp_resnet18 = interpolate_data(pr, resnet18, num_frames)
    _, y_interp_resnet101 = interpolate_data(pr, resnet101, num_frames)
    _, y_interp_esc = interpolate_data(pr, esc, num_frames)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(50, 100)

    # Increase tick size
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel('Prune Ratio', fontsize=16)

    ax.set_ylabel('Accuracy (%)', fontsize=16)
    ax.set_title('Accuracy vs. Prune Ratios', fontsize=20, fontweight='bold')

    # Plot lines for each model
    line_resnet18, = ax.plot([], [], color='dimgray', linestyle='-')
    line_resnet101, = ax.plot([], [], color='brown', linestyle='-')
    line_esc, = ax.plot([], [], color='purple', linestyle='-')

    # Plot markers for each model
    markers_resnet18, = ax.plot([], [], color='dimgray', marker='o', linestyle='None', label='ResNet18', markersize=12)
    markers_resnet101, = ax.plot([], [], color='brown', marker='s', linestyle='None', label='ResNet101', markersize=12)
    markers_esc, = ax.plot([], [], color='purple', marker='^', linestyle='None', label='ESC', markersize=12)

    ax.legend(fontsize=16, loc='lower left')
    ax.grid(True)

    # Initialization function for the animation
    def init():
        line_resnet18.set_data([], [])
        line_resnet101.set_data([], [])
        line_esc.set_data([], [])

        markers_resnet18.set_data([], [])
        markers_resnet101.set_data([], [])
        markers_esc.set_data([], [])
        return line_resnet18, line_resnet101, line_esc, markers_resnet18, markers_resnet101, markers_esc

    # Update function for the animation
    def update(frame):
        # Update line data for each model
        line_resnet18.set_data(x_interp[:frame], y_interp_resnet18[:frame])
        line_resnet101.set_data(x_interp[:frame], y_interp_resnet101[:frame])
        line_esc.set_data(x_interp[:frame], y_interp_esc[:frame])

        # Update marker data for each model: add markers only at the actual pr points that have been reached
        pr_reached = [x for x in pr if x <= x_interp[frame-1]]
        data_reached_resnet18 = [resnet18[i] for i in range(len(pr)) if pr[i] <= x_interp[frame-1]]
        data_reached_resnet101 = [resnet101[i] for i in range(len(pr)) if pr[i] <= x_interp[frame-1]]
        data_reached_esc = [esc[i] for i in range(len(pr)) if pr[i] <= x_interp[frame-1]]

        if frame > 0:  # Ensure markers are displayed only after the first frame
            markers_resnet18.set_data(pr_reached, data_reached_resnet18)
            markers_resnet101.set_data(pr_reached, data_reached_resnet101)
            markers_esc.set_data(pr_reached, data_reached_esc)
        else:
            markers_resnet18.set_data([], [])
            markers_resnet101.set_data([], [])
            markers_esc.set_data([], [])

        return line_resnet18, line_resnet101, line_esc, markers_resnet18, markers_resnet101, markers_esc

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames + 1, init_func=init,
                        blit=True, repeat=False, interval=300)

    # Save the animation
    ani.save('assets/figs/combined_model_animation.mp4', writer='ffmpeg', dpi=150, fps=30)
    plt.close(fig)

# Animate and save the plot with all models
animate_all_models()
