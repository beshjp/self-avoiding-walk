import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection


def animate_saw_2d(walk: np.ndarray) -> animation.FuncAnimation:
    """
    Animates and saves a two-dimensional self-avoiding walk.

    Parameters:
    walk (np.ndarray): The self avoiding walk.

    Returns:
    animation.FuncAnimation: The animation.
    """
    # Set up the figure
    x, y = walk.T
    aspect_ratio = (np.max(x) - np.min(x)) / (np.max(y) - np.min(y))
    fig, ax = plt.subplots(figsize=(aspect_ratio * 10, 10))

    # Set the limits
    ax.set_xlim((np.min(x) - 1, np.max(x) + 1))
    ax.set_ylim((np.min(y) - 1, np.max(y) + 1))

    # Remove axis and border
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Remove padding
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Create a grid of points to represent the lattice
    grid_x, grid_y = np.meshgrid(
        range(np.min(x), np.max(x) + 1), range(np.min(y), np.max(y) + 1)
    )
    ax.scatter(grid_x, grid_y, color="gray", s=1.5)

    # Create a set of line segments so that we can color each one
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, len(walk))
    lc = LineCollection(segments, cmap="hsv", norm=norm)
    lc.set_array(np.arange(len(walk)))
    lc.set_linewidth(2.5)

    # Initial scatter for traversed points, empty at the beginning
    traversed_scatter = ax.scatter([], [], color="black", s=3, zorder=3)

    # Adjust marker sizes and animation interval
    length = len(walk)
    start_stop_marker_size = 5
    interval = min(10000 // length, 100)

    (start_dot,) = ax.plot([x[0]], [y[0]], "go", markersize=start_stop_marker_size)
    (end_dot,) = ax.plot([x[0]], [y[0]], "ro", markersize=start_stop_marker_size)

    def update(frame: int):
        # Update the line collection up to the current frame
        lc.set_segments(segments[:frame])

        # Move the end dot to the current position
        end_dot.set_data([x[frame]], [y[frame]])

        # Update traversed points
        traversed_scatter.set_offsets(walk[:frame])

        return lc, end_dot, start_dot, traversed_scatter

    ani = animation.FuncAnimation(
        fig, update, frames=length, blit=True, interval=interval, repeat=False
    )

    # Save the animation as gif
    ani.save("saw.gif", writer="pillow", fps=60)

    return ani
