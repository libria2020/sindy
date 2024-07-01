import numpy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def plot_data(X, Y, u, filename):
    # Create a 3D surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the initial surface
    surface = ax.plot_surface(X, Y, u[:, :, 0], cmap='viridis', rstride=1, cstride=1, alpha=0.8, linewidth=0.5,
                              antialiased=True)

    # Set labels for axes
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x,y,t)')

    # Set axis limits and aspect ratio
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(numpy.min(u), numpy.max(u))
    ax.set_box_aspect([numpy.ptp(coord) for coord in [X, Y, u]])

    # Hide axis
    ax.set_axis_off()

    # Add a horizontal colorbar
    cbar = fig.colorbar(surface, ax=ax, shrink=0.65, aspect=35, label='u(x,y,t)', orientation='horizontal', pad=0.01)

    # Set time step and animation interval
    dt = 0.02
    frames = u.shape[2]

    def update(frame):
        # Clear the previous frame
        ax.cla()

        surface = ax.plot_surface(X, Y, u[:, :, frame], cmap='viridis', rstride=1, cstride=1, alpha=0.8, linewidth=0.5,
                                  antialiased=True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u(x,y,t)')

        # ax.set_xlim(-2, 2)
        # ax.set_ylim(-2, 2)
        ax.set_zlim(numpy.min(u), numpy.max(u))
        ax.set_box_aspect([numpy.ptp(coord) for coord in [X, Y, u]])

        ax.set_axis_off()

        # Update colorbar limits
        cbar.mappable.set_array(u[:, :, frame])
        cbar.update_normal(surface)

    # Create the animation
    animation = FuncAnimation(fig, update, frames=frames, interval=int(dt * 1000), repeat=False)

    # Save the animation as a GIF
    animation.save('../images/' + filename, writer='pillow')

    plt.show()
