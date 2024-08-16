import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotting_utilities


def plot_overall_scene(x: np.ndarray, y: np.ndarray, z: np.ndarray, vf: np.ndarray, path: str = None) -> None:
    """
    Plot the overall scene.
    :param x: X coordinates.
    :param y: Y coordinates.
    :param z: Z coordinates.
    :param vf: Vector field.
    :param path: Path to save the plot.
    """
    # matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x, y, z, vf[:, 0], vf[:, 1], vf[:, 2], length=0.25, normalize=True, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Vector field')
    if path is not None:
        fig.savefig(os.path.join(path, "overall.png"))


def plot_overall_scene(x: np.ndarray, y: np.ndarray, z: np.ndarray, vf: np.ndarray, path: str = None) -> None:
    """
    Plot the overall scene.
    :param x: X coordinates.
    :param y: Y coordinates.
    :param z: Z coordinates.
    :param vf: Vector field.
    :param path: Path to save the plot.
    """
    # matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x, y, z, vf[:, 0], vf[:, 1], vf[:, 2], length=0.25, normalize=True, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Vector field')
    if path is not None:
        fig.savefig(os.path.join(path, "overall.png"))

def plot_2d_slices(x: np.ndarray,
                   y: np.ndarray,
                   vector_field: np.ndarray,
                   z: float,
                   evaluation_dir: Optional[str] = None) -> None:
    """
    Plot 2D slices of the vector field.
    :param x: X coordinates.
    :param y: Y coordinates.
    :param vector_field: Vector field.
    :param z: Z coordinate.
    :param evaluation_dir: Evaluation directory.
    """
    plotting_utilities.set_figure_params(fontsize=18)
    # Create a color map. The color map is based on the norm of the vector field.
    # The lowest color is associated with the lowest norm and the highest color is associated with the highest norm.
    norm = np.linalg.norm(vector_field, axis=1)
    cmap = plt.cm.get_cmap('viridis')
    colors = cmap((norm - np.min(norm)) / (np.max(norm) - np.min(norm)))

    # Plot the vector field.
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.quiver(x, y, vector_field[:, 0], vector_field[:, 1], color=colors)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Vector field at z = {z:.2f}')

    # Plot the color bar. The minimum value of the color bar is the minimum norm of the vector field.
    # The maximum value of the color bar is the maximum norm of the vector field.
    norm = plt.Normalize(vmin=np.min(norm), vmax=np.max(norm))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax)

    # Save the figure.
    if evaluation_dir is not None:
        fig.savefig(os.path.join(evaluation_dir, f'vector_field_{z}.png'))
        fig.savefig(os.path.join(evaluation_dir, f'vector_field_{z}.pdf'), format='pdf', dpi=100, bbox_inches='tight')


def plot_3d_slices(x: np.ndarray,
                   y: np.ndarray,
                   vector_field: np.ndarray,
                   z: float,
                   evaluation_dir: Optional[str] = None,
                   scale: float = 0.0005) -> None:
    """
    Plot 3D slices of the vector field. The vector field is plotted as a 3D quiver plot.
    The z coordinate is fixed.
    :param x: X coordinates.
    :param y: Y coordinates.
    :param vector_field: Vector field.
    :param z: Z coordinate.
    :param evaluation_dir: Evaluation directory.
    :param scale: Scale of the vector field.
    """

    # Create a color map. The color map is based on the norm of the vector field.
    # The lowest color is associated with the lowest norm and the highest color is associated with the highest norm.
    norm = np.linalg.norm(vector_field, axis=1)
    cmap = plt.cm.get_cmap('viridis')
    colors = cmap((norm - np.min(norm)) / (np.max(norm) - np.min(norm)))

    # Plot the vector field.
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Repeat z so that it has the same shape as x and y.
    vector_field_copy = scale * vector_field.copy()
    z = np.repeat(z, x.shape[0])
    ax.quiver(x, y, z, vector_field_copy[:, 0], vector_field_copy[:, 1],
              vector_field_copy[:, 2], color=colors, capstyle='round')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'Vector field at z = {z[0]}')

    # Plot the color bar. The minimum value of the color bar is the minimum norm of the vector field.
    # The maximum value of the color bar is the maximum norm of the vector field.
    norm = plt.Normalize(vmin=np.min(norm), vmax=np.max(norm))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax)

    # Save the figure.
    if evaluation_dir is not None:
        fig.savefig(os.path.join(evaluation_dir, f'3dvector_field_{z[0]}.png'))


def show() -> None:
    """
    Show plots
    """
    plt.show()
