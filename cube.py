import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import patches


class Cube:
    """
    A 2x2x2 Rubik's cube.
    """

    def __init__(self):
        self.permutation = np.arange(8).reshape((2, 2, 2))
        self.orientation = np.zeros((2, 2, 2))

    def render(self, colors=None, border_color='k', border_width=2,
               background_color='#232323', mirror_gap=1.6):
        """
        Display an image of the cube in its current state.
        """
        if colors is None:
            colors = [['#ffff00',  # yellow
                       '#ffffff'],  # white
                      ['#0000ff',  # blue
                       '#008000'],  # green
                      ['#ffa500',  # orange
                       '#ff0000']]  # red

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(projection='3d')
        for layer, row, column in np.ndindex(self.permutation.shape):
            piece_index = self.permutation[layer, row, column]
            binary_piece_index = np.binary_repr(piece_index, width=3)
            piece_colors = np.array(
                [colors[axis][int(position)] for axis, position in
                 enumerate(binary_piece_index)])
            piece_orientation = self.orientation[layer, row, column]
            piece_colors = np.roll(piece_colors, piece_orientation)
            for axis, color in enumerate(piece_colors):
                if axis == 0:
                    x = mirror_gap + column
                    y = mirror_gap + row
                    z = 0 if layer == 0 else mirror_gap + 2
                    zdir = 'z'
                elif axis == 1:
                    x = mirror_gap + column
                    y = mirror_gap + layer
                    z = 0 if row == 0 else mirror_gap + 2
                    zdir = 'y'
                else:
                    x = mirror_gap + row
                    y = mirror_gap + layer
                    z = 0 if column == 0 else mirror_gap + 2
                    zdir = 'x'
                square = patches.Rectangle(
                    (x, y), 1, 1, edgecolor=border_color,
                    facecolor=color, linewidth=border_width)
                ax.add_patch(square)
                art3d.pathpatch_2d_to_3d(square, z=z, zdir=zdir)
        ax.set_xlim(0, mirror_gap + 2)
        ax.set_ylim(mirror_gap + 2, 0)
        ax.set_zlim(0, mirror_gap + 2)
        ax.axis('off')
        ax.set_facecolor(background_color)
        fig.tight_layout(pad=0)
        plt.show()
