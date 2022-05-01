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
        self.orientation = np.zeros((2, 2, 2), dtype=int)

    def turn_layer(self, axis: int, clockwise: bool, quarter_turn_count: int):
        """
        Turn a layer of the cube.

        The layer is specified by the axis. The 0, 1, and 2 axes correspond to
        the U, F, and R layers, respectively.
        """
        slices = tuple(slice(1, None) if axis_index == axis
                       else slice(None, None) for axis_index in range(3))
        k = -quarter_turn_count if clockwise else quarter_turn_count
        axes = [(1, 2), (2, 0), (0, 1)][axis]
        self.permutation[slices] = np.rot90(self.permutation[slices], k=k,
                                            axes=axes)
        self.orientation[slices] = np.rot90(self.orientation[slices], k=k,
                                            axes=axes)
        # Adjust the orientations of the pieces if the quarter turn count is
        # odd.
        if quarter_turn_count % 2 == 1:
            # Swap the two orientation values that are not equal to the axis.
            # For example, if the axis is 1, then change 0 orientation values
            # to 2 and 2 orientation values to 0.
            orientation_axes = np.arange(3)
            orientation_axes[[axes[0], axes[1]]] = (
                orientation_axes[[axes[1], axes[0]]])
            self.orientation[slices] = orientation_axes[
                self.orientation[slices]]

    def apply_move(self, move: str):
        """
        Apply a move to the cube.

        The move is a string such as "U", "F'", or "R2".
        """
        axis = {'U': 0, 'F': 1, 'R': 2}[move[0]]
        clockwise = not (len(move) == 2 and move[1] == "'")
        quarter_turn_count = 2 if len(move) == 2 and move[1] == "2" else 1
        self.turn_layer(axis, clockwise, quarter_turn_count)

    def apply_moves(self, moves: str):
        """
        Apply a sequence of moves to the cube.

        The moves are a string with a space between each move.
        """
        for move in moves.split():
            self.apply_move(move)

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
            # Calculate the Manhattan distance between the piece's current and
            # solved positions.
            distance = sum(int(binary_piece_index[i]) - position
                           for i, position in enumerate([layer, row, column]))
            if distance % 2 == 1:
                # Swap the two colors whose axes do not equal the piece's
                # orientation value.
                axes_to_swap = [[1, 2], [2, 0], [0, 1]][piece_orientation]
                piece_colors[[axes_to_swap[0], axes_to_swap[1]]] = (
                    piece_colors[[axes_to_swap[1], axes_to_swap[0]]])
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
