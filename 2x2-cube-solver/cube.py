import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
from matplotlib import patches


class Cube:
    """
    A 2x2x2 Rubik's cube.

    The three axes of the cube, numbered from 0 to 2, are the 0 axis (D to U),
    the 1 axis (B to F), and the 2 axis (L to R). For example, the DBL corner
    piece is located at (0, 0, 0) and the UBR corner piece is located at
    (1, 0, 1).

    Each piece is assigned a unique piece index from 0 to 7, with the 0 piece
    being the piece at (0, 0, 0) in the solved state, the 1 piece being the
    piece at (0, 0, 1) in the solved state, and so on. The permutation array is
    a 3D array of shape (2, 2, 2) that holds the piece indices. The piece index
    in each position of the array corresponds to the piece that currently
    occupies that position. For example, if the (0, 0, 0) element of the
    permutation array is 4, the piece that is currently at the DBL position is
    the piece with index 4.

    Each piece is also assigned an orientation value from 0 to 2. Multiple
    pieces can have the same orientation value. The orientation value of a
    piece is determined as follows. In the solved state, there are two faces of
    the cube that are normal to the 0 axis, the D and U  faces. For the
    standard color scheme, these faces are the yellow and white faces. The
    orientation value of a piece is the axis to which one of these two faces on
    the piece is normal. For example, if a piece of a scrambled cube with the
    standard color scheme has white on its F face, then its orientation value
    is 1 since the F face is normal to the 1 axis. The orientation array is a
    3D array of shape (2, 2, 2) that holds the orientation values of the pieces
    currently in each position.
    """

    def __init__(self, seed=None):
        self.rng = None
        self.permutation = None
        self.orientation = None
        # Store the solved state for resetting or checking if the cube is
        # solved.
        self.solved_permutation = np.arange(8).reshape((2, 2, 2))
        self.solved_orientation = np.zeros((2, 2, 2), dtype=int)
        self.reset(seed=seed)

    def reset(self, seed=None):
        """Reset the cube to its solved state."""
        if seed or self.rng is None:
            self.rng = np.random.default_rng(seed=seed)
        # Copy the arrays to prevent modifying the solved state.
        self.permutation = self.solved_permutation.copy()
        self.orientation = self.solved_orientation.copy()

    def is_solved(self) -> bool:
        """Check if the cube is solved."""
        return (np.array_equal(self.permutation, self.solved_permutation)
                and np.array_equal(self.orientation, self.solved_orientation))

    def turn_layer(self, axis: int, clockwise: bool, quarter_turn_count: int):
        """
        Turn a layer of the cube.

        The layer is specified by the axis. The 0, 1, and 2 axes correspond to
        the U, F, and R layers, respectively. Turning other layers is not
        supported.
        """
        # If the axis is the axis being turned, then the slice is [1:]
        # (one layer). If it is a different axis, then the slice is [:]
        # (all layers). The slices together form a 1x2x2 layer of the cube.
        slices = tuple(slice(1, None) if axis_index == axis
                       else slice(None, None) for axis_index in range(3))
        # k is the number of counterclockwise quarter turns.
        k = -quarter_turn_count if clockwise else quarter_turn_count
        # The rotation direction is from axes[0] to axes[1].
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
        clockwise = not move[-1] == "'"
        quarter_turn_count = 2 if move[-1] == '2' else 1
        self.turn_layer(axis, clockwise, quarter_turn_count)

    def apply_moves(self, moves: str):
        """
        Apply a sequence of moves to the cube.

        The moves are a string with a space between each move.
        """
        for move in moves.split():
            self.apply_move(move)

    def scramble(self, quarter_turn_count: int) -> str:
        """
        Scramble the cube using a random sequence moves of the given quarter
        turn count.

        Returns the scramble as a string.
        """
        moves = []
        for _ in range(quarter_turn_count):
            layers = ['U', 'F', 'R']
            # If the previous move was a double move, then the next move must
            # turn a different layer.
            if moves and moves[-1][-1] == '2':
                layers.remove(moves[-1][0])
            layer = self.rng.choice(layers)
            # If the chosen layer is the same as that of the previous move,
            # change the previous move to a double move instead of adding a new
            # move.
            if moves and layer == moves[-1][0]:
                moves[-1] = layer + '2'
            elif self.rng.choice([True, False]):
                moves.append(layer + "'")
            else:
                moves.append(layer)
        scramble = ' '.join(moves)
        self.apply_moves(scramble)
        return scramble

    def render(self, colors=None, border_color='k', border_width=2,
               background_color='#232323', mirror_gap=1.6):
        """
        Display an image of the cube in its current state.

        colors is an optional list of colors for each face. The order of the
        faces is [[D, U], [B, F], [L, R]]. A standard color scheme is provided
        by default.
        The D, B, and L faces are normally hidden from view, so to make them
        visible, they are shown as if they are reflected in a mirror.
        mirror_gap is the gap between the cube and the mirrored faces.
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
        # Iterate through each piece of the cube.
        for layer, row, column in np.ndindex(self.permutation.shape):
            # piece_index is a number from 0 to 7.
            piece_index = self.permutation[layer, row, column]
            # binary_piece_index is a binary string from '000' to '111'.
            binary_piece_index = np.binary_repr(piece_index, width=3)
            # piece_colors holds the 3 colors of the piece.
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
            # Plot a square patch for each face of the piece.
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
        fig.show()
