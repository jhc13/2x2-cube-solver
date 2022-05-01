import numpy as np
import gym
from gym import spaces

from cube import Cube


def one_hot_encode(array, category_count):
    """
    One-hot encode a 1D array of integers.

    Returns a 2D array.
    """
    return np.identity(category_count)[array]


class CubeEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        self.observation_space = spaces.Dict(
            {'cube': spaces.MultiBinary([6, 10]),
             'action_mask': spaces.MultiBinary(6)})
        self.action_space = spaces.Discrete(6)
        self.cube = Cube()
        self.moves = {
            0: 'U',
            1: "U'",
            2: 'F',
            3: "F'",
            4: 'R',
            5: "R'"
        }

    def get_observation(self):
        # The 0 piece is always solved because only U, F, and R moves are made.
        # The state of the 1 piece can be deduced from the states of the 2 to 7
        # pieces. Therefore, only the last 6 pieces need to be observed.

        # Subtract 1 from the permutation to transform the range of piece
        # indices from [1, 7] to [0, 6].
        permutation = self.cube.permutation.flatten()[2:] - 1
        orientation = self.cube.orientation.flatten()[2:]
        encoded_permutation = one_hot_encode(permutation, category_count=7)
        encoded_orientation = one_hot_encode(orientation, category_count=3)
        cube = np.hstack((encoded_permutation, encoded_orientation))
        action_mask = np.ones(6)
        observation = {'cube': cube, 'action_mask': action_mask}
        return observation

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        self.cube.reset(seed=seed)
        self.cube.scramble(options['scramble_quarter_turn_count'])
        observation = self.get_observation()
        info = None
        return (observation, info) if return_info else observation

    def step(self, action: int):
        move = self.moves[action]
        self.cube.apply_move(move)
        done = self.cube.is_solved()
        reward = 1 if done else 0
        observation = self.get_observation()
        info = None
        return observation, reward, done, info

    def render(self, mode='human'):
        self.cube.render()
