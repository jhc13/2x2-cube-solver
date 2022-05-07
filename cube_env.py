from typing import Union

import gym
import numpy as np
from gym import spaces

from cube import Cube


class CubeEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        # The observation includes the current state of the cube as well as the
        # action mask. The action mask is a Boolean array indicating the
        # validity of each action.
        self.observation_space = spaces.Dict(
            {'cube_state': spaces.MultiBinary([6, 10]),
             'action_mask': spaces.Box(low=False, high=True, shape=(6,),
                                       dtype=bool)})
        # There are 6 actions, corresponding to the 6 possible moves.
        self.action_space = spaces.Discrete(6)
        self.moves = {
            0: 'U',
            1: "U'",
            2: 'F',
            3: "F'",
            4: 'R',
            5: "R'"
        }
        self.cube = Cube()
        # The previous actions are needed for determining the action mask.
        self.previous_actions = []

    @staticmethod
    def get_undoing_action(action: int) -> int:
        """Get the action that undoes a given action."""
        return action + 1 if action % 2 == 0 else action - 1

    def get_action_mask(self) -> np.ndarray:
        """Get the action mask for the observation."""
        action_mask = np.full(6, True)
        # Undoing the previous move is not allowed.
        if self.previous_actions:
            previous_action = self.previous_actions[-1]
            undoing_action = (previous_action + 1 if previous_action % 2 == 0
                              else previous_action - 1)
            action_mask[undoing_action] = False
        # Repeating the same move 3 times is not allowed.
        if (len(self.previous_actions) >= 2
                and self.previous_actions[-2] == self.previous_actions[-1]):
            action_mask[self.previous_actions[-1]] = False
        return action_mask

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get the observation for the current state of the cube."""
        # The 0 piece is always solved because only U, F, and R moves are made.
        # The state of the 1 piece can be deduced from the states of the 2 to 7
        # pieces. Therefore, only the last 6 pieces need to be observed.

        # Subtract 1 from the permutation to transform the range of piece
        # indices from [1, 7] to [0, 6].
        permutation = self.cube.permutation.flatten()[2:] - 1
        orientation = self.cube.orientation.flatten()[2:]
        encoded_permutation = one_hot_encode(permutation, category_count=7)
        encoded_orientation = one_hot_encode(orientation, category_count=3)
        cube_state = np.hstack((encoded_permutation, encoded_orientation))
        action_mask = self.get_action_mask()
        observation = {'cube_state': cube_state, 'action_mask': action_mask}
        return observation

    def reset(self, seed=None, return_info=False, options=None) \
            -> Union[tuple[dict, dict], dict]:
        """
        Reset and scramble the cube, then return the observation.

        return_info determines whether the scramble is also returned.
        The scramble length must be provided inside options under the
        'scramble_length' key.
        """
        super().reset(seed=seed)
        self.cube.reset(seed=seed)
        self.previous_actions.clear()
        scramble = self.cube.scramble(options['scramble_length'])
        observation = self.get_observation()
        info = {'scramble': scramble}
        return (observation, info) if return_info else observation

    def step(self, action: int) -> tuple[dict, int, bool, None]:
        """
        Apply the given action to the cube and return the observation, the
        reward, and whether the cube has been solved. info is also returned,
        but it is always None.
        """
        self.previous_actions.append(action)
        move = self.moves[action]
        self.cube.apply_move(move)
        done = self.cube.is_solved()
        reward = 1 if done else 0
        observation = self.get_observation()
        info = None
        return observation, reward, done, info

    def render(self, mode='human'):
        """Display an image of the cube in its current state."""
        self.cube.render()


def one_hot_encode(array, category_count) -> np.ndarray:
    """One-hot encode a 1D array of integers and return a 2D array."""
    return np.identity(category_count)[array]
