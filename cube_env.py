import gym
from gym import spaces

from cube import Cube


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
        pass

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        self.cube.reset(seed=seed)
        self.cube.scramble(options['scramble_quarter_turn_count'])
        observation = self.get_observation()
        info = None
        return (observation, info) if return_info else observation

    def step(self, action):
        move = self.moves[action]
        self.cube.apply_move(move)
        done = self.cube.is_solved()
        reward = 1 if done else 0
        observation = self.get_observation()
        info = None
        return observation, reward, done, info

    def render(self, mode='human'):
        self.cube.render()
