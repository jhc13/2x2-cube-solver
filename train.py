import datetime
import random
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

import config
from cube_env import CubeEnv
from dueling_dqn import DuelingDQN
from solve import Solver


class Trainer:
    def __init__(self, seed=None):
        random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        print(f'Using device: {self.device}.')
        self.env = CubeEnv()
        self.env.reset(seed=seed, options={
            'scramble_length': config.INITIAL_SCRAMBLE_LENGTH})
        self.replay_buffer = deque(maxlen=config.REPLAY_BUFFER_SIZE)
        self.epsilon = config.EPSILON_START
        self.policy_network = DuelingDQN().to(self.device)
        self.target_network = DuelingDQN().to(self.device)
        self.target_network.eval()
        self.update_target_network()
        self.optimizer = optim.RMSprop(self.policy_network.parameters(),
                                       lr=config.LEARNING_RATE)
        self.criterion = nn.HuberLoss()
        self.solver = Solver(self.device, self.policy_network, self.env)
        self.scramble_length = config.INITIAL_SCRAMBLE_LENGTH
        self.scramble_lengths = [(0, config.INITIAL_SCRAMBLE_LENGTH)]
        self.epsilons = []
        self.running_rewards = []
        self.evaluations = []
        self.run_path = None

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def get_best_actions(self, cube_states, action_masks):
        self.policy_network.eval()
        with torch.no_grad():
            q_values = self.policy_network(cube_states)
        self.policy_network.train()
        q_values[~action_masks] = float('-inf')
        best_actions = q_values.max(dim=1).indices
        return best_actions

    def get_action(self, observation):
        cube_state = observation['cube_state']
        action_mask = observation['action_mask']
        if self.rng.random() < self.epsilon:
            return self.rng.choice(np.flatnonzero(action_mask))
        return self.get_best_actions(
            torch.as_tensor(cube_state, dtype=torch.float,
                            device=self.device).unsqueeze(0),
            torch.as_tensor(action_mask, device=self.device).unsqueeze(0)
        ).item()

    def get_experience_batch(self):
        # random.sample is much faster than numpy's self.rng.choice here.
        batch = random.sample(self.replay_buffer, config.BATCH_SIZE)
        observations, actions, next_observations, rewards, dones = [
            [experience[i] for experience in batch] for i in range(5)]
        observations, next_observations = [{
            key: torch.as_tensor(
                np.stack([observation[key] for observation in
                          observations]),
                dtype=(torch.bool if key == 'action_mask' else torch.float),
                device=self.device)
            for key in observations[0].keys()
        } for observations in (observations, next_observations)]
        actions = torch.as_tensor(actions, device=self.device).view(-1, 1)
        rewards = torch.as_tensor(rewards, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.int, device=self.device)
        return observations, actions, next_observations, rewards, dones

    def optimize_policy_network(self):
        observations, actions, next_observations, rewards, dones = (
            self.get_experience_batch())
        with torch.no_grad():
            target_network_next_q_values = self.target_network(
                next_observations['cube_state'])
        best_next_actions = self.get_best_actions(
            next_observations['cube_state'],
            next_observations['action_mask']).unsqueeze(1)
        best_next_q_values = (
            target_network_next_q_values.gather(1, best_next_actions).squeeze()
        )
        target_q_values = (rewards + (1 - dones) * config.DISCOUNT_FACTOR
                           * best_next_q_values)
        all_q_values = self.policy_network(observations['cube_state'])
        q_values = all_q_values.gather(1, actions).squeeze()

        loss = self.criterion(target_q_values, q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self, episode):
        self.epsilon = (
                config.EPSILON_START
                - (config.EPSILON_START - config.EPSILON_END)
                * (episode - config.OPTIMIZATION_START_EPISODE)
                / (config.EPSILON_END_EPISODE
                   - config.OPTIMIZATION_START_EPISODE))

    def evaluate(self, episode):
        self.policy_network.eval()
        solved_count = 0
        for _ in range(config.EVALUATION_SOLVE_COUNT):
            observation = self.env.reset(
                options={'scramble_length': self.scramble_length})
            solved, _, _ = self.solver.solve_cube(observation,
                                                  self.scramble_length)
            if solved:
                solved_count += 1
        self.policy_network.train()
        evaluation = solved_count / config.EVALUATION_SOLVE_COUNT
        self.evaluations.append((episode, evaluation))
        if (evaluation >= config.SCRAMBLE_LENGTH_INCREASE_EVAL_THRESHOLD
                and self.scramble_length < config.MAX_SCRAMBLE_LENGTH):
            self.scramble_length += 1
            self.scramble_lengths.append((episode, self.scramble_length))

    def save_config(self):
        with open('config.py', 'r') as config_file:
            config_string = config_file.read()
        config_string += (f'\n{self.criterion}\n\n{self.optimizer}'
                          f'\n\n{self.policy_network}\n')
        with open(f'{self.run_path}/config.txt', 'w') as config_file:
            config_file.write(config_string)

    def save_training_run_plot(self):
        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        ax.set_xlim(0, config.EPISODE_COUNT)
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlabel('Episode', fontsize=20)
        ax.tick_params(labelsize=12)
        ax.axvline(config.OPTIMIZATION_START_EPISODE, color='grey')
        for episode, scramble_length in self.scramble_lengths:
            ax.axvline(episode, linestyle='dotted')
            ax.text(episode + config.EVALUATION_INTERVAL / 10, 0.5,
                    f'scramble length: {scramble_length}', rotation='vertical',
                    verticalalignment='center')
        ax.plot(self.epsilons, color='g', label='Epsilon')
        ax.plot(self.running_rewards, color='orange', label='Running reward')
        episodes, evaluations = zip(*self.evaluations)
        ax.plot(episodes, evaluations, color='r', label='Evaluation')
        fig.legend(loc='lower right', fontsize=12, bbox_to_anchor=(1, 0),
                   bbox_transform=ax.transAxes)
        fig.tight_layout()
        plt.savefig(f'{self.run_path}/plot.png')

    def save_training_run(self):
        run_name = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        self.run_path = f'training-runs/{run_name}'
        Path(self.run_path).mkdir(parents=True)
        torch.save(self.policy_network.state_dict(),
                   f'{self.run_path}/state_dict.pt')
        self.save_config()
        self.save_training_run_plot()

    def train(self):
        progress_bar = tqdm(range(config.EPISODE_COUNT))
        reward = None
        running_reward = None
        for episode in progress_bar:
            observation = self.env.reset(
                options={'scramble_length': self.scramble_length})
            for step in range(self.scramble_length):
                action = self.get_action(observation)
                next_observation, reward, done, _ = self.env.step(action)
                self.replay_buffer.append(
                    (observation, action, next_observation, reward, done))
                observation = next_observation
                if done:
                    break
            self.epsilons.append(self.epsilon)
            running_reward = (reward if running_reward is None else
                              0.99 * running_reward + 0.01 * reward)
            self.running_rewards.append(running_reward)
            if episode >= config.OPTIMIZATION_START_EPISODE:
                self.optimize_policy_network()
                if episode <= config.EPSILON_END_EPISODE:
                    self.update_epsilon(episode)
            if episode % config.TARGET_NETWORK_UPDATE_INTERVAL == 0:
                self.update_target_network()
            if (episode % config.EVALUATION_INTERVAL == 0
                    or episode == config.EPISODE_COUNT - 1):
                self.evaluate(episode)
            progress_bar.set_postfix(
                {'scramble length': self.scramble_length,
                 'epsilon': f'{self.epsilon:.3f}',
                 'running reward': f'{running_reward:.3f}',
                 'last eval': f'{self.evaluations[-1][1]:.2f}'})
        self.save_training_run()


if __name__ == '__main__':
    trainer = Trainer(seed=13)
    trainer.train()
