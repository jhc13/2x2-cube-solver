import datetime
import math
# Python's random module is required in addition to NumPy's random number
# generator for fast sampling of tuples.
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
        # The scramble length does not matter here because the environment is
        # reset before each episode. The reset here is only for setting the
        # seed.
        self.env.reset(seed=seed, options={'scramble_length': 0})
        # A deque is used to automatically remove old experiences as new ones
        # are added.
        self.replay_buffer = deque(maxlen=config.REPLAY_BUFFER_SIZE)
        self.epsilon = config.EPSILON_START
        self.online_network = DuelingDQN().to(self.device)
        if config.RESUME_RUN_ID is not None:
            self.online_network.load_state_dict(torch.load(
                f'training-runs/{config.RESUME_RUN_ID}/state_dict.pt'))
        self.target_network = DuelingDQN().to(self.device)
        # Only the online network is directly trained, so set the target
        # network to evaluation mode.
        self.target_network.eval()
        # Copy the online network's state_dict to the target network.
        self.update_target_network()
        # Other optimizers could also be used.
        self.optimizer = optim.RMSprop(self.online_network.parameters(),
                                       lr=config.LEARNING_RATE)
        # Huber loss was chosen because of its robustness to outliers.
        self.criterion = nn.HuberLoss()
        self.solver = Solver(self.device, self.online_network, self.env)
        self.average_scramble_length = config.INITIAL_SCRAMBLE_LENGTH
        # Save the various updated values so that they can be plotted.
        self.scramble_lengths = [(0, config.INITIAL_SCRAMBLE_LENGTH)]
        self.epsilons = []
        self.running_rewards = []
        self.evaluations = []
        self.run_path = None

    def update_target_network(self):
        """
        Update the target network's state_dict by copying the online network's
        state_dict to it.
        """
        self.target_network.load_state_dict(self.online_network.state_dict())

    def get_scramble_length(self) -> int:
        """
        Get an integer scramble length based on the average scramble length.

        For example, if the average scramble length is 5.2, return 5 with a
        probability of 0.8 and 6 with a probability of 0.2.
        """
        return (math.ceil(self.average_scramble_length)
                if self.rng.random() < self.average_scramble_length % 1
                else math.floor(self.average_scramble_length))

    def get_best_actions(self, cube_states, action_masks) -> torch.Tensor:
        """
        Get the actions with the highest Q-value for each cube state, excluding
        the invalid actions specified by the action masks.
        """
        # In a Double DQN, the best action is chosen using the online network,
        # and the target network is only used during optimization to obtain the
        # target Q-values.
        self.online_network.eval()
        with torch.no_grad():
            q_values = self.online_network(cube_states)
        self.online_network.train()
        q_values[~action_masks] = float('-inf')
        best_actions = q_values.max(dim=1).indices
        return best_actions

    def get_action(self, observation) -> int:
        """
        Get an action for a given observation based on the current epsilon
        value. A random action is chosen with a probability of epsilon and the
        action with the highest Q-value is chosen with a probability of
        1 - epsilon.
        """
        cube_state = observation['cube_state']
        action_mask = observation['action_mask']
        if self.rng.random() < self.epsilon:
            # Randomly choose one of the valid actions.
            return self.rng.choice(np.flatnonzero(action_mask))
        # Unsqueeze the cube state to add a batch dimension. The batch size is
        # 1.
        return self.get_best_actions(
            torch.as_tensor(cube_state, dtype=torch.float,
                            device=self.device).unsqueeze(0),
            torch.as_tensor(action_mask, device=self.device).unsqueeze(0)
        ).item()

    def get_experience_batch(self) -> tuple[dict, torch.Tensor, dict,
                                            torch.Tensor, torch.Tensor]:
        """
        Get a batch of experiences from the replay buffer.

        Returns a tuple of the following:
        - A dictionary with a tensor of the cube states and a tensor of the
        action masks as values, corresponding to the "current observations" at
        the time the experience was sampled.
        - A tensor of the chosen actions.
        - A dictionary similar to the first dictionary, but corresponding to
        the "next observations" after the actions were taken.
        - A tensor of the rewards received as a result of the actions.
        - A tensor of done values, indicating whether the cube was solved after
        the actions were taken.
        """
        # random.sample is much faster than numpy's rng.choice here.
        batch = random.sample(self.replay_buffer, config.BATCH_SIZE)
        # Convert the batch of tuples to a tuple of batches.
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
        # The actions tensor is reshaped for use during optimization.
        actions = torch.as_tensor(actions, device=self.device).view(-1, 1)
        rewards = torch.as_tensor(rewards, device=self.device)
        # The dones tensor is converted to a tensor of integers to allow
        # subtraction during optimization.
        dones = torch.as_tensor(dones, dtype=torch.int, device=self.device)
        return observations, actions, next_observations, rewards, dones

    def optimize_online_network(self):
        """
        Optimize the online network using a batch of experiences from the
        replay buffer.
        """
        observations, actions, next_observations, rewards, dones = (
            self.get_experience_batch())
        # torch.no_grad() is used because the target network is not trained
        # directly.
        with torch.no_grad():
            next_q_values = self.target_network(
                next_observations['cube_state'])
        best_next_actions = self.get_best_actions(
            next_observations['cube_state'],
            next_observations['action_mask']).unsqueeze(1)
        best_next_action_q_values = (
            next_q_values.gather(1, best_next_actions).squeeze()
        )
        target_q_values = (rewards + (1 - dones) * config.DISCOUNT_FACTOR
                           * best_next_action_q_values)
        all_online_q_values = self.online_network(observations['cube_state'])
        online_q_values = all_online_q_values.gather(1, actions).squeeze()

        # Compute the loss and update the parameters of the online network.
        loss = self.criterion(target_q_values, online_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self, episode):
        """
        Update the epsilon value according to the configuration variables.
        A linear decay is used here, but other forms of decay could also be
        used.
        """
        self.epsilon = (
                config.EPSILON_START
                - (config.EPSILON_START - config.EPSILON_END)
                * (episode - config.OPTIMIZATION_START_EPISODE)
                / (config.EPSILON_END_EPISODE
                   - config.OPTIMIZATION_START_EPISODE))

    def evaluate(self, episode):
        """
        Evaluate the performance of the online network by making it solve
        randomly scrambled cubes.
        """
        self.online_network.eval()
        solved_count = 0
        for _ in range(config.EVALUATION_SOLVE_COUNT):
            scramble_length = self.get_scramble_length()
            observation = self.env.reset(
                options={'scramble_length': scramble_length})
            solved, _, _ = self.solver.solve_cube(observation, scramble_length)
            if solved:
                solved_count += 1
        self.online_network.train()
        evaluation = solved_count / config.EVALUATION_SOLVE_COUNT
        self.evaluations.append((episode, evaluation))
        # Increase the average scramble length if the evaluation result is
        # above the threshold. In this way, the model can be trained on
        # scrambles of incrementally increasing difficulty.
        if (evaluation >= config.SCRAMBLE_LENGTH_INCREASE_EVAL_THRESHOLD
                and self.average_scramble_length < config.MAX_SCRAMBLE_LENGTH):
            self.average_scramble_length = round(
                self.average_scramble_length + 0.1, 1)
            self.scramble_lengths.append(
                (episode, self.average_scramble_length))

    def save_config(self):
        """Save the training configuration to a text file."""
        with open('2x2-cube-solver/config.py', 'r') as config_file:
            config_string = config_file.read()
        config_string += (f'\n{self.criterion}\n\n{self.optimizer}'
                          f'\n\n{self.online_network}\n')
        with open(f'{self.run_path}/config.txt', 'w') as config_file:
            config_file.write(config_string)

    def save_training_run_plot(self):
        """
        Save a plot of the training run, showing the progression of epsilon,
        the running reward, and the evaluation along with the average scramble
        length.
        """
        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        ax.set_xlim(0, config.EPISODE_COUNT)
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlabel('Episode', fontsize=20)
        ax.tick_params(labelsize=12)
        ax.axvline(config.OPTIMIZATION_START_EPISODE, color='grey')
        for episode, scramble_length in self.scramble_lengths:
            # Plot vertical lines for all scramble lengths with the same
            # integer value as the final scramble length. For the previous
            # scramble lengths, only plot a line if the value is an integer.
            # For example, if the final scramble length is 7.3, plot vertical
            # lines for 7.3, 7.2, 7.1, 7.0, 6.0, 5.0, and so on.
            final_scramble_length = math.floor(self.scramble_lengths[-1][1])
            if (scramble_length != math.floor(scramble_length)
                    and math.floor(scramble_length) != final_scramble_length):
                continue
            ax.axvline(episode, linestyle='dotted')
            ax.text(episode + config.EVALUATION_INTERVAL / 10, 0.5,
                    f'average scramble length: {scramble_length:.1f}',
                    rotation='vertical', verticalalignment='center')
        ax.plot(self.epsilons, color='g', label='Epsilon')
        ax.plot(self.running_rewards, color='orange', label='Running reward')
        episodes, evaluations = zip(*self.evaluations)
        ax.plot(episodes, evaluations, color='r', label='Evaluation')
        fig.legend(loc='lower right', fontsize=12, bbox_to_anchor=(1, 0),
                   bbox_transform=ax.transAxes)
        fig.tight_layout()
        plt.savefig(f'{self.run_path}/plot.png')

    def save_training_run(self):
        """
        Save the state_dict, the training configuration, and a plot of the
        training run in the training-runs directory.
        """
        # Generate a unique run id using the current time.
        run_id = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        self.run_path = f'training-runs/{run_id}'
        Path(self.run_path).mkdir(parents=True)
        torch.save(self.online_network.state_dict(),
                   f'{self.run_path}/state_dict.pt')
        self.save_config()
        self.save_training_run_plot()

    def train(self):
        """Train the model and save the results."""
        progress_bar = tqdm(range(config.EPISODE_COUNT))
        reward = None
        running_reward = None
        for episode in progress_bar:
            scramble_length = self.get_scramble_length()
            observation = self.env.reset(
                options={'scramble_length': scramble_length})
            for step in range(scramble_length):
                action = self.get_action(observation)
                next_observation, reward, done, _ = self.env.step(action)
                self.replay_buffer.append(
                    (observation, action, next_observation, reward, done))
                observation = next_observation
                # Quit solving if the cube is solved.
                if done:
                    break
            self.epsilons.append(self.epsilon)
            running_reward = (reward if running_reward is None else
                              0.99 * running_reward + 0.01 * reward)
            self.running_rewards.append(running_reward)
            if episode >= config.OPTIMIZATION_START_EPISODE:
                self.optimize_online_network()
                if episode <= config.EPSILON_END_EPISODE:
                    self.update_epsilon(episode)
            if episode % config.TARGET_NETWORK_UPDATE_INTERVAL == 0:
                self.update_target_network()
            if (episode % config.EVALUATION_INTERVAL == 0
                    or episode == config.EPISODE_COUNT - 1):
                self.evaluate(episode)
            progress_bar.set_postfix(
                {'scramble length': f'{self.average_scramble_length:.1f}',
                 'epsilon': f'{self.epsilon:.3f}',
                 'running reward': f'{running_reward:.3f}',
                 'last eval': f'{self.evaluations[-1][1]:.3f}'})
        self.save_training_run()


if __name__ == '__main__':
    trainer = Trainer(seed=13)
    trainer.train()
