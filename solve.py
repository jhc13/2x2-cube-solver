from collections import namedtuple

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from cube_env import CubeEnv
from dueling_dqn import DuelingDQN

Solve = namedtuple('Solve', ['solved', 'scramble', 'solution',
                             'solution_length'])


class Solver:
    def __init__(self, device, model, env):
        self.device = device
        self.model = model
        self.env = env

    def solve_cube(self, observation, max_step_count):
        """
        Solve a given cube.
        """
        previous_states = []
        moves = []
        solved = False
        for _ in range(max_step_count):
            cube_state = torch.as_tensor(observation['cube_state'],
                                         dtype=torch.float, device=self.device)
            action_mask = observation['action_mask']
            with torch.no_grad():
                q_values = self.model(cube_state.unsqueeze(0)).squeeze()
            q_values[~action_mask] = float('-inf')
            _, actions = torch.sort(q_values, descending=True)
            actions = [action.item() for action in actions.squeeze()]
            action = None
            done = False
            for action in actions:
                observation, _, done, _ = self.env.step(action)
                cube_state = torch.as_tensor(observation['cube_state'],
                                             dtype=torch.float,
                                             device=self.device)
                if any(torch.equal(cube_state, previous_state)
                       for previous_state in previous_states):
                    self.env.step(self.env.get_undoing_action(action))
                    continue
                break
            previous_states.append(cube_state)
            moves.append(self.env.moves[action])
            if done:
                solved = True
                break
        solution = ' '.join(moves)
        solution_length = len(moves)
        return solved, solution, solution_length

    def solve_random_cubes(self, solve_count, scramble_quarter_turn_count,
                           max_step_count):
        """
        Solve a given number of randomly scrambled cubes.
        """
        solved_count = 0
        solves = []
        for _ in tqdm(range(solve_count)):
            observation, info = self.env.reset(return_info=True, options={
                'scramble_quarter_turn_count': scramble_quarter_turn_count})
            scramble = info['scramble']
            solved, solution, solution_length = self.solve_cube(
                observation, max_step_count)
            if solved:
                solved_count += 1
            solves.append(Solve(solved, scramble, solution, solution_length))
        return solved_count, solves


def print_solves(solves, print_type='all'):
    """
    Format and print a given list of solves.

    print_type determines which of the solves are printed. It can be one of
    "all", "solved", or "unsolved".
    """
    for solve in solves:
        if (print_type == 'all'
                or (print_type == 'solved' and solve.solved)
                or (print_type == 'unsolved' and not solve.solved)):
            print(f'Solved: {solve.solved}, scramble: {solve.scramble}, '
                  f'solution: {solve.solution}')


def plot_solution_lengths(solves):
    """
    Plot the distribution of solution lengths for a given list of solves.
    """
    solution_lengths = [solve.solution_length for solve in solves
                        if solve.solved]
    if not solution_lengths:
        return
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.hist(solution_lengths, bins=range(0, max(solution_lengths) + 1))
    ax.set_xlim(0, max(solution_lengths))
    ax.set_xlabel('Moves', fontsize=20)
    ax.set_ylabel('Count', fontsize=20)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.show()


def main():
    run_id = '220502012237'
    cube_count = 1000
    scramble_quarter_turn_count = 14
    max_step_count = 20

    # torch.equal in solve_cube is very slow on GPU, so use CPU.
    device = torch.device('cpu')
    print(f'Using device: {device}.')
    model = DuelingDQN().to(device)
    model.load_state_dict(torch.load(f'training-runs/{run_id}/state_dict.pt'))
    model.eval()
    env = CubeEnv()
    solver = Solver(device, model, env)
    solved_count, solves = solver.solve_random_cubes(
        cube_count, scramble_quarter_turn_count, max_step_count)
    print_solves(solves, print_type='all')
    print(f'{solved_count}/{cube_count} ({solved_count / cube_count:.2%}) '
          f'solved')
    plot_solution_lengths(solves)


if __name__ == '__main__':
    main()
