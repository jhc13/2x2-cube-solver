# 2x2 Cube Solver
A Dueling Double DQN implementation of a 2x2 Rubik's cube solver.

## Description
Inside the `2x2-cube-solver` directory, `cube.py` and `cube_env.py` define the 2x2 Rubik's cube and its Gym environment.

`dueling_dqn.py` is a simple 4-layer Dueling DQN model implemented in PyTorch, with separate outputs for the state value and the action advantages.

Training the model is done using `train.py`, and the trained model can be used to solve cubes through `solve.py`.

## Usage
### Training
To train a model, set the configuration variables in `config.py` and run `train.py`.
The trained model as well as the training configuration and a plot of the training run are saved in the `training-runs` directory.

Resuming a previous training run can be done by setting the `RESUME_RUN_ID` configuration variable.

### Solving
Two ways of solving cubes using a trained model are available in `solve.py`.

The first method uses the `evaluate_model` function and solves a large number of randomly scrambled cubes. Solve results and a plot of the solution length distribution are displayed.

The second method uses the `solve_scramble` function and solves a single cube
scrambled according to a given scramble. This can be used to examine a model's performance for a specific scramble of interest.

For solving, the configuration variables must be set inside the respective functions before calling them in `solve.py`.
