# 2x2 Cube Solver

A reinforcement learning agent that solves a 2x2 Rubik's cube, implemented
with a Dueling Double Deep Q-Network.

## Description

Inside the `2x2-cube-solver` directory, `cube.py` and `cube_env.py` define the
2x2 Rubik's cube and its Gym environment.

`dueling_dqn.py` is a simple 4-layer Dueling DQN model implemented in PyTorch,
with separate outputs for the state value and the action advantages.

Training the model is done using `train.py`, and the trained model can be used
to solve cubes through `solve.py`.

## Usage

### Training

To train a model, set the configuration variables in `config.py` and
run `train.py`.
The trained model as well as the training configuration and a plot of the
training run are saved in the `training-runs` directory.

Resuming a previous training run can be done by setting the `RESUME_RUN_ID`
configuration variable.

### Solving

Two ways of solving cubes using a trained model are available in `solve.py`.

The first method uses the `evaluate_model` function and solves a large number
of randomly scrambled cubes. Solve results and a plot of the solution length
distribution are displayed.

The second method uses the `solve_scramble` function and solves a single cube
scrambled according to a given scramble. This can be used to examine a model's
performance for a specific scramble of interest.

For solving, the configuration variables must be set inside the respective
functions before calling them in `solve.py`.

## Development and training

Initially, the scramble length was fixed for the entire training run. Because a
reward was only given when the cube was solved, and the chance of reaching the
solved state through nearly random moves from a sufficiently well-scrambled
initial state was extremely low, the model learned almost nothing.

To fix this, the training process was changed so that the scramble length was
initially set to a small number and was increased by 1 each time the model
reached a certain solution rate threshold during evaluation. This solved the
previous problem of the model not learning, but another problem that emerged
was [catastrophic forgetting](https://en.wikipedia.org/wiki/Catastrophic_interference).
Whenever the scramble length was increased, there was a good chance that the
model would rapidly forget most of its previously learned knowledge.

An attempt to mitigate this problem was to ensure that some scrambles of the
previous length were included along with the current length scrambles, but this
was unsuccessful. What finally worked was to introduce a concept of average
scramble length and to increase this value in increments of 0.1. The average
scramble length determined the proportion of scramble lengths given to the
model. For example, if the average scramble length was 5.2, 80% of the training
scrambles would be of length 5, and the remaining 20% would be of length 6.
This allowed a much more gradual change in the model's training data and
resulted in stable training.

The training runs for the final model can be found in the `training-runs`
directory. It was trained over 7 training runs and 12 million steps, starting
at an average scramble length of 3.0. The plot for the first training run is
shown below.

|      ![](training-runs/220504192153/plot.png)      |
|:--------------------------------------------------:|
| Plot of the first training run for the final model |

As can be seen in the plot, the rate of increase of the scramble length
decreased as the training progressed. Training was ultimately stopped at
an average scramble length of 7.1 as progress at that point was getting
stagnant.

## Results

The final model is able to solve the 2x2 Rubik's cube from a variety of
scrambled states. Shown below is an example of a randomly generated scramble
and the model's solution. The moves are written in standard
[Rubik's cube notation](https://en.wikipedia.org/wiki/Rubik%27s_Cube#Move_notation).

| ![](https://user-images.githubusercontent.com/39209141/167299726-121506a4-bfcd-41e3-9f2c-2e63e70cbdd2.png) | ![](https://user-images.githubusercontent.com/39209141/167299756-af3994e1-7f30-44c6-aaa5-76f833723aa3.png) |
|:----------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------:|
|                                               Scrambled cube                                               |                                                Solved cube                                                 |
|                                  Scramble: R2 F' R2 U' F R2 U' R' F R F'                                   |                                        Solution: U R F' U2 R' F' R                                         |

The performance of the model was evaluated with sets of 10,000 randomly
scrambled cubes. The results are summarized in the following table. The first
column is the length of the scramble in [quarter turn metric (QTM)](https://www.speedsolving.com/wiki/index.php/Metric#QTM),
and the second column is the maximum number of moves that was allowed for the
model to solve the cube, also in QTM. The solution rate in the third column is
the proportion of cubes that were successfully solved within the allowed move
count. Scrambles longer than 14 moves were not tested because a 2x2 cube in any
state can be solved in a maximum of 14 quarter turns (the [God's number](https://ruwix.com/the-rubiks-cube/gods-number/)).

| Scramble length | Maximum allowed solution length | Solution rate |
|:---------------:|:-------------------------------:|:-------------:|
|        4        |                4                |    100.0%     |
|        5        |                5                |     98.0%     |
|        6        |                6                |     92.6%     |
|        7        |                7                |     82.3%     |
|        7        |               20                |     88.6%     |
|        8        |                8                |     54.8%     |
|        9        |                9                |     38.0%     |
|       14        |               14                |     14.0%     |
|       14        |               20                |     18.6%     |
|       14        |               500               |    100.0%     |

The solution rate decreased as the scramble length increased, but increased
when the maximum allowed solution length was increased. Although the model was
never trained on scrambles longer than 8 moves, it solved many of them
successfully. In fact, it was able to solve all 10,000 of the length 14
scrambles when the maximum allowed solution length was set to 500. The solution
length distribution is shown below.

| ![](https://user-images.githubusercontent.com/39209141/167299769-12782ab3-9be0-4597-84d9-dcbc560309fa.png) |
|:----------------------------------------------------------------------------------------------------------:|
|                       Solution length distribution for 10,000 scrambles of length 14                       |

Part of this solving ability can be attributed to the fact that moves leading
to a repeated state were not allowed. For difficult scrambles, the model
presumably makes nearly random moves until it reaches a state that it is
familiar with. For comparison, a completely untrained model was given the same
task. It managed to solve none of the 10,000 scrambles within 1000 moves, even
with the restriction of no repeated states. This result demonstrates that the
model did indeed achieve some degree of solving ability.

## Future work

There exist several potential avenues of improvement, including the following:

- More experimentation with hyperparameter tuning
- Using a different network architecture, such as
  a [residual neural network](https://en.wikipedia.org/wiki/Residual_neural_network)
- Using a different reinforcement learning algorithm, possibly one that takes
  advantage of multiprocessing, such
  as [proximal policy optimization](https://openai.com/blog/openai-baselines-ppo/)

The use of a reinforcement learning library
like [Stable Baselines](https://stable-baselines3.readthedocs.io/en/master/index.html)
could also help save some time and effort.
