import torch
import torch.nn.functional as F
from torch import nn


class DuelingDQN(nn.Module):
    # A Dueling DQN has two outputs: one for the value of the state and one for
    # the advantages of each action. This allows the network to learn the state
    # values without having to learn the effect of each action.
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(60, 512)
        self.batch_norm_1 = nn.BatchNorm1d(512)
        self.linear_2 = nn.Linear(512, 256)
        self.batch_norm_2 = nn.BatchNorm1d(256)
        self.linear_3 = nn.Linear(256, 128)
        self.batch_norm_3 = nn.BatchNorm1d(128)
        self.output_1 = nn.Linear(128, 1)
        self.output_2 = nn.Linear(128, 6)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        # The ReLU activation function was also tried, but it produced much
        # worse results.
        x = self.batch_norm_1(F.elu(self.linear_1(x)))
        x = self.batch_norm_2(F.elu(self.linear_2(x)))
        x = self.batch_norm_3(F.elu(self.linear_3(x)))
        state_values = self.output_1(x)
        raw_advantages = self.output_2(x)
        # The advantage of the best action is 0.
        advantages = (raw_advantages
                      - raw_advantages.max(dim=1, keepdim=True).values)
        q_values = state_values + advantages
        return q_values
