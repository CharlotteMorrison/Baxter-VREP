import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """Implements a pair of Critic neural networks."""
        super(Critic, self).__init__()

        # Critic network 1
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

        # Critic network 2
        self.layer4 = nn.Linear(state_dim + action_dim, 400)
        self.layer5 = nn.Linear(400, 300)
        self.layer6 = nn.Linear(300, 1)

    def forward(self, x, u):
        """Returns the Q values for both Critic networks"""
        x = x.float()
        xu = torch.cat([x, u], 1)

        # Critic network 1 forward
        x1 = F.relu(self.layer1(xu))
        x1 = F.relu(self.layer2(x1))
        x1 = self.layer3(x1)

        # Critic network 2 forward
        x2 = F.relu(self.layer4(xu))
        x2 = F.relu(self.layer5(x2))
        x2 = self.layer6(x2)

        return x1, x2

    def get_q(self, x, u):
        """Returns the Q value for only Critic 1"""
        x = x.float()
        xu = torch.cat([x, u], 1)

        # Critic network 1 forward
        x1 = F.relu(self.layer1(xu))
        x1 = F.relu(self.layer2(x1))
        x1 = self.layer3(x1)

        return x1
