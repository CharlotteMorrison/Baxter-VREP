import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """Implements a pair of Critic neural networks."""
        super(Critic, self).__init__()

        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.lin_layer1 = nn.Linear(4032, 512)   # 3136
        self.lin_layer2 = nn.Linear(512, 1)

        self.conv_layer4 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv_layer5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv_layer6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.lin_layer3 = nn.Linear(7056, 512)
        self.lin_layer4 = nn.Linear(512, 1)

    def forward(self, x, u):
        """Returns the Q values for both Critic networks"""
        x = x.float()
        x = x / 255
        u = u.unsqueeze(1).expand(-1, 84, -1)

        xu = torch.cat([x, u], 2).unsqueeze(1)

        x1 = F.relu(self.conv_layer1(xu))
        x1 = F.relu(self.conv_layer2(x1))
        x1 = F.relu(self.conv_layer3(x1))

        x1 = x1.view(x1.size(0), -1)

        x1 = F.relu(self.lin_layer1(x1))
        x1 = self.lin_layer2(x1)

        x2 = F.relu(self.conv_layer4(xu))
        x2 = F.relu(self.conv_layer5(x2))
        x2 = F.relu(self.conv_layer6(x2))

        x2 = x.view(x2.size(0), -1)

        x2 = F.relu(self.lin_layer3(x2))
        x2 = self.lin_layer4(x2)
        return x1, x2

    def get_q(self, x, u):
        """Returns the Q value for only Critic 1"""
        x = x.float()
        x = x / 255
        u = u.unsqueeze(1).expand(-1, 84, -1)
        xu = torch.cat([x, u], 2).unsqueeze(1)

        x1 = F.relu(self.conv_layer1(xu))
        x1 = F.relu(self.conv_layer2(x1))
        x1 = F.relu(self.conv_layer3(x1))

        x1 = x1.view(x1.size(0), -1)

        x1 = F.relu(self.lin_layer1(x1))
        x1 = self.lin_layer2(x1)
        return x1
