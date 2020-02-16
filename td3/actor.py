import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        """Create the Actor neural network layers."""
        super(Actor, self).__init__()

        self.max_action = max_action

        # using DQN model with only 1 input image
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.lin_layer1 = nn.Linear(7*7*64, 512)
        self.lin_layer2 = nn.Linear(512, action_dim)

        self.activation = nn.ReLU()  # may use tanh

    def forward(self, x):
        """Forward pass in Actor neural network."""

        # normalize image
        x = x / 255
        x = self.conv_layer1(x)
        x = self.activation(x)
        x = self.conv_layer2(x)
        x = self.activation(x)
        x = self.conv_layer3(x)
        x = self.activation(x)

        x = x.view(x.size(0), -1)

        x = self.lin_layer1(x)
        x = self.activation(x)
        x = self.max_action * torch.tanh(self.lin_layer2(x))
        return x
