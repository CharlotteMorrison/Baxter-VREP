import torch.nn as nn
from dqn import dqn_constants as cons


class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):

        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=cons.num_frames_stacked, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.linear1 = nn.Linear(7 * 7 * 64, cons.hidden_layer)
        self.linear2 = nn.Linear(cons.hidden_layer, cons.output_size)

        # self.activation = nn.Tanh()
        self.activation = nn.ReLU()

    def forward(self, x):

        if cons.normalize_image:
            x = x / 255

        output_conv = self.conv1(x)
        output_conv = self.activation(output_conv)
        output_conv = self.conv2(output_conv)
        output_conv = self.activation(output_conv)
        output_conv = self.conv3(output_conv)
        output_conv = self.activation(output_conv)

        output_conv = output_conv.view(output_conv.size(0), -1)

        output_linear = self.linear1(output_conv)
        output_linear = self.activation(output_linear)
        output_linear = self.linear2(output_linear)

        return output_linear
