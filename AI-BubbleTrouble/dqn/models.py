import torch.nn as nn
import torch.nn.functional as F
from parameters import *


class CNN(nn.Module):
    def __init__(self, in_channels=4, n_actions=4):
        """
        Initialize Deep Q Network
        :param in_channels: Number of input channels
        :param n_actions: Number of outputs
        """

        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, NUM_FILTER_L1, kernel_size=KER_L1, stride=STRIDE_L1)
        self.bn1 = nn.BatchNorm2d(NUM_FILTER_L1)
        self.conv2 = nn.Conv2d(NUM_FILTER_L1, NUM_FILTER_L2, kernel_size=KER_L2, stride=STRIDE_L2)
        self.bn2 = nn.BatchNorm2d(NUM_FILTER_L2)
        self.conv3 = nn.Conv2d(NUM_FILTER_L2, NUM_FILTER_L3, kernel_size=KER_L3, stride=STRIDE_L3)
        self.bn3 = nn.BatchNorm2d(NUM_FILTER_L3)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        width_l1 = conv2d_size_out(WIDTH, KER_L1, STRIDE_L1)
        width_l2 = conv2d_size_out(width_l1, KER_L2, STRIDE_L2)
        convw = conv2d_size_out(width_l2, KER_L3, STRIDE_L3)

        heigt_l1 = conv2d_size_out(HEIGHT, KER_L1, STRIDE_L1)

        height_l2 = conv2d_size_out(heigt_l1, KER_L2, STRIDE_L2)
        convh = conv2d_size_out(height_l2, KER_L3, STRIDE_L3)

        linear_input_size = convw * convh * NUM_FILTER_L3

        self.fc4 = nn.Linear(linear_input_size, linear_input_size)
        self.head = nn.Linear(linear_input_size, n_actions)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        return self.head(x)
