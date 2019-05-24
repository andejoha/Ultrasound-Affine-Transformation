# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .affine_transformation import affine_transform


class PatchNet(nn.Module):
    def __init__(self):
        super(PatchNet, self).__init__()
        # Spatial transformer localization-network
        self.moving_localization = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=10),
            nn.AvgPool3d(2, stride=2),
            nn.ReLU(True),
            nn.Conv3d(8, 16, kernel_size=5),
            nn.AvgPool3d(2, stride=2),
            nn.ReLU(True)
        )

        self.target_localization = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=10),
            nn.AvgPool3d(2, stride=2),
            nn.ReLU(True),
            nn.Conv3d(8, 16, kernel_size=5),
            nn.AvgPool3d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 3 * 3 * 3, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        [moving, target] = torch.split(x, 1, 1)

        moving = self.moving_localization(moving)
        target = self.target_localization(target)

        xs = torch.cat((moving, target), 1)
        xs = xs.view(-1, 32*3*3*3)

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 4)

        return theta

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        return x


class FullNet(nn.Module):
    def __init__(self):
        super(FullNet, self).__init__()
        self.shape = ()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=10),
            nn.AvgPool3d(2, stride=2),
            nn.ReLU(True),
            nn.Conv3d(32, 32, kernel_size=5),
            nn.AvgPool3d(2, stride=2),
            nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=3),
            nn.AvgPool3d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 24 * 16 * 24, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 64 * 24 * 16 * 24)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 4)

        return theta

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stn(x)
        return x
