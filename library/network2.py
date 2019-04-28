# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from affine_transformation import affine_transform


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=7),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True),
            nn.Conv3d(8, 10, kernel_size=5),
            nn.MaxPool3d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 4)

        x = affine_transform(x, theta)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)
        # transform the input
        x = self.stn(x)

        return x
plt.show()
