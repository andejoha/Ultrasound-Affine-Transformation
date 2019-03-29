import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, features):
        super(ConvNet, self).__init__()

        # Moving image pipeline:
        #   nn.Conv3d(in_features, out_features, kernel_size)
        #   nn.ReLU()
        #   nn.MaxPool3d(kernel_size)
        self.conv1_m = nn.Conv3d(1, features, 3, padding=1).cuda()
        self.relu1_m = nn.ReLU().cuda()
        self.maxpool1_m = nn.MaxPool3d(2).cuda()
        self.conv2_m = nn.Conv3d(features, features, 3, padding=1).cuda()
        self.relu2_m = nn.ReLU().cuda()
        self.maxpool2_m = nn.MaxPool3d(2).cuda()
        self.conv3_m = nn.Conv3d(features, features, 3, padding=1).cuda()
        self.relu3_m = nn.ReLU().cuda()
        self.maxpool3_m = nn.MaxPool3d(2).cuda()
        self.conv4_m = nn.Conv3d(features, features, 3, padding=1).cuda()
        self.relu4_m = nn.ReLU().cuda()
        self.maxpool4_m = nn.MaxPool3d(2).cuda()
        self.conv5_m = nn.Conv3d(features, features, 3, padding=1).cuda()
        self.relu5_m = nn.ReLU().cuda()

        # Target image pipeline
        self.conv1_t = nn.Conv3d(1, features, 3, padding=1).cuda()
        self.relu1_t = nn.ReLU().cuda()
        self.maxpool1_t = nn.MaxPool3d(2).cuda()
        self.conv2_t = nn.Conv3d(features, features, 3, padding=1).cuda()
        self.relu2_t = nn.ReLU().cuda()
        self.maxpool2_t = nn.MaxPool3d(2).cuda()
        self.conv3_t = nn.Conv3d(features, features, 3, padding=1).cuda()
        self.relu3_t = nn.ReLU().cuda()
        self.maxpool3_t = nn.MaxPool3d(2).cuda()
        self.conv4_t = nn.Conv3d(features, features, 3, padding=1).cuda()
        self.relu4_t = nn.ReLU().cuda()
        self.maxpool4_t = nn.MaxPool3d(2).cuda()
        self.conv5_t = nn.Conv3d(features, features, 3, padding=1).cuda()
        self.relu5_t = nn.ReLU().cuda()

    def apply_downsample(self, x):
        # Do not apply downsampling when the tensor kernel is greater or equal to 6.
        # Why 6? Because we are using kernel size 3 and pooling_size 2 => 3*2=6
        if x.size()[2] >= 6:
            return F.max_pool3d(x, 2)
        else:
            return x

    def forward(self, x):
        [moving, target] = torch.split(x, 1, 1)

        moving_output = self.apply_downsample(self.relu1_m(self.conv1_m(moving)))
        moving_output = self.apply_downsample(self.relu2_m(self.conv2_m(moving_output)))
        moving_output = self.apply_downsample(self.relu3_m(self.conv3_m(moving_output)))
        moving_output = self.apply_downsample(self.relu4_m(self.conv4_m(moving_output)))
        moving_output = self.relu5_m(self.conv5_m(moving_output))
        moving_output = F.avg_pool3d(moving_output, moving_output.size()[2:]).cuda()

        target_output = self.apply_downsample(self.relu1_t(self.conv1_t(target)))
        target_output = self.apply_downsample(self.relu2_t(self.conv2_t(target_output)))
        target_output = self.apply_downsample(self.relu3_t(self.conv3_t(target_output)))
        target_output = self.apply_downsample(self.relu4_t(self.conv4_t(target_output)))
        target_output = self.relu5_t(self.conv5_t(target_output))
        target_output = F.avg_pool3d(target_output, target_output.size()[2:]).cuda()

        y = torch.cat((moving_output, target_output), 1)

        return y.view(-1, y.size()[1])


class Net(nn.Module):
    def __init__(self, features):
        super(Net, self).__init__()

        self.conv_net = ConvNet(features).cuda()
        self.fc1 = nn.Linear(features * 2, features).cuda()
        self.relu1 = nn.ReLU().cuda()
        self.fc2 = nn.Linear(features, 7).cuda()

    def forward(self, x):
        conv_output = self.conv_net(x)
        fc_output = self.relu1(self.fc1(conv_output))
        fc_output = self.fc2(fc_output)

        # Applying output constraints
        translation = fc_output[:, :3]
        scaling = fc_output[:, 3:6]
        rotation = fc_output[:, 6:]

        scaling = torch.clamp(scaling, 0.5, 1.5)
        rotation = torch.clamp(rotation, -math.pi/6, math.pi/6)

        return torch.cat((translation, scaling, rotation), 1)


if __name__ == '__main__':
    moving_image = torch.randn(1, 1, 15, 15, 15).cuda()
    target_image = torch.randn(1, 1, 15, 15, 15).cuda()
    x = torch.cat((moving_image, target_image), 1)

    net = Net(32).cuda()
    out = net(x)

    print out
