import torch
import torch.nn as nn


class NCC(nn.Module):
    def __init__(self):
        super(NCC, self).__init__()

    def forward(self, transformed_image, target_image):
        return ncc(transformed_image, target_image)


def ncc(transformed_image, target_image):
    warped = (transformed_image[:] - torch.mean(transformed_image, (1, 2, 3), keepdim=True)).cuda()
    target = (target_image[:] - torch.mean(target_image, (1, 2, 3), keepdim=True)).cuda()

    moving_std_div = torch.sqrt(torch.sum(torch.pow(warped, 2), (1, 2, 3))).cuda()
    target_std_div = torch.sqrt(torch.sum(torch.pow(target, 2), (1, 2, 3))).cuda()

    numerator = torch.sum(torch.mul(warped, target), (1, 2, 3)).cuda()
    denominator = torch.mul(moving_std_div, target_std_div).cuda()

    # To prevent division by 0
    epsilon = 0.0000001
    loss = - torch.div(numerator, denominator + epsilon).cuda()
    return loss


if __name__ == '__main__':
    torch.manual_seed(0)
    transformed_image = torch.randint(255, (64, 30, 30, 30), dtype=torch.float64)
    target_image = torch.randint(255, (64, 30, 30, 30), dtype=torch.float64)

    print(transformed_image.shape)

    loss = ncc(transformed_image, target_image)

    print(loss)
