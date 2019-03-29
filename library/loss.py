import torch
import torch.nn as nn


class NNCC(nn.Module):
    def __init__(self):
        super(NNCC, self).__init__()

    def forward(self, moving_image, target_image):
        N = moving_image.shape[0] * moving_image.shape[1]
        moving = moving_image - torch.mean(moving_image)
        target = target_image - torch.mean(target_image)

        moving_std_div = torch.sqrt(torch.sum(torch.pow(moving, 2)))
        target_std_div = torch.sqrt(torch.sum(torch.pow(target, 2)))

        numerator = torch.sum(torch.mul(moving, target))
        denominator = torch.mul(moving_std_div, target_std_div)

        return -1/N*torch.div(numerator, denominator)


if __name__ == '__main__':
    moving_image = torch.randn((4, 4), requires_grad=True)
    target_image = torch.randn((4, 4))

    criterion = NNCC()
    loss = criterion(moving_image, target_image)

    print loss