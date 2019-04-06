import torch
import torch.nn as nn
import torch.nn.functional as F
import affine_transformation as at


class NNCC(nn.Module):
    def __init__(self):
        super(NNCC, self).__init__()

    def forward(self, moving_image, target_image):
        n_dims = tuple([i for i in range(len(moving_image.size()))])

        moving = (moving_image - torch.mean(moving_image)).cuda()
        target = (target_image - torch.mean(target_image)).cuda()

        # print moving[0:5, 0:5]
        # print target[0:5, 0:5]

        moving_std_div = torch.sqrt(torch.sum(torch.pow(moving, 2))).cuda()
        target_std_div = torch.sqrt(torch.sum(torch.pow(target, 2))).cuda()

        # print moving_std_div
        # print target_std_div

        numerator = torch.sum(torch.mul(moving, target)).cuda()
        denominator = torch.mul(moving_std_div, target_std_div).cuda()

        # print numerator
        # print denominator

        return (torch.div(numerator, denominator).cuda())


if __name__ == '__main__':
    torch.manual_seed(0)
    moving_image = torch.randn(1, 4, 4, 4)
    target_image = torch.randn(1, 4, 4, 4)

    criterion = NNCC()
    loss = criterion(moving_image, target_image)

    print loss
