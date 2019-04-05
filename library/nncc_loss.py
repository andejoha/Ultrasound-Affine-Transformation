import torch
import torch.nn as nn
import torch.nn.functional as F
import affine_transformation as at


class NNCC(nn.Module):
    def __init__(self):
        super(NNCC, self).__init__()

    def forward(self, theta, target_image, moving_image, data_size):
        N, D, H, W = data_size

        # Adding channel element
        target_image = target_image.unsqueeze(1)
        moving_image = moving_image.unsqueeze(1)

        # Extending theta to include batches
        predicted_theta = torch.empty(N, theta.shape[0], theta.shape[1]).cuda()
        predicted_theta[:] = theta

        affine_grid = at.affine_grid_generator_3D(predicted_theta, (N, 1, D, H, W))
        predicted_image = F.grid_sample(moving_image, affine_grid)

        n_dims = tuple([i for i in range(len(moving_image.size()))])

        moving = (predicted_image - torch.mean(predicted_image)).cuda()
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
