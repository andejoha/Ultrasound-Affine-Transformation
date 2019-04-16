import torch
import torch.nn as nn
import torch.nn.functional as F
import affine_transformation as at
import matplotlib.pyplot as plt


class NNCC(nn.Module):
    def __init__(self):
        super(NNCC, self).__init__()

    def forward(self, warped_image, target_image):
        '''
        plt.subplot(121)
        plt.imshow(warped_image[0, int(warped_image.shape[1] / 2)].detach().cpu(), cmap='gray')
        plt.subplot(122)
        plt.imshow(target_image[0, int(target_image.shape[1] / 2)].detach().cpu(), cmap='gray')
        plt.show()
        '''

        warped = (warped_image[:] - torch.mean(warped_image, (1, 2, 3), keepdim=True)).cuda()
        target = (target_image[:] - torch.mean(target_image, (1, 2, 3), keepdim=True)).cuda()

        moving_std_div = torch.sqrt(torch.sum(torch.pow(warped, 2), (1, 2, 3))).cuda()
        target_std_div = torch.sqrt(torch.sum(torch.pow(target, 2), (1, 2, 3))).cuda()

        numerator = torch.sum(torch.mul(warped, target), (1, 2, 3)).cuda()
        denominator = torch.mul(moving_std_div, target_std_div).cuda()

        # To prevent division by 0
        epsilon = 0.0000001

        return 1 - torch.mean(torch.div(numerator, denominator + epsilon)).cuda()


if __name__ == '__main__':
    torch.manual_seed(0)
    warped_image = torch.randint(255, (2, 10, 10, 10), dtype=torch.float64)
    target_image = torch.randint(255, (2, 10, 10, 10), dtype=torch.float64)

    criterion = NNCC()
    loss = criterion(target_image, target_image)

    print loss
