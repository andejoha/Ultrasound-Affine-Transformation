import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class NNCC(nn.Module):
    def __init__(self):
        super(NNCC, self).__init__()
        self.loss_storage = torch.tensor([])

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
        loss = 1 - torch.mean(torch.div(numerator, denominator + epsilon)).cuda()
        self.loss_storage = torch.cat((self.loss_storage, loss.detach().cpu().unsqueeze(0)))

        return loss

    def plot_loss(self, n_epochs, output_name):
        x = np.linspace(1, n_epochs + 1, len(self.loss_storage))

        fig = plt.figure()
        plt.plot(x, self.loss_storage.numpy())
        for i in range(1, n_epochs+1):
            plt.axvline(i, color='gray', linestyle='--')

        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        fig.savefig(output_name, bbox_inches='tight')


if __name__ == '__main__':
    '''
    torch.manual_seed(0)
    warped_image = torch.randint(255, (2, 10, 10, 10), dtype=torch.float64)
    target_image = torch.randint(255, (2, 10, 10, 10), dtype=torch.float64)

    criterion = NNCC()
    loss = criterion(target_image, target_image)
    
    print loss
    '''
    a = torch.randn(100)
    criterion = NNCC()
    criterion.loss_storage = a
    criterion.plot_loss(5, '/home/anders/Ultrasound-Affine-Transformation/figures/patch_network_loss.png')