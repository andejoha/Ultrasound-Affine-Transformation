import torch
import torch.nn as nn
import affine_transformation as at


class NNCC(nn.Module):
    def __init__(self):
        super(NNCC, self).__init__()

    def forward(self, predicted_parameters, target_image, moving_image, data_size):
        predicted_image = torch.zeros(data_size[0], data_size[1], data_size[2], data_size[3]).cuda()

        translation_vector = predicted_parameters[:3].detach()
        scaling_vector = predicted_parameters[3:6].detach()
        rotation = predicted_parameters[6:].detach()

        print translation_vector
        print scaling_vector
        print rotation

        for i in range(len(moving_image)):
            transformed_image = at.translation(moving_image[i], translation_vector)
            transformed_image = at.scaling(transformed_image, scaling_vector)
            transformed_image = at.rotation(transformed_image, rotation)
            predicted_image[i] = torch.from_numpy(transformed_image)


        N = moving_image.shape[0] * moving_image.shape[1]

        n_dims = tuple([i for i in range(len(moving_image.size()))])
        print n_dims

        moving = (predicted_image - torch.mean(predicted_image)).cuda()
        target = (target_image - torch.mean(target_image)).cuda()

        #print moving[0:5, 0:5]
        #print target[0:5, 0:5]

        moving_std_div = torch.sqrt(torch.sum(torch.pow(moving, 2))).cuda()
        target_std_div = torch.sqrt(torch.sum(torch.pow(target, 2))).cuda()

        #print moving_std_div
        #print target_std_div

        numerator = torch.sum(torch.mul(moving, target)).cuda()
        denominator = torch.mul(moving_std_div, target_std_div).cuda()

        #print numerator
        #print denominator

        return (torch.div(numerator, denominator).cuda())


if __name__ == '__main__':
    torch.manual_seed(0)
    moving_image = torch.randn((4, 4))
    target_image = torch.randn((4, 4))

    criterion = NNCC()
    loss = criterion(moving_image, target_image)

    print loss
