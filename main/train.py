import sys
import time

sys.path.append('../library')
sys.path.append('../library/quicksilver')

# Custom libraries
import network
import util
import affine_transformation as at
from loss import NNCC

# External libraries
import torch
import torch.nn as nn
import torch.optim as optim


def create_net(features, continue_from_parameter=None):
    net = network.Net(features).cuda()

    if continue_from_parameter != None:
        print 'Loading existing weights!'
        weights = torch.load(continue_from_parameter)
        net.load_state_dict(weights['state_dict'])

    net.train()
    return net


def train_cur_data(epoch, data_index, moving_dataset, target_dataset, net, criterion, optimizer,
                   model_name, batch_size, stride=14):
    # Loading images
    moving = torch.load(moving_dataset).float()
    target = torch.load(target_dataset).float()

    data_size = moving.size()

    # Initializing batch variables
    input_batch = torch.zeros(batch_size, 2, patch_size, patch_size, patch_size, requires_grad=True).cuda()
    output_batch = torch.zeros(data_size[0], data_size[1], data_size[2], data_size[3], requires_grad=True).cuda()

    # Creates a flat array in indexes corresponding to patches in the target and moving images
    flat_idx = util.calculatePatchIdx3D(data_size[0], patch_size * torch.ones(3), data_size[1:],
                                        stride * torch.ones(3))
    flat_idx_select = torch.zeros(flat_idx.size())

    # Remove "black" patches
    for patch_idx in range(1, flat_idx.size()[0]):
        # Converts from fattened idx array to position in 3D image.
        patch_pos = util.idx2pos_4D(flat_idx[patch_idx], data_size[1:])
        moving_patch = moving[patch_pos[0],
                       patch_pos[1]:patch_pos[1] + patch_size,
                       patch_pos[2]:patch_pos[2] + patch_size,
                       patch_pos[3]:patch_pos[3] + patch_size]
        target_patch = target[patch_pos[0],
                       patch_pos[1]:patch_pos[1] + patch_size,
                       patch_pos[2]:patch_pos[2] + patch_size,
                       patch_pos[3]:patch_pos[3] + patch_size]

        # Check if "Black" patch
        if torch.sum(moving_patch) + torch.sum(target_patch) != 0:
            flat_idx_select[patch_idx] = 1

    flat_idx_select = flat_idx_select.byte()
    flat_idx = torch.masked_select(flat_idx, flat_idx_select)

    N = flat_idx.size()[0] / batch_size

    # Main training loop
    for iters in range(N):
        train_idx = torch.rand(batch_size).double() * flat_idx.size()[0]
        train_idx = torch.floor(train_idx).long()
        for slices in range(batch_size):
            patch_pos = util.idx2pos_4D(flat_idx[train_idx[slices]], data_size[1:])
            input_batch[slices, 0] = moving[patch_pos[0],
                                     patch_pos[1]:patch_pos[1] + patch_size,
                                     patch_pos[2]:patch_pos[2] + patch_size,
                                     patch_pos[3]:patch_pos[3] + patch_size].cuda()
            input_batch[slices, 1] = target[patch_pos[0],
                                     patch_pos[1]:patch_pos[1] + patch_size,
                                     patch_pos[2]:patch_pos[2] + patch_size,
                                     patch_pos[3]:patch_pos[3] + patch_size].cuda()

        # Zeroing gradients
        optimizer.zero_grad()

        # Forward pass and averaging over all batches
        predicted_parameters = net(input_batch).mean(0)
        translation_vector = predicted_parameters[:3].detach()
        scaling_vector = predicted_parameters[3:6].detach()
        rotation = predicted_parameters[6:].detach()

        for i in range(len(output_batch)):
            transformed_image = at.translation(moving[i], translation_vector)
            transformed_image = at.scaling(transformed_image, scaling_vector)
            transformed_image = at.rotation(transformed_image, rotation)
            output_batch[i] = torch.from_numpy(transformed_image)

        loss = criterion(output_batch, target.cuda())
        loss.backward(retain_graph=True)
        loss_value = loss.item()
        optimizer.step()
        print('====> Epoch: {}, datapart: {}, iter: {}/{}, loss: {}'.format(
            epoch + 1, data_index + 1, iters, N, loss_value / batch_size))

        if iters % 100 == 0 or iters == N - 1:
            cur_state_dict = net.state_dict()

            model_info = {'patch_size': patch_size,
                          'network_feature': features,
                          'state_dict': cur_state_dict}

            torch.save(model_info, model_name)


def train_network(moving_dataset, target_dataset, features, n_epochs, learning_rate, batch_size, model_name):
    net = create_net(features)
    net.train()
    criterion = NNCC().cuda()
    optimizer = optim.Adam(net.parameters(), learning_rate)

    for epoch in range(n_epochs):
        for data_index in range(len(moving_dataset)):
            train_cur_data(epoch,
                           data_index,
                           moving_dataset[data_index],
                           target_dataset[data_index],
                           net,
                           criterion,
                           optimizer,
                           model_name,
                           batch_size)


if __name__ == '__main__':
    # ===================================
    features = 32
    batch_size = 64
    patch_size = 15
    output_name = ['/home/anders/affine_registration/output/']
    model_name = '/home/anders/affine_registration/main/network_model.pht.tar'
    n_epochs = 1
    learning_rate = 0.0001
    # ===================================

    moving_dataset = ['/media/anders/TOSHIBA_EXT1/Training_set/moving_HA3C98PM.pth.tar']
    target_dataset = ['/media/anders/TOSHIBA_EXT1/Training_set/target_HA3C98PM.pth.tar']

    start = time.time()
    train_network(moving_dataset, target_dataset, features, n_epochs, learning_rate, batch_size, model_name)
    stop = time.time()
    print 'Time elapsed =', stop - start

    # TODO: Check that moving_dataset and target_dataset has equal size
