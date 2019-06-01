import gc
import time

# Custom libraries
from library.network import PatchNet
import library.quicksilver.util as util
from library.ncc_loss import NCC
from library.hdf5_file_process import HDF5Image
import library.affine_transformation as at
import library.ncc_loss as nccl

# External libraries
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


def create_net(continue_from_parameter=None):
    net = PatchNet().cuda()

    if continue_from_parameter != None:
        print('Loading existing weights!')
        weights = torch.load(continue_from_parameter)
        net.load_state_dict(weights['state_dict'])

    net.train()
    return net


def preform_validation(validation, target, batch_size, net, stride=29):
    if validation.shape[0] > target.shape[0]:
        diff = validation.shape[0] - target.shape[0]
        target = torch.cat((target.data, target.data[:diff]))
    data_size = validation.shape

    loss_storage = torch.tensor([]).float()

    # Initializing batch variables
    input_batch = torch.zeros(batch_size, 2, patch_size, patch_size, patch_size).cuda()

    # Creates a flat array in indexes corresponding to patches in the target and moving images
    flat_idx = util.calculatePatchIdx3D(data_size[0], patch_size * torch.ones(3), data_size[1:],
                                        stride * torch.ones(3))
    flat_idx_select = torch.zeros(flat_idx.size())

    # Remove "black" patches
    for patch_idx in range(1, flat_idx.size()[0]):
        # Converts from fattened idx array to position in 3D image.
        patch_pos = util.idx2pos_4D(flat_idx[patch_idx], data_size[1:])
        moving_patch = validation.data[patch_pos[0],
                       patch_pos[1]:patch_pos[1] + patch_size,
                       patch_pos[2]:patch_pos[2] + patch_size,
                       patch_pos[3]:patch_pos[3] + patch_size]
        target_patch = target.data[patch_pos[0],
                       patch_pos[1]:patch_pos[1] + patch_size,
                       patch_pos[2]:patch_pos[2] + patch_size,
                       patch_pos[3]:patch_pos[3] + patch_size]

        # Check if "Black" patch
        if torch.sum(moving_patch) + torch.sum(target_patch) != 0:
            flat_idx_select[patch_idx] = 1

    flat_idx_select = flat_idx_select.byte()
    flat_idx = torch.masked_select(flat_idx, flat_idx_select)

    N = int(flat_idx.size()[0] / batch_size)

    # Main training loop
    for iters in range(N):
        train_idx = torch.rand(batch_size).double() * flat_idx.size()[0]
        train_idx = torch.floor(train_idx).long()
        for slices in range(batch_size):
            patch_pos = util.idx2pos_4D(flat_idx[train_idx[slices]], data_size[1:])
            input_batch[slices, 0] = validation.data[patch_pos[0],
                                     patch_pos[1]:patch_pos[1] + patch_size,
                                     patch_pos[2]:patch_pos[2] + patch_size,
                                     patch_pos[3]:patch_pos[3] + patch_size].cuda()
            input_batch[slices, 1] = target.data[patch_pos[0],
                                     patch_pos[1]:patch_pos[1] + patch_size,
                                     patch_pos[2]:patch_pos[2] + patch_size,
                                     patch_pos[3]:patch_pos[3] + patch_size].cuda()

        # Forward pass and averaging over all batches
        predicted_theta = net(input_batch)

        # Affine transform
        predicted_patches = at.affine_transform(input_batch[:, 0], predicted_theta)
        idx = nccl.ncc(predicted_patches.squeeze(1), input_batch[:, 1]).argmax()
        predicted_image = at.affine_transform(validation.data[patch_pos[0]].unsqueeze(0).cuda(), predicted_theta[idx].unsqueeze(0))

        loss = nccl.ncc(predicted_image.squeeze(0), target.data[patch_pos[0]].unsqueeze(0))
        loss_value = loss.item()
        print('====> Validation loss: {}'.format(loss_value))

        loss_storage = torch.cat((loss_storage, loss.detach().cpu().unsqueeze(0)))
    return loss_storage.mean(0, keepdim=True)


def train_cur_data(epoch, data_index, moving, target, net, criterion, optimizer,
                   model_name, batch_size, patch_size, stride=29):
    # Making moving, target the same size
    if moving.shape[0] > target.shape[0]:
        diff = moving.shape[0] - target.shape[0]
        target = torch.cat((target.data, target.data[:diff]))

    loss_storage = torch.tensor([]).float()

    data_size = moving.shape

    # Initializing batch variables
    input_batch = torch.zeros(batch_size, 2, patch_size, patch_size, patch_size).cuda()

    # Creates a flat array in indexes corresponding to patches in the target and moving images
    flat_idx = util.calculatePatchIdx3D(data_size[0], patch_size * torch.ones(3), data_size[1:],
                                        stride * torch.ones(3))
    flat_idx_select = torch.zeros(flat_idx.size())

    # Remove "black" patches
    for patch_idx in range(1, flat_idx.size()[0]):
        # Converts from fattened idx array to position in 3D image.
        patch_pos = util.idx2pos_4D(flat_idx[patch_idx], data_size[1:])
        moving_patch = moving.data[patch_pos[0],
                       patch_pos[1]:patch_pos[1] + patch_size,
                       patch_pos[2]:patch_pos[2] + patch_size,
                       patch_pos[3]:patch_pos[3] + patch_size]
        target_patch = target.data[patch_pos[0],
                       patch_pos[1]:patch_pos[1] + patch_size,
                       patch_pos[2]:patch_pos[2] + patch_size,
                       patch_pos[3]:patch_pos[3] + patch_size]

        # Check if "Black" patch
        if torch.sum(moving_patch) + torch.sum(target_patch) != 0:
            flat_idx_select[patch_idx] = 1

    flat_idx_select = flat_idx_select.byte()
    flat_idx = torch.masked_select(flat_idx, flat_idx_select)

    N = int(flat_idx.size()[0] / batch_size)

    # Main training loop
    for iters in range(N):
        train_idx = torch.rand(batch_size).double() * flat_idx.size()[0]
        train_idx = torch.floor(train_idx).long()
        for slices in range(batch_size):
            patch_pos = util.idx2pos_4D(flat_idx[train_idx[slices]], data_size[1:])
            input_batch[slices, 0] = moving.data[patch_pos[0],
                                     patch_pos[1]:patch_pos[1] + patch_size,
                                     patch_pos[2]:patch_pos[2] + patch_size,
                                     patch_pos[3]:patch_pos[3] + patch_size].cuda()
            input_batch[slices, 1] = target.data[patch_pos[0],
                                     patch_pos[1]:patch_pos[1] + patch_size,
                                     patch_pos[2]:patch_pos[2] + patch_size,
                                     patch_pos[3]:patch_pos[3] + patch_size].cuda()

        # Zeroing gradients
        optimizer.zero_grad()

        # Forward pass
        predicted_theta = net(input_batch)

        # Affine transform
        predicted_patches = at.affine_transform(input_batch[:, 0], predicted_theta)
        idx = nccl.ncc(predicted_patches.squeeze(1), input_batch[:, 1]).argmax()
        predicted_image = at.affine_transform(moving.data[patch_pos[0]].unsqueeze(0).cuda(), predicted_theta[idx].unsqueeze(0))

        # plt.subplot(131)
        # plt.title('Moving')
        # plt.imshow(moving.data[patch_pos[0], int(data_size[1] / 2)].detach().cpu(), cmap='gray')
        # plt.subplot(132)
        # plt.title('Target')
        # plt.imshow(target.data[patch_pos[0], int(data_size[1] / 2)].detach().cpu(), cmap='gray')
        # plt.subplot(133)
        # plt.title('Transformed')
        # plt.imshow(predicted_image[0, 0, int(data_size[1] / 2)].detach().cpu(), cmap='gray')
        # plt.show()


        loss = criterion(predicted_image.squeeze(0), target.data[patch_pos[0]].unsqueeze(0))
        loss.backward()
        loss_value = loss.item()

        optimizer.step()
        print('====> Epoch: {}, datapart: {}, iter: {}/{}, loss: {}'.format(
            epoch + 1, data_index + 1, iters, N, loss_value))

        if iters % 100 == 0 or iters == N - 1:
            cur_state_dict = net.state_dict()

            model_info = {'patch_size': patch_size,
                          'state_dict': cur_state_dict}

            print('Saving model...')
            torch.save(model_info, model_name)
        loss_storage = torch.cat((loss_storage, loss.detach().cpu().unsqueeze(0)))
    return loss_storage


def train_network(moving_dataset, target_dataset, n_epochs, learning_rate, batch_size, patch_size, model_name):
    net = create_net()
    net.train()
    criterion = NCC().cuda()
    optimizer = optim.Adam(net.parameters(), learning_rate)

    training_loss_storage = torch.tensor([]).float()
    validation_loss_storage = torch.tensor([]).float()

    print('Loading images...')
    validation_dataset = moving_dataset.pop(0)
    validation_image = HDF5Image(validation_dataset)
    validation_image.histogram_equalization()
    validation_image.gaussian_blur(1.4)

    target_image = HDF5Image(target_dataset)
    target_image.histogram_equalization()
    target_image.gaussian_blur(1.4)

    for epoch in range(n_epochs):
        temp_training_loss = torch.tensor([]).float()
        for data_index in range(len(moving_dataset)):
            moving_image = HDF5Image(moving_dataset[data_index])
            moving_image.gaussian_blur(1.4),
            moving_image.histogram_equalization()

            training_loss = train_cur_data(epoch,
                                           data_index,
                                           moving_image.data,
                                           target_image.data,
                                           net.train(),
                                           criterion,
                                           optimizer,
                                           model_name,
                                           batch_size,
                                           patch_size)
            temp_training_loss = torch.cat((temp_training_loss, training_loss))
            gc.collect()
        training_loss_storage = torch.cat((training_loss_storage, temp_training_loss.mean(0, keepdim=True)))

        # Preform validation loss at end of epoch
        print('Preforming validation at end of epoch nr. {}...'.format(epoch + 1))
        with torch.no_grad():
            validation_loss = preform_validation(validation_image, target_image, batch_size, net.eval())
        validation_loss_storage = torch.cat((validation_loss_storage, validation_loss))

        training_x = np.linspace(0, epoch+1, len(training_loss_storage))
        validation_x = np.linspace(0, epoch+1, len(validation_loss_storage))

        fig = plt.figure()
        plt.subplot(211)
        plt.plot(training_x, training_loss_storage.numpy())
        plt.title('Training loss NCC \n Learning rate: ' + str(learning_rate))
        plt.ylabel('Loss')
        plt.grid()
        plt.subplot(212)
        plt.plot(validation_x, validation_loss_storage.numpy())
        plt.title('Validation loss NCC')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.show()
        fig.savefig('/home/anders/Ultrasound-Affine-Transformation/figures/' + time_string + '_NCC_patch_network_model.eps',
                    bbox_inches='tight')


if __name__ == '__main__':
    time_now = time.localtime()
    time_string = str(time_now[0]) + '.' + \
                  str(time_now[1]).zfill(2) + '.' + \
                  str(time_now[2]).zfill(2) + '_' + \
                  str(time_now[3]).zfill(2) + ':' + \
                  str(time_now[4]).zfill(2) + ':' + \
                  str(time_now[5]).zfill(2)

    # ===================================
    batch_size = 64
    patch_size = 30
    output_name = ['/home/anders/Ultrasound-Affine-Transformation/output/']
    model_name = '/home/anders/Ultrasound-Affine-Transformation/weights/' + time_string + '_patch_network_model.pht.tar'
    n_epochs = 150
    learning_rate = 0.00001
    # ===================================

    moving_dataset = ['/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70G.h5',
                      '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70I.h5',
                      '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70K.h5',
                      '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70M.h5']
    target_dataset = '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70E.h5'

    start = time.time()
    train_network(moving_dataset, target_dataset, n_epochs, learning_rate, batch_size, patch_size, model_name)
    stop = time.time()
    print('Time elapsed =', stop - start)

    # TODO: Check that moving_dataset and target_dataset has equal size?
