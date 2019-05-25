import gc
import time
import random

# Custom libraries
from library.network import FullNet
from library.ncc_loss import NCC
from library.hdf5_file_process import HDF5Image
import library.affine_transformation as at

# External libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def create_net(continue_from_parameter=None):
    net = FullNet().cuda()

    if continue_from_parameter != None:
        print('Loading existing weights...')
        weights = torch.load(continue_from_parameter)
        net.load_state_dict(weights['state_dict'])

    net.train()
    return net

def preform_validation(validation, target, net):
    if validation.shape[0] > target.shape[0]:
        diff = validation.shape[0] - target.shape[0]
        target = torch.cat((target.data, target.data[:diff]))
    data_size = validation.shape
    N = data_size[0]

    input_batch = torch.zeros(1, 2, data_size[1], data_size[2], data_size[3]).cuda()

    loss_batch = torch.tensor([]).float()
    for iters in range(N):
        input_batch[:, 0] = validation.data[iters]
        input_batch[:, 1] = target.data[iters]

        # Forward pass
        predicted_theta = net(input_batch)
        predicted_image = at.affine_transform(input_batch[:, 0], predicted_theta)

        validation_loss = F.binary_cross_entropy_with_logits(predicted_image.squeeze(1), input_batch[:, 1])
        validation_loss_value = validation_loss.item()
        print('====> Validation loss: {}, iter: {}/{}'.format(validation_loss_value, iters+1, N))
        loss_batch = torch.cat((loss_batch, validation_loss.detach().cpu().unsqueeze(0)))

    return loss_batch


def train_cur_data(epoch, data_index, moving, target, net, criterion, optimizer,
                   model_name):
    # Making moving, target the same size
    if moving.shape[0] > target.shape[0]:
        diff = moving.shape[0] - target.shape[0]
        target = torch.cat((target.data, target.data[:diff]))

    data_size = moving.shape
    N = data_size[0]

    # Initializing batch variables
    input_batch = torch.zeros(1, 2, data_size[1], data_size[2], data_size[3]).cuda()

    loss_batch = torch.tensor([]).float()

    # Main training loop
    for iters in range(N):
        input_batch[:, 0] = moving.data[iters]
        input_batch[:, 1] = target.data[iters]

        # Zeroing gradients
        optimizer.zero_grad()

        # Forward pass
        predicted_theta = net(input_batch)

        # Affine transform
        predicted_image = at.affine_transform(input_batch[:, 0], predicted_theta)

        '''
        plt.subplot(131)
        plt.title('Moving')
        plt.imshow(input_batch[0, 0, int(data_size[1] / 2)].detach().cpu(), cmap='gray')
        plt.subplot(132)
        plt.title('Target')
        plt.imshow(input_batch[0, 1, int(data_size[1] / 2)].detach().cpu(), cmap='gray')
        plt.subplot(133)
        plt.title('Transformed')
        plt.imshow(predicted_image[0, 0, int(data_size[1] / 2)].detach().cpu(), cmap='gray')
        plt.show()
        '''

        loss = criterion(predicted_image.squeeze(1), input_batch[:, 1])
        loss.backward()

        loss_value = loss.item()
        optimizer.step()
        print('====> Epoch: {}, datapart: {}, iter: {}/{}, loss: {}'.format(
            epoch + 1, data_index + 1, iters+1, N, loss_value))

        if iters % 100 == 0 or iters == N - 1:
            cur_state_dict = net.state_dict()

            model_info = {'patch_size': patch_size,
                          'network_feature': features,
                          'state_dict': cur_state_dict}

            print('Saving model...')
            torch.save(model_info, model_name)
        loss_batch = torch.cat((loss_batch, loss.detach().cpu().unsqueeze(0)))

    return loss_batch

def train_network(moving_dataset, target_dataset, n_epochs, learning_rate, batch_size, model_name):
    net = create_net()
    net.train()
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.Adam(net.parameters(), learning_rate)
    training_loss_storage = torch.tensor([]).float()
    validation_loss_storage = torch.tensor([]).float()
    for epoch in range(n_epochs):
        random_moving_dataset = moving_dataset.copy()
        random.shuffle(random_moving_dataset)
        validation_image = moving_dataset.pop(0)

        target = HDF5Image(target_dataset)
        for data_index in range(len(moving_dataset)):
            print('Loading images...')
            moving = HDF5Image(moving_dataset[data_index])

            moving.gaussian_blur(1),
            moving.histogram_equalization()

            target.gaussian_blur(1),
            target.histogram_equalization()

            training_loss = train_cur_data(epoch,
                                  data_index,
                                  moving,
                                  target,
                                  net.train(),
                                  criterion,
                                  optimizer,
                                  model_name)
            training_loss_storage = torch.cat((training_loss_storage, training_loss))
            gc.collect()

        # Preform validation loss at end of epoch
        validation = HDF5Image(validation_image)
        validation.gaussian_blur(1)
        validation.histogram_equalization()
        print('Preforming validation at end of epoch nr. {}...'.format(epoch+1))
        with torch.no_grad():
            validation_loss = preform_validation(validation, target, net)
        validation_loss_storage = torch.cat((validation_loss_storage, validation_loss))

    #criterion.plot_loss(n_epochs, '/home/anders/Ultrasound-Affine-Transformation/figures/test_img.png', learning_rate)
    training_x = np.linspace(0, n_epochs, len(training_loss_storage))
    validation_x = np.linspace(0, n_epochs, len(validation_loss_storage))

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
    fig.savefig('/home/anders/Ultrasound-Affine-Transformation/figures/' + time_string + '_BCE_network_model.eps', bbox_inches='tight')


if __name__ == '__main__':
    time_now = time.localtime()
    time_string = str(time_now[0]) + '.' + \
                  str(time_now[1]).zfill(2) + '.' + \
                  str(time_now[2]).zfill(2) + '_' + \
                  str(time_now[3]).zfill(2) + ':' + \
                  str(time_now[4]).zfill(2) + ':' + \
                  str(time_now[5]).zfill(2)

    # ===================================
    features = 32
    batch_size = 64
    patch_size = 30
    output_name = ['/home/anders/Ultrasound-Affine-Transformation/output/']
    model_name = '/home/anders/Ultrasound-Affine-Transformation/weights/' + time_string + '_network_model.pht.tar'
    n_epochs = 1
    learning_rate = 0.0001
    # ===================================

    moving_dataset = ['/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70E.h5',
                      '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70G.h5']
    target_dataset = '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70E.h5'

    start = time.time()
    train_network(moving_dataset, target_dataset, n_epochs, learning_rate, batch_size, model_name)
    stop = time.time()
    print('Time elapsed =', stop - start)

