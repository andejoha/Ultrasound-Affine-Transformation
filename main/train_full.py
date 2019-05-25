import gc
import sys
import time


# Custom libraries
from library.network import FullNet
import library.quicksilver.util as util
from library.ncc_loss import NCC
from library.hdf5_file_process import HDF5Image
import library.affine_transformation as at

# External libraries
import torch
import torch.nn as nn
import torch.optim as optim
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


def train_cur_data(epoch, data_index, moving, target, net, criterion, optimizer,
                   model_name, batch_size):
    # Making moving and target the same size
    if moving.shape[0] > target.shape[0]:
        diff = moving.shape[0] - target.shape[0]
        target = torch.cat((target.data, target.data[:diff]))

    data_size = moving.shape

    # Initializing batch variables
    input_batch = torch.zeros(1, 2, data_size[1], data_size[2], data_size[3]).cuda()

    N = data_size[0]

    loss_batch = torch.tensor([]).float()

    # Main training loop
    for iters in range(N):
        input_batch[:, 0] = moving.data[iters]
        input_batch[:, 1] = target.data[iters]

        # Zeroing gradients
        optimizer.zero_grad()

        # Forward pass and averaging over all batches
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
            epoch + 1, data_index + 1, iters, N, loss_value))

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
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), learning_rate)
    loss_storage = torch.tensor([]).float()
    for epoch in range(n_epochs):
        for data_index in range(len(moving_dataset)):
            moving = HDF5Image(moving_dataset[data_index])
            target = HDF5Image(target_dataset)

            moving.gaussian_blur(1),
            moving.histogram_equalization()

            target.gaussian_blur(1),
            target.histogram_equalization()

            loss = train_cur_data(epoch,
                                  data_index,
                                  moving,
                                  target,
                                  net,
                                  criterion,
                                  optimizer,
                                  model_name,
                                  batch_size)
            loss_storage = torch.cat((loss_storage, loss))
            gc.collect()
    #criterion.plot_loss(n_epochs, '/home/anders/Ultrasound-Affine-Transformation/figures/test_img.png', learning_rate)
    x = np.linspace(1, n_epochs + 1, len(loss_storage))

    fig = plt.figure()
    plt.plot(x, loss_storage.numpy())

    plt.title('Training loss BCEWithLogitsLoss \n Learning rate: ' + str(learning_rate))
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
    n_epochs = 200
    learning_rate = 0.0001
    # ===================================

    moving_dataset = ['/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70G.h5',
                      '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70I.h5',
                      '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70K.h5',
                      '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70M.h5']
    target_dataset = '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70E.h5'

    start = time.time()
    train_network(moving_dataset, target_dataset, n_epochs, learning_rate, batch_size, model_name)
    stop = time.time()
    print('Time elapsed =', stop - start)

