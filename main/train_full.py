import gc
import sys
import time


# Custom libraries
from library.network2 import FullNet
import library.quicksilver.util as util
from library.ncc_loss import NCC
from library.hdf5_file_process import HDF5Image
import library.affine_transformation as at

# External libraries
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


def create_net(features, continue_from_parameter=None):
    net = FullNet().cuda()

    if continue_from_parameter != None:
        print('Loading existing weights!')
        weights = torch.load(continue_from_parameter)
        net.load_state_dict(weights['state_dict'])

    net.train()
    return net


def train_cur_data(epoch, data_index, moving, target, net, criterion, optimizer,
                   model_name, batch_size):

    data_size = moving.size()

    # Initializing batch variables
    input_batch = torch.zeros(1, 2, data_size[1], data_size[2], data_size[3]).cuda()

    N = data_size[0]

    # Main training loop
    for iters in range(N):

        input_batch[:, 0] = moving[iters]
        input_batch[:, 1] = target[iters]

        print(input_batch.grad_fn)

        # Zeroing gradients
        optimizer.zero_grad()

        # Forward pass and averaging over all batches
        predicted_theta = net(input_batch[:,0])

        # Affine transform
        predicted_image = at.affine_transform(input_batch[:, 0], predicted_theta)

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


def train_network(files, features, n_epochs, learning_rate, batch_size, model_name):
    net = create_net(features)
    net.train()
    criterion = NCC()
    optimizer = optim.Adam(net.parameters(), learning_rate)
    for epoch in range(n_epochs):
        for data_index in range(len(files)):
            image = HDF5Image(files[data_index])

            moving_dataset = image.data[:-1]
            target_dataset = image.data[1:]

            train_cur_data(epoch,
                           data_index,
                           moving_dataset,
                           target_dataset,
                           net,
                           criterion,
                           optimizer,
                           model_name,
                           batch_size)
            gc.collect()


if __name__ == '__main__':
    # ===================================
    features = 32
    batch_size = 64
    patch_size = 30
    output_name = ['/home/anders/Ultrasound-Affine-Transformation/output/']
    model_name = '/home/anders/Ultrasound-Affine-Transformation/weights/network_model.pht.tar'
    n_epochs = 2
    learning_rate = 0.0001
    # ===================================

    moving_dataset = ['/media/anders/TOSHIBA_EXT1/Training_set/moving_HA3C98PM.pth.tar',
                      '/media/anders/TOSHIBA_EXT1/Training_set/moving_HA3C98PQ.pth.tar',
                      '/media/anders/TOSHIBA_EXT1/Training_set/moving_HA3C98Q4.pth.tar']
    target_dataset = ['/media/anders/TOSHIBA_EXT1/Training_set/target_HA3C98PM.pth.tar',
                      '/media/anders/TOSHIBA_EXT1/Training_set/target_HA3C98PQ.pth.tar',
                      '/media/anders/TOSHIBA_EXT1/Training_set/target_HA3C98Q4.pth.tar']

    files = ['/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr4_STolav1to4/p3122153/J1ECAT8E.h5',
             '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr4_STolav1to4/p3122153/J1ECAT8G.h5',
             '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr4_STolav1to4/p3122153/J1ECAT8I.h5',
             '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr4_STolav1to4/p3122153/J1ECAT9E.h5',
             '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr4_STolav1to4/p3122153/J1ECAT9G.h5',
             '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr4_STolav1to4/p3122153/J1ECAT9I.h5']


    start = time.time()
    train_network(files, features, n_epochs, learning_rate, batch_size, model_name)
    stop = time.time()
    print('Time elapsed =', stop - start)

    # TODO: Check that moving_dataset and target_dataset has equal size?
