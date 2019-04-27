import gc
import sys
import time

sys.path.append('../library')
sys.path.append('../library/quicksilver')

# Custom libraries
import network
from network2 import Net
import util
import affine_transformation as at
from nncc_loss import NNCC
from hdf5_file_process import HDF5Image

# External libraries
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def create_net(features, continue_from_parameter=None):
    net = Net().cuda()

    if continue_from_parameter != None:
        print 'Loading existing weights!'
        weights = torch.load(continue_from_parameter)
        net.load_state_dict(weights['state_dict'])

    net.train()
    return net


def train_cur_data(epoch, data_index, moving, target, net, criterion, optimizer,
                   model_name, batch_size, stride=14):
    # Loading images
    # moving = torch.load(moving_dataset).float()
    # target = torch.load(target_dataset).float()

    data_size = moving.size()

    loss_storage = torch.tensor([])

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

        #input_moving = moving.cuda()
        # input_target = target.cuda()

        # Zeroing gradients
        optimizer.zero_grad()

        # Forward pass and averaging over all batches
        predicted_image = net(input_batch[:,0])
        # Affine transform
        #predicted_image = at.affine_transform(input_batch[:, 0], predicted_theta)
        # print predicted_theta
        '''
        plt.subplot(131)
        plt.imshow(input_batch[0, 0, int(input_batch.shape[2] / 2)].detach().cpu(), cmap='gray')
        plt.subplot(132)
        plt.imshow(input_batch[0, 1, int(input_batch.shape[2] / 2)].detach().cpu(), cmap='gray')
        plt.subplot(133)
        plt.imshow(predicted_image[0, 0, int(predicted_image.shape[2] / 2)].detach().cpu(), cmap='gray')
        plt.show()
        '''
        loss = criterion(predicted_image.squeeze(1), input_batch[:, 1])
        loss_value = loss.item()
        loss.backward()
        loss_storage = torch.cat((loss_storage, loss.detach().cpu().unsqueeze(0)))

        optimizer.step()
        print('====> Epoch: {}, datapart: {}, iter: {}/{}, loss: {}'.format(
            epoch + 1, data_index + 1, iters, N, loss_value))

        if iters % 100 == 0 or iters == N - 1:
            cur_state_dict = net.state_dict()

            model_info = {'patch_size': patch_size,
                          'network_feature': features,
                          'state_dict': cur_state_dict}

            print 'Saving model...'
            torch.save(model_info, model_name)
    return loss_storage

def train_network(files, features, n_epochs, learning_rate, batch_size, model_name):
    net = create_net(features)
    criterion = NNCC()
    optimizer = optim.Adam(net.parameters(), learning_rate)
    loss_storage = torch.tensor([])
    for data_index in range(len(files)):
        for epoch in range(n_epochs):
            image = HDF5Image(files[data_index])
            #image.gaussian_blur(0.5)
            #image.histogram_equalization()
            #image.display_image()

            moving_dataset = torch.from_numpy(image.data[:-1])
            target_dataset = torch.from_numpy(image.data[1:])

            loss = train_cur_data(epoch,
                           data_index,
                           moving_dataset,
                           target_dataset,
                           net,
                           criterion,
                           optimizer,
                           model_name,
                           batch_size)

            loss_storage = torch.cat((loss_storage, loss.mean(0, keepdim=True)))

            gc.collect()
    #criterion.plot_loss(n_epochs, '/home/anders/Ultrasound-Affine-Transformation/figures/' + time_string + '_patch_network_loss.eps')
    x = np.linspace(1, n_epochs + 1, len(loss_storage))

    fig = plt.figure()
    plt.plot(x, loss_storage.numpy())
    #for i in range(1, n_epochs + 1):
    #    plt.axvline(i, color='gray', linestyle='--')

    plt.title('Training loss. Learning rate: ' + str(learning_rate))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    fig.savefig('/home/anders/Ultrasound-Affine-Transformation/figures/'+time_string+'_patch_network_loss.eps', bbox_inches='tight')
    plt.show()

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
    batch_size = 128
    patch_size = 30
    output_name = ['/home/anders/Ultrasound-Affine-Transformation/output/']
    model_name = '/home/anders/Ultrasound-Affine-Transformation/weights/' + time_string + '_patch_network_model.pht.tar'
    n_epochs = 1000
    learning_rate = 0.0001
    # ===================================

    moving_dataset = ['/media/anders/TOSHIBA_EXT1/Training_set/moving_HA3C98PM.pth.tar',
                      '/media/anders/TOSHIBA_EXT1/Training_set/moving_HA3C98PQ.pth.tar',
                      '/media/anders/TOSHIBA_EXT1/Training_set/moving_HA3C98Q4.pth.tar']
    target_dataset = ['/media/anders/TOSHIBA_EXT1/Training_set/target_HA3C98PM.pth.tar',
                      '/media/anders/TOSHIBA_EXT1/Training_set/target_HA3C98PQ.pth.tar',
                      '/media/anders/TOSHIBA_EXT1/Training_set/target_HA3C98Q4.pth.tar']

    files = ['/media/anders/TOSHIBA_EXT1/ultrasound_examples/NewData/gr4_STolav1to4/p3122153/J1ECAT8E.h5']

    start = time.time()
    train_network(files, features, n_epochs, learning_rate, batch_size, model_name)
    stop = time.time()
    print 'Time elapsed =', stop - start

    # TODO: Check that moving_dataset and target_dataset has equal size?
