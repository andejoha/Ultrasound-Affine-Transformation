import gc
import time

# Custom libraries
from library.network import PatchNet
from library.ncc_loss import NCC
import library.affine_transformation as at
from library.hdf5_image import HDF5Image
import library.quicksilver.util as util
import library.ncc_loss as nccl

# External libraries
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def create_net(weights):
    net = PatchNet().cuda()

    print('Loading weights...')
    weights = torch.load(weights)
    net.load_state_dict(weights['state_dict'])

    net.train()
    return net


def save_image(img, original_img, execution_time, loss, file_name):
    print('Saving predicted images to: {}...'.format(file_name))
    with h5py.File(file_name, 'w') as f:
        data = f.create_group('CartesianVolumes')
        for i in range(len(img)):
            name = 'vol' + str(i + 1).zfill(2)
            data.create_dataset(name, data=img[i].numpy(), shape=img[i].shape)

        # Inherent image geometry from original file.
        image_geometry = f.create_group('ImageGeometry')
        for ele in original_img.image_geometry:
            name = original_img.image_geometry[ele].name
            if 'filename' in name:
                image_geometry.create_dataset(name, data=file_name, shape=(1,))
            elif 'frameNumber' in name:
                image_geometry.create_dataset(name, data=img.shape[0], shape=(1,))
            else:
                image_geometry.create_dataset(name, data=original_img.image_geometry[ele].value)
        image_geometry.create_dataset('executionTime', data=execution_time.numpy())
        image_geometry.create_dataset('loss', data=loss.numpy())


def predict(moving, target, net, criterion, patch_size, output_name, stride=29):
    # Making moving and target the same size
    if moving.shape[0] > target.shape[0]:
        diff = moving.shape[0] - target.shape[0]
        target = torch.cat((target.data, target.data[:diff]))

    data_size = moving.shape
    N = data_size[0]

    output_batch = torch.zeros(data_size)
    loss_storage = torch.tensor([]).float()
    time_storage = torch.tensor([]).float()

    for i in range(N):
        start = time.time()*1000
        # Creates a flat array in indexes corresponding to patches in the target and moving images
        flat_idx = util.calculatePatchIdx3D(1, patch_size * torch.ones(3), data_size[1:],
                                            stride * torch.ones(3))
        flat_idx_select = torch.zeros(flat_idx.size())

        # Remove "black" patches
        for patch_idx in range(1, flat_idx.size()[0]):
            # Converts from fattened idx array to position in 3D image.
            patch_pos = util.idx2pos_4D(flat_idx[patch_idx], data_size[1:])
            moving_patch = moving.data[i,
                           patch_pos[1]:patch_pos[1] + patch_size,
                           patch_pos[2]:patch_pos[2] + patch_size,
                           patch_pos[3]:patch_pos[3] + patch_size]
            target_patch = target.data[i,
                           patch_pos[1]:patch_pos[1] + patch_size,
                           patch_pos[2]:patch_pos[2] + patch_size,
                           patch_pos[3]:patch_pos[3] + patch_size]

            # Check if "Black" patch
            if torch.sum(moving_patch) + torch.sum(target_patch) != 0:
                flat_idx_select[patch_idx] = 1

        flat_idx_select = flat_idx_select.byte()
        flat_idx = torch.masked_select(flat_idx, flat_idx_select)

        input_batch = torch.zeros(flat_idx.shape[0], 2, patch_size, patch_size, patch_size).cuda()

        for slices in range(flat_idx.shape[0]):
            patch_pos = util.idx2pos_4D(flat_idx[slices], data_size[1:])
            input_batch[slices, 0] = moving.data[i,
                                     patch_pos[1]:patch_pos[1] + patch_size,
                                     patch_pos[2]:patch_pos[2] + patch_size,
                                     patch_pos[3]:patch_pos[3] + patch_size].cuda()
            input_batch[slices, 1] = target.data[i,
                                     patch_pos[1]:patch_pos[1] + patch_size,
                                     patch_pos[2]:patch_pos[2] + patch_size,
                                     patch_pos[3]:patch_pos[3] + patch_size].cuda()

        # Forward pass
        predicted_theta = net(input_batch).mean(0, keepdim=True)

        # Affine transform
        predicted_image = at.affine_transform(moving.data[i].unsqueeze(0).cuda(), predicted_theta)
        stop = time.time() * 1000

        output_batch[i] = predicted_image[0, 0].detach().cpu()

        loss = criterion(predicted_image.squeeze(0), target.data[i].unsqueeze(0).cuda())
        loss_value = loss.item()
        print('====> Image predicted! Loss: {:.4f}, execution time [ms]: {}'.format(
            loss_value, int(stop - start)))

        time_storage = torch.cat((time_storage, torch.tensor([stop - start])))
        loss_storage = torch.cat((loss_storage, loss.detach().cpu().unsqueeze(0)))
    save_image(output_batch, moving, time_storage, loss_storage, output_name)
    return output_batch


def predict_image(moving_dataset, target_dataset, weights, patch_size, output_name):
    net = create_net(weights)
    net.train()
    criterion = NCC()

    target = HDF5Image(target_dataset)
    target.histogram_equalization()
    target.gaussian_blur(1.4)

    for data_index in range(len(moving_dataset)):
        start = time.time() * 1000
        print('Loading images...')
        moving = HDF5Image(moving_dataset[data_index])
        moving.histogram_equalization()
        moving.gaussian_blur(1.4),

        with torch.no_grad():
            predict(moving, target, net, criterion, patch_size, output_name[data_index])
        gc.collect()

if __name__ == '__main__':
    # ===================================
    patch_size = 30
    moving_dataset = ['/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70G.h5']
    target_dataset = '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70E.h5'
    output_name = ['/home/anders/Ultrasound-Affine-Transformation/output/J249J70G_patch_predicted_images.h5']

    weights = '/home/anders/Ultrasound-Affine-Transformation/weights/patch_network_weights.pht.tar'
    # ===================================

    predict_image(moving_dataset, target_dataset, weights, patch_size, output_name)
