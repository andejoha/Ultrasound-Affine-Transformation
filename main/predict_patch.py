import gc
import time

# Custom libraries
from library.network import FullNet, PatchNet
import library.affine_transformation as at
from library.hdf5_file_process import HDF5Image

# External libraries
import h5py
import torch
import torch.nn as nn


def create_net(weights):
    net = FullNet().cuda()

    print('Loading weights...')
    weights = torch.load(weights)
    net.load_state_dict(weights['state_dict'])

    net.train()
    return net


def save_image(img, original_img, execution_time, file_name, show_img=False, save_axis_img=False):
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
            print(name)
            if 'filename' in name:
                image_geometry.create_dataset(name, data=file_name, shape=(1,))
            elif 'frameNumber' in name:
                image_geometry.create_dataset(name, data=img.shape[0], shape=(1,))
            else:
                image_geometry.create_dataset(name, data=original_img.image_geometry[ele].value)
        image_geometry.create_dataset('executionTime', data=execution_time.numpy())


def predict(moving, target, net, criterion, output_name, save_image=True):
    # Making moving and target the same size
    if moving.shape[0] > target.shape[0]:
        diff = moving.shape[0] - target.shape[0]
        target = torch.cat((target.data, target.data[:diff]))

    data_size = moving.shape

    # Initializing batch variables
    input_batch = torch.zeros(1, 2, data_size[1], data_size[2], data_size[3]).cuda()
    output_batch = torch.zeros(data_size)

    N = data_size[0]
    time_storage = torch.tensor([]).float()

    # Main training loop
    print('Starting prediction...')
    for iters in range(N):
        input_batch[:, 0] = moving.data[iters]
        input_batch[:, 1] = target.data[iters]

        start = time.time() * 1000
        # Forward pass
        predicted_theta = net(input_batch)

        # Affine transform
        predicted_image = at.affine_transform(input_batch[:, 0], predicted_theta)
        stop = time.time() * 1000
        time_storage = torch.cat((time_storage, torch.tensor([stop - start])))

        output_batch[iters] = predicted_image[0, 0].detach().cpu()

        loss = criterion(predicted_image.squeeze(1), input_batch[:, 1])
        loss_value = loss.item()
        print('====> Image predicted! Loss: {}, execution time [ms]: {}'.format(
            loss_value, int(stop - start)))
    save_image(output_batch, moving, time_storage, output_name)
    return output_batch


def predict_image(moving_dataset, target_dataset, weights, output_name):
    net = create_net(weights)
    net.train()
    criterion = nn.BCEWithLogitsLoss()
    for data_index in range(len(moving_dataset)):
        print('Loading images...')
        moving = HDF5Image(moving_dataset[data_index])
        target = HDF5Image(target_dataset)

        moving.gaussian_blur(1),
        moving.histogram_equalization()

        target.gaussian_blur(1),
        target.histogram_equalization()

        predict(moving, target, net, criterion, output_name[data_index])
        gc.collect()


if __name__ == '__main__':
    # ===================================
    features = 32
    batch_size = 64
    patch_size = 15
    moving_dataset = ['/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70G.h5',
                      '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70I.h5',
                      '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70K.h5',
                      '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70M.h5']
    target_dataset = '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr5_STolav5to8/p7_3d/J249J70E.h5'
    output_name = ['/home/anders/Ultrasound-Affine-Transformation/output/J249J70G_predicted_images.h5',
                   '/home/anders/Ultrasound-Affine-Transformation/output/J249J70I_predicted_images.h5',
                   '/home/anders/Ultrasound-Affine-Transformation/output/J249J70K_predicted_images.h5',
                   '/home/anders/Ultrasound-Affine-Transformation/output/J249J70M_predicted_images.h5']

    weights = '/home/anders/Ultrasound-Affine-Transformation/weights/2019.05.24_18:20:42_network_model.pht.tar'
    # ===================================

    predict_image(moving_dataset, target_dataset, weights, output_name)