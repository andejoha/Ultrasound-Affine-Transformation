import sys
import os
import time

sys.path.append('library')
sys.path.append('library/quicksilver')

# Custom libraries
import network
import util
import affine_transformation as at

# External libraries
import cv2
import torch
import PyCA.Core as ca
import PyCA.Common as common
import numpy as np
import matplotlib.pyplot as plt


def preprocess(image):
    # TODO: Add other preprocess techniques such as gaussian blur, historgram equalization

    image_copy = common.AsNPCopy(image)

    # Removing possible 'NaN' variables
    nan_mask = np.isnan(image_copy)
    image_copy[nan_mask] = 0

    # Normalizing [0, 1]
    image_copy /= np.amax(image_copy)

    return image_copy


def create_net(features):
    net = network.Net(features).cuda()

    # weights = torch.load(parameter)
    # net.load_state_dict(weights['state_dict'])

    net.train()
    return net


def predict_deformation_parameters(moving_image, target_image, input_batch, batch_size, net, step_size=14):
    moving = torch.from_numpy(moving_image).cuda()
    target = torch.from_numpy(target_image).cuda()

    data_size = moving.size()

    # Creates a flat array in indexes corresponding to patches in the target and moving images
    flat_idx = util.calculatePatchIdx3D(1, patch_size * torch.ones(3), data_size, step_size * torch.ones(3))
    flat_idx_select = torch.zeros(flat_idx.size())

    # Remove "black" patches
    for patch_idx in range(flat_idx.size()[0]):
        # Converts from fattened idx array to position in 3D image.
        patch_pos = util.idx2pos(flat_idx[patch_idx], data_size)
        moving_patch = moving[patch_pos[0]:patch_pos[0] + patch_size,
                       patch_pos[1]:patch_pos[1] + patch_size,
                       patch_pos[2]:patch_pos[2] + patch_size]
        target_patch = target[patch_pos[0]:patch_pos[0] + patch_size,
                       patch_pos[1]:patch_pos[1] + patch_size,
                       patch_pos[2]:patch_pos[2] + patch_size]

        # Check if "Black" patch
        if torch.sum(moving_patch) + torch.sum(target_patch) != 0:
            flat_idx_select[patch_idx] = 1

    flat_idx_select = flat_idx_select.byte()
    flat_idx = torch.masked_select(flat_idx, flat_idx_select)

    # Initialize predicted parameter variables
    predicted_parameters = torch.zeros(7).cuda()

    # Prediction loop
    batch_idx = 0
    while (batch_idx < flat_idx.size()[0]):
        if (batch_idx + batch_size < flat_idx.size()[0]):
            cur_batch_size = batch_size
        else:
            cur_batch_size = flat_idx.size()[0] - batch_idx

        for slices in range(cur_batch_size):
            patch_pos = util.idx2pos(flat_idx[batch_idx], data_size)
            input_batch[slices, 0] = moving[patch_pos[0]:patch_pos[0] + patch_size,
                                     patch_pos[1]:patch_pos[1] + patch_size,
                                     patch_pos[2]:patch_pos[2] + patch_size]
            input_batch[slices, 1] = target[patch_pos[0]:patch_pos[0] + patch_size,
                                     patch_pos[1]:patch_pos[1] + patch_size,
                                     patch_pos[2]:patch_pos[2] + patch_size]

        output_batch = net(input_batch)

        for slices in range(cur_batch_size):
            predicted_parameters += output_batch[slices]

        batch_idx += cur_batch_size
    predicted_parameters /= len(flat_idx)
    return predicted_parameters


def predict_image(moving_list, target_list, use_GPU=True):
    if use_GPU:
        mType = ca.MEM_DEVICE
    else:
        mType = ca.MEM_HOST

    # Load network
    net = create_net(features)

    # Initialize batches
    input_batch = torch.zeros(batch_size, 2, patch_size, patch_size, patch_size, requires_grad=True).cuda()

    # Prediction loop
    for i in range(len(moving_list)):
        common.Mkdir_p(os.path.dirname(output_name[i]))

        # Loading images
        moving_image = common.LoadITKImage(moving_list[i], mType)
        target_image = common.LoadITKImage(target_list[i], mType)

        # Image preprocessing
        moving_image = preprocess(moving_image)
        target_image = preprocess(target_image)

        # Setup ITK options

        predicted_parameters = predict_deformation_parameters(moving_image, target_image, input_batch, batch_size, net)

        translation_vector = predicted_parameters[:3].detach()
        scaling_vector = predicted_parameters[3:6].detach()
        rotation = predicted_parameters[6:].detach()

        # Preform affine transformation
        transformed_image = at.translation(moving_image, translation_vector)
        transformed_image = at.scaling(transformed_image, scaling_vector)
        transformed_image = at.rotation(transformed_image, rotation)

        return transformed_image


if __name__ == '__main__':
    # ===================================
    features = 32
    batch_size = 64
    patch_size = 15
    moving_set = ['/media/anders/TOSHIBA_EXT1/ultrasound_examples/_0121835/NIFTI/HA3C98PM/HA3C98PM_4.nii']
    target_set = ['/media/anders/TOSHIBA_EXT1/ultrasound_examples/_0121835/NIFTI/HA3C98PM/HA3C98PM_5.nii']
    output_name = ['/home/anders/affine_registration/output/']
    # ===================================

    config = torch.load('/home/anders/affine_registration/library/OASIS_predict.pth.tar')

    start = time.time()

    test_image = torch.load(moving_set[0])
    #transformed_image = predict_image(moving_set, target_set)
    stop = time.time()
    print 'Time elapsed =', stop - start
