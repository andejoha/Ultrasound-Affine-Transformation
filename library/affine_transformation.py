import torch
import scipy
import numpy as np
import numpy.linalg
import scipy.ndimage as snd
import torch.nn.functional as F
import matplotlib.pyplot as plt

import time


def affine_grid_generator_3D(theta, size):
    N, C, D, H, W = size

    base_grid = theta.new(N, D, H, W, 4).cuda()

    w_points = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1])).cuda()
    h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1).cuda()
    d_points = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1).cuda()

    base_grid[:, :, :, :, 0] = w_points
    base_grid[:, :, :, :, 1] = h_points
    base_grid[:, :, :, :, 2] = d_points
    base_grid[:, :, :, :, 3] = 1
    grid = torch.bmm(base_grid.view(N, D * H * W, 4), theta.transpose(1, 2)).cuda()
    grid = grid.view(N, D, H, W, 3)

    return grid


def affine_grid_generator_2D(theta, size):
    N, C, H, W = size

    base_grid = theta.new(N, H, W, 3).cuda()

    linear_points = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]).cuda()
    base_grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(base_grid[:, :, :, 0])
    linear_points = torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1]).cuda()
    base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(base_grid[:, :, :, 1])
    base_grid[:, :, :, 2] = 1
    grid = torch.bmm(base_grid.view(N, H * W, 3), theta.transpose(1, 2)).cuda()
    grid = grid.view(N, H, W, 2)

    return grid


def affine_transform(moving_image, theta):
    N, D, H, W = moving_image.shape

    # Adding channel element
    moving_image = moving_image.unsqueeze(1)

    # Extending theta to include batches
    predicted_theta = torch.empty(N, theta.shape[0], theta.shape[1]).cuda()
    predicted_theta[:] = theta

    affine_grid = affine_grid_generator_3D(predicted_theta, (N, 1, D, H, W)).cuda()
    predicted_image = F.grid_sample(moving_image, affine_grid)
    return predicted_image


def trilinear_interpolate(im, x, y, z):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    z0 = np.floor(z).astype(int)
    z1 = z0 + 1

    x0 = np.clip(x0, 0, im.shape[0] - 1)
    x1 = np.clip(x1, 0, im.shape[0] - 1)
    y0 = np.clip(y0, 0, im.shape[1] - 1)
    y1 = np.clip(y1, 0, im.shape[1] - 1)
    z0 = np.clip(z0, 0, im.shape[2] - 1)
    z1 = np.clip(z1, 0, im.shape[2] - 1)

    c_000 = im[x0, y0, z0]
    c_100 = im[x1, y0, z0]
    c_010 = im[x0, y1, z0]
    c_001 = im[x0, y0, z1]
    c_110 = im[x1, y1, z0]
    c_101 = im[x1, y0, z1]
    c_011 = im[x0, y1, z1]
    c_111 = im[x1, y1, z1]

    w_000 = (x1 - x) * (y1 - y) * (z1 - z)
    w_100 = (x - x0) * (y1 - y) * (z1 - z)
    w_010 = (x1 - x) * (y - y0) * (z1 - z)
    w_001 = (x1 - x) * (y1 - y) * (z - z0)
    w_110 = (x - x0) * (y - y0) * (z1 - z)
    w_101 = (x - x0) * (y1 - y) * (z - z0)
    w_011 = (x1 - x) * (y - y0) * (z - z0)
    w_111 = (x - x0) * (y - y0) * (z - z0)

    return c_000 * w_000 + c_100 * w_100 + c_010 * w_010 + c_001 * w_001 + c_110 * w_110 + c_101 * w_101 + c_011 * w_011 + c_111 * w_111


if __name__ == '__main__':
    # start = time.time()

    np_image = scipy.misc.ascent()
    image = torch.empty(1, 1, np_image.shape[0], np_image.shape[1]).cuda()
    image[0, 0] = torch.tensor(np_image)

    # translation_vector = torch.ones(3).cuda() * -3
    # scaling_vector = torch.ones(3).cuda() * 0.5
    # theta = torch.ones(1).cuda() * np.pi / 12

    # transformed_image = translation(image, translation_vector)
    # transformed_image = scaling(image, scaling_vector)
    # transformed_image = rotation(image, theta)

    # plt.imshow(transformed_image, cmap='gray')

    # stop = time.time()
    # print 'Time elapsed =', stop - start
    # plt.show()

    # plt.imshow(transformed_image, cmap='gray')
    # plt.grid()
    # plt.show()

    theta_3D = torch.zeros(1, 3, 4).cuda()
    theta_3D[0, :, :-1] = torch.eye(3)*1.5
    input = torch.empty(1, 1, image.shape[2], image.shape[2], image.shape[3]).cuda()
    size_3D = input.shape

    for i in range(512):
        input[0, 0, i, :, :] = image[0, 0]

    affine_grid = affine_grid_generator_3D(theta_3D, size_3D)
    output = torch.nn.functional.grid_sample(input, affine_grid)
    print output

    plt.subplot(121)
    plt.imshow(input[0, 0, 256].cpu(), cmap='gray')
    plt.subplot(122)
    plt.imshow(output[0, 0, 256].cpu(), cmap='gray')
    plt.show()

    theta_2D = torch.zeros(1, 2, 3).cuda()
    theta_2D[0, :, :-1] = torch.eye(2)
    size_2D = (1, 1, image.shape[2], image.shape[3])

    affine_grid = affine_grid_generator_2D(theta_2D, size_2D)
    output = torch.nn.functional.grid_sample(image, affine_grid)

    # plt.imshow(output[0,0], cmap='gray')
    # plt.show()
