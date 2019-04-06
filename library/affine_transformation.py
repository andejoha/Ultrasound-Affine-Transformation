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


def affine_transform(moving_image, theta, data_size):
    N, D, H, W = data_size

    # Adding channel element
    moving_image = moving_image.unsqueeze(1)

    # Extending theta to include batches
    predicted_theta = torch.empty(N, theta.shape[0], theta.shape[1]).cuda()
    predicted_theta[:] = theta

    affine_grid = affine_grid_generator_3D(predicted_theta, (N, 1, D, H, W)).cuda()
    predicted_image = F.grid_sample(moving_image, affine_grid)
    return predicted_image


def translation(image, translation_vector):
    translation_vector = np.array(translation_vector.cpu())

    M = np.eye(4)
    M[:-1, -1:] = translation_vector.reshape((3, 1))

    transformed_image = snd.affine_transform(image, M)

    return transformed_image


def scaling(image, scaling_vector):
    s_x = scaling_vector[0].cpu().numpy()
    s_y = scaling_vector[1].cpu().numpy()
    s_z = scaling_vector[2].cpu().numpy()

    M = np.eye(3)
    M[0, 0] = s_x
    M[1, 1] = s_y
    M[2, 2] = s_z

    center = np.array(image.shape) * 0.5
    offset = center - center.dot(M)

    transformed_image = snd.affine_transform(image, M, offset=offset)

    return transformed_image


def old_scaling(image, scaling_vector):
    # TODO: This can be optimized by combining the two loops into one.

    s_x = scaling_vector[0].cpu().numpy()
    s_y = scaling_vector[1].cpu().numpy()
    s_z = scaling_vector[2].cpu().numpy()

    transformed_image = np.zeros((int(image.shape[0] * s_x), int(image.shape[1] * s_y), int(image.shape[2] * s_z)))
    transformed_image[:] = np.nan

    for i in range(image.shape[0] - 1):
        for j in range(image.shape[1] - 1):
            for k in range(image.shape[2] - 1):
                transformed_image[int(i * s_x), int(j * s_y), int(k * s_z)] = image[i, j, k]

    n_pixels_image = image.shape[0] * image.shape[1] * image.shape[2]
    n_pixels_transformed_image = transformed_image.shape[0] * transformed_image.shape[1] * transformed_image.shape[2]

    x = np.zeros(abs(n_pixels_transformed_image - n_pixels_image))
    y = np.zeros(abs(n_pixels_transformed_image - n_pixels_image))
    z = np.zeros(abs(n_pixels_transformed_image - n_pixels_image))

    n = 0
    for i in range(transformed_image.shape[0]):
        for j in range(transformed_image.shape[1]):
            for k in range(transformed_image.shape[2]):
                if np.isnan(transformed_image[i, j, k]):
                    x[n] = i
                    y[n] = j
                    z[n] = k
                    n += 1

    trilinear = trilinear_interpolate(image, x / s_x, y / s_y, z / s_z)

    for i in range(len(x)):
        transformed_image[int(x[i]), int(y[i]), int(z[i])] = trilinear[i]

    return transformed_image


def rotation(image, theta):
    theta = float(theta)

    R_x = np.array([np.array([1, 0, 0, 0]),
                    np.array([0, np.cos(theta), -np.sin(theta), 0]),
                    np.array([0, np.sin(theta), np.cos(theta), 0]),
                    np.array([0, 0, 0, 1])])

    R_y = np.array([np.array([np.cos(theta), 0, np.sin(theta), 0]),
                    np.array([0, 1, 0, 0]),
                    np.array([-np.sin(theta), 0, np.sin(theta), 0]),
                    np.array([0, 0, 0, 1])])

    R_z = np.array([np.array([np.cos(theta), -np.sin(theta), 0, 0]),
                    np.array([np.sin(theta), np.cos(theta), 0, 0]),
                    np.array([0, 0, 1, 0]),
                    np.array([0, 0, 0, 1])])

    R = np.array([np.array([np.cos(theta), -np.sin(theta), 0]),
                  np.array([np.sin(theta), np.cos(theta), 0]),
                  np.array([0, 0, 1])])

    transformed_image = snd.affine_transform(image, R_x)
    transformed_image = snd.affine_transform(transformed_image, R_y)
    transformed_image = snd.affine_transform(transformed_image, R_z)

    return transformed_image


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
