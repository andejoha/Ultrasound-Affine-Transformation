import torch
import scipy
import numpy as np
import numpy.linalg
import scipy.ndimage as snd
import matplotlib.pyplot as plt

import time


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
    start = time.time()

    image = scipy.misc.ascent()

    translation_vector = torch.ones(3).cuda() * -3
    scaling_vector = torch.ones(3).cuda() * 0.5
    theta = torch.ones(1).cuda() * np.pi / 12

    # transformed_image = translation(image, translation_vector)
    transformed_image = scaling(image, scaling_vector)
    # transformed_image = rotation(image, theta)

    plt.imshow(transformed_image, cmap='gray')

    stop = time.time()
    print 'Time elapsed =', stop - start
    plt.show()

    # plt.imshow(transformed_image, cmap='gray')
    # plt.grid()
    # plt.show()
