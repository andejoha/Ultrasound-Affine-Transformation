import math
import torch
import numpy as np
import torch.nn.functional as F
from .hdf5_file_process import HDF5Image
import cv2


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
    '''
    output shape: (N, C, H, W, D)
    '''
    N, D, H, W = moving_image.shape

    # Adding channel dimension
    moving_image = moving_image.unsqueeze(1)

    # Extending theta to include batches
    predicted_theta = torch.empty(N, theta.shape[1], theta.shape[2]).cuda()
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


def get_pixel_value(img, x,y,z):
    shape = img.shape
    N = shape[0]
    D = shape[2]
    H = shape[3]
    W = shape[4]

    batch_idx = torch.arange(N)
    batch_idx = batch_idx.view(N, 1, 1, 1)
    b = batch_idx.repeat(1,D,H,W)

    indices = torch.stack([b,z,y,x], dim=4)
    out = torch.index_select(4, )


def trilinear_sampler(img, x, y, z):
    img = img.cpu()
    x = x.cpu()
    y = y.cpu()
    z = z.cpu()


    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    z0 = torch.floor(z).long()
    z1 = z0 + 1

    x0 = torch.clamp(x0, min=0, max=img.shape[2] - 1)
    x1 = torch.clamp(x1, min=0, max=img.shape[2] - 1)
    y0 = torch.clamp(y0, min=0, max=img.shape[3] - 1)
    y1 = torch.clamp(y1, min=0, max=img.shape[3] - 1)
    z0 = torch.clamp(z0, min=0, max=img.shape[4] - 1)
    z1 = torch.clamp(z1, min=0, max=img.shape[4] - 1)


    test = get_pixel_value(img, x0, y0, z0)


    c_000 = img[:, :, x0, y0, z0]
    c_100 = img[:, :, x1, y0, z0]
    c_010 = img[:, :, x0, y1, z0]
    c_001 = img[:, :, x0, y0, z1]
    c_110 = img[:, :, x1, y1, z0]
    c_101 = img[:, :, x1, y0, z1]
    c_011 = img[:, :, x0, y1, z1]
    c_111 = img[:, :, x1, y1, z1]


    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()
    z0 = z0.float()
    z1 = z1.float()

    w_000 = (x1 - x) * (y1 - y) * (z1 - z)
    w_100 = (x - x0) * (y1 - y) * (z1 - z)
    w_010 = (x1 - x) * (y - y0) * (z1 - z)
    w_001 = (x1 - x) * (y1 - y) * (z - z0)
    w_110 = (x - x0) * (y - y0) * (z1 - z)
    w_101 = (x - x0) * (y1 - y) * (z - z0)
    w_011 = (x1 - x) * (y - y0) * (z - z0)
    w_111 = (x - x0) * (y - y0) * (z - z0)

    out = c_000 * w_000 + c_100 * w_100 + c_010 * w_010 + c_001 * w_001 + c_110 * w_110 + c_101 * w_101 + c_011 * w_011 + c_111 * w_111
    return out


if __name__ == '__main__':
    # start = time.time()
    image = HDF5Image('/media/anders/TOSHIBA_EXT1/ultrasound_examples/NewData/gr4_STolav1to4/p3122153/J1ECAT8E.h5')
    image = torch.from_numpy(image.data).cuda().float()
    image = image.unsqueeze(1)

    angle = math.pi/2
    theta = torch.tensor([[[1, 0, 0, -0.5],
                           [0, 1, 0, -0.5],
                           [0, 0, 1, 0]]])

    warped_image = affine_transform(image, theta)

    warped_image = np.array(warped_image.cpu(), dtype=np.uint8)

    cv2.imshow('x', warped_image[2,0, int(warped_image.shape[2]/2), :, :])
    cv2.imwrite('/home/anders/Ultrasound-Affine-Transformation/figures/affine_examples/trans_x.png', warped_image[2,0, int(warped_image.shape[2]/2), :, :])
    #cv2.imshow('y', warped_image[2,0, :, int(warped_image.shape[3]/2), :])
    #cv2.imwrite('/home/anders/Ultrasound-Affine-Transformation/figures/affine_examples/trans_y.png', warped_image[2,0, :, int(warped_image.shape[3]/2), :])
    #cv2.imshow('z', warped_image[2,0, :, :, int(warped_image.shape[4]/2)])
    #cv2.imwrite('/home/anders/Ultrasound-Affine-Transformation/figures/affine_examples/trans_z.png', warped_image[2,0, :, :, int(warped_image.shape[4]/2)])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
