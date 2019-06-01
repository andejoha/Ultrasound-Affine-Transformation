import time
import torch
import numpy as np
import h5py
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt


class HDF5Image:
    def __init__(self, filename):
        self.filename = filename
        img = h5py.File(filename, 'r')
        self.cartesian_volumes = img[list(img.keys())[0]]
        self.image_geometry = img[list(img.keys())[1]]

        self.shape = list(self.cartesian_volumes.values())[0].shape
        self.shape = (len(list(self.cartesian_volumes.values())),
                      self.shape[0],
                      self.shape[1],
                      self.shape[2])

        self.data = self.get_vol_data()

    def get_vol_data(self):
        data = torch.empty(self.shape)
        i = 0
        for ele in self.cartesian_volumes:
            data[i] = torch.from_numpy(self.cartesian_volumes[ele].value)
            i += 1
        return data

    def display_image(self, frame_number=0, axis=0):

        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        if axis == 0:
            middle_slice = self.data[frame_number, int(self.shape[1] / 2)]
            ax.imshow(middle_slice, cmap='gray')
        elif axis == 1:
            middle_slice = self.data[frame_number, :, int(self.shape[2] / 2)]
            ax.imshow(middle_slice, cmap='gray')
        elif axis == 1:
            middle_slice = self.data[frame_number, :, :, int(self.shape[3] / 2)]
            ax.imshow(middle_slice, cmap='gray')
        fig.show()

    def gaussian_blur(self, sigma):
        # Applies a Gaussian blur filter to the data
        for i in range(self.shape[0]):
            self.data[i] = torch.from_numpy(gaussian_filter(self.data[i], sigma)).float()


    def histogram_equalization(self):
        # Preforms a histogram_equalization to the data.
        data = self.data.numpy().astype('uint8')
        for i in range(data.shape[0]):
            hist, bins = np.histogram(data[i].flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf = np.ma.filled(cdf_m, 0).astype('uint8')
            data[i] = cdf[data[i]]
        self.data = torch.from_numpy(data).float()

    def to(self, device):
        self.data = self.data.to(device)

    def cuda(self):
        self.data = self.data.cuda()

    def cpu(self):
        self.data = self.data.cpu()

    def requires_grad_(self, bool=True):
        self.data = self.data.requires_grad_(bool)


if __name__ == '__main__':
    file1 = '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr4_STolav1to4/p3122153/J1ECAT8E.h5'
    file2 = '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/gr4_STolav1to4/p3122153/J1ECAT8I.h5'

    image1 = HDF5Image(file1)
    image2 = HDF5Image(file2)