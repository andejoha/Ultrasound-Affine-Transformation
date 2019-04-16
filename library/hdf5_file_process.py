import SimpleITK as sitk
import numpy as np
import h5py
import cv2
import os
import nibabel as nib
from scipy.ndimage.filters import gaussian_filter
import torch


class HDF5Image:
    def __init__(self, filename):
        self.filename = filename
        try:
            img = h5py.File(filename, 'r')
        except:
            print('An error occurred while trying to read ' + filename)
        img = h5py.File(filename, 'r')
        cartesian_volumes = img[img.keys()[0]]
        self.image_geometry = img[img.keys()[1]]
        vol = np.empty(len(cartesian_volumes), dtype=object)
        i = 0
        for ele in cartesian_volumes:
            vol[i] = cartesian_volumes[ele]
            i += 1
        self.data = self.get_vol_data(vol)
        self.shape = self.data.shape

    def get_vol_data(self, vol):
        if vol[0] != None:
            data = []
            i = 0
            for ele in vol:
                data.append(ele.value)
                i += 1
            return np.array(data)
        else:
            return [[[[None]]]]

    def display_image(self, frame_number=0, axis=0, multi_image=False):
        # When "multi_image=True" you can have multiple windows open at the same time, but remember to add
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # as this is not be included here.

        if axis == 0:
            cv2.imshow('HDF5Image (axis=0):' + self.filename,
                       np.squeeze(self.data[frame_number, int(round(self.shape[1] / 2)), :, :]))
            if not multi_image:
                print('Press any key to continue...')
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        elif axis == 1:
            cv2.imshow('HDF5Image (axis=1):' + self.filename,
                       np.squeeze(self.data[frame_number, :, int(round(self.shape[2] / 2)), :]))
            if not multi_image:
                print('Press any key to continue...')
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        elif axis == 2:
            cv2.imshow('HDF5Image (axis=2):' + self.filename,
                       np.squeeze(self.data[frame_number, :, :, int(round(self.shape[3] / 2))]))
            if not multi_image:
                print('Press any key to continue...')
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print('The "axis" argument has to be "0" for x-axis, "1" for y-axis or "2" for z-axis')

    # Applies a Gaussian blur filter to the data
    def gaussian_blur(self, sigma):
        self.data = gaussian_filter(self.data, sigma)

    # Preforms a histogram_equalization to the data.
    def histogram_equalization(self):
        for i in range(self.data.shape[0]):
            hist, bins = np.histogram(self.data[i].flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf = np.ma.filled(cdf_m, 0).astype('uint8')
            self.data[i] = cdf[self.data[i]]

    def convert_to_nii(self, data, output_filename):
        # OBS! ITK is using the axis orientation (x,y,z) while Numpy is using (z,y,x).
        # Swapping x-axis and z-axis to get ITK orientation.
        data = np.swapaxes(data, 0, 2)

        image = sitk.GetImageFromArray(data, isVector=False)

        # Updating header
        # origin = self.image_geometry['origin'].value
        # voxelsize = self.image_geometry['voxelsize'].value*1000
        # image.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2])))
        # image.SetSpacing((float(voxelsize[0]), float(voxelsize[1]), float(voxelsize[2])))

        print 'Creating file:', output_filename
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_filename)
        writer.Execute(image)


class NIFTIImage:
    def __init__(self, filename):
        self.filename = filename
        img = nib.load(filename)
        self.data = np.array(img.get_fdata(), np.uint8)
        self.shape = self.data.shape

    def display_image(self, axis=0, multi_image=False):
        # When "multi_image = True" you can have multiple windows open at the same time, but remember to add
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # as this is not be included here.

        if axis == 0:
            image_slice = (np.squeeze(self.data[int(round(self.shape[0] / 2)), :, :]))
            cv2.imshow('NIFTIImage (axis=0): ' + self.filename, image_slice)
            cv2.imwrite('Figures/' + self.filename.split('/')[-1][:-4] + '.png', image_slice)
            if not multi_image:
                print('Press any key to continue...')
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        elif axis == 1:
            image_slice = np.squeeze(self.data[:, int(round(self.shape[1] / 2)), :])
            cv2.imshow('NIFTIImage (axis=1): ' + self.filename, image_slice)
            cv2.imwrite('Figures/' + self.filename.split('/')[-1][:-4] + '.png', image_slice)
            if not multi_image:
                print('Press any key to continue...')
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        elif axis == 2:
            image_slice = np.squeeze(self.data[:, :, int(round(self.shape[2] / 2))])
            cv2.imshow('NIFTIImage (axis=2): ' + self.filename, image_slice)
            cv2.imwrite('Figures/' + self.filename.split('/')[-1][:-4] + '.png', image_slice)
            if not multi_image:
                print 'Press any key to continue...'
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print('The "axis" argument has to be "0" for x-axis, "1" for y-axis or "2" for z-axis')


def convert_all_to_nii(ultrasound_data_dir):
    # TODO: You have to run this function twice for some reason. Fix bug
    for root, dirs, files in os.walk(ultrasound_data_dir):
        if len(dirs) == 0 and 'NIFTI' not in root.split('/'):
            os.mkdir(root + '/NIFTI')
        if 'NIFTI' in dirs:
            for f in files:
                h5_image = HDF5Image(root + '/' + f)

                # Preposessing options
                h5_image.histogram_equalization()
                h5_image.gaussian_blur(1.5)

                if h5_image.data[0][0][0][0] != None:
                    if not os.path.isdir(root + '/NIFTI/' + f[:-3]):
                        os.mkdir(root + '/NIFTI/' + f[:-3])
                    for i in range(len(h5_image.data)):
                        h5_image.convert_to_nii(h5_image.data[i], root + '/NIFTI/' + f[:-3] + '/' + f[:-3] + '_' + str(i) + '.nii')


if __name__ == '__main__':
    file1 = '/media/anders/TOSHIBA_EXT1/ultrasound_examples/NewData/gr4_STolav1to4/p3122153/J1ECAT8E.h5'
    # file2 = '/media/anders/TOSHIBA_EXT1/ultrasound_examples/_0121835/NIFTI/HA3C98PM/HA3C98PM_5.nii'
    # file3 = '/home/anders/devel/test/HA3C98PM_4.nii'
    # file4 = '/home/anders/devel/test/HA3C98PM_5.nii'

    image1 = HDF5Image(file1)
    # image2 = NIFTIImage(file2)
    # image3 = NIFTIImage(file3)
    # image4 = NIFTIImage(file4)

    image1.display_image()
    # image3.display_image(multi_image=True)
    # image4.display_image()

    # FINISHED:
    # vol, image_geometry = read_h5(file)
    # data = get_vol_data(vol)
    # if len(data) != 0:
    #    display_image_from_array(data[0])
    # convert_to_nii(data[0], image_geometry, 'test.nii')
    # read_nii('ultrasound_examples/_0121835/NIFTI/HA3C98PM_0.nii')
    # convert_all_to_nii('/media/anders/TOSHIBA_EXT1/ultrasound_examples')

    # TODO: Do something with image geometry
    # h5_test_image = HDF5Image(file)
    # h5_test_image.display_image()

    # TODO: Needs work
    # convert_all_to_nii('ultrasound_examples')
    # TESTING:
