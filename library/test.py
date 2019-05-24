import gc
import sys
import time
import os

# Custom libraries
from library.network2 import FullNet
import library.quicksilver.util as util
from library.ncc_loss import NCC
from library.hdf5_file_process import HDF5Image
import library.affine_transformation as at

# External libraries
import torch
import torch.optim as optim
import matplotlib.pyplot as plt



root = '/media/anders/TOSHIBA_EXT/ultrasound_examples/NewData/'
for path, dir, files in os.walk(root):
    for f in files:
        name = path + '/' + f
        image = HDF5Image(name)
        print(name, '==>', image.shape)