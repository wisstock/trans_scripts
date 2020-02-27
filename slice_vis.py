#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Confocal image processing.

"""

import sys
import os
import glob
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage import data
from skimage import exposure
from skimage import filters
from skimage.filters import scharr
from skimage.external import tifffile
from skimage import restoration


sys.path.append('modules')
import oiffile as oif
import slicing as slc
import threshold as ts
# import deconvolute as dec


plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'


FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    filemode="w",
                    format=FORMAT)  # , filename="demo.log")



data_path = os.path.join(sys.path[0], 'confocal_data/HPCA-YFP/cell1_tif/20180718-1252-0001.tif')  # os.path.join(sys.path[0], '.temp/data/cell1.tif')
sample_path_1 = os.path.join(sys.path[0], 'confocal_data/HPCA-YFP/cell1_tif/20180718-1254-0002.tif')
sample_path_2 = os.path.join(sys.path[0], 'confocal_data/HPCA-YFP/cell1_tif/20180718-1255-0003.tif')

path_list = [data_path, sample_path_2]
img_list = []  # list of read TIF files as np arrays
frame_list = []  # list of separete frames
slice_list = []  # list of slices over extracted frames

frames = [5, 9, 16]  # indexes of frames

angle = 80
band_w = 2


for input_file in path_list:
    img_list.append(tifffile.imread(input_file))


for img in img_list:
    for frame_num in frames:
        frame_list.append(img[frame_num,:,:])


frame_shape = np.shape(frame_list[0])
img_cntr = [(frame_shape[1]-1)//2,
            (frame_shape[0]-1)//2]
# cntr = ts.cellMass(img)


xy0, xy1 = slc.lineSlice(frame_list[0], angle, img_cntr)

for frame in frame_list:
    slice_list.append(slc.bandExtract(frame, xy0, xy1, band_w))


i = 0
for i in range(len(frame_list)//2):

    ax1 = plt.subplot(3, 3, i+1)
    ax1.plot([xy0[0], xy1[0]], [xy0[1], xy1[1]])
    slc1 = ax1.imshow(frame_list[i])
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes('right', size='3%', pad=0.1)
    plt.colorbar(slc1, cax=cax1)
    ax1.set_title('Frame %s, i = 30' % frames[i])

    ax2 = plt.subplot(3, 3, i+4)
    ax2.plot([xy0[0], xy1[0]], [xy0[1], xy1[1]])
    slc2 = ax2.imshow(frame_list[i+3])
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes('right', size='3%', pad=0.1)
    plt.colorbar(slc2, cax=cax2)
    ax2.set_title('Frame %s, i = 1000' % frames[i])

    ax3 = plt.subplot(3, 3, i+7)
    ax3.plot(slice_list[i], label='i = 30')
    ax3.plot(slice_list[i+3], label='i = 1000', linestyle='dashed')
    ax3.legend(loc='upper left')

    i += 1 

plt.show()
