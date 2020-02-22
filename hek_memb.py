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
import deconvolute as dec


def getTiff(file_name, channel=0, frame_number=0, camera_offset=250):
    """ Function returns individual frame from image series and
    apply compensation of camera offset value for this frame.
    Separate two fluorecent channels (first start from 0, second - from 1).
    For Dual View system data.

    """

    path = os.getcwd() + '/temp/data/' + file_name

    tiff_tensor = tifffile.imread(path)
    # print(tiff_tensor.shape, np.max(tiff_tensor))

    channel_one = tiff_tensor[0::2, :, :]
    channel_two = tiff_tensor[1::2, :, :]
    # print(channel_one.shape)
    # print(channel_two.shape)

    if channel == 0:
    	frame_amount = channel_one.shape
    	try:
    		img = channel_one[frame_number] - camera_offset
    		return(img)
    	except ValueError:
    		if frame_number > frame_amount[0]:
    			print("Frame number out of range!")
    else:
    	frame_amount = channel_two.shape
    	try:
    		img = channel_two[frame_number] - camera_offset
    		return(img)
    	except ValueError:
    		if frame_number > frame_amount[0]:
    			print("Frame number out of range!")

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(filename="logs/sample.log",
                    level=logging.INFO,
                    filemode="w",
                    format=FORMAT)


# input_file = '/home/astria/Bio_data/s_C001Z007.tif'  # 'Fluorescence_435nmDD500_cell1.tiff'

oif_path = '/home/astria/Bio_data/HEK_mYFP/20180523_HEK_membYFP/cell1/20180523-1404-0003-250um.oif'
oif_raw = oif.OibImread(oif_path)
oif_img = oif_raw[0,:,:,:]

# img = getTiff(input_file, 0, 1)
img = oif_img[6,:,:]
# img = tifffile.imread(input_file)


# img = filters.gaussian(img, sigma=1)
img_mod = ts.cellEdge(img)

# IMAGEJ DATA VIS
# input_img = '/home/astria/Bio_data/Deconvoluted/s_C001Z009.tif'
# input_img_mod = '/home/astria/Bio_data/Deconvoluted/Deconvolved_23.tif'
# img = tifffile.imread(input_img)
# img_mod = tifffile.imread(input_img_mod)


angle = 10
band_w = 2
cntr = ts.cellMass(img)
xy0, xy1 = slc.lineSlice(img, angle, cntr)

# raw_slice = slc.lineExtract(img, xy0, xy1)  # for compare slice func
# mod_slice = slc.bandExtract(img, xy0, xy1, band_w)


raw_slice = slc.bandExtract(img, xy0, xy1, band_w)
mod_slice = slc.bandExtract(img_mod, xy0, xy1, band_w)

# mod_slice = slc.lineExtract(img_mod, xy0, xy1)

shape = np.shape(img)
cntr_img = [np.int((shape[1]-1)/2),
        np.int((shape[0]-1)/2)]



ax0 = plt.subplot(321)
ax0.imshow(img)  #, cmap='gray')
ax0.set_title('Raw image')
ax0.plot([xy0[0], xy1[0]], [xy0[1], xy1[1]], 'ro-')
ax0.scatter(cntr[0],cntr[1],color='r')
ax0.scatter(cntr_img[0],cntr_img[1])
# ax0.scatter(start[0]+5, start[1]+5)

ax1 = plt.subplot(322)
ax1.imshow(img_mod)  #, cmap='gray')
ax1.set_title('Deconvolute image')
ax1.plot([xy0[0], xy1[0]], [xy0[1], xy1[1]], 'ro-')
ax1.scatter(cntr[0],cntr[1],color='r')
ax1.scatter(cntr_img[0],cntr_img[1])
# ax0.scatter(start[0]+5, start[1]+5)

ax2 = plt.subplot(312)
ax2.set_title('Rav slice')
ax2.plot(raw_slice)

ax3 = plt.subplot(313)
ax3.set_title('Deconvolute slice')
ax3.plot(mod_slice)

# plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()