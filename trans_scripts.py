#!/usr/bin/env python3

"""
Copyright © 2020 Borys Olifirov

Confocal image processing.

"""

import sys
import os
import glob
import logging

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np

from skimage.external import tifffile
from skimage import data, img_as_float
from skimage import exposure
from skimage.filters import threshold_triangle
from skimage.filters import scharr

sys.path.append('modules')
import oiffile as oif
import slicing as slc
import threshold as ts


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


logging.basicConfig(filename="sample.log",  # logging options
                    level=logging.DEBUG,
                    filemode="w")


input_file = 'Fluorescence_435nmDD500_cell1.tiff'
angle = 200
# img = getTiff(input_file, 0, 1)

oif_path = '/home/astria/Bio_data/HEK_mYFP/20180523_HEK_membYFP/cell1/20180523-1404-0003-250um.oif'
oif_raw = oif.OibImread(oif_path)
oif_img = oif_raw[0,:,:,:]
img = oif_img[5,:,:]

print(np.shape(oif_img))



# with OifFile(oif_path) as oif:
# ...     filename = natural_sorted(oib.glob('*.tif'))[0]
# ...     image = oib.asarray(filename)


start, end = slc.lineSlice(img, angle)
line_slice = slc.lineExtract(img, start, end)

# img_eq = exposure.equalize_hist(img)


# img_perc = cellEdge(img, "percent", 95)
# img_eq_perc = exposure.equalize_hist(img_perc)


# raw_slice = slc.lineSlice(img, ends_coord)
# perc_slice = slc.lineSlice(img_perc, ends_coord)

shape = np.shape(img)
cntr = [np.int((shape[1]-1)/2),
        np.int((shape[0]-1)/2)]



# rot =  transforms.Affine2D().rotate_deg(90) # rotating to 90 degree

fig, (ax0, ax1) = plt.subplots(nrows=2,
                              ncols=1,
                              figsize=(8, 8))

ax0.imshow(img)  #, cmap='gray')
ax0.plot([start[0], end[0]], [start[1], end[1]], 'ro-')
ax0.scatter(cntr[0],cntr[1],color='r')
# ax0.scatter(start[0]+5, start[1]+5)

ax1.plot(line_slice)

# plt.gca().invert_yaxis()
plt.show()