#!/usr/bin/env python3

'''
Copyright © 2020 Borys Olifirov

Confocal image processing.

Provide inage preprocessing:
    - camera offset value compensation
    - offsets upper pixel intensiry limit (17000)
    - create threshold mask (avaliable methods - percentile, triangle)

'''

import sys
import os
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

sys.path.append('module')
from oiffile import OifFile
import slicing as slc
import threshold as ts



def getTiff(file_name, channel=0, frame_number=0, camera_offset=250):
    wd_path = os.path.split(os.getcwd())
    os.chdir(wd_path[0] + '/temp/data/')
    tiff_tensor = tifffile.imread(file_name)
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



input_file = 'Fluorescence_435nmDD500_cell1.tiff'
angle = 45


img = getTiff(input_file, 1, 10)
img_eq = exposure.equalize_hist(img)


img_perc = cellEdge(img, "percent", 95)
img_eq_perc = exposure.equalize_hist(img_perc)


raw_slice = slc.lineSlice(img, ends_coord)
perc_slice = slc.lineSlice(img_perc, ends_coord)



rot =  transforms.Affine2D().rotate_deg(90) # rotating to 90 degree

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1,
                                         ncols=4,
                                         figsize=(8, 8))

ax0.imshow(img)  #, cmap='gray')
ax0.plot([x0, x1], [y0, y1], 'ro-')
# ax0.axis("off")

ax1.plot(raw_slice)
# ax1.imshow(img_eq)
# ax1.axis("off")

ax2.imshow(img_perc)
ax2.plot([x0, x1], [y0, y1], 'ro-')
# ax2.axis("off")

ax3.plot(perc_slice)
# ax3.axis("off")

plt.show()