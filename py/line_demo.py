#!/usr/bin/env python3

'''
Copyright © 2020 Borys Olifirov

Script for extract pixel values by line

scipy.ndimage.measurements.center_of_mass

'''

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage.external import tifffile
from skimage import data, img_as_float


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

def lineSlice(img, coordinates=[0,0,100,100]):
    x0, y0, x1, y1 = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
    line_length = int(np.hypot(x1-x0, y1-y0))  # calculate line length
    x, y = np.linspace(x0, x1, line_length), np.linspace(y0, y1, line_length)  # calculate projection to axis

    output_img = img[x.astype(np.int), y.astype(np.int)]
    return(output_img)


# Generate some data...
# x, y = np.mgrid[-5:5:0.1, -5:5:0.1]
# z = np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2)


input_file = 'Fluorescence_435nmDD500_cell1.tiff'
ends_coord = [0, 500, 650, 500]

img = getTiff(input_file, 1, 10)
# data_shape = img.shape()
# print(data_shape[1], data_shape[2])

line_slice = lineSlice(img, ends_coord)
x0, y0, x1, y1 = ends_coord[0], ends_coord[1], ends_coord[2], ends_coord[3]


fig, axes = plt.subplots(nrows=2)

axes[0].imshow(img)
axes[0].plot([x0, x1], [y0, y1], 'ro-')
axes[0].axis('image')

axes[1].plot(line_slice)

plt.show()