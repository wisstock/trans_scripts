#!/usr/bin/env python3

'''
Copyright © 2020 Borys Olifirov

Script for cell edges detection.

Provide inage preprocessing:
    - camera offset value compensation
    - offsets upper pixel intensiry limit (17000)
    - create threshold mask (avaliable methods - percentile, triangle)

'''

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage.external import tifffile
from skimage import data, img_as_float

from skimage import exposure
from skimage.filters import threshold_triangle
from skimage.filters import scharr



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

def cellEdge(img, thbreshold_method="triangle", percent=90, seed_method="one"):
    '''
	seed methods:
        one - calculate threshold for one image
	    first - build threshold mask for first frame of series
	    max - build threshold mask for max intensity frame
	    mean - build threshold mask for mean intensity of all serie frames

    treshold methods:
        triangle - threshold_triangle
        percent - extract pixels abowe fix percentile value

	'''

    if thbreshold_method == "triangle":
        thresh_out = threshold_triangle(img)
        positive_mask = img > thresh_out  # create negative threshold mask
        threshold_mask = positive_mask * -1  # inversion threshold mask

        output_img = np.copy(img)
        output_img[threshold_mask] = 0

        return(output_img)

    elif thbreshold_method == "percent":
        percentile = np.percentile(img, percent)
        output_img = np.copy(img)
        output_img[output_img < percentile] = 0

        return(output_img)

    else:
        print("Incorrect treshold method!")

def lineSlice(img, coordinates=[0,0,100,100]):
    x0, y0, x1, y1 = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
    line_length = int(np.hypot(x1-x0, y1-y0))  # calculate line length
    x, y = np.linspace(x0, x1, line_length), np.linspace(y0, y1, line_length)  # calculate projection to axis

    output_img = img[x.astype(np.int), y.astype(np.int)]
    return(output_img)



input_file = 'Fluorescence_435nmDD500_cell1.tiff'

img = getTiff(input_file, 1, 10)
img_eq = exposure.equalize_hist(img)


img_perc = cellEdge(img, "percent", 95)
img_eq_perc = exposure.equalize_hist(img_perc)

ends_coord = [200, 200, 380, 550]
x0, y0, x1, y1 = ends_coord[0], ends_coord[1], ends_coord[2], ends_coord[3]

raw_slice = lineSlice(img, ends_coord)
perc_slice = lineSlice(img_perc, ends_coord)



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