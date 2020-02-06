#!/usr/bin/env python3

'''
Copyright © 2020 Borys Olifirov

'''

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage import data, img_as_float
from skimage import exposure
from skimage.external import tifffile
from skimage.filters import threshold_triangle
from skimage.filters import scharr



def getTiff(file_name, channel=0, frame_number=0, camera_offset=250):
    wd_path = os.path.split(os.getcwd())
    os.chdir(wd_path[0] + '/temp/data/')
    tiff_tensor = tifffile.imread(file_name)
    print(tiff_tensor.shape, np.max(tiff_tensor))

    channel_one = tiff_tensor[0::2, :, :]
    channel_two = tiff_tensor[1::2, :, :]
    print(channel_one.shape)
    print(channel_two.shape)

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

def cellThresh(img):
	cell_mask = threshold_triangle(img)
	out = img > cell_mask

	return(out)

def cellPercent(img, percent=80):
	percentile = np.percentile(img, percent)
	out = np.copy(img)
	out[out < percentile] = 0

	return(out)

def cellEdge(img, method="first"):
	'''
	method:
	    first - build threshold mask for first frame of series
	    max - build threshold mask for max intensity frame
	    mean - build threshold mask for mean intensity of all serie frames

	'''


input_file = 'Fluorescence_435nmDD500_cell1.tiff'

img = getTiff(input_file, 1, 5)
# img_eq = exposure.equalize_hist(img)
img_thresh = cellThresh(img)
img_perc = cellPercent(img, 98)
img_eq = exposure.equalize_hist(img_perc)


invert_mask = np.copy(img_thresh)  # create bollean mask for threshold result
invert_mask[:,:] = -1
thresh_mask = (img_thresh * invert_mask)^2

extract_thresh = np.copy(img)
extract_thresh[thresh_mask] = 0




fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1,
                                         ncols=4,
                                         figsize=(8, 8))

ax0.imshow(img)  #, cmap='gray')
ax0.axis("off")

ax1.imshow(img_perc)  #, cmap='gray')
ax1.axis("off")

ax2.imshow(thresh_mask)
ax2.axis("off")

ax3.imshow(extract_thresh)
ax3.axis("off")

plt.show()