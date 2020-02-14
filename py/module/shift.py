#!/usr/bin/env python3

'''
Copyright © 2020 Borys Olifirov

'''


import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage.external import tifffile

from skimage.filters import scharr


def oneTiff(file_name, channel=0, frame_number=0, camera_offset=250):
    wd_path = os.path.split(os.getcwd())
    start_path = os.getcwd()
    os.chdir(wd_path[0] + '/temp/data/')
    tiff_tensor = tifffile.imread(file_name)
    print(tiff_tensor.shape, np.max(tiff_tensor))

    channel_one = tiff_tensor[0::2, :, :]
    channel_two = tiff_tensor[1::2, :, :]

    if channel == 0:
    	frame_amount = channel_one.shape

    	try:
    		img = channel_one[frame_number] - camera_offset
    		os.chdir(start_path)
    		return(img)
    	except ValueError:
    		if frame_number > frame_amount[0]:
    			print("Frame number out of range!")
    else:
    	frame_amount = channel_two.shape

    	try:
    		img = channel_two[frame_number] - camera_offset
    		os.chdir(start_path)
    		return(img)
    	except ValueError:
    		if frame_number > frame_amount[0]:
    			print("Frame number out of range!")


input_file = 'Fluorescence_435nmDD500_cell1.tiff'

img_one = oneTiff(input_file, 0, 0)
img_two = oneTiff(input_file, 1, 0)

scharr_one = scharr(img_one)
scharr_two = scharr(img_two)

print(np.array_equal(scharr_one, scharr_two))

edge_diff = scharr_one - scharr_two


fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=1,
                                              ncols=5,
                                              figsize=(8, 8))

ax0.imshow(img_one)  #, cmap='gray')
ax0.axis("off")

ax1.imshow(img_two)  #, cmap='gray')
ax1.axis("off")

ax2.imshow(scharr_one)  #, cmap='gray')
ax2.axis("off")

ax3.imshow(scharr_two)
ax3.axis("off")

ax4.imshow(edge_diff)
ax4.axis("off")


plt.show()