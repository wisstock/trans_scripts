#!/usr/bin/env python3

'''
Copyright © 2020 Borys Olifirov

https://scipy-lectures.org/advanced/image_processing/
https://www.google.com/search?q=numpy%20array%20indexing


'''

import sys
import os
import glob


import numpy as np
import matplotlib.pyplot as plt

from skimage import img_as_float
from skimage import io
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from skimage.morphology import reconstruction
from skimage.external import tifffile

import oiffile  # custom lib for Olympus Image Files parsing




wd_path = os.path.split(os.getcwd())
os.chdir(wd_path[0] + '/temp/data/')  # go to DATA dir

# raw_series = []
# for current_image in glob.glob("*.tiff"):  # uploading input files as dict
	# b = current_image.split('.')[0]
	# a.append(b)

offset = 250  # camera offset value


gray_raw = tifffile.imread('Fluorescence_435nmDD500_cell1.tiff')

print(gray_raw.shape, np.min(gray_raw[0]), np.max(gray_raw[0]))
# gray_raw = rgb2gray(rgb_raw)  # Y = 0.2125 R + 0.7154 G + 0.0721 B
image = gray_raw[0] - offset
print(image.max(), image.min())

seed = np.copy(image)

seed[1:-1, 1:-1] = image.min()
mask = image
dilated = reconstruction(seed, mask, method = "dilation")

# print(mask.min(), mask.max())
print(seed.min(), seed.max())
print(dilated.min(), dilated.max())



# plt.imshow(image-dilated, cmap='gray')

yslice = 520

fig, (ax0, ax1, ax3) = plt.subplots(nrows=1,
                                    ncols=3,
                                    figsize=(8, 8))

ax0.imshow(dilated)  #, cmap='gray')
ax0.axhline(yslice, color='r', alpha=0.4)
ax0.axis("off")

ax1.imshow(image - dilated)  #, cmap='gray')
ax1.axhline(yslice, color='r', alpha=0.4)
ax1.axis("off")

ax3.plot(mask[yslice], '0.5', label='mask')
ax3.plot(seed[yslice], 'k', label='seed')
ax3.plot(dilated[yslice], 'r', label='dilated')
ax3.set_xticks([])
ax3.legend()

plt.show()





