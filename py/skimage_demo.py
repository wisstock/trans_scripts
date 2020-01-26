#!/usr/bin/env python3

'''
Copyright © 2020 Borys Olifirov


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


rgb_raw = tifffile.imread('composite.tiff')

gray_raw = rgb2gray(rgb_raw)  # Y = 0.2125 R + 0.7154 G + 0.0721 B
image = gray_raw[1]


# image = gaussian_filter(image, 2)

seed = np.copy(image)
seed = seed - .8
#seed[1:-1, 1:-1] = image.min()
mask = image

dilated = reconstruction(seed, mask, method='dilation')
result = image - dilated

print(seed.min(), seed.max())
print(result.min(), result.max())


fig, (ax0, ax1) = plt.subplots(nrows=1,
                               ncols=2,
                               figsize=(8, 2.5),
                               sharex=True,
                               sharey=True)

ax0.imshow(image, cmap='gray')
ax1.imshow(image - dilated, cmap='gray')

plt.show()

