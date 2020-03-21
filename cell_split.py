#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Split cell 4-5 to separate cell files.

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

input_path = os.path.join(sys.path[0], '.temp/cell4_5/cell4/')
output_path = os.path.join(sys.path[0], '.temp/cell4_5/cell4')

yfp_name = '20180718-1323-0010_ch2_4.tif'
hpca_name = '20180718-1323-0010_ch1_4.tif'


yfp = tifffile.imread(os.path.join(input_path, yfp_name))
hpca = tifffile.imread(os.path.join(input_path, hpca_name))


print(np.min(yfp))
print(np.shape(yfp))
print(np.min(hpca))
print(np.shape(hpca))

frame = (hpca-yfp)[2]
print(np.max(frame))
print(np.min(frame))

# print(type(hpca))


ax1 = plt.subplot()
ax1.plot([170, 170], [0, 320])
slc1 = ax1.imshow(frame)
div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slc1, cax=cax1)

plt.show()