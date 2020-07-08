#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Detecting cell edge and continue mebrane estimation

"""

import sys
import os
import logging

import numpy as np
import numpy.ma as ma

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

from skimage.exposure import histogram
from skimage import segmentation
from skimage import filters
from skimage.morphology import skeletonize
from skimage.feature import canny
from skimage.external import tifffile
from skimage.util import compare_images


sys.path.append('modules')
import threshold as ts
import membrane as memb
import readdata as rd


plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)

yfp_raw_stack = tifffile.imread(os.path.join(sys.path[0], 'data/yfp.tif'))
yfp_dec_stack = tifffile.imread(os.path.join(sys.path[0], 'data/yfp_dec_32.tif'))
hpca_raw_stack = tifffile.imread(os.path.join(sys.path[0], 'data/hpca.tif'))
hpca_dec_stack = tifffile.imread(os.path.join(sys.path[0], 'data/hpca_dec_32.tif'))

frame = 12
sigma = [25,28,1]
cell_roi = [70, 260, 70, 250]

yfp_raw_stack = yfp_raw_stack[:, cell_roi[0]:cell_roi[1], cell_roi[2]:cell_roi[3]]
yfp_dec_stack = yfp_dec_stack[:, cell_roi[0]:cell_roi[1], cell_roi[2]:cell_roi[3]]
hpca_raw_stack = hpca_raw_stack[:, cell_roi[0]:cell_roi[1], cell_roi[2]:cell_roi[3]]
hpca_dec_stack = hpca_dec_stack[:, cell_roi[0]:cell_roi[1], cell_roi[2]:cell_roi[3]]

a = 0
if a:
    yfp_raw_stack = ts.backCon(yfp_raw_stack)
    yfp_dec_stack = ts.backCon(yfp_dec_stack)
    hpca_raw_stack = ts.backCon(hpca_raw_stack)
    hpca_dec_stack = ts.backCon(hpca_dec_stack)


yfp_raw = yfp_raw_stack[frame,:,:]
hpca_raw = hpca_raw_stack[frame,:,:]

yfp_dec = yfp_dec_stack[frame,:,:]
hpca_dec = hpca_dec_stack[frame,:,:]

yfp_raw_sd = np.std(yfp_raw_stack[:,:20,:20])
yfp_dec_sd = np.std(yfp_dec_stack[:,:20,:20])
hpca_raw_sd = np.std(hpca_raw_stack[:,:20,:20])
hpca_dec_sd = np.std(hpca_dec_stack[:,:20,:20])

logging.info('Raw SD YFP={:.3f}, HPCA={:.3f}'.format(yfp_raw_sd, hpca_raw_sd))
logging.info('Dec SD YFP={:.3f}, HPCA={:.3f}'.format(yfp_dec_sd, hpca_dec_sd))



ax0 = plt.subplot(131)
slc0 = ax0.imshow(img_0)
div0 = make_axes_locatable(ax0)
cax0 = div0.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slc0, cax=cax0)
ax0.set_title('RAW')

ax1 = plt.subplot(132)
slc1 = ax1.imshow(img_1)
div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slc1, cax=cax1)
ax1.set_title('MASK')

ax2 = plt.subplot(133)
ax2.imshow(img_2, cmap=plt.cm.nipy_spectral)  # cmap=plt.cm.gray)

plt.show()


