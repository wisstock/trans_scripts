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

frame = 14
sigma = [25,28,1]
cell_roi = [70, 260, 70, 250]

yfp_raw_stack = yfp_raw_stack[:, cell_roi[0]:cell_roi[1], cell_roi[2]:cell_roi[3]]
yfp_dec_stack = yfp_dec_stack[:, cell_roi[0]:cell_roi[1], cell_roi[2]:cell_roi[3]]
hpca_raw_stack = hpca_raw_stack[:, cell_roi[0]:cell_roi[1], cell_roi[2]:cell_roi[3]]
hpca_dec_stack = hpca_dec_stack[:, cell_roi[0]:cell_roi[1], cell_roi[2]:cell_roi[3]]

a = 1
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


img_3 = yfp_raw
img_0 = yfp_dec

# low = yfp_dec_sd*3 # yfp raw 0.003
# high = 1200 # yfp raw 0.010

# img_1 = filters.apply_hysteresis_threshold(img_3, 300, 1500)  # ma.masked_less(yfp_dec, yfp_raw_sd*5)
img_2 = filters.apply_hysteresis_threshold(img_0, 200, 2200)


ax0 = plt.subplot(221)
slc0 = ax0.imshow(img_0)
div0 = make_axes_locatable(ax0)
cax0 = div0.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slc0, cax=cax0)
ax0.set_title('IMG DEC')

ax1 = plt.subplot(222)
ax1.imshow(img_2)  # cmap=plt.cm.gray)
ax1.set_title('RAW')

ax2 = plt.subplot(212)
ax2.plot (np.arange(0, np.shape(img_0)[1]), np.sum(img_0, axis=0),
          label='X')
ax2.plot (np.arange(0, np.shape(img_0)[0]), np.sum(img_0, axis=1),
          label='Y')
ax2.set(xlabel='px', ylabel='I')
legend_properties = {'weight':'bold'}
plt.legend(loc='upper right',
            prop=legend_properties)

plt.tight_layout()
plt.show()


