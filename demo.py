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

logging.info('Raw noise SD YFP={:.3f}, HPCA={:.3f}'.format(yfp_raw_sd, hpca_raw_sd))
logging.info('Dec noise SD YFP={:.3f}, HPCA={:.3f}'.format(yfp_dec_sd, hpca_dec_sd))



roi_start = [85, 85]
roi_lim = 30          


img_0 = yfp_raw


roi_cyto = np.mean(img_0[roi_start[0]:roi_start[0]+roi_lim,\
                   roi_start[1]:roi_start[1]+roi_lim])

logging.info('Cytoplasm ROI mean value {:.3f}'.format(roi_cyto))



img_1 = filters.gaussian(yfp_dec, sigma=3)
img_2 = filters.apply_hysteresis_threshold(img_1,
                                           low=0.1*np.max(img_1),
                                           high=0.6*np.max(img_1))

# mask = np.copy(img_2) * -1

img_3 = ma.masked_where(~img_2, img_0)
img_4 = ma.masked_where(a >= roi_cyto, img_3)

logging.info('Masked min {:.3f}'.format(np.min(img_3)))


ax0 = plt.subplot(231)
slc0 = ax0.imshow(img_0)
div0 = make_axes_locatable(ax0)
cax0 = div0.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slc0, cax=cax0)
ax0.set_title('IMG')
ax0.add_patch(patches.Rectangle((roi_start[0], roi_start[1]),
              roi_lim,
              roi_lim,
              linewidth=2,
              edgecolor='w',
              facecolor='none'))

ax2 = plt.subplot(232)
slc2 = ax2.imshow(img_3)
div2 = make_axes_locatable(ax2)
cax2 = div2.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slc2, cax=cax2)
ax2.set_title('HYST')

ax1 = plt.subplot(233)
ax1.imshow(img_2)  # cmap=plt.cm.gray)
ax1.set_title('HYST MASK')

ax3 = plt.subplot(234)
slc3 = ax3.imshow(img_4)
div3 = make_axes_locatable(ax3)
cax3 = div3.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slc3, cax=cax3)
ax3.set_title('2 SD')

# ax2 = plt.subplot(212)
# ax2.plot (np.arange(0, np.shape(img_0)[1]), np.sum(img_0, axis=0),
#           label='X')
# ax2.plot (np.arange(0, np.shape(img_0)[0]), np.sum(img_0, axis=1),
#           label='Y')
# ax2.set(xlabel='px', ylabel='I')
# legend_properties = {'weight':'bold'}
# plt.legend(loc='upper right',
#             prop=legend_properties)

plt.tight_layout()
plt.show()


