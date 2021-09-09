#!/usr/bin/env python3

""" Copyright © 2020-2021 Borys Olifirov

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
from skimage import morphology
from skimage.feature import canny
from skimage.external import tifffile
from skimage.util import compare_images


sys.path.append('modules')
import edge
# import membrane as memb
# import readdata as rd


plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)

yfp_raw_stack = tifffile.imread(os.path.join(sys.path[0], 'demo_data/yfp.tif'))

cell_roi = [70, 250, 70, 250]

yfp_raw_stack = yfp_raw_stack[:, cell_roi[0]:cell_roi[1], cell_roi[2]:cell_roi[3]]


frame = 13
roi_start = [85, 85]
roi_lim = 30    

img = yfp_raw_stack[frame,:,:]
# img = edge.backCon(img, dim=2)


# img_gauss = filters.gaussian(img, sigma=3)


# noise_sd = np.std(img[:20,:20])
# logging.info('Noise SD={:.3f}'.format(noise_sd))


# roi_cyto = np.mean(img[roi_start[0]:roi_start[0]+roi_lim,\
#                    roi_start[1]:roi_start[1]+roi_lim])
# logging.info('Cytoplasm ROI mean value {:.3f}'.format(roi_cyto))

# # low_mean = 0.45
# # low_2sd = 0.1
# # hyst_2sd = filters.apply_hysteresis_threshold(img_gauss,
# #                                            low=low_2sd*np.max(img_gauss),  # 0.07
# #                                            high=0.8*np.max(img_gauss))  # 0.8
# mask_hyst = filters.apply_hysteresis_threshold(img_gauss,
#                                            low=0.60*np.max(img_gauss),  # 0.07
#                                            high=0.8*np.max(img_gauss))  # 0.8


# # hyst_mean_masked = ma.masked_where(~hyst_mean, img_0)
# hyst_2sd_masked = ma.masked_where(~hyst_2sd, img_0)

# raw_2sd_masked = ma.masked_greater_equal(img_0, 2*noise_sd)
# raw_mean_masked = ma.masked_greater(img_0, roi_cyto)

# hyst_2sd_2sd = ma.masked_where(~hyst_2sd, raw_2sd_masked)
# hyst_mean_mean = ma.masked_where(~hyst_mean, raw_mean_masked)


# img_1 = raw_2sd_masked
# img_2 = raw_mean_masked

# img_3 = hyst_2sd_2sd
# img_4 = hyst_mean_mean





sd, mean, res = edge.hystMemb(img, roi_center=[100, 100], low_diff=100)  # raw_mean_masked

# sd = mask_hyst
# mean = segmentation.flood(mask_hyst, (0, 0)) + sd
# res = mean*-1

# print(np.all(mean))



# ax0 = plt.subplot(231)
# slc0 = ax0.imshow(img_0)
# div0 = make_axes_locatable(ax0)
# cax0 = div0.append_axes('right', size='3%', pad=0.1)
# plt.colorbar(slc0, cax=cax0)
# ax0.set_title('CYTO ROI')
# ax0.add_patch(patches.Rectangle((roi_start[0], roi_start[1]),
#               roi_lim,
#               roi_lim,
#               linewidth=2,
#               edgecolor='w',
#               facecolor='none'))

# ax2 = plt.subplot(323)
# slc2 = ax2.imshow(img_4)
# div2 = make_axes_locatable(ax2)
# cax2 = div2.append_axes('right', size='3%', pad=0.1)
# plt.colorbar(slc2, cax=cax2)
# ax2.set_title('MEAN+HYST (low={})'.format(low_mean))

# ax5 = plt.subplot(324)
# slc5 = ax5.imshow(img_3)
# div5 = make_axes_locatable(ax5)
# cax5 = div5.append_axes('right', size='3%', pad=0.1)
# plt.colorbar(slc5, cax=cax5)
# ax5.set_title('2SD+HYST (low={})'.format(low_2sd))

# ax3 = plt.subplot(321)
# slc3 = ax3.imshow(img_2)
# div3 = make_axes_locatable(ax3)
# cax3 = div3.append_axes('right', size='3%', pad=0.1)
# plt.colorbar(slc3, cax=cax3)
# ax3.set_title('MEAN')

# ax4 = plt.subplot(322)
# slc4 = ax4.imshow(img_1)
# div4 = make_axes_locatable(ax4)
# cax4 = div4.append_axes('right', size='3%', pad=0.1)
# plt.colorbar(slc4, cax=cax4)
# ax4.set_title('2SD')



# ax1 = plt.subplot(231)
# ax1.imshow(sd)
# ax2 = plt.subplot(232)
# ax2.imshow(mean)
ax3 = plt.subplot(122)  # 233)
ax3.imshow(res)

ax4 = plt.subplot(121)  # 223)
ax4.imshow(img)
# ax5 = plt.subplot(224)
# ax5.imshow(ma.masked_where(~res, img))

plt.tight_layout()
plt.show()
