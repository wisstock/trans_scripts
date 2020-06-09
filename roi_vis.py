#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Region vis

"""

import sys
import os
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib import transforms

from skimage.external import tifffile

sys.path.append('modules')
import oiffile as oif


FORMAT = '%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s'
logging.basicConfig(format=FORMAT,
                    level=logging.getLevelName('DEBUG'))


plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'


data_path = os.path.join(sys.path[0], 'data/hpca_dec_16.tif')
# data_path = os.path.join(sys.path[0], 'data/hpca/hpca.tif')


frame = 10
roi_start = [140, 200]  # [85, 150]
roi_lim = 50


roi_path = patches.Rectangle((roi_start[0],roi_start[1]),
                             roi_lim,
                             roi_lim,
                             linewidth=2,
                             edgecolor='w',
                             facecolor='none')

stack = tifffile.imread(data_path)
img = stack[frame,:,:]
roi = img[roi_start[1]:roi_start[1]+roi_lim,roi_start[0]:roi_start[0]+roi_lim]

logging.info('z-stack full intensity {:.3f}'.format(np.sum(stack)))

ax0 = plt.subplot(121)
slice_0 = ax0.imshow(img) 
divider_0 = make_axes_locatable(ax0)
cax = divider_0.append_axes("right", size="3%", pad=0.1)
plt.colorbar(slice_0, cax=cax)
ax0.set_title('Full image')
ax0.add_patch(roi_path)

ax1 = plt.subplot(122)
slice_1 = ax1.imshow(roi) 
divider_1 = make_axes_locatable(ax1)
cax = divider_1.append_axes("right", size="3%", pad=0.1)
plt.colorbar(slice_1, cax=cax)
ax1.set_title('ROI')

# plt.suptitle(samp = file.split('_')[0])
plt.show()
