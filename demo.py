#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Detecting cell edge and continue mebrane estimation

"""

import sys
import os
import logging

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

from skimage.exposure import histogram
from skimage.morphology import skeletonize
from skimage.feature import canny


sys.path.append('modules')
import threshold as ts
import membrane as memb
import readdata as rd


plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'magma'

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)

path_raw = os.path.join(sys.path[0], 'raw_data/3/')
path_dec = os.path.join(sys.path[0], 'dec_data/3/')

frame = 10
sigma = [25,28,1]

yfp_raw, hpca_raw = rd.readZ(path_raw)  # for raw data

yfp_stack, hpca_stack = rd.readZ(path_dec)  #
yfp_dec = yfp_stack[frame,:,:]              # for dec data
hpca_dec = hpca_stack[frame,:,:]            #


raw = yfp_raw
edge = ts.cellEdge(yfp_raw)
skel = skeletonize(edge)
cyto = hpca_raw

raw_noise = np.std(raw[0:20,0:20])
raw_ts = 2*raw_noise
logging.info('YFP noise {:.3f}'.format(raw_noise))

cyto_noise = np.std(cyto[0:20,0:20])
cyto_ts = 2*cyto_noise
logging.info('HPCA noise {:.3f}'.format(cyto_noise))





# xx, yy = np.mgrid[0:yfp_raw.shape[0], 0:yfp_raw.shape[1]]
# fig = plt.figure(figsize=(15,15))
# ax = fig.gca(projection='3d')
# ax.plot_surface(xx, yy, yfp_raw,
# 	            rstride=1, cstride=1,
# 	            cmap='inferno',
# 	            linewidth=2)
# ax.view_init(60, 10)
# plt.show()


# ax0 = plt.subplot()  # 131)
# ax0.imshow(raw)
# ax0.imshow(skel, alpha=0.2)
# ax0.axes.xaxis.set_visible(False)
# ax0.axes.yaxis.set_visible(False)

ax0 = plt.subplot(131)
slc0 = ax0.imshow(raw)
div0 = make_axes_locatable(ax0)
cax0 = div0.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slc0, cax=cax0)
ax0.set_title('membYFP')

ax1 = plt.subplot(132)
ax1.imshow(edge)
# ax1.plot(cntr_det[0], cntr_det[1], 'o')
ax1.set_title('Mask')

ax2 = plt.subplot(133)
slc2 = ax2.imshow(cyto)
div2 = make_axes_locatable(ax2)
cax2 = div2.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slc2, cax=cax2)
ax2.set_title('HPCA-TFP')

plt.show()


