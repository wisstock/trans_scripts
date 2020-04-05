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
plt.rcParams['image.cmap'] = 'inferno'

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)

path = os.path.join(sys.path[0], 'raw_data/3/')

frame = 10
sigma = [25,28,1]

yfp, hpca = rd.readZ(path)  # for raw data

# yfp_stack, hpca_stack = rd.readZ(path)  #
# yfp = yfp_stack[frame,:,:]              # for dec data
# hpca = hpca_stack[frame,:,:]            #

# yfp = ts.backCon(yfp, dim=2)
# hpca = ts.backCon(hpca, dim=2)


raw0 = yfp
edge0 = ts.cellEdge(raw0)
skeleton0 = skeletonize(edge0)

raw1 = hpca
edge1 = ts.cellEdge(raw1)
skeleton1 = skeletonize(edge1)



# xx, yy = np.mgrid[0:raw.shape[0], 0:raw.shape[1]]
# fig = plt.figure(figsize=(15,15))
# ax = fig.gca(projection='3d')
# ax.plot_surface(xx, yy, raw,
# 	            rstride=1, cstride=1,
# 	            cmap='inferno',
# 	            linewidth=2)
# ax.view_init(60, 10)
# plt.show()


ax0 = plt.subplot(131)
slc0 = ax0.imshow(raw0)
div0 = make_axes_locatable(ax0)
cax0 = div0.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slc0, cax=cax0)
ax0.set_title('membYFP')

ax1 = plt.subplot(132)
ax1.imshow(edge0)
ax1.set_title('Hessian mask')

ax2 = plt.subplot(133)
ax2.imshow(raw1)
ax2.imshow(skeleton0, alpha=0.4)
# ax2.imshow(skeleton1)
ax2.set_title('yfp_skeleton')

plt.show()


