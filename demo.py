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

from skimage.exposure import histogram

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

yfp, hpca = rd.readZ(path)  # for raw data

# yfp_stack, hpca_stack = rd.readZ(path)  #
# yfp = yfp_stack[frame,:,:]              # for dec data
# hpca = hpca_stack[frame,:,:]            #

yfp = ts.backCon(yfp, dim=2)
hpca = ts.backCon(hpca, dim=2)

yfp_hist = histogram(yfp)
hpca_hist = histogram(hpca)


ax0 = plt.subplot(121)
ax0.hist(yfp.ravel(),
	     bins=512)
	     # range=(2000.0, 12000.0))
# ax0.hist(hpca.ravel(),
# 	     bins=256,
# 	     range=(0.0, 1000.0),
# 	     label='HPCA-TFP')

ax1 = plt.subplot(122)
slc1 = ax1.imshow(yfp)
div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slc1, cax=cax1)
ax1.set_title('membYFP')

plt.show()


