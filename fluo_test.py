#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Test experiment with NP-EGTA + Fluo-4 in HEK cells.
24-27,07.2020

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
from skimage import segmentation
from skimage import filters
from skimage import morphology
from skimage.feature import canny
from skimage.external import tifffile
from skimage.util import compare_images


sys.path.append('modules')
import oifpars as op
import edge



plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.DEBUG,
                    format=FORMAT)



data_path = os.path.join(sys.path[0], 'fluo_data')

a = op.WDPars(data_path)
print(a[0].__dict__)
b = a[0]
c = b.img_series

print(np.shape(c))

