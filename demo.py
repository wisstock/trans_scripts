#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Prototyping script

"""

import sys
import os
import logging

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from skimage.external import tifffile
from skimage import filters
from skimage import measure

from scipy.ndimage import measurements as msr

sys.path.append('modules')
import oiffile as oif
import slicing as slc
import threshold as ts


plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno' 


FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)  # ,
                    # filemode="w",
                    # filename="oif_read.log")
                                        

input_path = os.path.join(sys.path[0], 'demo_data/dec')
raw_file = '20180718-1315-0007_ch2_dec_30.tif'
data_path = os.path.join(input_path, raw_file) 

frame = 10
band_w = 2

angle = 216  # 285 281 280


seq = tifffile.imread(data_path)
img = seq[frame,:,:]

cntr = ts.cellMass(img)
xy0, xy1 = slc.radiusSlice(img, angle, cntr)

band = slc.bandExtract(img, xy0, xy1, band_w)

print(band.max())


band_qual = ts.badSlc(band)

print(band_qual)


ax = plt.subplot()
ax.plot(band)
# ax.axvline(band_qual[0], ymin=0, ymax=band[band_qual[0]], linestyle='dashed')
# ax.axvline(band_qual[1], ymin=0, ymax=band[band_qual[1]], linestyle='dashed')
# ax.axvlines(band_qual[2], ymin=0, ymax=band[band_qual[2]], linestyle='dashed')


plt.show()








