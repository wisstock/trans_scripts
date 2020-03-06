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
import pandas as pd

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
                                        

# input_path = os.path.join(sys.path[0], 'demo_data/dec')
# raw_file = '20180718-1315-0007_ch2_dec_30.tif'
# data_path = os.path.join(input_path, raw_file) 

# frame = 10
# band_w = 2

<<<<<<< HEAD
angle = 2  # bad 285 281  good 90
=======
# angle = 216  # 285 281 280
>>>>>>> 84b0a05a307fb0cb517ceb5a755f47c5adabbc23


# seq = tifffile.imread(data_path)
# img = seq[frame,:,:]

# cntr = ts.cellMass(img)
# xy0, xy1 = slc.radiusSlice(img, angle, cntr)

# band = slc.bandExtract(img, xy0, xy1, band_w)

# print(band.max())


# band_qual = ts.badSlc(band)

# print(band_qual)



data_path = os.path.join(sys.path[0], 'dec/cell2/')

data_name = 'slice_20_frame_8.csv'
df_samp = pd.read_csv(os.path.join(data_path, data_name))

angl_list = []
[angl_list.append(i) for i in list(df_samp.angl) if i not in angl_list]  # generating of avaliable angles list


angl_val = angl_list[9]
angl_slice = df_samp.query('angl == @angl_val')
logging.info('Slice with angle %s in work' % angl_val)

slice_ch1 = np.asarray(angl_slice.val[angl_slice['channel'] == 'ch1'])
slice_ch2 = np.asarray(angl_slice.val[angl_slice['channel'] == 'ch2'])



ax = plt.subplot()
ax.plot(slice_ch1)
ax.plot(slice_ch2)
# ax.axvline(band_qual[0], ymin=0, ymax=band[band_qual[0]], linestyle='dashed')
# ax.axvline(band_qual[1], ymin=0, ymax=band[band_qual[1]], linestyle='dashed')
# ax.axvlines(band_qual[2], ymin=0, ymax=band[band_qual[2]], linestyle='dashed')


plt.show()








