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
# plt.rcParams['image.cmap'] = 'inferno' 


FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)
                    # filemode="w",
                    # format=FORMAT)  # , filename="demo.log")



wd_path = os.path.join(sys.path[0], 'confocal_data/HPCA-YFP/demo_dec')

logging.info('Dir path: %s' % data_path)

data = {}

frame = 10
angl = 120

for root, dirs, files in os.walk(wd_path):  # loop over the OIF files
    for file in files:
        if file.endswith('.tif'):

            img = tifffile.imread(os.path.join(wd_path, file))

            file_dict = {file.split('_')[1]: img[frame,:,:]}

            data.update(file_dict)

for ch in data:
    img = data[key]

    

plt.show()