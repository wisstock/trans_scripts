#!/usr/bin/env python3

'''
Copyright © 2020 Borys Olifirov

'''

import os
from skimage.external import tifffile
from skimage.filters import try_all_threshold

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


wd_path = os.path.split(os.getcwd())
os.chdir(wd_path[0] + '/temp/data/')  # go to DATA dir

offset = 250  # camera offset value
gray_raw = tifffile.imread('Fluorescence_435nmDD500_cell1.tiff')
img = gray_raw[0] - offset

fig, ax = try_all_threshold(img, figsize=(8, 10),
                                 verbose=False)
plt.show()