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
                    filemode="w",
                    format=FORMAT)  # , filename="demo.log")



# oif_path = '/home/astria/Bio_data/HEK_mYFP/20180523_HEK_membYFP/cell1/20180523-1404-0003-250um.oif'
# oif_raw = oif.OibImread(oif_path)
# oif_img = oif_raw[0,:,:,:]

# img = getTiff(input_file, 0, 1)
# img = oif_img[6,:,:]
# img = tifffile.imread(input_file)
# img_mod = ts.cellEdge(img)

data_path = os.path.join(sys.path[0], '.temp/data/cell1.tif')  # "/home/astria/Bio/Lab/scripts/trans_scripts/.temp/data/cell1.tif"


img_raw = tifffile.imread(data_path)

scaling_factor = 5  # substraction region part
ex_img = ts.backCon(img_raw, np.shape(img_raw)[1] // scaling_factor)

img = img_raw[:,0:50,0:50]
ext_img = ex_img[:,0:50,0:50]



img_dict = {'Raw, 3': img[2,:,:],
            'Extracted, 3': ext_img[2,:,:],
            'Raw, 10': img[9,:,:],
            'Extracted, 10': ext_img[9,:,:]}



pos = 1
for key in img_dict:
    img = img_dict[key]

    ax = plt.subplot(2, 2, pos)
    slc = ax.imshow(img)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='3%', pad=0.1)
    plt.colorbar(slc, cax=cax)
    ax.set_title(key)

    pos += 1 

plt.show()