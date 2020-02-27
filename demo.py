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



wd_path = os.path.join(sys.path[0], 'confocal_data/HPCA-YFP/')

sample_path = 'cell1_tif/'
data_path = os.path.join(wd_path, sample_path)

logging.info('Dir path: %s' % data_path)


frame = 12
frame_list = []


fig, ax = plt.subplots(1,3)
ax = ax.ravel()

i = 0
for root, dirs, files in os.walk(data_path):  # loop over the TIF files
    for file in files:
    	if file.endswith('.tif'):
    		logging.debug('File %s in work' % file)

    		file_path = os.path.join(root, file)

    		tif_raw = tifffile.imread(file_path)
    		# frame_list.append(tif_raw[frame,:,:])

    		img = tif_raw[frame,:,:]

    		ax[i].imshow(img)
    		ax[i].set_title('File %s' % file)

    		i += 1





# scaling_factor = 5  # substraction region part
# ex_img = ts.backCon(img_raw, np.shape(img_raw)[1] // scaling_factor)




# i = 0
# for i in range(len(frame_list)):

#     ax[i].axis('off') 
#     ax[i].imshow(frame_list[i])

#     i += 1

plt.show()