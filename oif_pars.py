#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

1-sr script

OIF image data automatic extractions
to TIF series 

"""

import sys
import os
import logging

import numpy as np
from skimage.external import tifffile
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append('modules')
import oiffile as oif


plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno' 


FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.DEBUG,
                    format=FORMAT)
                    # filename="oif_read.log")
                                        # filemode="w"


data_path = os.path.join(sys.path[0], '.temp/data/Cell3')
output_path =os.path.join(sys.path[0], '.temp/data/')

# cell = '20180718-1316-0008.oif.files/'
# cell_path = os.path.join(data_path, cell)
# logging.info('Dir path: %s' % cell_path)


for root, dirs, files in os.walk(data_path):  # loop over the OIF files
    for file in files:
    	if file.endswith('.oif'):
    		logging.debug('File %s in work' % file)

    		file_path = os.path.join(root, file)
   
    		oif_raw = oif.OibImread(file_path)
    		logging.debug(np.shape(oif_raw))

    		for i in range(np.shape(oif_raw)[0]):
    			tif_name = '%s_ch%s.tif' % (file.split('.')[0], i+1)
    			tifffile.imsave(os.path.join(output_path, tif_name), oif_raw[i,:,:,:])

    			logging.info('File %s, channel %s saved' % (tif_name, i+1))


# oif_raw = oif.OibImread(os.path.join(cell_path, '20180718-1255-0003.oif'))
# oif_img = oif_raw[0,:,:,:]

# tifffile.imsave(os.path.join(data_path, '0003_2.tif'), oif_img)
