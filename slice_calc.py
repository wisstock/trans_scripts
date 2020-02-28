#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Slice calc for deconvoluted HEK263 data

"""

import sys
import os
import glob
import logging
import csv

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage import data
from skimage import exposure
from skimage import filters
from skimage.filters import scharr
from skimage.external import tifffile
from skimage import restoration

sys.path.append('modules')
import slicing as slc
import threshold as ts


plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno' 


FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.DEBUG,
                    format=FORMAT)  # ,
                    # filemode="w",
                    # filename="oif_read.log")
                                        

input_path = os.path.join(sys.path[0], 'confocal_data/HPCA-YFP/dec')
output_path = os.path.join(sys.path[0], 'confocal_data/HPCA-YFP/')
output_name = 'slice_dec_30.csv'

angl = 0
angl_increment = 24
band_w = 2

ch1_list = []  # HPCA-TFP
ch2_list = []  # membYFP

with open(os.path.join(output_path, output_name), 'w') as csvfile:  # init CSF output file
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['sample', 'channel', 'frame', 'angl', 'val'])

i = 0
for root, dirs, files in os.walk(input_path):  # loop over the OIF files
    for file in files:
    	if file.endswith('.tif'):
    		logging.debug('File %s in work' % file)

    		file_path = os.path.join(root, file)
   
    		img = tifffile.imread(file_path)
    		frame_num = np.shape(img)[0] // 2  # frame of interes

    		logging.debug('Frame of interes: %s' % frame_num)

    		frame = img[frame_num,:,:]
    		cntr = ts.cellMass(frame)  # calc center of mass of the frame

    		while angl < 360:
    			xy0, xy1 = slc.lineSlice(frame, angl, cntr)

    			band = slc.bandExtract(frame, xy0, xy1, band_w)

    			for val in band:  # loop over the band slice values to write their to CSV
    			    with open(os.path.join(output_path, output_name), 'a') as csvfile: 
    				    writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    				    writer.writerow([file.split('_')[0], file.split('_')[1], frame_num+1, angl, val])

    			i += 1
    			angl += angl_increment

