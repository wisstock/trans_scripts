#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

3-rd script

Slice calc for deconvoluted HEK263 data

"""

import sys
import os
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
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)  # ,
                    # filemode="w",
                    # filename="oif_read.log")
                                        

input_path = os.path.join(sys.path[0], 'demo_data/dec')
output_path = os.path.join(sys.path[0], 'demo_data')

slice_num = 40

angl = 0
angl_increment = 360/slice_num
band_w = 2

ch1_list = []  # HPCA-TFP
ch2_list = []  # membYFP

output_name = 'slice_%s.csv' % (slice_num)  # output CSV file name
with open(os.path.join(output_path, output_name), 'w') as csvfile:  # init CSF output file
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['sample', 'channel', 'frame', 'angl', 'val'])


for root, dirs, files in os.walk(input_path):  # loop over the OIF files
    for file in files:
    	if file.endswith('.tif'):
    		priv_file = file

    		logging.info('File %s in work' % file)

    		file_path = os.path.join(root, file)
   
    		img = tifffile.imread(file_path)
    		frame_num = np.shape(img)[0] // 2 + 1  # frame of interes

    		logging.info('Frame of interes: %s' % frame_num)

    		frame = img[frame_num,:,:]
    		cntr = ts.cellMass(frame)  # calc center of mass of the frame

    		logging.info(angl)

    		while angl < 360:
    			xy0, xy1 = slc.radiusSlice(frame, angl, cntr)

    			band = slc.bandExtract(frame, xy0, xy1, band_w)

    			if not slcQual(band):  # checking band quality, if it's bad - going to the next one
    				continue

    			for val in band:  # loop over the band slice values to write their to CSV
    			    with open(os.path.join(output_path, output_name), 'a') as csvfile: 
    				    writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    				    writer.writerow([file.split('_')[0], file.split('_')[1], frame_num+1, angl, val])

    			angl += angl_increment

    		angl = 0

