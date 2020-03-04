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
import pandas as pd

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
                                        

input_path = os.path.join(sys.path[0], 'demo_data/dec/')
output_path = os.path.join(sys.path[0], 'demo_data')


slice_num = 360

angl = 0
angl_increment = 360/slice_num
band_w = 2

ch1_list = []  # HPCA-TFP
ch2_list = []  # membYFP

output_name = 'slice_%s.csv' % (slice_num)  # output CSV file name
data_frame = pd.DataFrame(columns=['sample', 'channel', 'frame', 'angl', 'val'])  # init pandas df


for root, dirs, files in os.walk(input_path):  # loop over the membYFP files
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


    		bad_angl = []
    		while angl < 360:
    			xy0, xy1 = slc.radiusSlice(frame, angl, cntr)

    			band = slc.bandExtract(frame, xy0, xy1, band_w)

    			if file.split('_')[1] == 'ch2':  # slice quality control, for YFP channel only
    			    if ts.badSlc(band):
    				    bad_angl.append(angl)
    				    angl += angl_increment
    				    continue

    			band_df = pd.DataFrame(columns=['sample', 'channel', 'frame', 'angl', 'val'])                    
    			for i in range(len(band)):  # loop over the band slice values to write their to CSV
    				val = band[i]
    				band_df.loc[i] = [file.split('_')[0], file.split('_')[1], frame_num+1, angl, val]

    			data_frame = pd.concat([data_frame, band_df])

    			angl += angl_increment

    		logging.info('Bad slices (slice angle) %s \n' % bad_angl)

    		angl = 0

for bad_val in bad_angl:  # delete bad slice in HPCA channel
	data_frame = data_frame[data_frame.angl != bad_val]


data_frame.to_csv(os.path.join(output_path, output_name), index=False)