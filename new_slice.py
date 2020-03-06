#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Slice calc for deconvoluted HEK263 data

New variant (much better)

"""

import sys
import os
import logging
import csv
from timeit import default_timer as timer

import numpy as np
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


FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)  # ,
                    # filemode="w",
                    # filename="oif_read.log")


slice_num = 10  # frame of interes num (indexing start from 1)
frame = 9

input_path = os.path.join(sys.path[0], 'dec/cell3/')
output_path = os.path.join(sys.path[0], 'dec/cell3/')

output_name = 'slice_%s_frame_%s.csv' % (slice_num, frame_int)  # output CSV file name
data_frame = pd.DataFrame(columns=['sample', 'channel', 'frame', 'angl', 'val'])  # init pandas df

stat_name = 'stat_slice_%s_frame_%s.csv' % (slice_num, frame_int)  # output stat CSV file name
stat_df = pd.DataFrame(columns=['sample', 'prot', 'loc','mean', 'sd'])  # init pandas df for slice data

rel_name = 'rel_slice_%s_frame_%s.csv' % (slice_num, frame_int)  # relative amounts in membrane output file
rel_df = pd.DataFrame(columns=['sample', 'prot', 'mean', 'sd'])


start_time = timer()

for root, dirs, files in os.walk(input_path):
    for file in files:
    	if file.endswith('.tif'):

    		file_path = os.path.join(root, file)
    		img = tifffile.imread(file_path)

    		if file.split('_')[1] == 'ch1':
    			hpca = img
    		elif file.split('_')[1] == 'ch2':
    			yfp = img
    		else:
    			logging.error('INCORRECT channels notation!')
    			sys.exit()

# data = np.append(ch1, ch2, axis=0)
# a, b = np.split(data, 2)


cntr = ts.cellMass(hpca[frame-1,:,:])  # calc center of mass of the frame

angl = 0  # slice angle starting value
angl_increment = 360/slice_num

yfp_frame = yfp[frame-1,:,:]  # slices of membYFP channel
hpca_frame = hpca[frame-1,:,:]  # slice of HPCA-TFP channel

bad_angl = []
while angl < 360:   
	logging.info('Slice with angle %s in work' % angl)
	xy0, xy1 = slc.radiusSlice(yfp_frame, angl, cntr)

	yfp_band = slc.bandExtract(yfp_frame, xy0, xy1)

	if ts.badSlc(yfp_band):
		bad_angl.append(angl)
		angl += angl_increment
		logging.info('Slice wih angle %s is not so good, it was discarded!' % angl)
		continue

	hpca_band = slc.bandExtract(hpca_frame, xy0, xy1)

