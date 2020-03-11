#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Slice calc for deconvoluted HEK263 data.

For diameter slice statistic computation.

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
                    # filename="results.log")


slice_num = 15  # number of diameter slices
frame = 10  # frame of interes num (indexing start from 1)


input_path = os.path.join(sys.path[0], 'dec/cell1/')
output_path = os.path.join(sys.path[0], 'dec/cell1/')

output_name = 'slice_%s_frame_%s.csv' % (slice_num, frame)  # output CSV file name
data_frame = pd.DataFrame(columns=['channel', 'angl', 'val'])  # init pandas df

stat_name = 'stat_slice_%s_frame_%s.csv' % (slice_num, frame)  # output stat CSV file name
stat_df = pd.DataFrame(columns=['prot', 'loc','mean', 'sd'])  # init pandas df for slice data

rel_name = 'rel_slice_%s_frame_%s.csv' % (slice_num, frame)  # relative amounts in membrane output file
rel_df = pd.DataFrame(columns=['prot', 'mean', 'sd'])


start_time = timer()

for root, dirs, files in os.walk(input_path):
    for file in files:
    	if file.endswith('.tif'):

            samp = file.split('_')[0]
            file_path = os.path.join(root, file)
            img = tifffile.imread(file_path)

            if file.split('_')[1] == 'ch1':
            	hpca = img
            elif file.split('_')[1] == 'ch2':
            	yfp = img
            else:
            	logging.error('INCORRECT channels notation!')
            	sys.exit()

cntr = ts.cellMass(hpca[frame-1,:,:])  # calc center of mass of the frame

angl = 0  # slice angle starting value
angl_increment = 180/slice_num


yfp_frame = yfp[frame-1,:,:]  # slices of membYFP channel
hpca_frame = hpca[frame-1,:,:]  # slice of HPCA-TFP channel

cell_hpca, memb_hpca, rel_memb_hpca = [], [], []
cell_yfp, memb_yfp, rel_memb_yfp = [], [], []

