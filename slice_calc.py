#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

3-rd script

Slice calc for deconvoluted HEK263 data

"""

import sys
import os
import logging
import csv
from timeit import default_timer as timer

import numpy as np
import pandas as pd

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
                                        

input_path = os.path.join(sys.path[0], 'dec/cell3/')
output_path = os.path.join(sys.path[0], 'dec/cell3/')


slice_num = 10  # frame of interes num (indexing start from 1)

angl = 0
angl_increment = 360/slice_num
band_w = 2
frame_int = 9

# ch1_list = []  # HPCA-TFP
# ch2_list = []  # membYFP

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

    		logging.debug('File %s in work' % file)

    		file_path = os.path.join(root, file)
   
    		img = tifffile.imread(file_path)

    		frame = img[frame_int-1,:,:]
    		cntr = ts.cellMass(frame)  # calc center of mass of the frame


    		bad_angl = []
    		i = 0  # file counter
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
    				band_df.loc[i] = [file.split('_')[0], file.split('_')[1], frame_int, angl, val]

    			data_frame = pd.concat([data_frame, band_df])

    			angl += angl_increment
    			i += 1

    		logging.info('Bad slices (slice angle) %s \n' % bad_angl)

    		angl = 0

for bad_val in bad_angl:  # delete bad slice in HPCA channel
	data_frame = data_frame[data_frame.angl != bad_val]



cell_hpca = []
memb_hpca = []
rel_memb_hpca = []

cell_yfp = []
memb_yfp = []
rel_memb_yfp = []


samp_lvl = []
[samp_lvl.append(i) for i in list(data_frame.loc[:, 'sample']) if i not in samp_lvl]
samp = samp_lvl[0]

angl_list = []
[angl_list.append(i) for i in list(data_frame.angl) if i not in angl_list]  # generating of avaliable angles list

for angl_val in angl_list:  # loop over the all slices in one sample
    angl_slice = data_frame.query('angl == @angl_val')

    logging.info('Slice with angle %s in work' % angl_val)

    slice_ch1 = np.asarray(angl_slice.val[angl_slice['channel'] == 'ch1'])
    slice_ch2 = np.asarray(angl_slice.val[angl_slice['channel'] == 'ch2'])

    coord = ts.membDet(slice_ch2)  # menbrane coord calc

    if not coord:
        logging.fatal('In slice %s mebrane NOT detected!' % angl_val)
        continue

    c_hpca = slice_ch1[0: coord[0]]
    m_hpca = slice_ch1[coord[0]: coord[1]]

    rel_memb_hpca.append(np.sum(m_hpca)/(np.sum(m_hpca) + np.sum(c_hpca)))
    cell_hpca.append(np.sum(c_hpca))
    memb_hpca.append(np.sum(m_hpca))

    c_yfp = slice_ch2[0: coord[0]]
    m_yfp = slice_ch2[coord[0]: coord[1]]

    rel_memb_yfp.append(np.sum(m_yfp)/(np.sum(m_yfp) + np.sum(c_yfp)))
    cell_yfp.append(np.sum(c_yfp))
    memb_yfp.append(np.sum(m_yfp))

logging.info('HPCA-TFP cytoplasm: %s, sd %s' % (np.mean(cell_hpca), np.std(cell_hpca)))
logging.info('HPCA-TFP membrane: %s, sd %s' % (np.mean(memb_hpca), np.std(memb_hpca)))
logging.info('HPCA-TFP in membrane: %s, sd %s \n' % (np.mean(rel_memb_hpca), np.std(rel_memb_hpca)))

logging.info('membYFP cytoplasm: %s, sd %s' % (np.mean(cell_yfp), np.std(cell_yfp)))
logging.info('membYFP membrane: %s, sd %s' % (np.mean(memb_yfp), np.std(memb_yfp)))
logging.info('membYFP in membrane: %s, sd %s \n' % (np.mean(rel_memb_yfp), np.std(rel_memb_yfp)))

stat_row = [[samp, 'HPCA', 'cyto', np.mean(cell_hpca), np.std(cell_hpca)],
            [samp, 'HPCA', 'memb', np.mean(memb_hpca), np.std(memb_hpca)],
            [samp, 'YFP', 'cyto', np.mean(cell_yfp), np.std(cell_yfp)],
            [samp, 'YFP', 'memb', np.mean(memb_yfp), np.std(memb_yfp)]]

stat_df = stat_df.append(stat_row, ignore_index=True)


rel_row = [[samp, 'HPCA', np.mean(rel_memb_hpca), np.std(rel_memb_hpca)],
           [samp, 'YFP', np.mean(rel_memb_yfp), np.std(rel_memb_yfp)]]

rel_df = rel_df.append(rel_row, ignore_index=True)


data_frame.to_csv(os.path.join(output_path, output_name), index=False)
stat_df.to_csv(os.path.join(output_path, stat_name), index=False)
rel_df.to_csv(os.path.join(output_path, rel_name), index=False)

end_time = timer()
logging.info('%s sample processed in %d second' % (samp, int(end_time - start_time)))
