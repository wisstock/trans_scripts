#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Slice calc for deconvoluted HEK263 data.

New variant (much better) for radial slice statistic computation.

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


slice_num = 15  # frame of interes num (indexing start from 1)
frame = 10

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

# data = np.append(ch1, ch2, axis=0)
# a, b = np.split(data, 2)


cntr = ts.cellMass(hpca[frame-1,:,:])  # calc center of mass of the frame

angl = 0  # slice angle starting value
angl_increment = 360/slice_num

yfp_frame = yfp[frame-1,:,:]  # slices of membYFP channel
hpca_frame = hpca[frame-1,:,:]  # slice of HPCA-TFP channel

cell_hpca, memb_hpca, rel_memb_hpca = [], [], []
cell_yfp, memb_yfp, rel_memb_yfp = [], [], []

bad_angl = []
while angl < 360:   
    logging.info('Slice with angle %s in work' % angl)
    xy0, xy1 = slc.radiusSlice(yfp_frame, angl, cntr)

    yfp_band = slc.bandExtract(yfp_frame, xy0, xy1)

    if ts.badRad(yfp_band):
        bad_angl.append(angl)
        logging.warning('Slice wih angle %s is not so good, it was discarded!' % angl)
        angl += angl_increment
        continue

    # print([angl for i in range(len(yfp_band))])
    hpca_band = slc.bandExtract(hpca_frame, xy0, xy1)

    coord = ts.membDet(yfp_band)  # detecting membrane peak in membYFP slice
    if not coord:
        logging.error('In slice with angle %s mebrane NOT detected!' % angl)
        continue

    c_hpca = hpca_band[0: coord[0]]
    m_hpca = hpca_band[coord[0]: coord[1]]
    c_yfp = yfp_band[0: coord[0]]
    m_yfp =yfp_band[coord[0]: coord[1]]

    rel_memb_hpca.append(np.sum(m_hpca)/(np.sum(m_hpca) + np.sum(c_hpca)))
    cell_hpca.append(np.sum(c_hpca))
    memb_hpca.append(np.sum(m_hpca))

    rel_memb_yfp.append(np.sum(m_yfp)/(np.sum(m_yfp) + np.sum(c_yfp)))
    cell_yfp.append(np.sum(c_yfp))
    memb_yfp.append(np.sum(m_yfp))

    angl += angl_increment

if bad_angl:
    logging.warning('Slices with angles %s discarded, %s slices was calculated successfully!\n' % (bad_angl, slice_num-len(bad_angl)))

logging.info('Sample {}'.format(samp))

logging.info('membYFP cytoplasm: {:.3f}, sd {:.3f}'.format(np.mean(cell_yfp), np.std(cell_yfp)))
logging.info('membYFP membrane: {:.3f}, sd {:.3f}\n'.format(np.mean(memb_yfp), np.std(memb_yfp)))

logging.info('HPCA-TFP cytoplasm: {:.3f}, sd {:.3f}'.format(np.mean(cell_hpca), np.std(cell_hpca)))
logging.info('HPCA-TFP membrane: {:.3f}, sd {:.3f}\n'.format(np.mean(memb_hpca), np.std(memb_hpca)))

logging.info('membYFP relative amount in membrane: {:.3f} percent, sd {:.3f}'.format(np.mean(rel_memb_yfp)*100, np.std(rel_memb_yfp)*100))
logging.info('HPCA-TFP relative amount in membrane: {:.3f} percent, sd {:.3f}'.format(np.mean(rel_memb_hpca)*100, np.std(rel_memb_hpca)*100))
