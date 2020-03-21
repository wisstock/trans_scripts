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
# import pandas as pd

from skimage import data
from skimage import exposure
from skimage import filters
from skimage.filters import scharr
from skimage.external import tifffile
from skimage import restoration

sys.path.append('modules')
import slicing as slc
import threshold as ts

input_path = os.path.join(sys.path[0], 'dec/cell4_5/cell5')

frame = 9  # frame of interes num (indexing start from 1)
slice_num = 10  # number of diameter slices

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)  #,
                    # filemode="w",
                    # filename=os.path.join(input_path, "results_{}.log".format(frame+1)))



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

cntr = ts.cellMass(hpca[frame,:,:])  # calc center of mass of the frame

angl = 0  # slice angle starting value
angl_increment = 180/slice_num


yfp_frame = yfp[frame,:,:]  # slices of membYFP channel
hpca_frame = hpca[frame,:,:]  # slice of HPCA-TFP channel

cell_hpca, memb_hpca, rel_memb_hpca = [], [], []
cell_yfp, memb_yfp, rel_memb_yfp = [], [], []


bad_angl = []
bad_memb = []
while angl < 180:   
    logging.info('Slice with angle %s in work' % angl)
    xy0, xy1 = slc.lineSlice(yfp_frame, angl, cntr)

    yfp_band = slc.bandExtract(yfp_frame, xy0, xy1)

    if ts.badDiam(yfp_band):
        bad_angl.append(angl)
        angl += angl_increment
        continue

    hpca_band = slc.bandExtract(hpca_frame, xy0, xy1)

    coord,_ = ts.membDet(yfp_band, mode='diam')  # detecting membrane peak in membYFP slice
    if not coord:
        logging.error('In slice with angle %s mebrane NOT detected!\n' % angl)
        bad_memb.append(angl)
        angl += angl_increment
        continue

    m_yfp_l = yfp_band[coord[0][0]: coord[0][1]]
    m_yfp_r = yfp_band[coord[1][0]: coord[1][1]]
    c_yfp =yfp_band[coord[0][1]: coord[1][0]]

    m_hpca_l = hpca_band[coord[0][0]: coord[0][1]]
    m_hpca_r = hpca_band[coord[1][0]: coord[1][1]]
    c_hpca = hpca_band[coord[0][1]: coord[1][0]]

    m_yfp_total = np.sum(m_yfp_l) + np.sum(m_yfp_r)
    yfp_total = np.sum(c_yfp) + m_yfp_total

    m_hpca_total = np.sum(m_hpca_l) + np.sum(m_hpca_r)
    hpca_total = np.sum(c_hpca) + m_hpca_total

    rel_memb_yfp.append(m_yfp_total/yfp_total)

    rel_memb_hpca.append(m_hpca_total/hpca_total)

    # logging.info('mYFP {}'.format(m_yfp_total/yfp_total))
    # logging.info('HPCA-TFP {} \n'.format(m_hpca_total/hpca_total))

    angl += angl_increment

if bad_angl:
    logging.warning('Slices with angles %s discarded, bad peaks!\n' % (bad_angl))
if bad_memb:
    logging.warning('Slices with angles %s discarded, no membrane detected!\n' % (bad_memb))

logging.info('{} slices successful complited!\n'.format(slice_num-(len(bad_memb)+len(bad_angl))))

logging.info('Sample {}, frame {}'.format(samp, (frame+1)))
logging.info('membYFP relative amount in membrane: {:.3f} percent, sd {:.3f}'.format(np.mean(rel_memb_yfp)*100, np.std(rel_memb_yfp)*100))
logging.info('HPCA-TFP relative amount in membrane: {:.3f} percent, sd {:.3f}'.format(np.mean(rel_memb_hpca)*100, np.std(rel_memb_hpca)*100))
