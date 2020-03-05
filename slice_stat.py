#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

4-th script

Stat calc for the results of slicing (slice_*.csv)

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

from timeit import default_timer as timer

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


data_path = os.path.join(sys.path[0], 'confocal_data/HPCA-YFP/dec/cell3')

data_name = 'slice_10_frame_10.csv'
df_samp = pd.read_csv(os.path.join(data_path, data_name))



stat_name = 'stat_%s.csv' % data_name.split('_')[3]  # output stat CSV file name
stat_df = pd.DataFrame(columns=['sample', 'prot', 'loc','mean', 'sd'])  # init pandas df for slice data

rel_name = 'rel_%s.csv' % data_name.split('_')[3]  # relative amounts in membrane output file
rel_df = pd.DataFrame(columns=['sample', 'prot', 'mean', 'sd'])


cell_hpca = []
memb_hpca = []
rel_memb_hpca = []

cell_yfp = []
memb_yfp = []
rel_memb_yfp = []


samp_lvl = []
[samp_lvl.append(i) for i in list(df_samp.loc[:, 'sample']) if i not in samp_lvl]
samp = samp_lvl[0]
print(samp_lvl)

# i = 0  # sample counter
# start_time = timer()
# for samp in samp_list:

# 	logging.info('Sample %s in work' % samp)
# 	i +=1

# 	df_samp = df.query('sample == @samp')

start_time = timer()
angl_list = []
[angl_list.append(i) for i in list(df_samp.angl) if i not in angl_list]  # generating of avaliable angles list

for angl_val in angl_list:  # loop over the all slices in one sample
	angl_slice = df_samp.query('angl == @angl_val')

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


# cell_hpca = []
# memb_hpca = []
# rel_memb_hpca = []

# cell_yfp = []
# memb_yfp = []
# rel_memb_yfp = []


stat_df.to_csv(os.path.join(data_path, stat_name), index=False)
rel_df.to_csv(os.path.join(data_path, rel_name), index=False)


end_time = timer()
logging.info('%s samples processed in %d second' % (samp, int(end_time - start_time)))

