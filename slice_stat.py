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


data_path = os.path.join(sys.path[0], 'demo_data')
img_path = os.path.join(sys.path[0], 'demo_data/dec')

data_name = 'slice_360.csv'

df = pd.read_csv(os.path.join(data_path, data_name))
# df = df.dropna()
# df = df.reset_index(drop=True)

 
angl_num = 5

angl_list = []
[angl_list.append(i) for i in list(df.angl) if i not in angl_list]  # remove duplications and generate angl list

logging.info('%s angle values is avaliable: %s' % (len(angl_list), angl_list))

# angl_val = 297
try:
	angl_val = angl_list[angl_num]
except IndexError:
	sys.exit('Angle value NOT avaliable!')


samp = '20180718-1315-0007'
df_demo = df.query('sample == @samp')


cells_hpca = []
membs_hpca = []

cells_yfp = []
membs_yfp = []
rel_hpca = []
rel_yfp = []

for angl_val in angl_list:  # loop over the all slices in one sample
	angl_slice = df_demo.query('angl == @angl_val')

	logging.info('Slice with angle %s in work' % angl_val)

	slice_ch1 = np.asarray(angl_slice.val[df['channel'] == 'ch1'])
	slice_ch2 = np.asarray(angl_slice.val[df['channel'] == 'ch2'])

	coord = ts.membDet(slice_ch2)  # menbrane coord calc

	cell_hpca = slice_ch1[0: coord[0]]
	memb_hpca = slice_ch1[coord[0]: coord[1]]

	rel_hpca.append(np.sum(memb_hpca)/np.sum(cell_hpca))
	cells_hpca.append(np.sum(cell_hpca))
	membs_hpca.append(np.sum(memb_hpca))

	cell_yfp = slice_ch2[0: coord[0]]
	memb_yfp = slice_ch2[coord[0]: coord[1]]

	rel_yfp.append(np.sum(cell_yfp)/np.sum(memb_yfp))
	cells_yfp.append(np.sum(cell_yfp))
	membs_yfp.append(np.sum(memb_yfp))


logging.info('HPCA-TFP cytoplasm: %s, sd %s' % (np.mean(cells_hpca), np.std(cells_hpca)))
logging.info('HPCA-TFP membrane: %s, sd %s' % (np.mean(membs_hpca), np.std(membs_hpca)))
logging.info('HPCA-TFP memb/cell ratio: %s, sd %s \n' % (np.mean(rel_hpca), np.std(rel_hpca)))

logging.info('membYFP cytoplasm: %s, sd %s' % (np.mean(cells_yfp), np.std(cells_yfp)))
logging.info('membYFP membrane: %s, sd %s' % (np.mean(membs_yfp), np.std(membs_yfp)))
logging.info('membYFP cell/memb ratio: %s, sd %s \n \n' % (np.mean(rel_yfp), np.std(rel_yfp)))


# slice_demo = df_demo.loc[df['angl'] == angl_val]  # and df['sample'] == samp]

# ch1 = np.array(angl_slice.query('channel == "ch1"'))
# ch2 = np.asarray(slice_demo.val[df['channel'] == 'ch2'])

# logging.info('Slice angle %s, size %s px' % (angl_val, np.shape(ch2)[0]))


# memb_loc = ts.membDet(ch2)

# print(memb_loc)


# A-A-A-A-A-A-A-A, eto uzhasno
# bar = []  # memb plot by membDet results
# for i in range(np.shape(ch1)[0]):
# 	if i <= memb_loc[0]:
# 		bar.append(0)
# 	elif i > memb_loc[0] and i < memb_loc[1]:
# 		bar.append(memb_loc[2])
# 	elif i >= memb_loc[1]:
# 		bar.append(0)

# ax = plt.subplot()
# ax.plot(ch1, label='HPCA-TFP')
# ax.plot(ch2, label='membYFP')  # , linestyle='dashed')
# # ax.plot(bar, label='Memb. estimation', linestyle='dashed')
# ax.legend(loc='upper left')

# plt.title('File %s, frame 10' % samp)

# plt.show()
