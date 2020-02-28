#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

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


data_path = os.path.join(sys.path[0], 'confocal_data/HPCA-YFP/')
img_path = os.path.join(sys.path[0], 'confocal_data/HPCA-YFP/demo_dec/')

data_name = 'slice_dec_30.csv'

df = pd.read_csv(os.path.join(data_path, data_name))
df = df.dropna()
df = df.reset_index(drop=True)

samp = '20180718-1315-0007'



df_demo = df.loc[df['sample'] == samp]

angl_list = [] 
[angl_list.append(i) for i in list(df.angl) if i not in angl_list]  # remove duplications and generate angl list

logging.info('Avaliable angles: %s' % angl_list)

for angl_slice in angl_list:  # loop over the slices with particular angles
	pass


slice_demo = df.loc[df['angl'] == angl_list[5]]

print(slice_demo)


ch1 = np.asarray(slice_demo.val[df['channel'] == 'ch1'])
ch2 = np.asarray(slice_demo.val[df['channel'] == 'ch2'])

logging.info('Slice size, ch 1: %s, ch 2: %s' % (np.shape(ch1)[0], np.shape(ch2)[0]))

ax = plt.subplot()
ax.plot(ch1, label='HPCA-TFP')
ax.plot(ch2, label='membYFP', linestyle='dashed')
ax.legend(loc='upper left')

plt.title('File %s, frame 10' % samp)


plt.show()
