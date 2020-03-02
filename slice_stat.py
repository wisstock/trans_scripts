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

data_name = 'slice_20.csv'

df = pd.read_csv(os.path.join(data_path, data_name))
df = df.dropna()
df = df.reset_index(drop=True)

 
angl_num = 4

angl_list = []
[angl_list.append(i) for i in list(df.angl) if i not in angl_list]  # remove duplications and generate angl list

angl_val = angl_list[angl_num]

logging.info('Avaliable angles: %s' % angl_list)


try:
	angl_val = angl_list[angl_num]
except IndexError:
	print('Angle value NOT avaliable!')
	sys.exit()



for angl_slice in angl_list:  # loop over the slices with particular angles
	pass


samp = '20180718-1315-0007'

df_demo = df.loc[df['sample'] == samp]
slice_demo = df_demo.loc[df_demo['angl'] == angl_val]


ch1 = np.asarray(slice_demo.val[df['channel'] == 'ch1'])
ch2 = np.asarray(slice_demo.val[df['channel'] == 'ch2'])

logging.info('Slice angle %s, size %s px' % (angl_val, np.shape(ch2)[0]))


memb_loc = ts.membDet(ch2)

print(memb_loc)


# ax = plt.subplot()
# ax.plot(ch1, label='HPCA-TFP')
# ax.plot(ch2, label='membYFP', linestyle='dashed')
# ax.legend(loc='upper left')

# plt.title('File %s, frame 10' % samp)


# plt.show()
