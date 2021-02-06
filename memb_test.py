#!/usr/bin/env python3
""" Copyright © 2021 Borys Olifirov
Registration of co-transferenced (HPCA-TagRFP + EYFP-Mem) cells.

"""

import sys
import os
import logging

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('modules')
import oifpars as op
import edge

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)

data_path = os.path.join(sys.path[0], 'data')
res_path = os.path.join(sys.path[0], 'memb_res')

if not os.path.exists(res_path):
    os.makedirs(res_path)

all_cells = op.WDPars(wd_path=data_path, mode='z',
	                  middle_frame=6,  # MembData parameters
                      sigma=2, kernel_size=5, sd_area=40, sd_lvl=0.5, high=0.8, low_init=0.05, mask_diff=50)  # hystTools parameters

df = pd.DataFrame(columns=['file', 'cell', 'feature', 'time', 'int'])
for cell_num in range(0, len(all_cells)):
    cell = all_cells[cell_num]
    logging.info('Image {} in progress'.format(cell.img_name))

    plt.figure()
    ax0 = plt.subplot(121)
    img0 = ax0.imshow(cell.max_frame)
    ax0.axis('off')
    ax0.text(10,10,f'max int frame',fontsize=8)
    ax1 = plt.subplot(122)
    img1 = ax1.imshow(cell.mask_series[0])
    ax1.axis('off')
    ax1.text(10,10,f'binary mask',fontsize=8)
    plt.savefig(f'{res_path}/{cell.img_name}_ctrl.png')
    logging.info(f'{cell.img_name} control image saved!\n')

# df.to_csv(f'{res_path}/results.csv', index=False)
