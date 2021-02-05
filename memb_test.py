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

all_cells = op.WDPars(wd_path=data_path, mode='memb',
	                  max_frame=1,  # MembData parameters
                      sigma=2, kernel_size=5, sd_area=40, sd_lvl=0.5, high=0.8, low_init=0.05, mask_diff=50)  # hystTools parameters
