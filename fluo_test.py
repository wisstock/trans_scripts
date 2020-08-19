#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Test experiment with NP-EGTA + Fluo-4 in HEK cells.
24-27,07.2020

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



data_path = os.path.join(sys.path[0], 'fluo_data')

all_cells = op.WDPars(data_path)
one_cell = 6


df = pd.DataFrame(columns=['cell', 'exp', 'cycl', 'time', 'int'])
for cell_num in range(0, len(all_cells)):
    cell = all_cells[cell_num]
    series_int = cell.relInt()
    for single_num in range(len(series_int)):
        single_int = series_int[single_num]
        df = df.append(pd.Series([int(cell_num+1), cell.exposure, cell.cycles, int(single_num+1), single_int],
                       index=df.columns),
                       ignore_index=True)

df.to_csv('bleaching.csv', index=False)


ax0 = plt.subplot(121)
slc0 = ax0.imshow(all_cells[one_cell].max_frame)
slc0.set_clim(vmin=0, vmax=np.max(all_cells[one_cell].max_frame)) 
div0 = make_axes_locatable(ax0)
cax0 = div0.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slc0, cax=cax0)
ax0.set_title('0 s')

ax1 = plt.subplot(122)
ax1.imshow(all_cells[one_cell].relInt(mask=True))
ax1.set_title('mask')

plt.tight_layout()
plt.show()


