#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Test experiment with NP-EGTA + Fluo-4 in old HEK cells.
12-14.10.2020

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
one_cell = 0

# # Fluo-4 bleachin experiment
# df = pd.DataFrame(columns=['cell', 'exp', 'cycl', 'time', 'int'])
# for cell_num in range(0, len(all_cells)):
#     cell = all_cells[cell_num]
#     series_int = cell.relInt()
#     for single_num in range(len(series_int)):
#         single_int = series_int[single_num]
#         df = df.append(pd.Series([int(cell_num+1), cell.exposure, cell.cycles, int(single_num+1), single_int],
#                        index=df.columns),
#                        ignore_index=True)


# PA in loading solution experiment (1 - no PA, 2 - with PA)
df = pd.DataFrame(columns=['cell', 'area', 'time', 'int'])
for cell_num in range(0, len(all_cells)):
    cell = all_cells[cell_num]
    logging.info('Image {} in progress'.format(cell.img_name))

    series_int, mask, gauss = cell.relInt(high_lim=0.8, init_low=0.05, mask_diff=40, sigma=3, noise_size=40)

    # try:                            # register exceptions from lowHyst function
    # 	series_int, mask, gauss = cell.relInt(high_lim=0.8, init_low=0.05, mask_diff=40, sigma=3, noise_size=40)
    # except RuntimeError:
    # 	logging.fatal('For image {} relative intensity DON`T calculated, RE!\n'.format(cell.img_name))
    # 	continue
    # except ValueError:
    # 	logging.fatal('For image {} relative intensity DON`T calculated, VE!\n'.format(cell.img_name))
    # 	continue

    feature = cell.feature
    for single_num in range(len(series_int)):
        single_int = series_int[single_num]
        df = df.append(pd.Series([int(cell_num+1), feature, int(single_num+1), single_int],
                       index=df.columns),
                       ignore_index=True)

df.to_csv('results_1-2.csv', index=False)


ax0 = plt.subplot(231)
slc0 = ax0.imshow(all_cells[one_cell].max_frame)
slc0.set_clim(vmin=0, vmax=np.max(all_cells[one_cell].max_frame)) 
div0 = make_axes_locatable(ax0)
cax0 = div0.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slc0, cax=cax0)
ax0.set_title(all_cells[one_cell].img_name)

ax1 = plt.subplot(233)
ax1.imshow(all_cells[one_cell].cell_mask)
ax1.set_title('mask')

ax2 = plt.subplot(232)
slc2 = ax2.imshow(all_cells[one_cell].max_gauss)
# slc2.set_clim(vmin=0, vmax=np.max(all_cells[one_cell].max_frame)) 
div2 = make_axes_locatable(ax2)
cax2 = div2.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slc2, cax=cax2)
ax2.set_title('gauss')

# int_curve = plt.plot(212)



plt.tight_layout()
plt.show()


