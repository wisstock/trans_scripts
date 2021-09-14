 #!/usr/bin/env python3
""" Copyright © 2020-2021 Borys Olifirov
Registrations of Fluo-4 loaded cells.

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
from hyst import hystTool as h


plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)


data_path = os.path.join(sys.path[0], 'fluo_data')
res_path = os.path.join(sys.path[0], 'fluo_res')

if not os.path.exists(res_path):
    os.makedirs(res_path)


# options
max_frame_number = 5  # frames after stimulation
cell_name_suffix = '_30_10' # suffix with registration date       
frame_reg_time = 1.0   # frame registration time in seconds
save_csv = False

# hystTool global settings set up
h.settings(sigma=1.5, kernel_size=20, sd_lvl=3, high=0.8, low_init=0.05, mask_diff=30)

# records reading
all_cells = op.WDPars(data_path,
                      max_frame=max_frame_number, name_suffix=cell_name_suffix)

df = pd.DataFrame(columns=['cell', 'power', 'stimul', 'frame', 'time', 'int', 'delta'])
for cell_num in range(0, len(all_cells)):
    cell = all_cells[cell_num]
    logging.info('Image {} in progress'.format(cell.img_name))

    cell_path = f'{res_path}/{cell.img_name}'
    if not os.path.exists(cell_path):
        os.makedirs(cell_path)

    # blue/red derivate images
    der_int = edge.series_derivate(cell.img_series,
                                   mask= 'full_frame',  # cell.mask_series[cell.max_frame_num],
                                   sd_mode='cell',
                                   sd_tolerance=False,
                                   sigma=1, kernel_size=3,
                                   left_w=1, space_w=0, right_w=1)  #,
                                   # output_path=cell_path)
    amp_series = [np.sum(np.abs(der_int[i])) for i in range(len(der_int))]

    # control image of the cell with native image of max frame and hysteresis binary mask
    cell.save_ctrl_img(path=res_path)
    cell_int = cell.max_mask_int(plot_path=cell_path)

    # calc delta F for mask series
    cell_delta = edge.deltaF(cell_int, f_0_win=max_frame_number-1)

    # group all results by mask
    maskres_dict = {'cell':[cell_int, cell_delta]}

    # saving results to CSV
    # NOT READY!
    if save_csv:
      for val_num in range(len(cell_int)):
            int_val = cell_int[val_num]
            delta_val = cell_delta[val_num]
            df = df.append(pd.Series([cell.img_name,  # cell file name
                                      cell.feature,  # 405 nm power
                                      cell.max_frame_num+1,  # number of stimulation frame
                                      int(val_num+1),  # number of frame
                                      round((frame_reg_time * int(val_num+1)) - (frame_reg_time * int(cell.max_frame_num+1)), 2),  # relative time, 0 at stimulation point
                                      int_val,  # mask mean value
                                      delta_val],  # mask delta F value
                           index=df.columns),
                           ignore_index=True)

    plt.figure(figsize=(20,10))
    ax1 = plt.subplot(312)
    ax1.set_title('cell mask deltaF')
    img1 = ax1.plot(cell_delta)

    ax5 = plt.subplot(311)
    ax5.set_title('cell mask')
    img5 = ax5.plot(cell_int)

    ax6 = plt.subplot(313)
    ax6.set_title('abs amplitude')
    img6 = ax6.plot(amp_series)

    plt.suptitle(f'{cell.img_name}, 405 nm power {cell.feature}%')
    plt.savefig(f'{res_path}/{cell.img_name}_summary_plot.png')
    plt.close('all')

if save_csv:
  df.to_csv(f'{res_path}/results.csv', index=False)
  logging.info('CSV file saved')