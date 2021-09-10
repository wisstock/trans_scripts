 #!/usr/bin/env python3
""" Copyright © 2020-2021 Borys Olifirov
Registration of homogeneous fluoresced cells (Fluo-4, low range of the HPCA translocation) analysis.

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
res_path = os.path.join(sys.path[0], 'fluo_res')

if not os.path.exists(res_path):
    os.makedirs(res_path)


# main options
max_frame_number = 20  # frames after stimulation
cell_name_suffix = '_27_01' # suffix with registration date       
frame_reg_time = 1.0   # frame registration time in seconds
save_csv = False

# for single file registrations
all_cells = op.WDPars(data_path,
                      max_frame=max_frame_number, name_suffix=cell_name_suffix,    # FluoData parameters
                      sigma=1, kernel_size=3, sd_area=40, sd_lvl=5, high=0.8, low_init=0.005, mask_diff=50)  # hystTool parameters

# # for multiple file registrations, merge all files one by one
# all_registrations = op.WDPars(data_path, restrict=True)

df = pd.DataFrame(columns=['cell', 'power', 'stimul', 'frame', 'time', 'mask', 'int', 'delta', 'rel'])
for cell_num in range(0, len(all_cells)):
    cell = all_cells[cell_num]
    logging.info('Image {} in progress'.format(cell.img_name))

    cell_path = f'{res_path}/{cell.img_name}'
    if not os.path.exists(cell_path):
        os.makedirs(cell_path)

    # alex mask images
    alex_up, alex_down, base_win, max_win = edge.alex_delta(cell.img_series,
                                                            mask=cell.max_frame_mask,
                                                            win_index=[10, cell.max_frame_num+1, 5],
                                                            spacer=2,
                                                            sigma=False, kernel_size=10,
                                                            mode='multiple',
                                                            output_path=cell_path)
    up_int, down_int = cell.updown_mask_int(up_mask=alex_up, down_mask=alex_down, plot_path=cell_path)


    # blue/red derivate images
    der_int = edge.series_derivate(cell.img_series,
                                   mask= 'full_frame',  # cell.mask_series[cell.max_frame_num],
                                   sd_mode='cell',
                                   sd_tolerance=False,
                                   sigma=1, kernel_size=3,
                                   left_w=1, space_w=0, right_w=1)  # ,
                                   # output_path=cell_path)
    amp_series = [np.sum(np.abs(der_int[i])) for i in range(len(der_int))]

    # control image of the cell with native image of max frame and hysteresis binary mask
    cell.save_ctrl_img(path=res_path)
    cell_int = cell.max_mask_int(plot_path=cell_path)

    # calc relative amount in up mask
    up_rel = edge.deltaF(up_int / cell_int, f_0_win=10)
    down_rel = edge.deltaF(down_int / cell_int, f_0_win=10)

    # calc delta F for mask series
    cell_delta = edge.deltaF(cell_int, f_0_win=10)
    up_delta = edge.deltaF(up_int, f_0_win=10)
    down_delta = edge.deltaF(down_int, f_0_win=10)
    # group all results by mask
    maskres_dict = {'cell':[cell_int, cell_delta, cell_int],
                 'up':[up_int, up_delta, up_rel],
                 'down':[down_int, down_delta, down_rel]}

    # saving results to CSV
    if save_csv:
      for val_num in range(len(cell_int)):
          for maskres_key in maskres_dict.keys():
              maskres = maskres_dict.get(maskres_key)
              int_val = maskres[0][val_num]
              delta_val = maskres[1][val_num]
              rel_val = maskres[2][val_num]
              df = df.append(pd.Series([cell.img_name,  # cell file name
                                        cell.feature,  # 405 nm power
                                        cell.max_frame_num+1,  # number of stimulation frame
                                        int(val_num+1),  # number of frame
                                        round((frame_reg_time * int(val_num+1)) - (frame_reg_time * int(cell.max_frame_num+1)), 2),  # relative time, 0 at stimulation point
                                        maskres_key,  # mask type
                                        int_val,  # mask mean value
                                        delta_val,  # mask delta F value
                                        rel_val],  # up/down mask intensity relative to cell mask intensity 
                             index=df.columns),
                             ignore_index=True)

    plt.figure(figsize=(20,10))
    ax1 = plt.subplot(426)
    ax1.set_title('up mask rel')
    img1 = ax1.plot(up_rel)

    ax2 = plt.subplot(428)
    ax2.set_title('down mask rel')
    img2 = ax2.plot(down_rel)

    ax3 = plt.subplot(425)
    ax3.set_title('up mask')
    img3 = ax3.plot(up_int)

    ax4 = plt.subplot(427)
    ax4.set_title('down mask')
    img4 = ax4.plot(down_int)

    ax5 = plt.subplot(411)
    ax5.set_title('cell mask')
    ax5.fill_between(x=range(base_win[0], base_win[1], 1),
                     y1=min(cell_int),
                     y2=max(cell_int),
                     alpha=0.35,
                     label='baseline')
    ax5.fill_between(x=range(max_win[0], max_win[1], 1),
                     y1=min(cell_int),
                     y2=max(cell_int),
                     alpha=0.35,
                     label='max')
    img5 = ax5.plot(cell_int)

    ax6 = plt.subplot(412)
    ax6.set_title('abs amplitude')
    img6 = ax6.plot(amp_series)
    plt.savefig(f'{res_path}/{cell.img_name}_rel_updown.png')
    plt.close('all')

if save_csv:
  df.to_csv(f'{res_path}/results.csv', index=False)
  logging.info('CSV file saved')