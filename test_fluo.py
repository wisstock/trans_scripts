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

# for single file registrations
all_cells = op.WDPars(data_path,
                      max_frame=20,    # FluoData parameters
                      sigma=1, kernel_size=3, sd_area=40, sd_lvl=5, high=0.8, low_init=0.005, mask_diff=50)  # hystTool parameters

# # for multiple file registrations, merge all files one by one
# all_registrations = op.WDPars(data_path, restrict=True)

df = pd.DataFrame(columns=['file', 'cell', 'frame', 'time', 'int'])
for cell_num in range(0, len(all_cells)):
    cell = all_cells[cell_num]
    logging.info('Image {} in progress'.format(cell.img_name))

    cell_path = f'{res_path}/{cell.img_name}'
    if not os.path.exists(cell_path):
        os.makedirs(cell_path)

    alex_up, alex_down = edge.alex_delta(cell.img_series,
                                         mask=cell.max_frame_mask,
                                         baseline_frames=5,
                                         max_frames=[cell.max_frame_num, 5],
                                         sigma=1, kernel_size=3,
                                         output_path=cell_path)

    up_int, down_int = cell.updown_mask_int(up_mask=alex_up, down_mask=alex_down, plot_path=cell_path)

    cell_int = cell.max_mask_int(plot_path=cell_path)
    # control image of the cell with native image of max frame and hysteresis binary mask
    cell.save_ctrl_img(path=cell_path)

    # frame_int = cell.frame_mask_int(plot_path=res_path)

    # # pixel-wise F-FO/F0 images
    # delta_int = edge.series_point_delta(cell.img_series,
    #                                     mask=cell.mask_series,
    #                                     sigma=1, kernel_size=1,
    #                                     baseline_frames=15,
    #                                     output_path=cell_path)
    
    # blue/red derivate images
    # der_int = edge.series_derivate(cell.img_series,
    #                                mask= 'full_frame',  # cell.mask_series[cell.max_frame_num],
    #                                sd_mode='cell',
    #                                sd_tolerance=False,
    #                                sigma=1, kernel_size=5,
    #                                left_w=1, space_w=0, right_w=1)  # ,
    #                                # output_path=cell_path)
    # abs sum of derivate images intensity for derivate amplitude plot

    plt.figure(figsize=(20,10))
    ax1 = plt.subplot(222)
    ax1.set_title('up mask rel')
    img1 = ax1.plot(up_int/cell_int)
    ax2 = plt.subplot(224)
    ax2.set_title('down mask rel')
    img2 = ax2.plot(down_int/cell_int)
    ax3 = plt.subplot(221)
    ax3.set_title('up mask')
    img3 = ax3.plot(up_int)
    ax4 = plt.subplot(223)
    ax4.set_title('down mask')
    img4 = ax4.plot(down_int)
    plt.savefig(f'{res_path}/{cell.img_name}_rel_updown.png')
    plt.close('all')

    # plt.close('all')
#     for single_num in range(len(series_int)):
#         single_int = series_int[single_num]
#         df = df.append(pd.Series([cell.img_name, cell.feature, int(single_num+1), single_int],  # ['file', 'cell', 'frame', 'time', 'int']
#                        index=df.columns),
#                        ignore_index=True)

# df.to_csv(f'{res_path}/results.csv', index=False)