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

df = pd.DataFrame(columns=['file', 'cell', 'feature', 'time', 'int'])
for cell_num in range(0, len(all_cells)):
    cell = all_cells[cell_num]
    logging.info('Image {} in progress'.format(cell.img_name))

    cell_path = f'{res_path}/{cell.img_name}'
    if not os.path.exists(cell_path):
        os.makedirs(cell_path)

    alex_up, alex_down = edge.alex_delta(cell.img_series,
                                         mask=cell.max_frame_mask,
                                         baseline_frames=8,
                                         max_frames=[19, 30],
                                         sd_tolerance=1,
                                         output_path=cell_path)

    cell.updown_mask_int(up_mask=alex_up, down_mask=alex_down, plot_path=cell_path)

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
                                   # output_path=cell_path)
    # abs sum of derivate images intensity for derivate amplitude plot
    # der_amp = [np.sum(np.abs(der_int[i,:,:])) for i in range(len(der_int))]


    # pixel-wise F/F0 images
    # delta_int = edge.series_point_delta(cell.img_series, mask_series=cell.mask_series, 
    #                                     baseline_frames=18,
    #                                     delta_min=-0.75, delta_max=0.75,
    #                                     sigma=1, kernel_size=3,
    #                                     output_path=cell_path)

    cell.max_mask_int(plot_path=cell_path)

    # control image of the cell with native image of max frame and hysteresis binary mask,
    cell.save_ctrl_img(path=res_path)
    # frame_int = cell.frame_mask_int(plot_path=res_path)

    # derivate amplitude plot of image series
    # plt.figure()
    # ax = plt.subplot()
    # img = ax.plot(der_amp)
    # plt.savefig(f'{res_path}/{cell.img_name}_der_amp.png')

    # plt.close('all')

# df.to_csv(f'{res_path}/results.csv', index=False)