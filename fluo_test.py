#!/usr/bin/env python3
""" Copyright © 2020 Borys Olifirov
Test experiment with NP-EGTA + Fluo-4 in new HEK cells.
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


def deltaF(int_list, f_0_win=2):
    """ Function for colculation ΔF/F0 for data series.
    f_0_win - window for F0 calculation (mean of first 2 values by defoult).

    """
    f_0 = np.mean(int_list[:f_0_win])
    return [(i - f_0)/f_0 for i in int_list[f_0_win:]]


data_path = os.path.join(sys.path[0], 'fluo_data')
res_path = os.path.join(sys.path[0], 'fluo_res')

if not os.path.exists(res_path):
    os.makedirs(res_path)

all_cells = op.WDPars(data_path, max_frame=1,    # FluoData parameters
                      sigma=2, kernel_size=9, sd_area=40, sd_lvl=1.5, high=0.8, low_init=0.05, mask_diff=50, output_path=res_path)  # hystTools parameters

df = pd.DataFrame(columns=['file', 'cell', 'feature', 'time', 'int'])
for cell_num in range(0, len(all_cells)):
    cell = all_cells[cell_num]
    logging.info('Image {} in progress'.format(cell.img_name))

    cell_path = f'{res_path}/{cell.img_name}'
    if not os.path.exists(cell_path):
        os.makedirs(cell_path)

    # delta_int = edge.series_point_delta(cell.img_series, mask_series=cell.mask_series, 
    #                                     baseline_frames=18,
    #                                     delta_min=-0.75, delta_max=0.75,
    #                                     sigma=1, kernel_size=3,
    #                                     output_path=cell_path)

    # der_int = edge.series_derivate(cell.img_series, mask_series=cell.mask_series, mask_num=cell.max_frame_num,
    #                                left_w=4, space_w=2, right_w=4,
    #                                sigma=1, kernel_size=3,
    #                                output_path=cell_path)

    # series_int = cell.sum_int()
    # series_int = deltaF(series_int, f_0_win=3)
    # series_int = cell.frame_mask_int()
    # for single_num in range(len(series_int)):
    #     single_int = series_int[single_num]
    #     df = df.append(pd.Series([cell.img_name, cell.feature, int(single_num+1), single_int],
    #                    index=df.columns),
    #                    ignore_index=True)

    plt.figure()
    ax0 = plt.subplot(121)
    img0 = ax0.imshow(cell.max_frame)
    ax0.axis('off')
    ax0.text(10,10,f'max int frame',fontsize=8)
    ax1 = plt.subplot(122)
    img1 = ax1.imshow(cell.mask_series[0])
    ax1.axis('off')
    ax1.text(10,10,f'binary mask',fontsize=8)
    plt.savefig(f'{cell_path}/{cell.img_name}_mask.png')
    logging.info(f'{cell.img_name} ctrl img saved!\n')


# df.to_csv(f'{res_path}/results.csv', index=False)

















# for cell_img in all_cells:
#   plt.figure()
#   ax0 = plt.subplot(121)
#   img0 = ax0.imshow(cell_img.max_gauss)
#   ax0.text(10,10,cell_img.img_name,fontsize=10)
#   ax0.axis('off')
  
#   ax1 = plt.subplot(122)
#   img1 = ax1.imshow(cell_img.cell_mask)
#   ax1.axis('off')

#   plt.savefig(f'fluo_res/{cell_img.img_name}_max_frame.png')
#   logging.info(f'Frame {cell_img.img_name} saved!')


# ax0 = plt.subplot(131)
# slc0 = ax0.imshow(all_cells[one_cell].max_frame)
# slc0.set_clim(vmin=0, vmax=np.max(all_cells[one_cell].max_frame)) 
# div0 = make_axes_locatable(ax0)
# cax0 = div0.append_axes('right', size='3%', pad=0.1)
# plt.colorbar(slc0, cax=cax0)
# ax0.set_title(all_cells[one_cell].img_name)

# ax1 = plt.subplot(133)
# ax1.imshow(all_cells[one_cell].cell_mask)
# ax1.set_title('mask')

# ax2 = plt.subplot(132)
# slc2 = ax2.imshow(all_cells[one_cell].max_gauss)
# # slc2.set_clim(vmin=0, vmax=np.max(all_cells[one_cell].max_frame)) 
# div2 = make_axes_locatable(ax2)
# cax2 = div2.append_axes('right', size='3%', pad=0.1)
# plt.colorbar(slc2, cax=cax2)
# ax2.set_title('gauss')


# plt.tight_layout()
# plt.show()


