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
plt.rcParams['figure.facecolor'] = '#0000002b'
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
                      sigma=3, kernel_size=21, sd_area=40, sd_lvl=1, mean=[140,130], high=0.8, low_init=0.05, mask_diff=10)  # hystTools parameters

df = pd.DataFrame(columns=['file', 'cell', 'feature', 'time', 'int'])
for cell_num in range(0, len(all_cells)):
    cell = all_cells[cell_num]
    logging.info('Image {} in progress'.format(cell.img_name))

    plt.figure()
    ax0 = plt.subplot()
    img0 = ax0.imshow(cell.label_middle_frame)
    roi = patches.Rectangle((cell.custom_center[0] - 5,
                             cell.custom_center[1] - 5),
                            10,10, linewidth=1,edgecolor='r',facecolor='none')
    ax0.add_patch(roi)
    ax0.text(10,10,f'label, frame {cell.middle_frame_num}',fontsize=8)

    plt.savefig(f'{res_path}/{cell.img_name}_ROI.png')
    logging.info(f'{cell.img_name} ROI image saved!\n')


    plt.close('all')


    plt.figure()
    ax0 = plt.subplot(141)
    img0 = ax0.imshow(cell.label_middle_frame)
    ax0.axis('off')
    ax0.text(10,10,f'label, frame {cell.middle_frame_num}',fontsize=8)

    ax1 = plt.subplot(142)
    img1 = ax1.imshow(cell.target_middle_frame)
    ax1.axis('off')
    ax1.text(10,10,f'target',fontsize=8)

    ax3 = plt.subplot(143)
    img3 = ax3.imshow(cell.detection_mask)
    ax3.axis('off')
    ax3.text(10,10,'det mask',fontsize=8)

    ax2 = plt.subplot(144)
    img2 = ax2.imshow(cell.cells_labels)
    ax2.axis('off')
    ax2.text(10,10,f'binary mask',fontsize=8)
    
    plt.savefig(f'{res_path}/{cell.img_name}_ctrl.png')
    logging.info(f'{cell.img_name} control image saved!\n')


    plt.close('all')


    plt.figure()
    ax0 = plt.subplot(141)
    img0 = ax0.imshow(cell.memb_det_masks[0])
    ax0.axis('off')
    ax0.text(10,10,'SD mask',fontsize=8)

    ax1 = plt.subplot(142)
    img1 = ax1.imshow(cell.memb_det_masks[1])
    ax1.axis('off')
    ax1.text(10,10,'ROI mask',fontsize=8)

    ax2 = plt.subplot(143)
    img2 = ax2.imshow(cell.memb_det_masks[2])
    ax2.axis('off')
    ax2.text(10,10,'cytoplasm mask',fontsize=8)

    ax3 = plt.subplot(144)
    img3 = ax3.imshow(cell.memb_det_masks[3])
    ax3.axis('off')
    ax3.text(10,10,'membrane mask',fontsize=8)

    plt.savefig(f'{res_path}/{cell.img_name}_masks.png')
    logging.info(f'{cell.img_name} masks saved!\n')

# df.to_csv(f'{res_path}/results.csv', index=False)
