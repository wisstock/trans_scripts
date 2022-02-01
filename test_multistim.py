#!/usr/bin/env python3
""" Copyright © 2020-2022 Borys Olifirov
Registrations of cells with both Fluo-4 loading and HPCA-TagRFP transfection and repetitive stimulations.

"""

import sys
import os
import logging
import yaml

import numpy as np
import numpy.ma as ma
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage import filters
from skimage import measure

sys.path.append('modules')
# import oiffile as oif
import reg_type as rt
import edge


plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)


# I/O
data_path = os.path.join(sys.path[0], 'data')
res_path = os.path.join(sys.path[0], 'results')

if not os.path.exists(res_path):
    os.makedirs(res_path)

# options
date_name_suffix = '_11_27_2021' # suffix with recording date       
frame_reg_time = 2.0   # frame rate, inter frame time in seconds
save_csv = True

# metadata YAML file reading
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith('.yml') or file.endswith('.yaml'):
            meta_file_path = os.path.join(root, file)
            with open(meta_file_path) as f:
                meta_data = yaml.safe_load(f)
            logging.info(f'Metadata file {file} uploaded')

# records uploading
record_list = []
for root, dirs, files in os.walk(data_path):  # loop over OIF files
    for one_dir in dirs:
        if one_dir in meta_data.keys():
            one_record = rt.MultiData(oif_path=os.path.join(root, one_dir),
                                      img_name=one_dir,
                                      meta_dict=meta_data[one_dir])
            record_list.append(one_record)

for record in record_list:
    record_path = f'{res_path}/{record.img_name}{date_name_suffix}'
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    # DERIVATE SEXTION
    # derivate series building
    ca_der_series, ca_der_profile = edge.series_derivate(record.ca_series,
                                                         mask= 'full_frame',
                                                         sd_mode='cell',
                                                         sd_tolerance=False,
                                                         sigma=1, kernel_size=3,
                                                         left_w=1, space_w=0, right_w=1)
    prot_der_series, prot_der_profile = edge.series_derivate(record.prot_series,
                                                             mask= 'full_frame',
                                                             sd_mode='cell',
                                                             sd_tolerance=False,
                                                             sigma=1, kernel_size=3,
                                                             left_w=1, space_w=0, right_w=1)
    # derivate plot saving
    plt.figure()
    ax0 = plt.subplot(211)
    ax0.set_title('Ca dye channel derivate')
    img0 = ax0.plot(ca_der_profile)

    ax1 = plt.subplot(212)
    ax1.set_title('FP channel derivate')
    img1 = ax1.plot(prot_der_profile)

    plt.suptitle(f'{record.img_name}{date_name_suffix}, {record.stim_power}%, {record.baseline_frames}|{record.stim_loop_num}x {record.stim_frames}|{record.tail_frames}')
    plt.tight_layout()
    plt.savefig(f'{record_path}/{record.img_name}{date_name_suffix}_total_der.png')
    plt.close('all')

    # ANALYSIS SECTION
    # cell mask building, Ca dye channel
    record.get_master_mask(sigma=3.5, kernel_size=10, mask_ext=5)
    record.ca_profile()
    record.prot_profile()
    record.get_delta_mask(sigma=3.5, kernel_size=10, stim_shift=1, loop_win_frames=2, path=record_path)
    record.save_ctrl_img(path=record_path, time_scale=0.5)

    # save_ctrl_img(path=res_path)

    # # up/down masks building, fluorescent protein channel
    # alex_up, alex_down, base_win, max_win = edge.alex_delta(record.prot_series,
    #                                                         mask=record.master_mask,
    #                                                         baseline_win=[0, 5],
    #                                                         max_win=[25, 30],
    #                                                         sigma=2, kernel_size=20,
    #                                                         output_path=record_path)