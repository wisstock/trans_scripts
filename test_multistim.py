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
import type_multi as rt  # registration type module
import edge              # util functions module


plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)


# I/O options
data_path = os.path.join(sys.path[0], 'data')
res_path = os.path.join(sys.path[0], 'results')

if not os.path.exists(res_path):
    os.makedirs(res_path)

# options
date_name_suffix = '_02_2_2022' # suffix with recording date       
frame_reg_time = 2.0   # frame rate, inter frame time in seconds
save_csv = False

# data frame init
df_profile = pd.DataFrame(columns=['ID',           # recording ID
                                   'power',        # 405 nm stimulation power (%)
                                   'ch',           # channel (FP or Ca dye)
                                   'frame',        # frame num
                                   'time',         # frame time (s)
                                   'mask',         # mask type (master, up, down)
                                   'mask_region',  # mask region (1 for master or down)
                                   'mean',         # mask mean intensity
                                   'delta',        # mask ΔF/F
                                   'rel'])         # mask sum / master mask sum

df_area = pd.DataFrame(columns=['ID',          # recording ID
                                'stim_num',    # stimulation number
                                'stim_frame',  # stimulation frame number
                                'mask',        # mask type (up or down)
                                'area',        # mask region area (in px)
                                'rel_area'])   # mask relative area (mask / master mask)

df_px = pd.DataFrame(columns=['ID',           # recording ID
                              'stim',         # stimulus number
                              'mask_region',  # mask region (1 for master or down)
                              'int',          # px mean intensity
                              'delta'])       # px ΔF/F

# metadata YAML file uploading
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
                                      meta_dict=meta_data[one_dir],
                                      time_scale=0.5)
            record_list.append(one_record)

# records analysis
for record in record_list:
    logging.info(f'Record {record.img_name} in progress')
    record_path = f'{res_path}/{record.img_name}{date_name_suffix}'
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    record.get_master_mask(mask_ext=3, nuclear_ext=3, multi_otsu_nucleus_mask=True)
    record.find_stimul_peak()
    record.peak_img_diff(sigma=1.5, kernel_size=20, baseline_win=6, stim_shift=2, stim_win=3,
                         up_min_tolerance=0.2, up_max_tolerance=0.7,
                         down_min_tolerance=-0.2, down_max_tolerance=-0.1)
    # record.peak_img_deltaF(sigma=1.5, kernel_size=20, baseline_win=6, stim_shift=2, stim_win=3,
    #                        deltaF_up=0.1, deltaF_down=-0.1)
    record.diff_mask_segment(segment_num=6, segment_min_area=30)

    # DEMO FUN
    record.segment_dist_calc()

    # RESULTS OUTPUT
    # record.save_ctrl_profiles(path=record_path)
    # record.save_ctrl_img(path=record_path)

    # df_profile = df_profile.append(record.save_profile_df(id_suffix=date_name_suffix), ignore_index=True)
    # df_area = df_area.append(record.save_area_df(id_suffix=date_name_suffix), ignore_index=True)
    # df_px = df_px.append(record.save_px_df(id_suffix=date_name_suffix), ignore_index=True)

# data frames saving
# df_profile.to_csv(f'{res_path}/profile{date_name_suffix}.csv', index=False)
# df_area.to_csv(f'{res_path}/area{date_name_suffix}.csv', index=False)
# df_px.to_csv(f'{res_path}/px{date_name_suffix}.csv', index=False)