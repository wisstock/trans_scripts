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
save_csv = False

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
    logging.info(f'Record {record.img_name} in progress')
    record_path = f'{res_path}/{record.img_name}{date_name_suffix}'
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    record.get_master_mask(mask_ext=5)
    record.find_stimul_peak()
    record.peak_img_diff(sigma=2, kernel_size=20, baseline_win=6, stim_shift=2, stim_win=3, path=record_path)
    record.peak_img_deltaF(sigma=1, kernel_size=20, baseline_win=6, stim_shift=2, stim_win=3, path=record_path)
    record.save_ctrl_img(path=record_path, time_scale=0.5)

    # plt.figure(figsize=(15,4))
    # plt.plot(edge.deltaF(record.prot_profile()), label='FP')
    # plt.plot(edge.deltaF(record.ca_profile()), label='Ca')
    # plt.grid(visible=True, linestyle=':')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
