#!/usr/bin/env python3

'''
Copyright © 2020 Borys Olifirov

Read tiff files from data folder and z-stack data manipulations

'''

import os
import sys
import logging

# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np

from skimage.external import tifffile

from skimage.filters import scharr


def oneTiff(file_name, channel=0, frame_number=0, camera_offset=250):
    wd_path = os.path.split(os.getcwd())
    start_path = os.getcwd()
    os.chdir(wd_path[0] + '/temp/data/')
    tiff_tensor = tifffile.imread(file_name)
    print(tiff_tensor.shape, np.max(tiff_tensor))

    channel_one = tiff_tensor[0::2, :, :]
    channel_two = tiff_tensor[1::2, :, :]

    if channel == 0:
    	frame_amount = channel_one.shape

    	try:
    		img = channel_one[frame_number] - camera_offset
    		os.chdir(start_path)
    		return(img)
    	except ValueError:
    		if frame_number > frame_amount[0]:
    			print("Frame number out of range!")
    else:
    	frame_amount = channel_two.shape

    	try:
    		img = channel_two[frame_number] - camera_offset
    		os.chdir(start_path)
    		return(img)
    	except ValueError:
    		if frame_number > frame_amount[0]:
    			print("Frame number out of range!")

def getTiff(file_name, channel=0, frame_number=0, camera_offset=250):
    """ Function returns individual frame from image series and
    apply compensation of camera offset value for this frame.
    Separate two fluorecent channels (first start from 0, second - from 1).
    For Dual View system data.

    """

    path = os.getcwd() + '/temp/data/' + file_name

    tiff_tensor = tifffile.imread(path)
    # print(tiff_tensor.shape, np.max(tiff_tensor))

    channel_one = tiff_tensor[0::2, :, :]
    channel_two = tiff_tensor[1::2, :, :]
    # print(channel_one.shape)
    # print(channel_two.shape)

    if channel == 0:
        frame_amount = channel_one.shape
        try:
            img = channel_one[frame_number] - camera_offset
            return(img)
        except ValueError:
            if frame_number > frame_amount[0]:
                print("Frame number out of range!")
    else:
        frame_amount = channel_two.shape
        try:
            img = channel_two[frame_number] - camera_offset
            return(img)
        except ValueError:
            if frame_number > frame_amount[0]:
                print("Frame number out of range!")

def readZ(path):
    """ Read two z-stacks (membYFP & HPCA-TFP) from input folder (path)
    in two numpy arrays.

    Default name format:
    "HPCATFP" - HPCA-TFP channel file
    "membYFP" - membYFP channel file

    """

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.tif'):

                samp = file.split('_')[0]
                file_path = os.path.join(root, file)
                img = tifffile.imread(file_path)

                if file.split('.')[0] == 'HPCATFP':
                    hpca = img
                    logging.info('HPCA-TFP data uploaded')
                elif file.split('.')[0] == 'membYFP':
                    yfp = img
                    logging.info('membYFP data uploaded')
                else:
                    logging.error('INCORRECT channels notation!')
                    sys.exit()

    return(yfp, hpca)


if __name__=="__main__":
    pass


# That's all!