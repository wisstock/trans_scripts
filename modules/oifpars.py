#!/usr/bin/env python3

""" Copyright © 2020-2021 Borys Olifirov

OIF test

"""

import sys
import os
import logging

import numpy as np
import numpy.ma as ma
import yaml
import pandas as pd
from skimage import filters
from skimage import measure

import oiffile as oif
import edge


def WDPars(wd_path, **kwargs):
    """ Parser for data dirrectory and YAML methadata file

    """
    fluo_list = []
    for root, dirs, files in os.walk(wd_path):
        for file in files:
            if file.endswith('.yml') or file.endswith('.yaml'):
                file_path = os.path.join(root, file)

                with open(file_path) as f:
                    data_metha = yaml.safe_load(f)

                logging.info(f'Methadata file {file} uploaded')

    for root, dirs, files in os.walk(wd_path):  # loop over OIF files
        for file in files:
            data_name = file.split('.')[0]
            data_path = os.path.join(root, file)

            if data_name in data_metha.keys():
                data_path = os.path.join(root, file)
                logging.info(f'File {data_name} in progress')
                fluo_list.append(FluoData(oif_path=data_path,
                                          img_name=data_name,
                                          feature=data_metha[data_name],
                                          **kwargs))
    return fluo_list


class FluoData():
    """ Time series of homogeneous fluoresced cells (Fluo-4,  low range of the HPCA translocation).

    """
    def __init__(self, oif_path, img_name, feature=False, max_frame=6,
                 background_rm=True, 
                 restrict=False,
                 img_series=False,
                 **kwargs):
        # restricted variant for multiple file registration, next step - connect all registration to one FluoData object with fluo_ext function
        if restrict:
            pass
        # NOT READY!
        #     self.img_name = img_name
        #     self.img_series = oif.OibImread(oif_path)[0,:,:,:]                          # z-stack frames series
        #     if background_rm:                                                           # background remove option
        #         for frame in range(0, np.shape(self.img_series)[0]):
        #         self.img_series[frame] = edge.back_rm(self.img_series[frame],
        #                                               edge_lim=10,
        #                                               dim=2)
        #     self.feature = feature
        # # full variant for one-file registration
        else:
            if not img_series:
                self.img_series = oif.OibImread(oif_path)[0,:,:,:]                          # z-stack frames series
                if background_rm:                                                           # background remove option
                    for frame in range(0, np.shape(self.img_series)[0]):
                        self.img_series[frame] = edge.back_rm(self.img_series[frame],
                                                              edge_lim=10,
                                                              dim=2)
            else:
                self.img_series = img_series
            self.img_name = img_name
            self.max_frame_num = max_frame                                                     # file name
            self.max_frame = self.img_series[max_frame,:,:]                              # first frame after 405 nm exposure (max intensity) or first frame (for FP)
            # self.feature = feature                                                       # variable parameter value from YAML file (loading type, stimulation area, exposure per px et. al)
            # self.noise_sd = np.std(self.max_frame[:noise_size, :noise_size])             # noise sd in max intensity frame in square region
            # self.max_gauss = filters.gaussian(self.max_frame, sigma=sigma)               # create gauss blured image for thresholding

            self.cell_detector = edge.hystTool(self.max_frame, **kwargs)  # detect all cells in max frame
            # self.max_frame_mask, self.all_cells_mask = self.cell_detector.cell_mask(mode='multi')
            self.mask_series = self.cell_detector.cell_mask(self.img_series)

    def max_mask_int(self):
        """ Calculation mean intensity  in masked area along frames series.
        Mask was created by max_frame image.

        """
        return edge.series_sum_int(self.img_series, self.max_frame_mask)

    def frame_mask_int(self):
        """ Calculation mean intensity  in masked area along frames series.
        Mask was created for each frame individually.

        """
        mean_series = []
        for i in range(len(self.img_series)):
            img = self.img_series[i]
            mask = self.mask_series[i]
            mean_series.append(round(np.sum(ma.masked_where(~mask, img)) / np.sum(mask), 3))
        return mean_series

    def save_int():
        """ Saving mask intensity results to CSV table.

        """
        pass


def fluo_ext(reg_list, **kwargs):
    """ Extension for FluoData class, require list of FluoData objects to create single output image series.
    Return new FluoData object.
    """
    cells_name = list(dict.fromkeys([reg.img_name.split('_')[0] for reg in reg_list]))  # create list of cells name (remove repeated names)
    print(cells_name)

    return [FluoData(kwargs) ]
    

class YFPmem():
    """ Registration of co-transferenced (TFP + EYFP-Mem) cells.
    T-axis of file represent excitation laser combination (1 - 456+488, 456, 488).

    """
    def __init__(self):
        pass


class FRETData():
    """ Time series of FRET registration. Include both donor and acceptor image series.

    """
    def __init__(self):
        pass
        


if __name__=="__main__":
  pass


# That's all!
