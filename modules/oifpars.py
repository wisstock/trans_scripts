#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

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


class OifExt(oif.OifFile):
    """ Inheritance and extension of the functionality
    of the OfFice class. Methods for extracting file metadata.

    """
    @property
    def geometry(self):
        """Return linear size in um from mainfile"""
        size = {
            self.mainfile[f'Axis {i} Parameters Common']['AxisCode']:
            float(self.mainfile[f'Axis {i} Parameters Common']['EndPosition'])
            for i in [0, 1]
        }
        size['Z'] = float(self.mainfile['Axis 3 Parameters Common']['Interval']*self.mainfile['Axis 3 Parameters Common']['MaxSize'] / 1000)

        return tuple(size[ax] for ax in ['X', 'Y', 'Z'])

    @property
    def px_size(self):
        """Return linear px size in nm from mainfile"""
        size = {
            self.mainfile[f'Axis {i} Parameters Common']['AxisCode']:
            float(self.mainfile[f'Axis {i} Parameters Common']['EndPosition'] / self.mainfile[f'Axis {i} Parameters Common']['MaxSize'] * 1000)
            for i in [0, 1]     
        }
        size['Z'] = float(self.mainfile['Axis 3 Parameters Common']['Interval'] / 1000)

        return tuple(size[ax] for ax in ['X', 'Y', 'Z'])

    @property
    def lasers(self):
        """Return active lasers and theyir transmissivity lyst from mainfile"""
        laser = {
            self.mainfile[f'Laser {i} Parameters']['LaserWavelength']:
            int(self.mainfile[f'Laser {i} Parameters']['LaserTransmissivity']/10)
            for i in range(5) 
        }
        laser_enable = [self.mainfile[f'Laser {i} Parameters']['LaserWavelength']
                        for i in range(5)
                        if self.mainfile[f'Laser {i} Parameters']['Laser Enable'] == 1]


        return tuple([i, laser[i]] for i in laser.keys() if i in laser_enable)

    @property
    def channels(self):
        """Return list of active channel (return laser WL and intensity) from minefile"""
        active_ch = [self.mainfile[f'GUI Channel {i} Parameters']['ExcitationWavelength']
                     for i in range(1, 4)
                     if self.mainfile[f'GUI Channel {i} Parameters']['CH Activate'] == 1]
        laser = {
            self.mainfile[f'GUI Channel {i} Parameters']['ExcitationWavelength']:
            int(self.mainfile[f'GUI Channel {i} Parameters']['LaserNDLevel']/10)
            for i in range(1, 4)
        }

        return tuple([ch, laser[ch]] for ch in active_ch)


class FluoData:
    """ Time series of homogeneous fluoresced cells (Fluo-4,  low range of the HPCA translocation).

    """
    def __init__(self, oif_path, img_name, feature=0, max_frame=6, sigma=3, noise_size=40,
                 background_rm=True,
                 **kwargs):
        self.img_series = oif.OibImread(oif_path)[0,:,:,:]                          # z-stack frames series
        if background_rm:                                                           # background remove option
            for frame in range(0, np.shape(self.img_series)[0]):
                self.img_series[frame] = edge.back_rm(self.img_series[frame],
                                                      edge_lim=10,
                                                      dim=2)
        self.img_name = img_name                                                     # file name
        self.max_frame = self.img_series[max_frame,:,:]                              # first frame after 405 nm exposure (max intensity) or first frame (for FP)
        self.feature = feature                                                       # variable parameter value from YAML file (loading type, stimulation area, exposure per px et. al)
        self.noise_sd = np.std(self.max_frame[:noise_size, :noise_size])             # noise sd in max intensity frame in square region
        self.max_gauss = filters.gaussian(self.max_frame, sigma=sigma)               # create gauss blured image for thresholding

        self.cell_detector = edge.hystTool(self.max_frame, sigma, self.noise_sd, **kwargs)  # detect all cells in max frame
        # self.max_frame_mask, self.all_cells_mask = self.cell_detector.cell_mask(mode='multi')

        if self.cell_detector.cells_num == 1:
            self.mask_series = []
            for i in range(np.shape(self.img_series)[0]):
                self.frame_img =self.img_series[i]
                self.frame_cell = edge.hystTool(self.frame_img, sigma, self.noise_sd, **kwargs)
                self.frame_mask, self.frame_whole_mask = self.frame_cell.cell_mask()
                self.mask_series.append(self.frame_mask)
        elif self.cell_detector.cells_num > 1:
            self.cells = self.cell_detector.cells_labels

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


class FRETData():
    """ Time series of FRET registration. Include both donor and acceptor image series.

    """
    def __init__(self):
        pass
        


if __name__=="__main__":
  pass


# That's all!
