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


def WDPars(wd_path, mode='fluo'):
    """ Parser for data dirrectory and YAML methadata file

    """

    logging.debug('WD path {}'.format(wd_path))

    fluo_list = []

    if mode == 'fluo':
        for root, dirs, files in os.walk(wd_path):
            for file in files:
                if file.endswith('.yml'):
                    file_path = os.path.join(root, file)

                    with open(file_path) as f:
                        data_metha = yaml.safe_load(f)

                    logging.info('Methadata file {} uploaded!'.format(file))



        for root, dirs, files in os.walk(wd_path):  # loop over OIF files
            for file in files:
                data_name = file.split('.')[0]
                data_path = os.path.join(root, file)

                if data_name in data_metha.keys():
                    logging.info(f'File {data_name} prepating!')
                    data_path = os.path.join(root, file)
                    fluo_list.append(FluoData(data_path,
                                              img_name=data_name,
                                              feature=data_metha[data_name],
                                              max_frame=6))

                    logging.info('File {} uploaded!\n'.format(data_path))

    return fluo_list


class OifPars(oif.OifFile):
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
    """ Time series in NP-EGTA + Fluo-4 test experiment series

    """
    def __init__(self, oif_input, img_name, exposure=10, cycles=1, max_frame=6, feature=0, background_rm=True):
        self.img_series = oif.OibImread(oif_input)[0,:,:,:]  # z-stack frames series

        if background_rm:  # background remove option
            for frame in range(0, np.shape(self.img_series)[0]):
                self.img_series[frame] = edge.backCon(self.img_series[frame],
                                                      edge_lim=10,
                                                      dim=2)
            logging.info('Background removed!')
        
        self.img_name = img_name                             # file name
        self.max_frame = self.img_series[max_frame,:,:]      # first frame after 405 nm exposure (max intensity)
        self.exposure = exposure                             # exposure of 405 nm
        self.cycles = cycles                                 # exposures cucles number
        self.feature = feature                               # variable parameter value from YAML file (loading type, stimulation area et. al)

    def relInt(self, high_lim=0.8, init_low=0.05, mask_diff=50, sigma=3, noise_size=40):
        self.noise_sd = np.std(self.max_frame[:noise_size, :noise_size])  # calc noise sd in max imtensity frame in square region
        self.max_gauss = filters.gaussian(self.max_frame, sigma=sigma)

        low_lim = edge.hystLow(self.max_frame, self.max_gauss, sd=self.noise_sd,
                               mode='cell', diff=mask_diff, init_low=init_low, gen_high=high_lim)

        self.cell_mask = filters.apply_hysteresis_threshold(self.max_gauss,
                                                            low=low_lim['2sd']*np.max(self.max_gauss),
                                                            high=high_lim*np.max(self.max_gauss))

        self.rel_int = [round(np.sum(ma.masked_where(~self.cell_mask, img)) / np.sum(self.cell_mask), 3)
                        for img in self.img_series]

        return self.rel_int, self.cell_mask, self.max_gauss


# class FRETData:
#   """ One registation in co-transfected cell after uncaging
#   """

#   def __init__(self, series, native, **kwargs):
#       self.input_series = OifPars(series)  # experimental series data path
#       self.input_native = OifPars(native)  # image at native state data path

#       self.microimg = {
#           'I_DA': oif.OibImread(series)[0,:,0::2,:,:],  # donor excitation/acceptor registration time series
#           'I_DD': oif.OibImread(series)[1,:,0::2,:,:],  # donor excitation/donor registration time series
#           'I_AA': oif.OibImread(series)[0,:,1::2,:, :],  # acceptor excitation/acceptor registration time series
#           'I_DA_native': ,  # donor excitation/acceptor registration at native state image
#           'I_DD_native': ,  # donor excitation/donor registration at native state image
#           'I_AA_native': ,  # acceptor excitation/acceptor registration at native state image
#           'size': ,  # image size in um (x, y, z)
#           'shape': ,  # image shape in px (x, y, z)
#           'lat_res': ,  # lateral resolution in um/px
#           'axi_res': ,  # axial resolution in um/z-step
#           'D_exc': ,  # donor excitation wavelength
#           'D_ems': ,  # donor emission wavelength 
#           'A_exc': ,  # acceptor excitation wavelength
#           'A_ems': ,  # accetor emission wavelength

#       }




if __name__=="__main__":
  pass


# That's all!
