#!/usr/bin/env python3

""" Copyright © 2020-2021 Borys Olifirov

OIFfile package extension & OIF files parsing

"""

import sys
import os
import logging
import yaml

import oiffile as oif
import reg_type


def WDPars(wd_path, mode='single', name_suffix=None, **kwargs):
    """ Parser for data dirrectory and YAML methadata file

    """
    data_list = []
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

            if data_name in data_metha.keys():
                data_path = os.path.join(root, file)
                logging.info(f'File {data_name} in uploading')
                if mode == 'single':
                    data_list.append(reg_type.FluoData(oif_path=data_path,
                                              img_name=data_name+name_suffix,
                                              feature=data_metha[data_name],
                                              **kwargs))
                elif mode == 'z':
                    data_list.append(reg_type.MembZData(oif_path=data_path,
                                              img_name=data_name+name_suffix,
                                              feature=data_metha[data_name],
                                              **kwargs))
                elif mode == 'memb':
                    data_list.append(reg_type.MembData(oif_path=data_path,
                                                       img_name=data_name+name_suffix,
                                                       feature=data_metha[data_name],
                                                       **kwargs))
                elif mode == 'FRET':
                    data_list.append(reg_type.FRETData(oif_path=data_path,
                                                       img_name=data_name+name_suffix,
                                                       feature=data_metha[data_name],
                                                       **kwargs))
                else:
                    logging.fatal('Incorrect registration type!')

    return data_list


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


if __name__=="__main__":
  pass


# That's all!