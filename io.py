#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

OIF test

"""

import sys
import os
import logging

import numpy as np

sys.path.append('modules')
import oiffile as oif

data_path = os.path.join(sys.path[0], 'raw_data/cell1/cell1_01.oif')


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
		size['Z'] = float(self.mainfile['Axis 3 Parameters Common']['Interval'])

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


# class MicroImg:
# 	def __init__(self, filename, **kwargs):
# 		self.input_file = OifPars(filename)

# 		self.microimg = {
# 		    'donor_ch'
# 		    'acceptor_ch'
# 		}






data = OifPars(data_path)

print(data.axes)
print(data.geometry)
print(data.shape)
print(data.channels)