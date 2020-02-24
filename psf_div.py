#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov
Estimate distimct between other PSF calculation models

"""

import sys
import logging

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from skimage import data
from skimage import exposure
from skimage import filters
from skimage.filters import scharr
from skimage.external import tifffile
from skimage import restoration

import psf

sys.path.append('modules')
import gila

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'


def psfGen(setting, ems=False):
	""" Calculate Richards-Wolf PSF model

	"""
	args = {'shape': (256, 256),
	        'dims': (5.0, 5.0),
	        'ex_wavelen': 488.0,
	        'em_wavelen': 520.0,
	        'num_aperture': 1.2,
	        'refr_index': 1.333,
	        'magnification': 1.0,
	        'pinhole_radius': 0.05,
	        'pinhole_shape': 'square'}

	args.update(setting)

	if ems:
		em_psf = psf.PSF(psf.ISOTROPIC| psf.EMISSION, **args)
		args.update({'empsf': em_psf})

	confocal_psf = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
	return confocal_psf


psf_size =[255, 255, 255]

args = {'shape': (128, 128),  # number of samples in z and r direction
        'dims': (5, 2.0),   # size in z and r direction in micrometers
        'ex_wavelen': 488.0,  # excitation wavelength in nanometers
        'em_wavelen': 512.0,  # emission wavelength in nanometers
        'num_aperture': 0.9,
        'refr_index': 1.333,
        'magnification': 60.0,
        'pinhole_radius': 0.01,  # in micrometers
        'pinhole_shape': 'round'}

args_1 = {'shape': (128, 128),  # number of samples in z and r direction
          'dims': (2.0, 2.0),   # size in z and r direction in micrometers
          'ex_wavelen': 454.0,  # excitation wavelength in nanometers
          'em_wavelen': 512.0,  # emission wavelength in nanometers
          'num_aperture': 0.9,
          'refr_index': 1.333,
          'magnification': 60.0,
          'pinhole_radius': 1,  # in micrometers
          'pinhole_shape': 'round'}


psf_0 = psfGen(args)  # Richards-Wolf model 
psf_0 = psf_0.volume()  # extract PSF as np array

# psf_1 = psfGen(args_1, True)  # Richards-Wolf model with diff options (arg_1)
# psf_1 = psf_1.volume()

psf_1 = gila.demo(psf_size)  # Gibson-Lanni model
psf_size = np.shape(psf_0)

xy_slice = 128
slice_coord = int(psf_size[1] / 2)

psf_xz = psf_0[:,slice_coord,:]
psf_xy = psf_0[xy_slice,:,:]

psf_xz_1 = psf_1[:,slice_coord,:]

model_div = psf_xz_1 - psf_xz


ax = plt.subplot()
slic = ax.imshow(model_div)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slic, cax=cax)

plt.show()