#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov
PSF visualisation

Usefull links:
https://www.researchgate.net/post/How_are_you_deconvolving_your_confocal_LSM_stacks_for_dendritic_spine_morphological_analyses
https://svi.nl/NyquistCalculator
http://bigwww.epfl.ch/algorithms/psfgenerator/

https://photutils.readthedocs.io/en/stable/psf_matching.html

http://kmdouglass.github.io/posts/implementing-a-fast-gibson-lanni-psf-solver-in-python/


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
	""" Calculate PSF,
	avalianble two models:
	gila - Gibson-Lanni model

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
    # args.update(kwargs)
    # obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
    # return obsvol



psf_size =[216, 216, 19]

args = {
        'shape': (128, 128),  # number of samples in z and r direction
        'dims': (2.0, 2.0),   # size in z and r direction in micrometers
        'ex_wavelen': 488.0,  # excitation wavelength in nanometers
        'em_wavelen': 512.0,  # emission wavelength in nanometers
        'num_aperture': 0.9,
        'refr_index': 1.333,
        'magnification': 60.0,
        'pinhole_radius': 1,  # in micrometers
        'pinhole_shape': 'round',
    }

psf = psfGen(args)  # gila.demo(psf_size)
psf = psf.volume()

psf_size = np.shape(psf)


xy_slice = 128
slice_coord = int(psf_size[1] / 2)

psf_xz = psf[:,slice_coord,:]
psf_xy = psf[xy_slice,:,:]


ax0 = plt.subplot(1,2,1)
slice_0 = ax0.imshow(psf_xy) 
divider_0 = make_axes_locatable(ax0)
cax = divider_0.append_axes("right", size="3%", pad=0.1)
plt.colorbar(slice_0, cax=cax)
ax0.set_title('X-Y, %s px' % xy_slice)

ax1 = plt.subplot(1,2,2)
slice_1 = ax1.imshow(psf_xz)
ax1.plot([0, psf_size[1]-1], [xy_slice, xy_slice])
divider_1 = make_axes_locatable(ax1)
cax = divider_1.append_axes("right", size="3%", pad=0.1)
plt.colorbar(slice_1, cax=cax)
ax1.set_title('X-Z, middle')

plt.show()


# That's all!