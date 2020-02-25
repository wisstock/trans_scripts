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


def psfRiWo(setting, ems=False):
	""" Calculate Richards-Wolf PSF model

    return PSF as numpy array, values was normalised from 0 to 1

	"""
	args = {'shape': (256, 256),  # number of samples in z and r direction
	        'dims': (5.0, 5.0),  # size in z and r direction in micrometers
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
	return confocal_psf.volume()

def pdfGila(setting):
    """ Calculate Gibson-Lanni PSF model
    require gila.py module

    return PSF as numpy array, values was normalised from 0 to 1

    """
    args = {'size_x': 128,
            'size_y': 128,
            'size_z': 128,
            'num_basis': 100,  # Number of rescaled Bessels that approximate the phase function
            'num_samples': 1000,  # Number of pupil samples along radial direction
            'oversampling': 2,  # Defines the upsampling ratio on the image space grid for computations
            'NA': 0.9,
            'wavelength': 0.512,  # microns
            'M': 60,  # magnification
            'ns': 1.33,  # specimen refractive index (RI)
            'ng0': 1.5,  # coverslip RI design value
            'ng': 1.5,  # coverslip RI experimental value
            'ni0': 1.33,  # immersion medium RI design value
            'ni': 1.33,  # immersion medium RI experimental value
            'ti0': 150,  # microns, working distance (immersion medium thickness) design value
            'tg0' :170,  # microns, coverslip thickness design value
            'tg': 170,  # microns, coverslip thickness experimental value
            'res_lateral': 0.1,  # microns
            'res_axial': 0.1,  # microns
            'pZ': 2,  # microns, particle distance from coverslip
            'min_wavelength': 0.488}  # scaling factors for the Fourier-Bessel series expansion, microns

    args.update(setting)

    return gila.generate(args)



psf_size =[255, 255, 17]

args = {'shape': (9, 100),  # number of samples in z and r direction
        'dims': (5, 2.0),   # size in z and r direction in micrometers
        'ex_wavelen': 488.0,  # excitation wavelength in nanometers
        'em_wavelen': 512.0,  # emission wavelength in nanometers
        'num_aperture': 0.9,
        'refr_index': 1.333,
        'magnification': 60.0,
        'pinhole_radius': 0.01,  # in micrometers
        'pinhole_shape': 'round'}

args_1 = {'shape': (9, 128),  # number of samples in z and r direction
          'dims': (2.0, 2.0),   # size in z and r direction in micrometers
          'ex_wavelen': 454.0,  # excitation wavelength in nanometers
          'em_wavelen': 512.0,  # emission wavelength in nanometers
          'num_aperture': 0.9,
          'refr_index': 1.333,
          'magnification': 60.0,
          'pinhole_radius': 1,  # in micrometers
          'pinhole_shape': 'round'}


psf_rw_file = '/home/astria/Bio/Lab/scripts/trans_scripts/.temp/data/psf_rw.tif'
psf_gl_file = '/home/astria/Bio/Lab/scripts/trans_scripts/.temp/data/psf_gl.tif'


psf_0 = psfGen(args)  # Richards-Wolf model 
psf_0 = psf_0.volume()  # extract PSF as np array

tifffile.imsave(psf_rw_file, psf_0)

# psf_1 = psfGen(args_1, True)  # Richards-Wolf model with diff options (arg_1)
# psf_1 = psf_1.volume()

psf_1 = gila.demo(psf_size)  # Gibson-Lanni model
psf_size = np.shape(psf_0)

xy_slice = 9
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


# That's all!