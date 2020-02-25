#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov
PDF generating as TIF files
and stimation of distincts between different PSF calculation models.

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

sys.path.append('modules')
import getpsf as psf

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'


rw_args = {'shape': (64, 64),  # number of samples in z and r direction
        'dims': (5, 2.0),   # size in z and r direction in micrometers
        'ex_wavelen': 488.0,  # excitation wavelength in nanometers
        'em_wavelen': 512.0,  # emission wavelength in nanometers
        'num_aperture': 0.9,
        'refr_index': 1.333,
        'magnification': 60.0,
        'pinhole_radius': 0.01,  # in micrometers
        'pinhole_shape': 'round'}

gl_args = {'size_x': 127,
           'size_y': 127,
           'size_z': 127,
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


psf_rw_file = '/home/astria/Bio/Lab/scripts/trans_scripts/.temp/data/psf_rw.tif'
psf_gl_file = '/home/astria/Bio/Lab/scripts/trans_scripts/.temp/data/psf_gl.tif'


psf_rw = psf.psfRiWo(rw_args)  # Richards-Wolf model 
psf_gl = psf.psfGiLa(gl_args)  # Gibson-Lanni model

tifffile.imsave(psf_rw_file, psf_rw)  # save models
tifffile.imsave(psf_gl_file, psf_gl)

psf_size = gl_args['size_x']

# xy_slice = 9
slice_coord = int(psf_size / 2)

psf_rw_xz = psf_rw[:,slice_coord,:]
# psf_xy = psf_0[xy_slice,:,:]

psf_gl_xz = psf_gl[:,slice_coord,:]

model_div = psf_rw_xz - psf_gl_xz


ax = plt.subplot()
slic = ax.imshow(model_div)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slic, cax=cax)

plt.show()


# That's all!