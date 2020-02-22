#!/usr/bin/env python3

""" Deconvolutio experiments

PSF visualisation

Usefull links:
https://www.researchgate.net/post/How_are_you_deconvolving_your_confocal_LSM_stacks_for_dendritic_spine_morphological_analyses
https://svi.nl/NyquistCalculator
http://bigwww.epfl.ch/algorithms/psfgenerator/

https://photutils.readthedocs.io/en/stable/psf_matching.html

http://kmdouglass.github.io/posts/implementing-a-fast-gibson-lanni-psf-solver-in-python/


"""

import os
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

import g_l


FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(filename="deconvolute.log",
                    level=logging.DEBUG,
                    filemode="w",
                    format=FORMAT)


# psf = np.ones((5, 5)) / 25

# print(psf)


def psf_example(cmap='hot', savebin=False, savetif=False, savevol=False,
                plot=True, **kwargs):
    """Calculate, save, and plot various point spread functions."""

    args = {
        'shape': (512, 512),  # number of samples in z and r direction
        'dims': (5.0, 5.0),   # size in z and r direction in micrometers
        'ex_wavelen': 488.0,  # excitation wavelength in nanometers
        'em_wavelen': 520.0,  # emission wavelength in nanometers
        'num_aperture': 1.2,
        'refr_index': 1.333,
        'magnification': 1.0,
        'pinhole_radius': 0.05,  # in micrometers
        'pinhole_shape': 'square',
    }
    args.update(kwargs)

    return(psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args))


psf = g_l.demo()

input_img = '/home/astria/Bio_data/Deconvoluted/s_C001Z009.tif'
img = tifffile.imread(input_img)

xy_slice = 0

psf_xz = psf[:,63,:]
psf_xy = psf[xy_slice,:,:]
# img_mod = restoration.richardson_lucy(img, psf, iterations=10)

# print(np.shape(img_mod))
# plt.imshow(img_mod)
ax0 = plt.subplot(121)
slice_0 = ax0.imshow(psf_xy, cmap = 'inferno') 
divider_0 = make_axes_locatable(ax0)
cax = divider_0.append_axes("right", size="3%", pad=0.1)
plt.colorbar(slice_0, cax=cax)
ax0.set_title('X-Y')


ax1 = plt.subplot(122)
slice_1 = ax1.imshow(psf_xz, cmap = 'inferno')
ax1.plot([0, 127], [xy_slice, xy_slice])

divider_1 = make_axes_locatable(ax1)
cax = divider_1.append_axes("right", size="3%", pad=0.1)
plt.colorbar(slice_1, cax=cax)
ax1.set_title('X-Z, middle')


plt.show()

