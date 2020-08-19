#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov
Generating PSF estimation by two models,
Richards-Wolf's (PSF library)
and Gibson-Lanni (flowdec library, module gila.py)

"""


import sys
import logging

import numpy as np

import psf
import gila

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

def psfGiLa(setting):
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


if __name__=="__main__":
    pass


# That's all!