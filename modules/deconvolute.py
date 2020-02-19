#!/usr/bin/env python3

""" Deconvolutio experiments

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
import numpy as np

import psf


FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(filename="deconvolute.log",
                    level=logging.INFO,
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

if __name__=="__main__":
  pass
