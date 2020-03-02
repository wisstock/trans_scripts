#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

2-nd script

Deconvolution seq for HEK263 data

"""

import sys
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
# from skimage import io
from skimage.external import tifffile

from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from flowdec import psf as fd_psf

sys.path.append('modules')
import threshold as ts
import getpsf as psf


FORMAT = '%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s'
logging.basicConfig(format=FORMAT,
                    level=logging.getLevelName('DEBUG'))
logger = logging.getLogger('DeconvolutionCLI')


n_iter = 30
scaling_factor = 5  # substraction region part for background calculaion


input_path = os.path.join(sys.path[0], 'confocal_data/HPCA-YFP/input')
output_path = os.path.join(sys.path[0], 'confocal_data/HPCA-YFP/dec')

psf_path = os.path.join(sys.path[0], 'confocal_data/HPCA-YFP/dec/psf')


start_time = timer()

i = 0
for root, dirs, files in os.walk(input_path):  # loop over the OIF files
    for file in files:
    	if file.endswith('.tif'):
    		logger.info('Upload data file "{}"'.format(os.path.join(input_path, file)))

    		file_path = os.path.join(root, file)
    		img = tifffile.imread(file_path)

    		processed_img = ts.backCon(img, np.shape(img)[0] // scaling_factor)  # background extraction

    		z_scale = np.shape(processed_img)[0]  # z direcion size for PSF calclation
    		rw_args = {'shape': (z_scale // 2, 160),  # number of samples in z and r direction
                       'dims': (1, 4),   # size in z and r direction in micrometers
                       'ex_wavelen': 488.0,  # excitation wavelength in nanometers
                       'em_wavelen': 512.0,  # emission wavelength in nanometers
                       'num_aperture': 0.9,
                       'refr_index': 1.333,
                       'magnification': 60.0,
                       'pinhole_radius': 0.1,  # in micrometers
                       'pinhole_shape': 'round'}

    		psf_rw = psf.psfRiWo(rw_args)  # calcilating PSF

    		logger.debug('PSF calclation done, z = %s' % z_scale)

    		psf_name = 'psf_z%s' % z_scale
    		tifffile.imsave(os.path.join(psf_path, psf_name), psf_rw)

    		acq = fd_data.Acquisition(data=processed_img,
                          kernel=psf_rw)

    		logger.debug('Loaded data with shape {} and psf with shape {}'.format(acq.data.shape, acq.kernel.shape))

    		algo = fd_restoration.RichardsonLucyDeconvolver(n_dims=acq.data.ndim, pad_min=[1, 1, 1]).initialize()
    		res = algo.run(acq, niter=n_iter)

    		output_name = '%s_dec_%s.tif' % (file.split('.')[0], n_iter)
    		tifffile.imsave(os.path.join(output_path, output_name), res.data)

    		logger.info('Deconvoluted file %s saved' % output_name)


end_time = timer()
logger.info('Deconvolution complete (in {:.3f} seconds)'.format(end_time - start_time))