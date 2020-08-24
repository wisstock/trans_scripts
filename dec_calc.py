#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

2-nd script

Deconvolution seq for HEK 293 data

"""

import sys
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from skimage import io
# from skimage.external import tifffile

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


# n_iter = 16

iter_list = [32, 64, 128]

scaling_factor = 5  # subtraction region part for background calculation


input_path = os.path.join(sys.path[0], 'data/hpca')
output_path = os.path.join(sys.path[0], 'data')

psf_path = os.path.join(sys.path[0], 'data')

for n_iter in iter_list:
    logging.info('Deconvolution with {} iteration starting'.format(n_iter))

    start_time = timer()

    i = 0
    for root, dirs, files in os.walk(input_path):  # loop over the OIF files
        for file in files:
          if file.endswith('.tif'):
              logger.info('Upload data file "{}"'.format(os.path.join(input_path, file)))

              file_path = os.path.join(root, file)
              img = io.imread(file_path)

              processed_img = ts.backCon(img, np.shape(img)[1] // scaling_factor)  # background extraction

              z_scale = np.shape(processed_img)[0]  # z direction size for PSF calculation
              rw_args = {'shape': (z_scale // 2, 159),  # number of samples in z and r direction
                       'dims': (11, 16),   # size in z (1.2um*9slices) and r(0.1um*160px) direction in micrometers
                       'ex_wavelen': 462.0,  # excitation wavelength in nanometers
                       'em_wavelen': 492.0,  # emission wavelength in nanometers
                       'num_aperture': 1.0,
                       'refr_index': 1.333,
                       'magnification': 60.0,
                       'pinhole_radius': 0.250,  # in mm
                       'pinhole_shape': 'round'}

              psf_rw = psf.psfRiWo(rw_args)

              psf_time = timer()
              logger.info('PSF (z = {}) calclation complete in {:.3f} seconds'.format( z_scale, psf_time - start_time))

              psf_name = 'psf_z%s.tiff' % z_scale
              io.imsave(os.path.join(psf_path, psf_name), psf_rw)

              acq = fd_data.Acquisition(data=processed_img,
                          kernel=psf_rw)

              logger.debug('Loaded data with shape {} and psf with shape {}'.format(acq.data.shape, acq.kernel.shape))

              algo = fd_restoration.RichardsonLucyDeconvolver(n_dims=acq.data.ndim, pad_min=[1, 1, 1]).initialize()
              res = algo.run(acq, niter=n_iter)

              output_name = '%s_dec_%s.tif' % (file.split('.')[0], n_iter)
              io.imsave(os.path.join(output_path, output_name), res.data)

              logger.info('Deconvolve file %s saved\n' % output_name)
    i += 1
    logging.info('Deconvolution with {} iteration complete'.format(n_iter))


end_time = timer()
logger.info('Deconvolution of {} files complete in {:.3f} seconds'.format(i, end_time - start_time))
