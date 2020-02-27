#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Deconvolution functions demo experiments with lib flowdec

https://github.com/wisstock/flowdec

"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from skimage import io
from skimage.external import tifffile
import logging

from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from flowdec import psf as fd_psf

sys.path.append('modules')
import oiffile as oif
import slicing as slc
import threshold as ts



def resolve_psf(config, logger):
    # if args["psf_path"] and args["psf_config_path"]:
    #     raise ValueError('Must supply PSF file path or PSF config path but not both')
    # if not args.psf_path and not args.psf_config_path:
    #     raise ValueError('Must supply either PSF file path or PSF config path')

    # # If a PSF data file was given, load it directly
    # if args.psf_path:
    #     return io.imread(args["psf_path"])
    # # Otherwise, load PSF configuration file and generate a PSF from that
    # else:
    psf = fd_psf.GibsonLanni.load(config)
    logger.info('Loaded psf with configuration: {}'.format(psf.to_json()))
    return psf.generate()


FORMAT = '%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s'
logging.basicConfig(format=FORMAT,
                    level=logging.getLevelName('DEBUG'))
logger = logging.getLogger('DeconvolutionCLI')


n_iter = 1000


data_path = os.path.join(sys.path[0], '.temp/data/cell1.tif')
output_path = os.path.join(sys.path[0], '.temp/data/dec_rw_30.tif')

psf_model_path = os.path.join(sys.path[0], '.temp/data/psf_rw.tif')
psf_model = tifffile.imread(psf_model_path)

# psf_config_path = os.path.join(sys.path[0], '.temp/data/psf.json')
psf_model = tifffile.imread(psf_model_path)
# try:
#     psf_model = tifffile.imread(psf_model_path)
# else:
#     resolve_psf(psf_config_path, logger)



input_img = io.imread(data_path)

scaling_factor = 5  # substraction region part
processed_img = ts.backCon(input_img, np.shape(input_img)[1] // scaling_factor)  # extracimg background


acq = fd_data.Acquisition(data=processed_img,
                          kernel=psf_model)# resolve_psf(psf_config_path,logger))

logger.debug('Loaded data with shape {} and psf with shape {}'.format(acq.data.shape, acq.kernel.shape))

logger.info('Beginning deconvolution of data file "{}"'.format(data_path))

start_time = timer()

# Initialize deconvolution with a padding minimum of 1, which will force any images with dimensions
# already equal to powers of 2 (which is common with examples) up to the next power of 2
algo = fd_restoration.RichardsonLucyDeconvolver(n_dims=acq.data.ndim, pad_min=[10, 10, 10]).initialize()
res = algo.run(acq, niter=n_iter)

end_time = timer()
logger.info('Deconvolution complete (in {:.3f} seconds)'.format(end_time - start_time))

io.imsave(output_path, res.data)
logger.info('Result saved to "{}"'.format(output_path))

