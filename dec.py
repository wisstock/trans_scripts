#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Deconvolutio functions demo experiments with lib flowdec

https://github.com/wisstock/flowdec

"""

import sys

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



logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s',
                    level=logging.getLevelName('DEBUG'))
logger = logging.getLogger('DeconvolutionCLI')

# oif_input = '/home/astria/Bio_data/HEK_mYFP/20180523_HEK_membYFP/cell2/20180523-1414-0011-500um.oif'
data_path = "/home/astria/Bio/Lab/scripts/trans_scripts/.temp/data/cell1.tif"
output_path = "/home/astria/Bio/Lab/scripts/trans_scripts/.temp/data/res.tif"
psf_config_path = "/home/astria/Bio/Lab/scripts/trans_scripts/.temp/data/psf.json"
n_iter = 10



acq = fd_data.Acquisition(data=io.imread(data_path),
                          kernel=resolve_psf(psf_config_path,logger))

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

