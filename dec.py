#!/usr/bin/env python3

""" Deconvolutio experiments with lib flowdec

https://github.com/wisstock/flowdec

"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.external import tifffile
from scipy import ndimage, signal
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration

sys.path.append('modules')
import oiffile as oif

import g_l


psf = g_l.demo()


oif_path = '/home/astria/Bio_data/HEK_mYFP/20180523_HEK_membYFP/cell1/20180523-1404-0003-250um.oif'
oif_raw = oif.OibImread(oif_path)
oif_img = oif_raw[0,:,:,:]
data = oif_img[:,:,:]

tifffile.imsave('cell2.tif', data)

# algo = fd_restoration.RichardsonLucyDeconvolver(data.ndim).initialize()

# res = algo.run(fd_data.Acquisition(data=data, kernel=psf), niter=10).data


# fig, axs = plt.subplots(1, 3)
# axs = axs.ravel()
# fig.set_size_inches(18, 12)
# center = tuple([slice(None), slice(10, -10), slice(10, -10)])
# titles = ['Original Image', 'Blurred Image', 'Reconstructed Image']
# for i, d in enumerate([actual, data, res]):
#     img = exposure.adjust_gamma(d[center].max(axis=0), gamma=.2)
#     axs[i].imshow(img)
#     axs[i].set_title(titles[i])
#     axs[i].axis('off')

# plt.show()