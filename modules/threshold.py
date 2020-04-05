#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Functions for cell detecting and ROI extraction.
Optimysed for confocal images of the individual HEK 293 cells.
(mYFP-HPCA project).

"""

import os
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage.external import tifffile
from skimage import filters
from skimage import measure

from scipy.ndimage import measurements as msr
from scipy import signal


def cellMask(img, threshold_method="triangle", percent=90):
    """ Extract cells using symple mask.

    Treshold methods:
    triangle - threshold_triangle;
    percent - extract pixels abowe fix percentile value.

	"""

    if threshold_method == "triangle":
        thresh_out = filters.threshold_triangle(img)
        positive_mask = img > thresh_out  # create negative threshold mask
        threshold_mask = positive_mask * -1  # inversion threshold mask

        output_img = np.copy(img)
        output_img[threshold_mask] = 0

        return output_img 

    elif threshold_method == "percent":
        percentile = np.percentile(img, percent)
        output_img = np.copy(img)
        output_img[output_img < percentile] = 0

        return output_img

    else:
        logging.warning("Incorrect treshold method!")

def cellMass(img):
    """ Calculating of the center of mass coordinate using threshold mask
    for already detected cell.

    Treshold function use modifyed Hessian filter.
    This method optimysed for confocal image of HEK 293 cells with fluorecent-
    labelled protein who located into the membrane.

    Results of this method for fluorecent microscop images
    or fluorecent-labelled proteins with cytoplasmic localization
    may by unpredictable and incorrect.

    """

    mass_mask = filters.hessian(img, sigmas=range(10, 28, 1))
    mass_cntr = msr.center_of_mass(mass_mask)
    mass_coord = [np.int(mass_cntr[1]), np.int(mass_cntr[0])]

    logging.info("Image center of mass: %s" % mass_coord)

    return mass_coord

def cellEdge(img):
    """ Returns the cell edge mask generating by modifyed Hessian filter.

    """
    output = filters.hessian(img, sigmas=range(10, 28, 1))
    
    return output

def backCon(img, edge_lim=20, dim=3):
    """ Background extraction in TIFF series

    For confocal Z-stacks only!
    dem = 2 for one frame, 3 for z-stack

    """

    if dim == 3:
        edge_stack = img[:,:edge_lim,:edge_lim]
        mean_back = np.mean(edge_stack)

        logging.info('Mean background, %s px region: %s' % (edge_lim, mean_back))

        img = np.copy(img)
        img = img - mean_back
        img[img < 0] = 0

        return img
    elif dim == 2:
        edge_fragment = img[:edge_lim,:edge_lim]
        mean_back = np.mean(edge_fragment)

        logging.info('Mean background, %s px region: %s' % (edge_lim, mean_back))

        img = np.copy(img)
        img = img - mean_back
        img[img < 0] = 0

        return img


if __name__=="__main__":
    pass


# That's all!