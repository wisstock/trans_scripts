#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Functions for cell detecting and ROI extraction.
Optimysed for confocal images
 of the individual HEK 293 cells.
(mYFP-HPCa project).

"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage.external import tifffile
from skimage import filters
from skimage import measure

from scipy.ndimage import measurements as msr


def cellMask(img, thbreshold_method="triangle",
             percent=90, seed_method="one"):
    """ Extract cells using symple mask.
    
    Treshold methods:
    triangle - threshold_triangle;
    percent - extract pixels abowe fix percentile value.

	"""

    if thbreshold_method == "triangle":
        thresh_out = filters.threshold_triangle(img)
        positive_mask = img > thresh_out  # create negative threshold mask
        threshold_mask = positive_mask * -1  # inversion threshold mask

        output_img = np.copy(img)
        output_img[threshold_mask] = 0

        return output_img 

    elif thbreshold_method == "percent":
        percentile = np.percentile(img, percent)
        output_img = np.copy(img)
        output_img[output_img < percentile] = 0

        return output_img

    else:
        print("Incorrect treshold method!")

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
    # print(type(mass_mask))
    mass_cntr = msr.center_of_mass(mass_mask)

    return [np.int(mass_cntr[1]), np.int(mass_cntr[0])]

def cellEdge(img):
    output = filters.hessian(img, sigmas=range(10, 28, 1))
    
    return output


if __name__=="__main__":
    wd_path = os.path.split(os.getcwd())
    os.chdir(wd_path[0] + '/temp/data/')  # go to DATA dir

    offset = 250  # camera offset value
    gray_raw = tifffile.imread('Fluorescence_435nmDD500_cell1.tiff')
    img = gray_raw[0] - offset

    fig, ax = try_all_threshold(img, figsize=(8, 10),
                                     verbose=False)
    plt.show()