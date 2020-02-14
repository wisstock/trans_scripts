#!/usr/bin/env python3

'''
Copyright © 2020 Borys Olifirov

'''

import os
from skimage.external import tifffile
from skimage.filters import try_all_threshold

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def cellEdge(img, thbreshold_method="triangle", percent=90, seed_method="one"):
    '''
	seed methods:
        one - calculate threshold for one image
	    first - build threshold mask for first frame of series
	    max - build threshold mask for max intensity frame
	    mean - build threshold mask for mean intensity of all serie frames

    treshold methods:
        triangle - threshold_triangle
        percent - extract pixels abowe fix percentile value

	'''

    if thbreshold_method == "triangle":
        thresh_out = threshold_triangle(img)
        positive_mask = img > thresh_out  # create negative threshold mask
        threshold_mask = positive_mask * -1  # inversion threshold mask

        output_img = np.copy(img)
        output_img[threshold_mask] = 0

        return(output_img)

    elif thbreshold_method == "percent":
        percentile = np.percentile(img, percent)
        output_img = np.copy(img)
        output_img[output_img < percentile] = 0

        return(output_img)

    else:
        print("Incorrect treshold method!")


wd_path = os.path.split(os.getcwd())
os.chdir(wd_path[0] + '/temp/data/')  # go to DATA dir

offset = 250  # camera offset value
gray_raw = tifffile.imread('Fluorescence_435nmDD500_cell1.tiff')
img = gray_raw[0] - offset

fig, ax = try_all_threshold(img, figsize=(8, 10),
                                 verbose=False)
plt.show()