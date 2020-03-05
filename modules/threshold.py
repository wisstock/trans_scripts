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


def getTiff(file_name, channel=0, frame_number=0, camera_offset=250):
    """ Function returns individual frame from image series and
    apply compensation of camera offset value for this frame.
    Separate two fluorecent channels (first start from 0, second - from 1).
    For Dual View system data.

    """

    path = os.getcwd() + '/temp/data/' + file_name

    tiff_tensor = tifffile.imread(path)
    # print(tiff_tensor.shape, np.max(tiff_tensor))

    channel_one = tiff_tensor[0::2, :, :]
    channel_two = tiff_tensor[1::2, :, :]
    # print(channel_one.shape)
    # print(channel_two.shape)

    if channel == 0:
        frame_amount = channel_one.shape
        try:
            img = channel_one[frame_number] - camera_offset
            return(img)
        except ValueError:
            if frame_number > frame_amount[0]:
                print("Frame number out of range!")
    else:
        frame_amount = channel_two.shape
        try:
            img = channel_two[frame_number] - camera_offset
            return(img)
        except ValueError:
            if frame_number > frame_amount[0]:
                print("Frame number out of range!")

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

def backCon(img, edge_lim=50):
    """ Background extraction in TIFF series

    For confocal Z-stacks only!

    """

    edge_stack = img[:,0:edge_lim,0:edge_lim]
    mean_back = np.mean(edge_stack)

    logging.info('Mean background, %s px region: %s' % (edge_lim, mean_back))

    img = np.copy(img)
    img = img - mean_back
    img[img < 0] = 0

    return img

def membDet(slc, h=2, mode='rad'):
    """ Finding membrane maxima by membYFP data
    and calculating full width at set height of maxima

    Mode:
    'rad' for radius slices (radiusSlice fun)
    'diam' for diameter slices (lineSlice fun)

    Split slice to two halfs and find maxima in each half separately
    (left and right)

    Return list of two list, first value is coordinate for left peak
    second - coordinate for right
    and third - upper limit.

    """

    # if mode == 'diam':
    #     slc_l, slc_r = np.split(slc, 2)

    #     peak_l = np.int(np.argsort(slc_l)[-1:])

    #     peak_r = np.int(np.shape(slc_l)[0] + np.argsort(slc_r)[-1])

    #     peaks = {peak_l: np.int(slc[peak_l]),
    #              peak_r: np.int(slc[peak_r])}

    #     logging.info('Diam. mode, peaks coordinates %s, %s' % (peak_l, peak_r))

    #     maxima_int = []

    #     for key in peaks:
    #         loc = key  # peack index in slice 
    #         val = peaks[key]
    #         lim = val / h
    #         interval = []
    #         logging.info('Full width at 1/%s of height (%s) for peack %s' %
    #                     (h, lim, key))

    #         while val > lim:  # left shift
    #             val = slc[loc]
    #             loc -= 1
    #         interval.append(loc)

    #         loc = key
    #         val = peaks[key]

    #         while val > lim:  # right shift
    #             val = slc[loc]
    #             loc += 1
    #         interval.append(loc)
    #         interval.append(lim)

    #         maxima_int.append(interval)

    # elif mode == 'rad':

    peak = np.argsort(slc)[-1:]

    try:
        val = int(slc[peak])
    except TypeError:
        return False


    
    lim = val / h
    loc = int(peak)
    maxima_int = []

    logging.debug('Peak coordinate %s and height %s' % (loc, val))

    while val >= lim:
        try:
            val = slc[loc]
            loc -= 1
        except IndexError:
            return False

    maxima_int.append(int(loc))

    loc = peak
    val = int(slc[peak])

    while val >= lim:
        try:
            val = slc[loc]
            loc += 1
        except IndexError:
            return False

        
    maxima_int.append(int(loc))

    logging.debug('Peak width %s at 1/%d height \n' % (maxima_int, h))

    return maxima_int

def badSlc(slc, cutoff_lvl=0.5, n=800):
    """ Slice quality control.
    Slice will be discarded if it have more than one peak
    with height of more than the certain percentage (cutoff_lvl) of the slice maximum
    with no interceptions of full width at set height of maxima with others peaks

    Return True if bad

    """

    up_cutoff = slc.max()  # upper limit for peak detecting, slice maxima
    down_cutoff = up_cutoff * cutoff_lvl  # lower limit for peak detecting, percent of maxima

    max_pos = int(np.argsort(slc)[-1:])

    peaks_pos, _ = signal.find_peaks(slc, [down_cutoff, up_cutoff])
    peaks_val = slc[peaks_pos]

    loc_rel = []

    for peak in peaks_pos:  # peak grouping estimation
        loc_rel.append([i for i in peaks_pos if i > peak-slc[peak]/n and i < peak+slc[peak]/n])

    loc_div = []
    [loc_div.append(i) for i in [len(a) for a in loc_rel] if i not in loc_div]

    if not [i for i in peaks_pos if i == max_pos]:  # if maxima is not a peak
        return True
    elif len(loc_div) > 1:
        return True
    else:
        return False


if __name__=="__main__":
    pass


# That's all!