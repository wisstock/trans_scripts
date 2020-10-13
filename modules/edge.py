#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Functions for cell detecting and ROI extraction.
Functions for embrane detection and membrane regions extraction with hysteresis filter.
Optimysed for confocal images of the individual HEK 293 cells.

"""

import os
import logging

import numpy as np
import numpy.ma as ma

from skimage.external import tifffile
from skimage import filters
from skimage import measure
from skimage import segmentation

from scipy.ndimage import measurements as msr
from scipy import signal


def backCon(img, edge_lim=20, dim=3):
    """ Background extraction in TIFF series

    For confocal Z-stacks only!
    dem = 2 for one frame, 3 for z-stack

    """

    if dim == 3:
        edge_stack = img[:,:edge_lim,:edge_lim]
        mean_back = np.mean(edge_stack)

        logging.debug('Mean background, {} px region: {:.3f}'.format(edge_lim, mean_back))

        img_out = np.copy(img)
        img_out = img_out - mean_back
        img_out[img_out < 0] = 0

        return img_out
    elif dim == 2:
        edge_fragment = img[:edge_lim,:edge_lim]
        mean_back = np.mean(edge_fragment)

        logging.debug('Mean background, %s px region: %s' % (edge_lim, mean_back))

        img = np.copy(img)
        img = img - mean_back
        img[img < 0] = 0

        return img


def hystLow(img, img_gauss, sd=0, mean=0, diff=40, init_low=0.05, gen_high=0.8, mode='memb'):
    """ Lower treshold calculations for hysteresis membrane detection function hystMemb.

    diff - int, difference (in px number) between hysteresis mask and img without greater values
    delta_diff - int, tolerance level (in px number) for diff value
    gen_high, sd, mean - see hystMemb

    mode - 'cell': only sd treshold calc, 'memb': both tresholds calc

    """
    if mode == 'memb':
        masks = {'2sd': ma.masked_greater_equal(img, 2*sd),  # values greater then 2 noise sd 
                 'mean': ma.masked_greater(img, mean)}       # values greater then mean cytoplasm intensity
    elif mode == 'cell':
        masks = {'2sd': ma.masked_greater_equal(img, 2*sd)}

    logging.info('masks: {}'.format(masks.keys()))

    low_val = {}
    control_diff = False
    for mask_name in masks:
        mask_img = masks[mask_name]

        logging.info('Mask {} lower treshold fitting in progress'.format(mask_name))

        mask_hyst = filters.apply_hysteresis_threshold(img_gauss,
                                                      low=init_low*np.max(img_gauss),
                                                      high=gen_high*np.max(img_gauss))
        diff_mask = np.sum(ma.masked_where(~mask_hyst, mask_img) > 0)

        if diff_mask < diff:
            raise ValueError('Initial lower threshold is too low!')
        logging.info('Initial masks difference {}'.format(diff_mask))

        low = init_low

        i = 0
        control_diff = 1
        while diff_mask >= diff:
            mask_hyst = filters.apply_hysteresis_threshold(img_gauss,
                                                          low=low*np.max(img_gauss),
                                                          high=gen_high*np.max(img_gauss))
            diff_mask = np.sum(ma.masked_where(~mask_hyst, mask_img) > 0)

            low += 0.01

            i += 1
            # is cytoplasm mean mask at initial lower threshold value closed? prevent infinit cycle
            if i == 75:
                logging.fatal('Lower treshold for {} mask {:.2f}, control difference {}px'.format(mask_name, low, control_diff))
                raise RuntimeError('Membrane in mean mask doesn`t detected at initial lower threshold value!')
    

        # is cytoplasm mask at setted up difference value closed?
        if mask_name == 'mean':
            control_diff = np.all((segmentation.flood(mask_hyst, (0, 0)) + mask_hyst))
            if control_diff == True:
                logging.fatal('Lower treshold for {} mask {:.2f}, masks difference {}px'.format(mask_name, low, diff_mask))
                raise ValueError('Membrane in {} mask doesn`t closed, mebrane unlocated at this diff value (too low)!'.format(mask_name))

        low_val.update({mask_name : low})
    logging.info('Lower tresholds {}\n'.format(low_val))

    return low_val


def hystMemb(img, roi_center, roi_size=30, noise_size=20, low_diff=40, gen_high=0.8, sigma=3):
    """ Function for membrane region detection with hysteresis threshold algorithm.
    Outdide edge - >= 2sd noise
    Inside edge - >= cytoplasm mean intensity

    Require hystLow function for lower hysteresis threshold calculations.

    img - imput z-stack frame;
    roi_center - list of int [x, y], coordinates of center of the cytoplasmic ROI for cytoplasm mean intensity calculation;
    roi_size - int, cutoplasmic ROI side size in px (ROI is a square area);
    noise_size - int, size in px of region for noise sd calculation (square area witf start in 0,0 coordinates);
    sd_low - float, hysteresis algorithm lower threshold for outside cell edge detection,
             > 2sd of noise (percentage of maximum frame intensity);
    mean_low - float, hysteresis algorithm lower threshold for inside cell edge detection,
             > cytoplasmic ROI mean intensity (percentage of maximum frame intensity);
    gen_high - float,  general upper threshold for hysteresis algorithm (percentage of maximum frame intensity);
    sigma - int, sd for gaussian filter.

    Returts membrane region boolean mask for input frame.

    """
    img = backCon(img, dim=2)
    img_gauss = filters.gaussian(img, sigma=sigma)

    noise_sd = np.std(img[:noise_size, :noise_size])
    logging.info('Frame noise SD={:.3f}'.format(noise_sd))

    roi_mean = np.mean(img[roi_center[0] - roi_size//2:roi_center[0] + roi_size//2, \
                           roi_center[1] - roi_size//2:roi_center[1] + roi_size//2])  # cutoplasmic ROI mean celculation
    logging.info('Cytoplasm ROI mean intensity {:.3f}'.format(roi_mean))

    low_val = hystLow(img, img_gauss, sd=noise_sd, mean=roi_mean, diff=low_diff, gen_high=gen_high)

    mask_2sd = filters.apply_hysteresis_threshold(img_gauss,
                                                  low=low_val['2sd']*np.max(img_gauss),
                                                  high=gen_high*np.max(img_gauss))
    mask_roi_mean = filters.apply_hysteresis_threshold(img_gauss,
                                                      low=low_val['mean']*np.max(img_gauss),
                                                      high=gen_high*np.max(img_gauss))
    # filling external space and create cytoplasmic mask 
    mask_cytoplasm = mask_roi_mean + segmentation.flood(mask_roi_mean, (0, 0))

    return mask_2sd, mask_roi_mean, ma.masked_where(~mask_cytoplasm, mask_2sd)



if __name__=="__main__":
    pass


# That's all!