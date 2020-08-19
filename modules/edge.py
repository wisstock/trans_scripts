#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Functions for cell detecting and ROI extraction.
Functions for slices quality control, membrane detection
and membrane regions extraction.
Optimysed for confocal images of the individual HEK 293 cells.
(mYFP-HPCA project).

"""

import os
import logging

import numpy as np

from skimage.external import tifffile
from skimage import filters
from skimage import measure

from scipy.ndimage import measurements as msr
from scipy import signal


def cellMask(img, method="triangle", percent=90):
    """ Extract cells using symple mask.

    Treshold methods:
    triangle - threshold_triangle;
    percent - extract pixels abowe fix percentile value.

	"""

    if method == "triangle":
        thresh_out = filters.threshold_triangle(img)
        positive_mask = img > thresh_out  # create negative threshold mask
        threshold_mask = positive_mask * -1  # inversion threshold mask

        output_img = np.copy(img)
        output_img[threshold_mask] = 0

        return output_img 

    elif method == "percent":
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

    mass_mask = filters.hessian(img, sigmas=range(20, 28, 1))
    mass_cntr = msr.center_of_mass(mass_mask)
    mass_coord = [np.int(mass_cntr[1]), np.int(mass_cntr[0])]

    logging.info("Image center of mass: %s" % mass_coord)

    return mass_coord


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


def hystCell(img, low_lim=0.05, high_lim=0.8, sigma=3):
    """ Detection of cell  membrane using hysteresis threshold method.
    
    """

    img_gauss = filters.gaussian(img, sigma=sigma)

    return filters.apply_hysteresis_threshold(img_gauss,
                                              low=low_lim*np.max(img_gauss),
                                              high=high_lim*np.max(img_gauss))


def membMaxDet(slc, mode='rad', h=0.5):
    """ Finding membrane maxima in membYFP data
    and calculating full width at set height of maxima
    for identification membrane regions.

    Mode:
    'rad' for radius slices (radiusSlice fun from slicing module)
    'diam' for diameter slices (lineSlice fun from slicing module)

    In diam mode we split slice to two halves and find maxima in each half separately
    (left and right).

    Return list of two list, first value is coordinate for left peak
    second - coordinate for right
    and third - upper limit.

    """

    if mode == 'diam':
        if (np.shape(slc)[0] % 2) != 0:  # parity check
            slc = slc[:-1]

        slc_l, slc_r = np.split(slc, 2)

        peak_l = np.int(np.argsort(slc_l)[-1:])

        peak_r = np.int(np.shape(slc_l)[0] + np.argsort(slc_r)[-1])

        peaks_val = [np.int(slc[peak_l]), np.int(slc[peak_r])]

        peaks = {peak_l: peaks_val[0],
                 peak_r: peaks_val[1]}

        logging.info('Diam. mode, peaks coordinates %s, %s' % (peak_l, peak_r))

        maxima_int = []

        for key in peaks:
            loc = key  # peack index in slice 
            
            try:
                val = peaks[key]
            except TypeError:
                return False

            lim = val * h
            interval = []

            while val > lim:  # left shift
                try:
                    val = slc[loc]
                    loc -= 1
                except IndexError:
                    return False
            interval.append(loc)

            loc = key
            val = peaks[key]

            while val > lim:  # right shift
                try:
                    val = slc[loc]
                    loc += 1
                except IndexError:
                    return False                
            interval.append(loc)
            # interval.append(lim)

            maxima_int.append(interval)

    elif mode == 'rad':
        peak = np.argsort(slc)[-1:]

        try:
            val = int(slc[peak])
            peaks_val = [val]
        except TypeError:
            return False

        lim = val / h
        loc = int(peak)
        maxima_int = []

        logging.debug('Rad. mode, peak coordinate %s and height %s' % (loc, val))

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

    logging.info('Peak width %s at 1/%d height \n' % (maxima_int, h))

    return maxima_int, peaks_val


def membOutDet(input_slc, cell_mask=10, outer_mask=30, det_cutoff=0.75):
    """ Detection of mYFP maxima in the line of interest.
    Algorithm is going from outside to inside cell
    and finding first outer maxima of the membrane.

    "cell_mask" - option for hiding inner cell region
    for ignoring possible cytoplasmic artefacts of fluorescence,
    number of pixels to be given to zero.

    "outer_mask" - option for hiding extracellular artefacts of fluorescence,
    numbers of pexels

    Working with diam slice only!

    Returns two indexes of membrane maxima.

    """

    slc = np.copy(input_slc)

    if (np.shape(slc)[0] % 2) != 0:  # parity check for correct splitting slice by two half
        slc = slc[:-1]

    slc_left, slc_right = np.split(slc, 2)
    # slc_right = np.flip(slc_right)

    logging.info('Slice splitted!')

    slc_left[-cell_mask:] = 0   # mask cellular space
    slc_right[:cell_mask] = 0  #

    slc_left[:outer_mask] = 0   # mask extracellular space
    slc_right[-outer_mask:] = 0  #

    left_peak, _ = signal.find_peaks(slc_left,
                                     height=[slc_left.max()*det_cutoff,
                                             slc_left.max()],
                                     distance=10)

    logging.info('Left peak val {:.2f}'.format(slc_left[left_peak[0]]))

    right_peak, _ = signal.find_peaks(slc_right,
                                      height=[slc_right.max()*det_cutoff,
                                              slc_right.max()],
                                      distance=10)

    logging.info('Right peak val {:.2f}'.format(slc_right[right_peak[0]]))

    memb_peaks = []

    try:
        memb_peaks.append(left_peak[0])
    except IndexError:
        logging.error('LEFT membrane peak NOT DETECTED!')
        memb_peaks.append(0)

    try:
        memb_peaks.append(int(len(slc)/2+right_peak[0]))
    except IndexError:
        logging.error('RIGHT membrane peak NOT DETECTED!')
        memb_peaks.append(0)

    logging.info('L {}, R {}'.format(memb_peaks[0], memb_peaks[1]))

    output_slc = np.concatenate((slc_left, slc_right))

    return output_slc, memb_peaks


def membExtract(slc, memb_loc, cutoff_sd=2, noise_region=15, noise_dist=25, roi_val=False):
    """ Base on exact locatiom of the mebrane peak (membYFP channel data)
    this function estimate mebrane fraction of the HPCA-TFP.

    Return summ of mebrane fraction
    and summ of cytoplasm fraction (from peak to peak region).

    For diam slice only!

    """

    memb_left = memb_loc[0]
    memb_right = memb_loc[1]

    

    print(type(memb_right))
    # logging.info('Membrane interval {}px'.format(memb_left-memb_right))


    left_noise_roi = slc[memb_left-noise_region-noise_dist \
                         :memb_left-noise_dist]
    left_noise = np.std(left_noise_roi)
    left_cutoff = left_noise * cutoff_sd

    logging.info('Left side LOI noise {}, left cutoff {}'.format(left_noise, left_cutoff))

    left_lim = memb_left
    while slc[left_lim] >= left_cutoff:
        left_lim -= 1


    right_noise_roi = slc[memb_right+noise_dist \
                          :memb_right+noise_dist+noise_region]
    right_noise = np.std(right_noise_roi)
    right_cutoff = right_noise * cutoff_sd

    logging.info('Right side LOI noise {}, right cutoff {}'.format(right_noise, right_cutoff))

    right_lim = memb_right
    while slc[right_lim] >= right_cutoff:
        right_lim += 1

    memb_frac = np.sum(slc[left_lim:memb_left])*2 + np.sum(slc[memb_right:right_lim])*2

    if roi_val:
        logging.info('Membrane interval {}px'.format(memb_right - memb_left))
        cell_frac = roi_val * (memb_right - memb_left)
    else:
        logging.info('Cytoplasm fraction extracted!')
        cell_frac = np.sum(slc[memb_left:memb_right])

    return(cell_frac, memb_frac, [left_lim, right_lim])


def badRad(slc, cutoff_lvl=0.5, n=800):
    """ Radial slice quality control.
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


def badDiam(slc, cutoff_lvl=0.2, d=35, n=50):
    """ Diameter slice quality control.
    Slice will be discarded if it have more than one peak
    with height of more than the certain percentage (cutoff_lvl) of the slice maximum
    with no interceptions of full width at set height of maxima with others peaks

    Return True if bad

    """

    up_cutoff = slc.max()  # upper limit for peak detecting, slice maxima
    down_cutoff = up_cutoff * cutoff_lvl  # lower limit for peak detecting, percent of maxima

    max_pos = int(np.argsort(slc)[-1:])

    peaks_pos, _ = signal.find_peaks(slc,
                                     height=[down_cutoff, up_cutoff],
                                     distance=d)

    logging.debug('Detecting peaks positions: {}'.format(peaks_pos))



    if not [i for i in peaks_pos if i == max_pos]:
        logging.warning('Maxima out of peak!\n')
        return True
    elif len(peaks_pos) > 2:
        logging.warning('More then two peaks!\n')
        return True
    else:
        logging.info('Slice is OK')
        return False


if __name__=="__main__":
    pass


# That's all!