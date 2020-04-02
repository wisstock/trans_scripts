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


def membMaxDet(slc, mode='rad', h=0.5):
    """ Finding membrane maxima by membYFP data
    and calculating full width at set height of maxima
    for identification membrane regions.

    Mode:
    'rad' for radius slices (radiusSlice fun)
    'diam' for diameter slices (lineSlice fun)

    Split slice to two halfs and find maxima in each half separately
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

def membOutDet(slc):
    """ Detection of mYFP maxima in the line of interest.
    Algorithm is going from outside to inside cell
    and finding first outer maxima of the membrane.

    Working with diam slice only!

    Returns two indexes of membrane maxima.

    """




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