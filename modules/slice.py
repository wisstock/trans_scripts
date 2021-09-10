#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Functions for extract pixel values by line.

"""

import sys
import os
import logging

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage.external import tifffile
from skimage import data, img_as_float


def lineSlice(img, angle=1, cntr_coord="center"):
    """ Returns coordinates of intersection points  for the image edges
    and the line which go through point with set up coordinates
    at the set up angle.
    Requires cell image mass center coordinates
    (default set up frame center)
    and angle value (in degree).


    Angles annotation:
                

           Vertical line (v)     
           180 
    Up(u)   | mass centre coord
            |/
    270 ----+---- 90 Horizontal line (h) 
            |
    Down(d) |
            0

    y = a * x + b
    a = tg alpha
    b = y | x = 0
    

    IF deltha y < lim y

      y
      ^
      |
     Y|------------------
      |                  |
      |                  |
      |                  |
     B| *                |
      |   *              |
      |     *            |
      |       *          |
      |        a* O      |
     A|-----------*------|A'
      |           | *    |
      |           |   *  |
      |           |     *|B'
     S+-----------------------> x
                         X
    LEFT
    AB = AO * tg(a)

    RIGHT
    A'B' = OA' * tg(a)



    IF deltha y > lim y

      y
      ^
      |
     Y| *
      |  *
      |   *                
      |    *               
      |     *C              
     B|------* ----------               
      |       *          |
      |        *         |
      |         *        |
      |         a* O     |
     A|-----------*------|A'
      |           |*     |
      |           | *    |
      |           |  *   |B'
     S+---------------*-------> x
      |              C'* | 
      |                 *|X
      |

    LEFT
    BC = AO - AB/tg(a)

    RIGHT
    B'C' = OA' - A'B'/tg(a)                    

    """

    def anglPars(angl):
        """Parse input angle value.
        Real angle range is from 0 till 180 degree
        (because slice is diameter, not radius)

        """

        if 0 < angl < 90 or 180 < angl < 270:
            logging.debug("anglPars done! d")
            return("d", math.radians(90 - angl))

        elif 90 < angl < 180 or 270 < angl < 360:
            logging.debug("anglPars done! u")
            return("u", math.radians(angl-90))

        elif angl == 0 or angl == 180 or angl == 360:
            logging.debug("anglPars done! v")
            return("v", 0)

        elif angl == 90 or angl == 270:
            logging.debug("anglPars done! h")
            return("h", 90)


    x0, y0, x1, y1 = 0, 0, 0, 0  # init line ends coordinates
    img_shape = np.shape(img)

    logging.info("Frame shape: %s", img_shape)
    logging.info("Slice angle: %s", angle)

    x_lim = img_shape[1]-1  # create global image size var 
    y_lim = img_shape[0]-1  # "-1" because pixels indexing starts from 0

    indicator, angl_rad = anglPars(angle)

    if cntr_coord == "center":
        cntr_coord = [np.int(x_lim/2),
                      np.int(y_lim/2)]  # [x, y]

        logging.info("Center mode, coord: %s" % cntr_coord)

    AO_left = cntr_coord[0]
    OA_right = x_lim - cntr_coord[0]

    x_cntr = cntr_coord[0]
    y_cntr = cntr_coord[1]
    

    logging.info("Custom center, coord: %s" % cntr_coord)


    if indicator == "h":  # boundary cases check
        x0, y0 = 0, cntr_coord[1]
        x1, y1 = x_lim, cntr_coord[1] 

        logging.debug("coordinate h")
        logging.debug("0 point %s, 1 point %s" % ([x0, y0], [x1, y1]))

        return([x0, y0], [x1, y1])

    elif indicator == "v":  # boundary cases check
        x0, y0 = cntr_coord[0], y_lim
        x1, y1 = cntr_coord[0], 0

        logging.debug("coordinate v")
        logging.debug("0 point %s, 1 point %s" % ([x0, y0], [x1, y1]))

        return([x0, y0], [x1, y1])


    elif indicator == "d":
        AB_left = np.int(cntr_coord[0]*math.tan(angl_rad))  # AB, see on comment above
        
        logging.debug("AB: %s" % AB_left)

        if  cntr_coord[1] - AB_left > 0:  # calculate down left (0-90)
            x0 = 0
            y0 = cntr_coord[1] - AB_left

            logging.debug("AB > 0. x0, y0 = %s, %s" % (x0, y0))
        elif cntr_coord[1] - AB_left <= 0:
            BO_left = np.int(cntr_coord[1]/math.tan(angl_rad))  # BO, see comment above
            x0 = cntr_coord[0] - BO_left
            y0 = 0

            logging.debug("AB < y max. x0, y0 = %s, %s" % (x0, y0))

        AB_right = np.int((x_lim - cntr_coord[0])*math.tan(angl_rad))  # A'B', see comment above

        logging.debug("A'B': %s" % AB_right)
        
        if AB_right < y_lim - cntr_coord[1]:  # calculate down right
            x1 = x_lim
            y1 = cntr_coord[1] + AB_right

            logging.debug("A'B' > y_max. x1, y1 = %s, %s" % (x1, y1))
        elif AB_right >= y_lim - cntr_coord[1]:
            OC_right = np.int((y_lim - cntr_coord[1])/math.tan(angl_rad))
            BC_right = ((x_lim - cntr_coord[0]) - OC_right)  # B'C', see comment above
            x1 = x_lim - BC_right
            y1 = y_lim

            logging.debug("A'B' > y max. x1, y1 = %s, %s" % (x1, y1))


    elif indicator == "u":
        AB_left = np.int(cntr_coord[0]*math.tan(angl_rad))  # AB, see comment above

        logging.debug("AB: %s" % AB_left)

        if  AB_left < y_lim - cntr_coord[1]:  # calculate up left (90-180)
            x0 = 0
            y0 = cntr_coord[1] + AB_left
            logging.debug("AB < y max. x0, y0 = %s, %s" % (x0, y0))
        elif AB_left >= y_lim - cntr_coord[1]:
            BO_left = np.int((y_lim - cntr_coord[1])/math.tan(angl_rad))  # BC, see comment above
            x0 = cntr_coord[0] - BO_left
            y0 = y_lim
            logging.debug("AB > y max. x0, y0 = %s, %s" % (x0, y0))

        AB_right = np.int((x_lim - cntr_coord[0])*math.tan(angl_rad))  # A'B', see comment above

        logging.debug("A'B': %s" % AB_right)

        if cntr_coord[1] - AB_right > 0:  # calculate up right
            x1 = x_lim
            y1 = cntr_coord[1] - AB_right

            logging.debug("A'B' > 0. x1, y1 = %s, %s" % (x1, y1))
        elif cntr_coord[1] - AB_right <= 0:
            BC_right = (cntr_coord[0] - np.int(cntr_coord[1]/math.tan(angl_rad)))  # B'C', see comment above
            x1 = x_lim - BC_right
            y1 = 0

            logging.debug("A'B' < 0. x1, y1 = %s, %s" % (x1, y1))

    return([x0, y0], [x1, y1])


def radiusSlice(img, angl=1, cntr_coord="center"):
    """ Returns coordinates of intersection points for the image edges
    and the line starting from center point at the set up angle. 

    Algorithm is the same as the function lineSlice

     y^
      |
      |
      |------------------------
      |                        |
      |                        |
      |                        |
      |                        |
      |                        |
     A|--------- ** * O        |
      |       **   *|          |
      |    **    *  |          |
     a|**      *    |          |
      |      *      |          |
      |    *        |          |
      +-------------------------------->
          b        B                  x


     II | III
        |
    ----+----
        |
      I | IV
    0

    """ 

    def anglPars(angl):
        """Parse input angle value.
        Real angle range is from 0 till 180 degree
        (because slice is diameter, not radius)

        """

        if 0 <= angl <90:
            return("I")  # , math.radians(angl))

        elif 90 <= angl < 180:
            return("II")  # , math.radians(angl))

        elif 180 <= angl < 270:
            return("III")  # , math.radians(angl))

        elif 270 <= angl <= 360:
            return("IV")  # , math.radians(angl))

    img_shape = np.shape(img)

    x_lim = img_shape[1]-1  # create global image size var 
    y_lim = img_shape[0]-1  # "-1" because pixels indexing starts from 0

    if cntr_coord == "center":
        cntr_coord = [np.int(x_lim/2),
                      np.int(y_lim/2)]  # [x, y]

        logging.debug("Center mode, coord: %s" % cntr_coord)
    else:
        logging.debug("Custom center, coord: %s" % cntr_coord)

    x0, y0 = 0, 0  # init line ends coordinates

    x_cntr = cntr_coord[0]  # AO_left = x_cntr, OA_right = x_lim - x_cntr
    y_cntr = cntr_coord[1]  # BO_down = y_cntr, OB_up = y_lim - y_cntr

    indicator = anglPars(angl)
    logging.debug("Square quartile: %s" % indicator)

    logging.debug("Frame shape: %s, slice angle: %s" % (img_shape, angl))

    if indicator == 'I':
      Bb = np.int(y_cntr * math.tan(math.radians(angl)))

      logging.debug("Bb = %s" % Bb)

      if Bb <= x_cntr:
        x0 = x_cntr - Bb
        y0 = 0

      if Bb > x_cntr:
        Aa = np.int(x_cntr * math.tan(math.radians(90 - angl)))
        
        logging.debug("Aa = %s" % Aa)

        x0 = 0
        y0 = y_cntr - Aa


    elif indicator == 'II':
      Aa = np.int(x_cntr * math.tan(math.radians(angl - 90)))

      logging.debug("Aa = %s" % Aa)

      if Aa <= y_cntr + Aa:
        x0 = 0
        y0 = y_cntr + Aa

      if Aa > y_cntr:
        Bb = np.int((y_lim-y_cntr) * math.tan(math.radians(180 - angl)))
        
        logging.debug("Bb = %s" % Bb)

        x0 = x_cntr - Bb
        y0 = y_lim

    elif indicator == 'III':
      Bb = np.int((y_lim - y_cntr) * math.tan(math.radians(angl - 180)))

      logging.debug('Bb = %s' % Bb)

      if Bb <= x_lim - x_cntr:
        x0 = x_cntr + Bb
        y0 = y_lim

      if Bb > x_lim - x_cntr:
        Aa = np.int((x_lim - x_cntr) * math.tan(math.radians(270 - angl)))

        logging.debug('Aa = %s' % Aa)

        x0 = x_lim
        y0 = y_cntr + Aa

    elif indicator == 'IV':
      Aa = np.int((x_lim - x_cntr) * math.tan(math.radians(angl - 270)))

      logging.debug('Aa = %s' % Aa)

      if Aa <= y_cntr:
        x0 = x_lim
        y0 = y_cntr - Aa

      if Aa > y_cntr:
        Bb = np.int(y_cntr * math.tan(math.radians(360 - angl)))

        logging.debug('Bb = %s' % Bb)

        x0 = x_cntr + Bb
        y0 = 0

    return([x_cntr, y_cntr], [x0, y0])


def lineExtract(img, start_coors, end_coord):
    """ Returns values of pixels intensity along the line
    with specified ends coordinates.

    Requires cell image, and line ends coordinate (results of lineSlice).

    """
    x0, y0 = start_coors[1], start_coors[0]
    x1, y1 = end_coord[1], end_coord[0]
    line_length = int(np.hypot(x1-x0, y1-y0))  # calculate line length

    # logging.debug("line length: %s" % (np.hypot(x1-x0, y1-y0)))

    x, y = np.linspace(x0, x1, line_length), np.linspace(y0, y1, line_length)  # calculate projection to axis

    logging.debug("X and Y length: %s, %s" % (np.shape(x)[0], np.shape(y)[0]))

    output = np.array(img[x.astype(np.int), y.astype(np.int)])

    return output


def bandExtract(img, start_coors, end_coord, band_width=2, mode="mean"):
    """ Returns values ​​of pixels intensity in fixed neighborhood (in pixels)
    for each points in the the line with specified ends coordinates.
    
    Requires cell image, and line ends coordinate (results of lineSlice).

    """

    def neighborPix(img, coord, shift):
      """ Mean value of matrix with setting shifr.
      Default 4x4.

      """
      sub_img = img[coord[0]-shift:coord[0]+shift:1,
                    coord[1]-shift:coord[1]+shift:1]

      mean_val = np.mean(sub_img)

      return mean_val

    def paralellPix(img, coord, length, shift):
      """ Calculate mean value for points with same indexes in several
      slice lines. Ends coordinates for each line 
      """
      pass


    x0, y0 = start_coors[1], start_coors[0]
    x1, y1 = end_coord[1], end_coord[0]
    line_length = int(np.hypot(x1-x0, y1-y0))  # calculate line length

    x, y = np.linspace(x0, x1, line_length), np.linspace(y0, y1, line_length)  # calculate projection to axis

    if mode == "mean":

      i = 0
      res = []

      while i < np.shape(x)[0]:
        x_point = np.int(x[i])
        y_point = np.int(y[i])

        res.append(neighborPix(img, [x_point, y_point], band_width))

        i += 1

      np_res = np.array(res)
      nan_res = np.isnan(np_res)  # removing NaN values (errors on ends)
      not_nan_res =~ nan_res
      output = np_res[not_nan_res]

      return output

    elif mode == 'parallel':
      pass  


def membMaxDet(slc, mode='rad', h=0.5):
    """ Finding membrane maxima in membYFP data
    and calculating full width at set height of maxima
    for identification membrane regions.

    Mode:
    'rad' for radius slices (radiusSlice fun from slicing module)
    'diam' for diameter slices (lineSlice fun from slicing module)

    In diam mode we split slice to two halves and find maxima in each half separately
    (left and right).

    Returns list of two list, first value is coordinate for left peak
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

    Returns True if bad

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



if __name__=="__main__":
  pass


# That's all!