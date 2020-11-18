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

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.external import tifffile
from skimage import filters
from skimage import measure
from skimage import segmentation

from scipy.ndimage import measurements as msr
from scipy import signal
from scipy import ndimage as ndi


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


def s_der(gauss_series, mask, sigma=4,  sd_area=50, sd_tolerance=1, left_w=1, space_w=0, right_w=1, output_path=False, norm=True):
    """ Calculating derivative image series (difference between two windows of interes).

    Pixels greater than noise sd * sd_tolerance set equal to 1;
    Pixels less than -noise sd * sd_tolerance set equal to -1.

    """
    # w = 3
    # gauss_series = np.asarray([filters.gaussian(series[i], sigma=sigma, truncate=(((w - 1)/2)-0.5)/sigma) for i in range(np.shape(series)[0])])

    logging.info(f'Derivate sigma={sigma}')

    der_series = []
    for i in range(np.shape(gauss_series)[0] - (left_w+space_w+right_w)):
        der_frame = np.mean(gauss_series[i+left_w+space_w:i+left_w+space_w+right_w], axis=0) - np.mean(gauss_series[i:i+left_w], axis=0) 
        # der_sd = np.std(der_frame[:sd_area, sd_area])
        # der_frame[der_frame > der_sd * sd_tolerance] = 1
        # der_frame[der_frame < -der_sd * sd_tolerance] = -1
        der_series.append(ma.masked_where(~mask, der_frame))    
    logging.info(f'Derivative series len={len(der_series)} (left WOI={left_w}, spacer={space_w}, right WOI={right_w})')

    if output_path:
        save_path = f'{output_path}/blue_red'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        norm = lambda x, min_val, max_val: (x-min_val)/(max_val-min_val)  # normilize derivate series values to 0-1 range
        vnorm = np.vectorize(norm)

        for i in range(len(der_series)):
            raw_frame = der_series[i]
            frame = vnorm(raw_frame, np.min(der_series), np.max(der_series))

            plt.figure()
            ax = plt.subplot()
            img = ax.imshow(frame, cmap='bwr')
            img.set_clim(vmin=0., vmax=1.) 
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', size='3%', pad=0.1)
            plt.colorbar(img, cax=cax)
            ax.text(10,10,i+1,fontsize=10)
            ax.axis('off')

            file_name = save_path.split('/')[-1]
            plt.savefig(f'{save_path}/{file_name}_frame_{i}.png')
            logging.info('Frame {} saved!'.format(i))
        return np.asarray(der_series)
    else:
        return np.asarray(der_series)


class hystTool():
    """ Apply hysteresis thresholding for cell detection in confocal image.
    Optimised for image of the individual cells.

    """
    def __init__(self, img, sigma, sd, mean=0, sd_lvl=2, high=0.8, low_init=0.05, mask_diff=50):
        """ Lower threshold calculations for hysteresis membrane detection functions.

        """
        self.img = img
        self.gauss = filters.gaussian(self.img, sigma=sigma)

        mask_img = ma.masked_greater_equal(img, sd_lvl*sd)
        low = low_init
        diff = np.size(self.img)

        while diff > mask_diff:
            mask_hyst = filters.apply_hysteresis_threshold(self.gauss,
                                                          low=low*np.max(self.gauss),
                                                          high=high*np.max(self.gauss))
            diff = np.sum(ma.masked_where(~mask_hyst, mask_img) > 0)

            if all([diff < mask_diff, low == low_init]):
                raise ValueError('Initial lower threshold is too low!')
            low += 0.01
            if low >= high:
                logging.fatal('LOW=HIGH, thresholding failed!')
                break
        logging.info(f'Lower threshold {round(low, 2)}')
        self.low = low

    def cell_mask(self, high=0.8, mode='single'):
        """ Creating binary mask for homogeneous fluoresced cell by SD thresholding and hysteresis smoothing.
        Detecting one cell in frame, with largest area ('single' mode) or all thresholded areas ('multi' mode).
        """
        raw_mask = filters.apply_hysteresis_threshold(self.gauss,
                                                      low=self.low*np.max(self.gauss),
                                                      high=high*np.max(self.gauss))
        labels_cells, cells_conunt = ndi.label(raw_mask)
        logging.info(f'{cells_conunt} cells detected')
        if mode == 'single':
            if cells_conunt > 1:
                size_list = [np.sum(ma.masked_where(labels_cells ==  cell_num, labels_cells).mask) for cell_num in range(cells_conunt)]
                logging.info(f'Cells sizes {size_list}')
                mask = ma.masked_where(labels_cells == size_list.index(max(size_list))+1, labels_cells).mask
            else:
                mask = raw_mask
            return mask, labels_cells
        elif mode == 'multi':
            return raw_mask, labels_cells

    def memb_mask(self):
        pass


# def hystLow(img, img_gauss, sd=0, sd_threshold=2, mean=0, diff=40, init_low=0.05, gen_high=0.8, mode='memb'):
#     """ Lower treshold calculations for hysteresis membrane detection function hystMemb.

#     diff - int, difference (in px number) between hysteresis mask and img without greater values
#     gen_high, sd, mean - see hystMemb

#     mode - 'cell': only sd treshold calc, 'memb': both tresholds calc

#     """
#     if mode == 'memb':
#         masks = {'2sd': ma.masked_greater_equal(img, sd_threshold*sd),  # values greater then 2 noise sd 
#                  'mean': ma.masked_greater(img, mean)}       # values greater then mean cytoplasm intensity
#     elif mode == 'cell':
#         masks = {'2sd': ma.masked_greater_equal(img, sd_threshold*sd)}

#     low_val = {}
#     control_diff = False
#     for mask_name in masks:
#         mask_img = masks[mask_name]

#         logging.info(f'Mask {mask_name} lower treshold fitting in progress')

#         mask_hyst = filters.apply_hysteresis_threshold(img_gauss,
#                                                       low=init_low*np.max(img_gauss),
#                                                       high=gen_high*np.max(img_gauss))
#         diff_mask = np.sum(ma.masked_where(~mask_hyst, mask_img) > 0)

#         if diff_mask < diff:
#             raise ValueError('Initial lower threshold is too low!')
#         logging.info('Initial masks difference {}'.format(diff_mask))

#         low = init_low
#         # control_diff = 1
#         while diff_mask >= diff:
#             mask_hyst = filters.apply_hysteresis_threshold(img_gauss,
#                                                           low=low*np.max(img_gauss),
#                                                           high=gen_high*np.max(img_gauss))
#             diff_mask = np.sum(ma.masked_where(~mask_hyst, mask_img) > 0)

#             low += 0.01
#             # is cytoplasm mean mask at initial lower threshold value closed? prevent infinit cycle
#             if low >= gen_high:
#                 logging.fatal('Lower treshold for {} mask {:.2f}, control difference {}px'.format(mask_name, low, diff_mask))
#                 break
#                 raise RuntimeError('Membrane in mean mask doesn`t detected at initial lower threshold value!')
    

#         # is cytoplasm mask at setted up difference value closed?
#         if mask_name == 'mean':
#             control_diff = np.all((segmentation.flood(mask_hyst, (0, 0)) + mask_hyst))
#             if control_diff == True:
#                 logging.fatal('Lower treshold for {} mask {:.2f}, masks difference {}px'.format(mask_name, low, diff_mask))
#                 raise ValueError('Membrane in {} mask doesn`t closed, mebrane unlocated at this diff value (too low)!'.format(mask_name))

#         low_val.update({mask_name : low})
#     logging.info(f'Lower tresholds {low_val}')

#     return low_val


# def hystMemb(img, roi_center, roi_size=30, noise_size=20, low_diff=40, gen_high=0.8, sigma=3):
#     """ Function for membrane region detection with hysteresis threshold algorithm.
#     Outdide edge - >= 2sd noise
#     Inside edge - >= cytoplasm mean intensity

#     Require hystLow function for lower hysteresis threshold calculations.

#     img - imput z-stack frame;
#     roi_center - list of int [x, y], coordinates of center of the cytoplasmic ROI for cytoplasm mean intensity calculation;
#     roi_size - int, cutoplasmic ROI side size in px (ROI is a square area);
#     noise_size - int, size in px of region for noise sd calculation (square area witf start in 0,0 coordinates);
#     sd_low - float, hysteresis algorithm lower threshold for outside cell edge detection,
#              > 2sd of noise (percentage of maximum frame intensity);
#     mean_low - float, hysteresis algorithm lower threshold for inside cell edge detection,
#              > cytoplasmic ROI mean intensity (percentage of maximum frame intensity);
#     gen_high - float,  general upper threshold for hysteresis algorithm (percentage of maximum frame intensity);
#     sigma - int, sd for gaussian filter.

#     Returts membrane region boolean mask for input frame.

#     """
#     img = backCon(img, dim=2)
#     img_gauss = filters.gaussian(img, sigma=sigma)

#     noise_sd = np.std(img[:noise_size, :noise_size])
#     logging.info('Frame noise SD={:.3f}'.format(noise_sd))

#     roi_mean = np.mean(img[roi_center[0] - roi_size//2:roi_center[0] + roi_size//2, \
#                            roi_center[1] - roi_size//2:roi_center[1] + roi_size//2])  # cutoplasmic ROI mean celculation
#     logging.info('Cytoplasm ROI mean intensity {:.3f}'.format(roi_mean))

#     low_val = hystLow(img, img_gauss, sd=noise_sd, mean=roi_mean, diff=low_diff, gen_high=gen_high)

#     mask_2sd = filters.apply_hysteresis_threshold(img_gauss,
#                                                   low=low_val['2sd']*np.max(img_gauss),
#                                                   high=gen_high*np.max(img_gauss))
#     mask_roi_mean = filters.apply_hysteresis_threshold(img_gauss,
#                                                       low=low_val['mean']*np.max(img_gauss),
#                                                       high=gen_high*np.max(img_gauss))
#     # filling external space and create cytoplasmic mask 
#     mask_cytoplasm = mask_roi_mean + segmentation.flood(mask_roi_mean, (0, 0))

#     return mask_2sd, mask_roi_mean, ma.masked_where(~mask_cytoplasm, mask_2sd)


if __name__=="__main__":
    pass


# That's all!