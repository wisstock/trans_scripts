#!/usr/bin/env python3

""" Copyright © 2020-2021 Borys Olifirov
Toolkit.
- Function for series analysis (require image series):
  back_rm
  series_sum_int
  series_point_delta
  series_derivate
- Cell detection with hysteresis thresholding (require one image):
  hystTools class

Optimised at confocal images of HEK 293 cells.

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


def back_rm(img, edge_lim=20, dim=3):
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


def series_sum_int(img_series, mask):
    """ Calcualtion of summary intensity of masked region along time series frames

    """
    return [round(np.sum(ma.masked_where(~mask, img)) / np.sum(mask), 3) for img in img_series]


def series_point_delta(series, mask=False, mask_series=False, baseline_frames=3, sigma=4, kernel_size=5, output_path=False):
    trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for gaussian fliter according to sigma value and kernel size
    img_series = np.asarray([filters.gaussian(series[i], sigma=sigma, truncate=trun(kernel_size, sigma)) for i in range(np.shape(series)[0])])

    baseline_img = np.mean(img_series[:baseline_frames,:,:], axis=0)

    delta = lambda f, f_0: (f - f_0)/f_0 if f_0 > 0 else f_0 
    vdelta = np.vectorize(delta)

    if mask_series:
        delta_series = [ma.masked_where(~mask_series[i], vdelta(img_series[i], baseline_img)) for i in range(len(img_series))]
    elif mask:
        delta_series = [ma.masked_where(~mask, vdelta(i, baseline_img)) for i in img_series]
    else:
        raise TypeError('NO mask available!')

    if output_path:
        save_path = f'{output_path}/delta_F'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in range(len(delta_series)):
            frame = delta_series[i]

            plt.figure()
            ax = plt.subplot()
            img = ax.imshow(frame, cmap='jet')
            img.set_clim(vmin=-1., vmax=1.) 
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', size='3%', pad=0.1)
            plt.colorbar(img, cax=cax)
            ax.text(10,10,i+1,fontsize=10)
            ax.axis('off')

            file_name = save_path.split('/')[-1]
            plt.savefig(f'{save_path}/{file_name}_frame_{i+1}.png')
            logging.info('Delta F frame {} saved!'.format(i))
            plt.close('all')
        return np.asarray(delta_series)
    else:
        return np.asarray(delta_series)


def series_derivate(series, mask=False, mask_series=False, mask_num=0, sigma=4, kernel_size=3,  sd_area=50, sd_tolerance=False, left_w=1, space_w=0, right_w=1, output_path=False):
    """ Calculation of derivative image series (difference between two windows of interes).

    """
    trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for gaussian fliter according to sigma value and kernel size
    gauss_series = np.asarray([filters.gaussian(series[i], sigma=sigma, truncate=trun(kernel_size, sigma)) for i in range(np.shape(series)[0])])

    logging.info(f'Derivate sigma={sigma}')

    der_series = []
    for i in range(np.shape(gauss_series)[0] - (left_w+space_w+right_w)):
        der_frame = np.mean(gauss_series[i+left_w+space_w:i+left_w+space_w+right_w], axis=0) - np.mean(gauss_series[i:i+left_w], axis=0) 
        if sd_tolerance:
            der_sd = np.std(der_frame[:sd_area, sd_area])
            der_frame[der_frame > der_sd * sd_tolerance] = 1
            der_frame[der_frame < -der_sd * sd_tolerance] = -1
        if mask_series:
            der_series.append(ma.masked_where(~mask_series[mask_num], der_frame)) 
        elif mask:
            der_series.append(ma.masked_where(~mask, der_frame))
        else:
            raise TypeError('NO mask available!')
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
            plt.savefig(f'{save_path}/{file_name}_frame_{i+1}.png')
            logging.info(f'Derivate frame {i+1} saved!')
            plt.close('all')
        return np.asarray(der_series)
    else:
        return np.asarray(der_series)


class hystTool():
    """ Cells detection with hysteresis thresholding.

    """
    def __init__(self, img, sd_area=20, mean=0, sd_lvl=2, high=0.8, low_init=0.05, low_detection=0.3, mask_diff=50, sigma=4, kernel_size=3):
        """ Detection of all cells with init lower threshold and save center of mass coordinates for each cell.

        """
        self.img = img
        self.high = high
        self.low_init = low_init
        self.low_detection = low_detection
        self.mask_diff = mask_diff
        self.sd_area = sd_area
        self.sd_lvl = sd_lvl
        self.mean = mean
        self.kernel_size = kernel_size
        self.sigma = sigma

        trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for gaussian fliter according to sigma value and kernel size
        self.truncate = trun(self.kernel_size, self.sigma)
        self.gauss = filters.gaussian(self.img, sigma=sigma, truncate= self.truncate)

        self.detection_mask = filters.apply_hysteresis_threshold(self.gauss,
                                                                 low=self.low_detection*np.max(self.gauss),
                                                                 high=self.high*np.max(self.gauss))

        self.cells_labels, self.cells_num = ndi.label(self.detection_mask)
        if self.cells_num == 0:
            logging.fatal(f'In file {self.img_name} cells DOESN`T detected! You should try increase low_detection')
            raise ValueError

        cells_center_float = ndi.center_of_mass(self.cells_labels, self.cells_labels, range(1, self.cells_num+1))
        self.cells_center = [[int(x) for x in i] for i in cells_center_float]
        self.cells_center_dict = dict(zip(range(1, self.cells_num+1), self.cells_center))
        logging.info(f'Detected {self.cells_num} cells with center of mass coord. {self.cells_center_dict}')

    def __low_calc(self, img, gauss, sd):
        """ Lower threshold calculations for hysteresis membrane detection functions.

        """
        mask_img = ma.masked_greater_equal(img, self.sd_lvl*sd)
        low = self.low_init
        diff = np.size(img)

        while diff > self.mask_diff:
            mask_hyst = filters.apply_hysteresis_threshold(gauss,
                                                          low=low*np.max(gauss),
                                                          high=self.high*np.max(gauss))
            diff = np.sum(ma.masked_where(~mask_hyst, mask_img) > 0)
            if all([diff < self.mask_diff, low == self.low_init]):
                logging.fatal('Initial lower threshold is too low!')
                break
            low += 0.01
            if low >= self.high:
                logging.fatal('LOW=HIGH, thresholding failed!')
                break
        logging.info(f'Lower threshold {round(low, 2)}')
        return low

    def cell_mask(self, img_series):
        """ Create series of masks for each frame.
        If there are more than one cells per frame, create dict with mask series for each cell. 
        """
        if self.cells_num == 1:
            mask_series = []
            for i in range(np.shape(img_series)[0]):
                frame = img_series[i]
                frame_gauss = filters.gaussian(frame, sigma=self.sigma, truncate= self.truncate)
                frame_sd = np.std(frame[:self.sd_area, :self.sd_area])
                frame_mask = filters.apply_hysteresis_threshold(frame_gauss,
                                                                low=self.__low_calc(frame, frame_gauss, frame_sd)*np.max(frame_gauss),
                                                                high=self.high*np.max(frame_gauss))
                mask_series.append(frame_mask)
            return mask_series
        else:
            pass

    def huge_cell_mask(self):
        """ Creating binary mask for homogeneous fluoresced cell by SD thresholding and hysteresis smoothing.
        Detecting one cell in frame, with largest area.

        """
        raw_mask = filters.apply_hysteresis_threshold(self.gauss,
                                                      low=self.__low_calc(self.img)*np.max(self.gauss),
                                                      high=self.high*np.max(self.gauss))
        labels_cells, cells_conunt = ndi.label(raw_mask)
        logging.info(f'{cells_conunt} cells detected')
        if cells_conunt > 1:
            size_list = [np.sum(ma.masked_where(labels_cells ==  cell_num, labels_cells).mask) for cell_num in range(cells_conunt)]
            logging.info(f'Cells sizes {size_list}')
            mask = ma.masked_where(labels_cells == size_list.index(max(size_list))+1, labels_cells).mask
        else:
            mask = raw_mask
        return mask, labels_cells

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