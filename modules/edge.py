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

# from skimage.external import tifffile
from skimage import filters
from skimage import measure
from skimage import segmentation

from scipy.ndimage import measurements as msr
from scipy import signal
from scipy import ndimage as ndi


def deltaF(int_list, f_0_win=2):
    """ Function for colculation ΔF/F0 for data series.
    f_0_win - window for F0 calculation (mean of first 2 values by defoult).

    """
    f_0 = np.mean(int_list[:f_0_win])
    return [(i - f_0)/f_0 for i in int_list[f_0_win:]]


def back_rm(img, edge_lim=20, dim=3):
    """ Background extraction in TIFF series

    For confocal Z-stacks only!
    dem = 2 for one frame, 3 for z-stack

    """

    if dim == 3:
        mean_back = np.mean(img[:,:edge_lim,:edge_lim])

        logging.debug('Mean background, {} px region: {:.3f}'.format(edge_lim, mean_back))

        img_out = np.copy(img)
        img_out = img_out - mean_back
        img_out[img_out < 0] = 0
        return img_out
    elif dim == 2:
        mean_back = np.mean(img[:edge_lim,:edge_lim])

        logging.debug(f'Mean background, {edge_lim} px region: {mean_back}')

        img = np.copy(img)
        img = img - mean_back
        img[img < 0] = 0
        return img


# def series_sum_int(img_series, mask):
#     """ Calculation of summary intensity of masked region along time series frames

#     """
#     return [round(np.sum(ma.masked_where(~mask, img)) / np.sum(mask), 3) for img in img_series]


def alex_delta(series, mask=False, baseline_frames=5, max_frames=[10, 15], sd_tolerance=2, output_path=False):
    """ Detecting increasing and decreasing areas, detection limit - sd_tolerance * sd_cell.

    """
    baseline_img = np.mean(series[:baseline_frames,:,:], axis=0)
    max_img = np.mean(series[max_frames[0]:max_frames[1],:,:], axis=0)

    cell_sd = np.std(ma.masked_where(~mask, series[max_frames[0],:,:]))
    logging.info(f'Cell area SD={round(cell_sd, 2)}')

    delta_img = max_img - baseline_img  

    delta_img[delta_img > cell_sd * sd_tolerance] = 1
    delta_img[delta_img < -cell_sd * sd_tolerance] = -1
    delta_img = ma.masked_where(~mask, delta_img)


    if output_path:
        plt.figure()
        ax = plt.subplot()
        img = ax.imshow(delta_img, cmap='bwr')
        img.set_clim(vmin=-1., vmax=1.) 
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img, cax=cax)
        ax.axis('off')

        plt.savefig(f'{output_path}/alex_mask.png')
        logging.info('Alex F/F0 mask saved!')
        plt.close('all')

        return np.asarray(delta_img)
    else:
        return np.asarray(delta_img)


def series_point_delta(series, mask=False, baseline_frames=3, sigma=4, kernel_size=5, output_path=False):
    trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for gaussian fliter according to sigma value and kernel size
    img_series = np.asarray([filters.gaussian(series[i], sigma=sigma, truncate=trun(kernel_size, sigma)) for i in range(np.shape(series)[0])])

    baseline_img = np.mean(img_series[:baseline_frames,:,:], axis=0)

    delta = lambda f, f_0: (f - f_0)/f_0 if f_0 > 0 else f_0 
    vdelta = np.vectorize(delta)

    if type(mask) is list:
        delta_series = [ma.masked_where(~mask[i], vdelta(img_series[i], baseline_img)) for i in range(len(img_series))]
    elif mask == 'full_frame':
        delta_series = [vdelta(i, baseline_img) for i in img_series]
    elif mask:
        delta_series = [ma.masked_where(~mask, vdelta(i, baseline_img)) for i in img_series]
    else:
        raise TypeError('INCORRECT mask option!')

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


def series_derivate(series, mask=False, mask_num=0, sigma=4, kernel_size=3, sd_mode='cell', sd_area=50, sd_tolerance=False, left_w=1, space_w=0, right_w=1, output_path=False, abs_amplitude=False):
    """ Calculation of derivative image series (difference between two windows of interes).

    SD mode - noise SD/'noise' (in sd_area region) or cell region SD/'cell' (in mask region)

    """
    trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for gaussian fliter according to sigma value and kernel size
    if sigma == 'no_gauss':
        gauss_series = series
    else:
        gauss_series = np.asarray([filters.gaussian(series[i], sigma=sigma, truncate=trun(kernel_size, sigma)) for i in range(np.shape(series)[0])])

    der_series = []
    for i in range(np.shape(gauss_series)[0] - (left_w+space_w+right_w)):
        der_frame = np.mean(gauss_series[i+left_w+space_w:i+left_w+space_w+right_w], axis=0) - np.mean(gauss_series[i:i+left_w], axis=0) 
        if sd_tolerance:
            if sd_mode == 'cell':
                logging.info('Cell region SD mode selected')
                der_sd = np.std(ma.masked_where(~mask, der_frame))
            else:
                logging.info('Outside region SD mode selected')
                der_sd = np.std(der_frame[:sd_area, sd_area])
            der_frame[der_frame > der_sd * sd_tolerance] = 1
            der_frame[der_frame < -der_sd * sd_tolerance] = -1
        if type(mask) is not str:
            der_series.append(ma.masked_where(~mask, der_frame))
        elif mask == 'full_frame':
            der_series.append(der_frame)
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
            ax.text(10,10,i+1,fontsize=10, color='w')
            ax.axis('off')

            file_name = save_path.split('/')[-1]
            plt.savefig(f'{save_path}/{file_name}_frame_{i+1}.png')
            plt.close('all')
        logging.info('Derivate images saved!')
        return np.asarray(der_series)
    else:
        return np.asarray(der_series)


class hystTool():
    """ Cells detection with hysteresis thresholding.

    """
    def __init__(self, img, sd_area=20, roi_area=10, mean=False, sd_lvl=2, high=0.8, low_init=0.05, low_detection=0.5, mask_diff=50, inside_mask_diff=50, sigma=3, kernel_size=5):
        """ Detection of all cells with init lower threshold and save center of mass coordinates for each cell.

        """
        self.img = img                            # image for hystTool initialization and cell counting                   
        self.high = high                          # high threshold for all methods
        self.low_init = low_init                  # initial low threshold for cell detection
        self.low_detection = low_detection        # low threshold for cell counting during initialization
        self.mask_diff = mask_diff                # difference between fixed-value and hysteresis masks for outside mask (2SD mask)
        self.inside_mask_diff = inside_mask_diff  # difference between fixed-value and hysteresis masks for inside mask (cytoplasm mean mask)
        self.sd_area = sd_area                    # area in px for frame SD calculation
        self.roi_area = roi_area                  # area in px for square ROI creating and mean intensity calculation
        self.sd_lvl = sd_lvl                      # multiplication factor of noise SD value for outside fixed-value mask building
        self.mean = mean                          # coordinate of cytoplasm ROI center
        self.kernel_size = kernel_size            # kernel size of the Gaussian filter
        self.sigma = sigma                        # sigma of the Gaussian filter

        trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for Gaussian fliter according to sigma value and kernel size
        self.truncate = trun(self.kernel_size, self.sigma)
        self.gauss = filters.gaussian(self.img, sigma=sigma, truncate= self.truncate)

        self.detection_mask = filters.apply_hysteresis_threshold(self.gauss,
                                                                 low=self.low_detection*np.max(self.gauss),
                                                                 high=self.high*np.max(self.gauss))

        self.cells_labels, self.cells_num = ndi.label(self.detection_mask)
        if self.cells_num == 0:
            logging.warning(f'Cells DOESN`T detected! You should try increase low_detection')
            raise ValueError

        cells_center_float = ndi.center_of_mass(self.cells_labels, self.cells_labels, range(1, self.cells_num+1))
        self.cells_center = [[int(x) for x in i] for i in cells_center_float]
        self.cells_center_dict = dict(zip(range(1, self.cells_num+1), self.cells_center))

        if self.cells_num == 1:               # just for fun c:
            logging_cell_num = 'cell'
        else:
            logging_cell_num = 'cells_labels'
        logging.info(f'Detected {self.cells_num} {logging_cell_num} with center of mass coord. {self.cells_center_dict}')

    def __low_calc(self, img, gauss, threshold_value):
        """ Lower threshold calculations for hysteresis detection functions.

        """
        mask_img = ma.masked_greater_equal(img, threshold_value)
        
        # fixed-value masked image saving, for debuging only
        plt.figure()
        ax0 = plt.subplot()
        img0 = ax0.imshow(mask_img)
        plt.savefig(f'mask_{int(threshold_value)}.png')

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
        logging.debug(f'Lower threshold {round(low, 2)}')

        # final masks difference, for debuging only
        plt.figure()
        ax0 = plt.subplot()
        img0 = ax0.imshow(ma.masked_where(~mask_hyst, mask_img))
        plt.savefig(f'mask_low_{int(threshold_value)}.png')
        return low

    def __create_sd_mask(self, img):
        """ Create SD mask for image.
        """
        img_gauss = filters.gaussian(img, sigma=self.sigma, truncate= self.truncate)
        sd = np.std(img[:self.sd_area, :self.sd_area])
        img_mask = filters.apply_hysteresis_threshold(img_gauss,
                                                     low=self.__low_calc(img, img_gauss, self.sd_lvl*sd) * np.max(img_gauss),
                                                     high=self.high*np.max(img_gauss))
        # logging.info(f'{mode} mask builded successfully, {np.shape(img_mask)}')
        plt.close('all')
        return img_mask, sd

    def __create_roi_mask(self, img):
        """ Create ROI mean  mask for image, default create ROI across of cell center of mass.
        """
        img_gauss = filters.gaussian(img, sigma=self.sigma, truncate= self.truncate)
        if self.mean:
            roi_center = self.mean
            logging.info(f'Custom ROI center {roi_center}')
        else:
            roi_center = self.cells_center[0]
            logging.info(f'CoM ROI center {roi_center}')
        roi_mean = np.mean(img[roi_center[0] - self.roi_area//2:roi_center[0] + self.roi_area//2, \
                       roi_center[1] - self.roi_area//2:roi_center[1] + self.roi_area//2])
        img_mask = filters.apply_hysteresis_threshold(img_gauss,
                                                      low=self.__low_calc(img, img_gauss, roi_mean) * np.max(img_gauss),
                                                      high=self.high*np.max(img_gauss))
        return img_mask, roi_mean

    def cell_mask(self, frame):
        """ Create series of masks for each frame.
        If there are more than one cells per frame, create dict with mask series for each cell. 
        """
        if self.cells_num == 1:
            frame_mask, frame_sd = self.__create_sd_mask(frame)
            logging.info(f'Noise SD={round(frame_sd, 3)}')
            return frame_mask
        else:
            # NOT READY FOR MULTIPLE CELLS!
            logging.fatal('More then one cell, CAN`T create masks series!')

    def huge_cell_mask(self):
        """ Creating binary mask for homogeneous fluoresced cell by SD thresholding and hysteresis smoothing.
        Detecting one cell in frame, with largest area.

        """
        # NOW DOESN'T WORKING, NEED UPDATE!

        raw_mask = filters.apply_hysteresis_threshold(self.gauss,
                                                      low=self.__low_calc(self.img)*np.max(self.gauss),
                                                      high=self.high*np.max(self.gauss))
        logging.info('Mask builded successfully')
        labels_cells, cells_conunt = ndi.label(raw_mask)
        logging.info(f'{cells_conunt} cells detected')
        if cells_conunt > 1:
            size_list = [np.sum(ma.masked_where(labels_cells == cell_num, labels_cells).mask) for cell_num in range(cells_conunt)]
            logging.info(f'Cells sizes {size_list}')
            mask = ma.masked_where(labels_cells == size_list.index(max(size_list))+1, labels_cells).mask
        else:
            mask = raw_mask
        return mask, labels_cells

    def memb_mask(self, frame, roi_size=10):
        """ Membrane region detection.
        Outside edge - >= 2sd noise
        Inside edge - >= cytoplasm mean intensity

       img - imput z-stack frame;
       roi_center - list of int [x, y], coordinates of center of the cytoplasmic ROI for cytoplasm mean intensity calculation;
       roi_size - int, cytoplasmic ROI side size in px (ROI is a square area);
       noise_size - int, size in px of region for noise sd calculation (square area with start in 0,0 coordinates);
       sd_low - float, hysteresis algorithm lower threshold for outside cell edge detection,
                > 2sd of noise (percentage of maximum frame intensity);
       mean_low - float, hysteresis algorithm lower threshold for inside cell edge detection,
                > cytoplasmic ROI mean intensity (percentage of maximum frame intensity);
       gen_high - float,  general upper threshold for hysteresis algorithm (percentage of maximum frame intensity);
       sigma - int, sd for Gaussian filter.

        """
        # THIS IS TEMPORARY METHOD, FOR ONE CELL AT IMAGE ONLY!
        sd_mask, frame_sd = self.__create_sd_mask(frame)
        roi_mask, frame_roi = self.__create_roi_mask(frame)
        logging.info(f'Noise SD={round(frame_sd, 3)}, ROI mean intensity={round(frame_roi, 3)}')

        # filling external space and create cytoplasmic mask 
        cytoplasm_mask = roi_mask + segmentation.flood(roi_mask, (0, 0))
        if np.all(cytoplasm_mask):
            logging.fatal('Cytoplasm mask is NOT closed, CAN`T create correct membrane mask!')

        membrane_mask = ma.masked_where(~cytoplasm_mask, sd_mask)

        return [sd_mask, roi_mask, cytoplasm_mask, membrane_mask]

    def ctrl_imsave():
        """ Save control images: fixed-value masks, cells labels etc.

        """
        pass

if __name__=="__main__":
    pass


# That's all!