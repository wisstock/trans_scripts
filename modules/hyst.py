#!/usr/bin/env python3

""" Copyright © 2020-2021 Borys Olifirov
Cell detection with hysteresis thresholding (require one image), hystTools class.

"""

import os
import logging

import numpy as np
import numpy.ma as ma

import matplotlib
import matplotlib.pyplot as plt

# from skimage.external import tifffile
from skimage import filters
from skimage import measure
from skimage import segmentation

from scipy.ndimage import measurements as msr
from scipy import signal
from scipy import ndimage as ndi

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
        
        # # fixed-value masked image saving, for debuging only
        # plt.figure()
        # ax0 = plt.subplot()
        # img0 = ax0.imshow(mask_img)
        # plt.savefig(f'mask_{int(threshold_value)}.png')

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

        # # final masks difference, for debuging only
        # plt.figure()
        # ax0 = plt.subplot()
        # img0 = ax0.imshow(ma.masked_where(~mask_hyst, mask_img))
        # plt.savefig(f'mask_low_{int(threshold_value)}.png')
        return low

    def __create_sd_mask(self, img):
        """ Create SD mask for image.
        """
        img_gauss = filters.gaussian(img, sigma=self.sigma, truncate= self.truncate)
        sd = np.std(img[:self.sd_area, :self.sd_area])
        threshold_sd = self.sd_lvl*sd
        logging.info(f'SD mask threshold {threshold_sd}')
        img_mask = filters.apply_hysteresis_threshold(img_gauss,
                                                     low=self.__low_calc(img, img_gauss, threshold_sd) * np.max(img_gauss),
                                                     high=self.high*np.max(img_gauss))
        # logging.info(f'{mode} mask builded successfully, {np.shape(img_mask)}')
        plt.close('all')
        return img_mask, sd

    def __create_roi_mask(self, img):
        """ Create ROI mean  mask for image, default create ROI across of cell center of mass.
        """
        img_gauss = filters.gaussian(img, sigma=self.sigma, truncate=self.truncate)
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