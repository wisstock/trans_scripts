#!/usr/bin/env python3
""" Copyright © 2020-2022 Borys Olifirov
Time series with simultaneous HPCA transfection and calcium dye loading.
Multiple stimulation during registration.

Does NOT requires oifpars module.

Useful links:
https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/ImageProcessing-html/topic4.htm

"""

        # fig, ax = plt.subplots()
        # ax.imshow(self.total_mask_ctrl_img, cmap='jet')
        # plt.show()

import sys
import os
import logging

import numpy as np
import numpy.ma as ma

from scipy.signal import find_peaks
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import distance_transform_cdt
import scipy.ndimage as ndi

import yaml
import pandas as pd

from skimage import filters
from skimage import measure
from skimage import morphology
from skimage.color import label2rgb
from skimage import util
from skimage import exposure
from skimage import transform

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

import oiffile as oif
import edge


class MultiData():
    """ Time series with simultaneous HPCA transfection and calcium dye loading.
    Multiple stimulation during registration.

    Does NOT requires oifpars module.

    """
    def __init__(self, oif_path, img_name, meta_dict, id_suffix, time_scale=1, baseline_frames=5, prot_bleach_correction=True):
        self.img_name = img_name
        self.id_suffix = id_suffix
        self.stim_power = meta_dict['power']

        base_path = f'{oif_path}/{img_name}_01.oif'  # default OIF index 01
        self.img_series = oif.OibImread(base_path)
        logging.info(f'Record {self.img_name} uploaded!')  # ({self.stim_power}%, {self.baseline_frames}|{self.stim_loop_num}x {self.stim_frames}|{self.tail_frames}) uploaded')

        # channel separation
        self.ca_series = np.array(edge.back_rm(self.img_series[0]), dtype='float')    # calcium dye channel array (Fluo-4)
        self.prot_series = np.array(edge.back_rm(self.img_series[1]), dtype='float')  # fluorescent labeled protein channel array (HPCA-TagRFP)

        # frames time line
        self.time_scale = time_scale
        self.time_line = np.asarray([i/time_scale for i in range(0,len(self.ca_series))])

        self.baseline_frames = baseline_frames

        # FP bleaching corrections
        if prot_bleach_correction:
            native_total_intensity = np.sum(np.mean(self.prot_series[:2], axis=0))
            self.prot_series = np.asarray([frame / (np.sum(frame)/native_total_intensity) for frame in self.prot_series])

        # create derevative series for Ca dye series
        zero_end = np.vstack((self.ca_series, np.zeros_like(self.ca_series[0:1])))                     # array with last 0 frame
        zero_start = np.vstack((np.zeros_like(self.ca_series[0:1]), self.ca_series))                   # array with first 0 frame
        self.derivate_series = np.subtract(zero_end, zero_start)[1:-1]                                 # derivative frames series
        self.derivate_profile = np.asarray([np.sum(np.abs(frame)) for frame in self.derivate_series])  # sum of abs values of derivative frames
        
        # # LOOP STIMULATION PROTOCOL WITH SEPARATED OIF-FILES FOR EACH RECORDING PART
        # self.baseline_frames = meta_dict['base']
        # self.stim_frames = meta_dict['stimul']
        # self.stim_loop_num = meta_dict['loop']
        # self.tail_frames = meta_dict['tail']
        # self.max_ca_frame = self.baseline_frames + self.stim_frames * self.stim_loop_num # index of frame after last stimulation

        # # record OIF files combining
        # base_path = f'{oif_path}/{img_name}_01.oif'
        # self.img_series = oif.OibImread(base_path)  # read baseline record
        # self.img_series = self.img_series.astype(int) 

        # for loop_num in range(2, self.stim_loop_num+2):
        #     loop_path = f'{oif_path}/{img_name}_0{loop_num}.oif'
        #     self.img_series = np.concatenate((self.img_series, oif.OibImread(loop_path)), axis=1)  # add each stimulation loop record
        
        # tail_path = f'{oif_path}/{img_name}_0{loop_num+1}.oif'
        # self.img_series = np.concatenate((self.img_series, oif.OibImread(tail_path)), axis=1)  # add tail record

    @classmethod
    def _select_larg_mask(self, raw_mask):
        # get larger mask element
        _element_label = measure.label(raw_mask)
        _element_area = {_element.area : _element.label for _element in measure.regionprops(_element_label)}
        _larger_mask = _element_label == _element_area[max(_element_area.keys())]
        return _larger_mask

    @classmethod
    def _get_mask_rim(self, raw_mask, rim_th=2):
        # get binary mask rim
        f_print = morphology.disk(rim_th)
        _dilate_mask = morphology.binary_dilation(raw_mask, footprint=morphology.disk(rim_th))
        _erode_mask = morphology.binary_erosion(raw_mask, footprint=morphology.disk(rim_th))
        return np.logical_and(_dilate_mask, ~_erode_mask)

    # cell masking
    def get_master_mask(self, sigma=1, kernel_size=5, mask_ext=5, nuclear_ext=2, multi_otsu_nucleus_mask=True):
        """ Whole cell mask building by Ca dye channel data with Otsu thresholding.
        Filters greater element of draft Otsu mask and return master mask array.

        Also create nuclear distances images (euclidean and chess metrics)

        mask_ext - otsu mask extension value in px

        """
        trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for gaussian fliter according to sigma value and kernel size

        self.detection_img = filters.gaussian(np.mean(self.ca_series, axis=0), sigma=sigma, truncate=trun(kernel_size, sigma))

        # multi Otsu mask for nucleus detection
        self.multi_otsu_nucleus_mask = multi_otsu_nucleus_mask
        if self.multi_otsu_nucleus_mask:
            multi_otsu = filters.threshold_multiotsu(self.detection_img, classes=3)
            self.element_label = np.digitize(self.detection_img, bins=multi_otsu)  # 1 - cell elements, 2 - intracellular and nuclear elements

            # get larger multi Otsu cellular element
            cell_element = (self.element_label == 1) | (self.element_label == 2)
            cell_border_mask = self._select_larg_mask(raw_mask=cell_element)
            self.cell_distances, _ = distance_transform_edt(~cell_border_mask, return_indices=True)
            self.cell_mask = self.cell_distances <= mask_ext

            # get larger multi Otsu intracellular element
            nuclear_element = self.element_label == 2
            nuclear_element_border = self._select_larg_mask(raw_mask=nuclear_element)
            self.nuclear_distances, _ = distance_transform_edt(~nuclear_element_border, return_indices=True)
            self.nuclear_distances_chess = distance_transform_cdt(~nuclear_element_border, metric='chessboard')
            self.nuclear_mask = self.nuclear_distances <= nuclear_ext

            self.master_mask = np.copy(self.cell_mask)
            self.master_mask[self.nuclear_mask] = 0
        else:
            logging.fatal('multi_otsu_nucleus_mask=False, please DO NOT this!')
            raise ValueError('incorrect master mask option')
            # # please, DON'T use this option!
            # otsu = filters.threshold_otsu(self.detection_img)
            # draft_mask = self.detection_img > otsu
            # self.element_label, self.element_num = measure.label(draft_mask, return_num=True)
            # logging.info(f'{self.element_num} Otsu mask elements detected')

            # detection_label = np.copy(self.element_label)
            # element_area = {element.area : element.label for element in measure.regionprops(detection_label)}
            # self.master_mask = detection_label == element_area[max(element_area.keys())]

            # # mask expansion
            # self.cell_distances, _ = distance_transform_edt(~self.master_mask, return_indices=True)
            # self.master_mask = self.cell_distances <= mask_ext

        self.total_byte_prot_img = filters.gaussian(util.img_as_ubyte(np.mean(self.prot_series, axis=0)/np.max(np.abs(np.mean(self.prot_series, axis=0)))), sigma=sigma, truncate=trun(kernel_size, sigma))
        self.total_mask_ctrl_img = label2rgb(self.master_mask, image=self.total_byte_prot_img, colors=['blue'], alpha=0.2)

    def find_stimul_peak(self, h=0.15, d=3, l_lim=5, r_lim=18):
        """ Require master_mask, results of get_master_mask!
        Find peaks on deltaF/F Fluo-4 derivate profile.
        h - minimal peaks height, in ΔF/F0
        d - minimal distance between peaks, in frames

        """
        self.stim_peak = find_peaks(edge.deltaF(self.derivate_profile), height=h, distance=d)[0]
        self.stim_peak = self.stim_peak[(self.stim_peak >= l_lim) & (self.stim_peak <= r_lim)]  # filter outer peaks
        logging.info(f'Detected peaks: {self.stim_peak}')

    # translocation regions masking
    def peak_img_diff(self, sigma=1, kernel_size=5, baseline_win=3, stim_shift=0, stim_win=3, up_min_tolerance=0.2, up_max_tolerance=0.75, down_min_tolerance=2, down_max_tolerance=0.75, path=False):
        """ Mask for up and down regions of FP channel data.
        baseline_win - indexes of frames for baseline image creation
        stim_shift - additional value for loop_start_index
        tolerance - tolerance value in au for mask creation, down < -tolerance, up > tolerance

        REQUIRE peaks position (find_stim_peak).

        """
        trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for gaussian fliter according to sigma value and kernel size
        prot_series_sigma = [filters.gaussian(i, sigma=sigma, truncate=trun(kernel_size, sigma)) for i in self.prot_series]
        baseline_prot_img = np.mean(prot_series_sigma[:baseline_win], axis=0)

        self.peak_diff_series = []
        self.up_diff_mask = []
        self.down_diff_mask = []
        self.up_diff_mask_prop = []
        self.comb_diff_mask = []
        for stim_position in self.stim_peak:
            diff_frames_start = stim_position + stim_shift
            diff_frames_end = stim_position + stim_shift + stim_win
            stim_mean_img = np.mean(prot_series_sigma[diff_frames_start:diff_frames_end], axis=0)
            stim_diff_img = stim_mean_img - baseline_prot_img

            # creating and normalization of differential image
            stim_diff_img[self.cell_distances >= 30] = 0 
            stim_diff_img = stim_diff_img/np.max(np.abs(stim_diff_img))
            self.peak_diff_series.append(stim_diff_img)

            # up regions thresholding
            frame_diff_up_mask = filters.apply_hysteresis_threshold(stim_diff_img,
                                                                    low=up_min_tolerance,
                                                                    high=up_max_tolerance)
            frame_diff_up_mask_elements, frame_diff_up_mask_elements_num = measure.label(frame_diff_up_mask, return_num=True)
            self.up_diff_mask.append(frame_diff_up_mask_elements)  # up mask elements labeling
            # logging.info(f'In stim frame {stim_position} finded {frame_diff_up_mask_elements_num} up regions')

            # down regions thresholding
            frame_diff_down_mask = filters.apply_hysteresis_threshold(stim_diff_img,
                                                                    low=down_min_tolerance,
                                                                    high=down_max_tolerance)
            self.down_diff_mask.append(frame_diff_down_mask)

            self.comb_diff_mask.append((frame_diff_up_mask*2) + (frame_diff_down_mask-2)*-1)

        # fig, ax = plt.subplots()
        # ax.imshow(self.up_diff_mask[0], c)
        # plt.show()

        # find better up mask (with maximal area)
        self.best_up_mask_index = np.argmax([np.sum(u_m != 0) for u_m in self.up_diff_mask])
        logging.info(f'Best up mask {self.best_up_mask_index+1} (stim frame {self.stim_peak[self.best_up_mask_index]})')

    def peak_img_deltaF(self, mode='delta', sigma=1, kernel_size=5, baseline_win=3, stim_shift=0, stim_win=3, deltaF_up=0.14, deltaF_down=-0.1, path=False):
        """ Pixel-wise ΔF/F0 calculation.
        baseline_frames - numbers of frames for mean baseline image calculation (from first to baseline_frames value frames)

        mode: `delta` - pixel-wise ΔF/F0, `diff` - differential image

        WARNING! The function is sensitive to cell shift during long registrations!

        """
        trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for gaussian fliter according to sigma value and kernel size
        delta = lambda f, f_0: (f - f_0)/f_0 if f_0 > 0 else f_0 
        vdelta = np.vectorize(delta)

        # px-wise delta for FP ch
        prot_series_sigma = [filters.gaussian(i, sigma=sigma, truncate=trun(kernel_size, sigma)) for i in self.prot_series]
        baseline_prot_img = np.mean(prot_series_sigma[:baseline_win], axis=0)
        self.deltaF_series = np.asarray([vdelta(i, baseline_prot_img) for i in prot_series_sigma])

        # px-wise delta for Ca dye ch
        ca_series_sigma = [filters.gaussian(i, sigma=sigma, truncate=trun(kernel_size, sigma)) for i in self.ca_series]
        baseline_ca_img = np.mean(ca_series_sigma[:baseline_win], axis=0) 
        self.deltaF_ca_series = np.asarray([vdelta(i, baseline_ca_img) for i in ca_series_sigma])

        self.stim_mean_series = []
        for stim_position in self.stim_peak:
            peak_frames_start = stim_position + stim_shift
            peak_frames_end = stim_position + stim_shift + stim_win
            self.stim_mean_series.append(np.mean(prot_series_sigma[peak_frames_start:peak_frames_end], axis=0))

        self.peak_deltaF_series = np.asarray([ma.masked_where(~self.master_mask, vdelta(i, baseline_prot_img)) for i in self.stim_mean_series])
        self.up_delta_mask = np.copy(self.peak_deltaF_series)
        self.up_delta_mask = self.up_delta_mask >= deltaF_up
        self.up_delta_mask[~np.broadcast_to(self.master_mask, np.shape(self.up_delta_mask))] = 0

    # advanced segmentation
    def diff_mask_segment(self, segment_num=3, segment_min_area=30):
        """ Up regions mask segmentation.

        """
        element_total_mask = self.up_diff_mask[self.best_up_mask_index]  # [60:180, 100:150]  # test element of cell7 up mask
        segment_mask = np.zeros_like(element_total_mask)

        for element_num in measure.regionprops(element_total_mask):
            element_mask = element_total_mask == element_num.label
            element_segment = np.copy(element_mask).astype('int32')

            if element_num.area // segment_num < segment_min_area:  # check element area limitation, if lower - no segmentation
                logging.warning(f'Element {element_num.label} segmentation range is too low ({element_num.area // segment_num}px < lim {segment_min_area}px)')
                np.putmask(segment_mask, element_mask, element_segment)
                continue
            else:
                px_num = 1
                for i, j in np.ndindex(element_segment.shape):
                    if element_segment[i, j] != 0:
                        element_segment[i, j] = px_num
                        px_num += 1
                segment_range = px_num // segment_num
                segment_lab_dict = {segment_i:[segment_i * segment_range - segment_range + 1, segment_i * segment_range]
                                    for segment_i in range(1, segment_num+1)}

                logging.info(f'Element {element_num.label} segmentation range is {segment_range}px')

                # outlier pixels handling, addition it to the last segment
                segment_lab_dict[list(segment_lab_dict)[-1]][-1] = px_num

                for segment_lab in segment_lab_dict.keys():
                    range_limits = segment_lab_dict[segment_lab]
                    element_segment[(element_segment >= range_limits[0]) & (element_segment <= range_limits[1])] = segment_lab
                np.putmask(segment_mask, element_mask, element_segment)
                
        self.up_segments_mask_array = np.stack((element_total_mask, segment_mask), axis=0)  # 0 - elements mask, 1 - segments mask
        self.up_segments_mask_ctrl_img = label2rgb(self.up_segments_mask_array[1], image=self.total_byte_prot_img, alpha=0.4)
        logging.info(f'Up mask elements segmented, segment num={segment_num}')

    def segment_dist_calc(self):
        """ Calculation of distance between nucleus and mask elements/segments.
        Require multi Otsu cell without nucleus region and previously segmented up mask elements mask (diff_mask_segment).

        """
        demo_ring_mask = np.copy(self.master_mask)
        demo_nuc_mask = np.copy(self.nuclear_mask)

        self.cytoplasm_dist = np.zeros(demo_ring_mask.shape, dtype='float64')  # np.zeros(((np.ndim(demo_ring_mask),) + demo_ring_mask.shape), dtype='int32')
        distance_transform_edt(~self.nuclear_mask, return_indices=True, distances=self.cytoplasm_dist)
        self.nuclear_dist = np.copy(self.cytoplasm_dist)
        self.cytoplasm_dist[~self.master_mask] = 0  # distancion mask

        up_element_mask = self.up_segments_mask_array[0]
        up_segment_mask = self.up_segments_mask_array[1]

        self.segment_dist = np.copy(up_element_mask)
        self.dist_df = pd.DataFrame(columns=['ID',            # recording ID
                                             'mask_element',  # up mask element number
                                             'segment',       # up mask element's segment number
                                             'dist'])         # segment distance

        for element_num in measure.regionprops(up_element_mask):
            element_mask = up_element_mask == element_num.label
            one_element = np.copy(up_segment_mask)
            one_element[~element_mask] = 0
            for segment_num in measure.regionprops(one_element):
                one_segment = np.copy(one_element)
                one_segment = one_segment == segment_num.label  # mask for individual segment
                one_segment_dist = round(np.mean(ma.masked_where(~one_segment, self.nuclear_dist)), 3)
                np.putmask(self.segment_dist, one_segment, one_segment_dist)  # distances image updating

                dist_series = pd.Series([f'{self.img_name}{id_suffix}',
                                         element_num.label,
                                         segment_num.label,
                                         one_segment_dist],
                                        index=self.dist_df.columns)
                self.dist_df = self.dist_df.append(dist_series, ignore_index=True)

        return self.dist_df

        fig, ax = plt.subplots()
        ax.imshow(self.segment_dist, cmap='jet')  # self.up_segments_mask_ctrl_img
        plt.show()

    # extract mask profile
    def ca_profile(self, mask=False):
        if not mask:
            mask = self.master_mask
        return np.asarray([round(np.sum(ma.masked_where(~mask, img)) / np.sum(mask), 3) for img in self.ca_series])

    def prot_profile(self, mask=None):
        """ Test

        """
        return np.asarray([round(np.sum(ma.masked_where(~mask, img)) / np.sum(mask), 3) for img in self.prot_series])

    # data frame saving
    def save_profile_df(self, id_suffix):
        """ Masked regions intensity profiles data frame

        """
        self.profile_df = pd.DataFrame(columns=['ID',           # recording ID
                                                'power',        # 405 nm stimulation power (%)
                                                'ch',           # channel (FP or Ca dye)
                                                'frame',        # frame number
                                                'time',         # frame time (s)
                                                'mask',         # mask type (master, up, down)
                                                'mask_region',  # mask region (1 for master or down)
                                                'mean',         # mask mean intensity
                                                'delta',        # mask ΔF/F
                                                'rel'])         # mask sum / master mask sum
        # Ca dye
        ca_profile = self.ca_profile()
        ca_profile_delta = edge.deltaF(ca_profile, f_0_win=self.baseline_frames)
        for ca_val in range(0, len(self.ca_series)):
            point_series = pd.Series([f'{self.img_name}{id_suffix}', # recording ID
                                              self.stim_power,               # 405 nm stimulation power (%)
                                              'ca',                          # channel (FP or Ca dye)
                                              ca_val+1,                      # frame number
                                              self.time_line[ca_val],        # frame time (s)
                                              'master',                      # mask type (only master for Ca dye channel)
                                              1,                             # mask region (1 for master or down)
                                              ca_profile[ca_val],            # mask mean intensity
                                              ca_profile_delta[ca_val],      # mask ΔF/F
                                              1],                            # mask sum / master mask sum (1 for Ca dye channel)
                                      index=self.profile_df.columns)
            self.profile_df = self.profile_df.append(point_series, ignore_index=True)
        
        # FP intensity
        # FP master
        fp_master_profile = self.prot_profile(mask=self.master_mask)
        fp_master_profile_delta = edge.deltaF(fp_master_profile, f_0_win=self.baseline_frames)
        fp_master_sum = np.asarray([round(np.sum(ma.masked_where(~self.master_mask, img)), 3) for img in self.prot_series]) # intensity sum for master mask

        for fp_val in range(0, len(self.prot_series)):
            point_series = pd.Series([f'{self.img_name}{id_suffix}',            # recording ID
                                              self.stim_power,                  # 405 nm stimulation power (%)
                                              'fp',                             # channel (FP or Ca dye)
                                              fp_val+1,                         # frame number
                                              self.time_line[fp_val],           # frame time (s)
                                              'master',                         # mask type (only master for Ca dye channel)
                                              1,                                # mask region (1 for master or down)
                                              fp_master_profile[fp_val],        # mask mean intensity
                                              fp_master_profile_delta[fp_val],  # mask ΔF/F
                                              1],                               # mask sum / master mask sum (1 for Ca dye channel)
                                     index=self.profile_df.columns)
            self.profile_df = self.profile_df.append(point_series, ignore_index=True)

        # FP down regions
        fp_down_profile = self.prot_profile(mask=self.down_diff_mask[self.best_up_mask_index])
        fp_down_profile_delta = edge.deltaF(fp_down_profile, f_0_win=self.baseline_frames)
        fp_down_sum = np.asarray([round(np.sum(ma.masked_where(~self.down_diff_mask[self.best_up_mask_index], img)), 3) for img in self.prot_series])
        for fp_val in range(0, len(self.prot_series)):
            point_series = pd.Series([f'{self.img_name}{id_suffix}',                               # recording ID
                                              self.stim_power,                                     # 405 nm stimulation power (%)
                                              'fp',                                                # channel (FP or Ca dye)
                                              fp_val+1,                                            # frame number
                                              self.time_line[fp_val],                              # frame time (s)
                                              'down',                                              # mask type (only master for Ca dye channel)
                                              1,                                                   # mask region (1 for master or down)
                                              fp_down_profile[fp_val],                             # mask mean intensity
                                              fp_down_profile_delta[fp_val],                       # mask ΔF/F
                                              fp_down_sum[fp_val]/fp_master_sum[fp_val]],  # mask sum / master mask sum (1 for Ca dye channel)
                                     index=self.profile_df.columns)
            self.profile_df = self.profile_df.append(point_series, ignore_index=True)

        # FP up regions
        best_up_mask = self.up_diff_mask[self.best_up_mask_index]
        best_up_mask_prop = measure.regionprops(best_up_mask)

        fp_up_profile_dict = {}
        for i in best_up_mask_prop:  # calculate profiles for each up region
            best_up_mask_region = best_up_mask == i.label
            fp_up_profile = self.prot_profile(mask=best_up_mask_region)
            fp_up_profile_delta = edge.deltaF(fp_up_profile, f_0_win=self.baseline_frames)
            fp_up_sum = np.asarray([round(np.sum(ma.masked_where(~best_up_mask_region, img)), 3) for img in self.prot_series])

            fp_up_profile_dict.update({i.label: [fp_up_profile, fp_up_profile_delta, fp_up_sum]})
        for fp_val in range(0, len(self.prot_series)):
            for up_region_key in fp_up_profile_dict.keys():
                up_region = fp_up_profile_dict.get(up_region_key)

                point_series = pd.Series([f'{self.img_name}{id_suffix}',                    # recording ID
                                          self.stim_power,                                  # 405 nm stimulation power (%)
                                          'fp',                                             # channel (FP or Ca dye)
                                          fp_val+1,                                         # frame number
                                          self.time_line[fp_val],                           # frame time (s)
                                          'up',                                             # mask type (only master for Ca dye channel)
                                          up_region_key,                                    # mask region (1 for master or down)
                                          up_region[0][fp_val],                             # mask mean intensity
                                          up_region[1][fp_val],                             # mask ΔF/F
                                          up_region[2][fp_val]/fp_master_sum[fp_val]],      # mask sum / master mask sum (1 for Ca dye channel)
                                        index=self.profile_df.columns)
                self.profile_df = self.profile_df.append(point_series, ignore_index=True)

        logging.info(f'Recording profile data frame {self.profile_df.shape} created')
        return self.profile_df

        # for val_num in range(len(self.ca_series)):

    def save_area_df(self, id_suffix):
        """ FP channel masked regions area data frame

        """
        self.area_df = pd.DataFrame(columns=['ID',          # recording ID
                                             'stim_num',    # stimulation number
                                             'stim_frame',  # stimulation frame number
                                             'mask',        # mask type (up or down)
                                             'area',        # mask area (in px)
                                             'rel_area'])   # mask relative area (mask / master mask) 

        # down mask area
        for mask_num in range(0, len(self.stim_peak)):
            down_mask_area = np.sum(self.down_diff_mask[mask_num])
            cell_region_down_mask = np.copy(self.down_diff_mask[mask_num])
            cell_region_down_mask[cell_region_down_mask != self.master_mask] = 0
            down_mask_rel_area = np.sum(cell_region_down_mask) / np.sum(self.master_mask)  # relative area only for master mask region
            point_series = pd.Series([f'{self.img_name}{id_suffix}',  # recording ID
                                      mask_num+1,                       # stimulation number
                                      self.stim_peak[mask_num],       # stimulation frame number
                                      'down',                         # mask type (up or down)
                                      down_mask_area,                 # mask area (in px)
                                      down_mask_rel_area],            # mask relative area (mask / master mask) 
                                    index=self.area_df.columns)
            self.area_df = self.area_df.append(point_series, ignore_index=True)

        # up mask area
        for mask_num in range(0, len(self.stim_peak)):
            up_mask_area = np.sum(self.up_diff_mask[mask_num] != 0)
            cell_region_up_mask = np.copy(self.up_diff_mask[mask_num])
            cell_region_up_mask != 0
            cell_region_up_mask[cell_region_up_mask != self.master_mask] = 0
            up_mask_rel_area = np.sum(cell_region_up_mask) / np.sum(self.master_mask)
            point_series = pd.Series([f'{self.img_name}{id_suffix}',  # recording ID
                                      mask_num+1,                       # stimulation number
                                      self.stim_peak[mask_num],       # stimulation frame number
                                      'up',                           # mask type (up or down)
                                      up_mask_area,                   # mask area (in px)
                                      up_mask_rel_area],              # mask relative area (mask / master mask) 
                                    index=self.area_df.columns) 
            self.area_df = self.area_df.append(point_series, ignore_index=True)

        # master mask area
        self.area_df = self.area_df.append(pd.Series([f'{self.img_name}{id_suffix}',  # recording ID
                                                      0,                              # stimulation number
                                                      0,                              # stimulation frame number
                                                      'master',                       # mask type (up or down)
                                                      np.sum(self.master_mask),       # mask area (in px)
                                                      1],                             # mask relative area (mask / master mask) 
                                                    index=self.area_df.columns),
                                           ignore_index=True)
        

        logging.info(f'Mask area data frame {self.area_df.shape} created')
        return self.area_df

    def save_px_df(self, id_suffix):
        """ Save up regions intensity pixel-wise, for best mask and corresponding pixel-wise ΔF/F image

        """
        self.px_df = pd.DataFrame(columns=['ID',            # recording ID 
                                           'stim',          # stimulus number
                                           'mask_element',  # mask region (1 for master or down)
                                           'd',             # distances from nucleus border
                                           'int',           # px intensity
                                           'delta'])        # px ΔF/F

        best_up_mask = self.up_diff_mask[self.best_up_mask_index]
        best_up_mask_prop = measure.regionprops(best_up_mask)

        for stim_img_num in range(len(self.stim_mean_series)):
            stim_mean_img = self.stim_mean_series[stim_img_num]
            stim_deltaF_img = self.peak_deltaF_series[stim_img_num]
            for i in best_up_mask_prop:  # calculate profiles for each up region
                best_up_mask_region = best_up_mask == i.label

                # loop over masked px in images 
                for px_int, px_delta, px_d in zip(ma.compressed(ma.masked_where(~best_up_mask_region, stim_mean_img)),
                                                  ma.compressed(ma.masked_where(~best_up_mask_region, stim_deltaF_img)),
                                                  ma.compressed(ma.masked_where(~best_up_mask_region, self.nuclear_distances))): 
                    point_series = pd.Series([f'{self.img_name}{id_suffix}',  # recording ID
                                              stim_img_num+1,                 # stimulus number  
                                              i.label,                        # mask region
                                              px_d,                           # px distances from nucleus
                                              px_int,                         # px intensity
                                              px_delta],                      # px ΔF/F
                                            index=self.px_df.columns)
                    self.px_df = self.px_df.append(point_series, ignore_index=True)

        logging.info(f'px-wise intensity data frame {self.px_df.shape} created')
        return self.px_df

    # image saving
    def save_ctrl_profiles(self, path, baseline_frames=5):
        """ Masks intensity profiles

        """
        # MASTER MASK PROFILES + DERIVATE PROFILE
        ca_deltaF = edge.deltaF(self.ca_profile())
        prot_deltaF = edge.deltaF(self.prot_profile(mask=self.master_mask))
        derivate_deltaF = edge.deltaF(self.derivate_profile)

        plt.figure(figsize=(12,3))
        plt.plot(self.time_line, ca_deltaF, label='Ca2+ dye')
        plt.plot(self.time_line, prot_deltaF, label='FP')
        plt.plot(self.time_line[1:], derivate_deltaF, label='Ca2+ dye derivative', linestyle='--')
        plt.plot(np.take(self.time_line[1:], self.stim_peak), np.take(derivate_deltaF, self.stim_peak), 'v',
                 label='Stimulation', markersize=10, color='red')
        
        plt.grid(visible=True, linestyle=':')
        plt.xlabel('Time, s')
        plt.ylabel('ΔF/F')
        plt.xticks(np.arange(0, np.max(self.time_line)+2, step=1/self.time_scale))
        plt.legend()
        plt.tight_layout()
        # plt.suptitle(f'Master mask int profile, {self.img_name}, {self.stim_power}%', fontsize=20)
        plt.savefig(f'{path}/{self.img_name}_profile_ca.png', dpi=300)

        # UP REGIONS PROFILES + CA PROFILE
        best_mask = self.up_diff_mask[self.best_up_mask_index]
        best_mask_total = best_mask != 0
        # best_mask_down = self.down_diff_mask[self.best_up_mask_index]

        plt.figure(figsize=(15, 4))
        plt.plot(self.time_line, edge.deltaF(self.prot_profile(mask=best_mask_total)),
                 label='total up mask', linestyle='--', linewidth=3, color='white')
        # plt.plot(self.time_line, edge.deltaF(self.prot_profile(mask=best_mask_down)),
        #          label='total down mask', linestyle=':', linewidth=3, color='white')
        plt.plot(np.take(self.time_line[1:], self.stim_peak), [0.5 for i in self.stim_peak], 'v',
                 label='stimulation peak', markersize=10, color='red')

        for up_region_lab in range(1, np.max(best_mask)+1):
            up_region_mask = best_mask == up_region_lab
            up_region_profile = edge.deltaF(self.prot_profile(mask=up_region_mask))
            plt.plot(self.time_line, up_region_profile, label=f'region {up_region_lab}', linewidth=0.75)

        plt.grid(visible=True, linestyle=':')
        plt.xlabel('Time (s)')
        plt.ylabel('ΔF/F')
        plt.xticks(np.arange(0, np.max(self.time_line)+2, step=1/self.time_scale))
        plt.legend()
        plt.tight_layout()
        plt.suptitle(f'Best up mask regions profiles, {self.img_name}, {self.stim_power}%', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{path}/{self.img_name}_profile_up.png')


        plt.close('all')
        logging.info(f'{self.img_name} control profiles saved!')
        
    def save_ctrl_img(self, path, baseline_frames=5):
        # BASELINE IMG
        plt.figure(figsize=(15,8))

        ax0 = plt.subplot(121)
        ax0.set_title('Ca dye base img')
        img0 = ax0.imshow(np.mean(self.ca_series[0:baseline_frames], axis=0))
        div0 = make_axes_locatable(ax0)
        cax0 = div0.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img0, cax=cax0)
        ax0.axis('off')

        ax1 = plt.subplot(122)
        ax1.set_title('FP base img')
        img1 = ax1.imshow(np.mean(self.prot_series[0:baseline_frames], axis=0))
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img1, cax=cax1)
        ax1.axis('off')

        plt.suptitle(f'{self.img_name}, {self.stim_power}%, baseline image ({baseline_frames+1} frames)', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{path}/{self.img_name}_baseline_img.png', dpi=300)

        # MASTER MASK
        plt.figure(figsize=(13,8))

        ax0 = plt.subplot(121)
        ax0.set_title('Otsu elements')
        ax0.imshow(self.element_label, cmap='jet')
        ax0.axis('off')

        # ax1 = plt.subplot(132)
        # ax1.set_title('Effective master mask')
        # ax1.imshow(self.master_mask, cmap='jet')
        # ax1.axis('off')

        ax2 = plt.subplot(122)
        ax2.set_title('Final mask & FP img overlap')
        ax2.imshow(self.total_mask_ctrl_img)
        ax2.axis('off')

        plt.suptitle(f'{self.img_name} master mask', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{path}/{self.img_name}_master_mask.png', dpi=300)  

        # COMPARISON IMG
        centr = lambda img: abs(np.max(img)) if abs(np.max(img)) > abs(np.min(img)) else abs(np.min(img))  # center cmap around zero
        cdict_blue_red = {
                          'red':(
                            (0.0, 0.0, 0.0),
                            (0.52, 0.0, 0.0),
                            (0.55, 0.3, 0.3),
                            (1.0, 1.0, 1.0)),
                          'blue':(
                            (0.0, 0.0, 0.0),
                            (1.0, 0.0, 0.0)),
                          'green':(
                            (0.0, 1.0, 1.0),
                            (0.45, 0.3, 0.3),
                            (0.48, 0.0, 0.0),
                            (1.0, 0.0, 0.0))
                            }

        for peak_num in range(0, len(self.stim_peak)):
            delta_img = self.peak_deltaF_series[peak_num]
            diff_img = self.peak_diff_series[peak_num]

            plt.figure(figsize=(15,8))

            ax0 = plt.subplot(121)
            ax0.set_title('ΔF/F0')
            img0 = ax0.imshow(ma.masked_where(~self.master_mask, delta_img), cmap='jet')  # , norm=colors.LogNorm(vmin=-1.0, vmax=1.0))
            img0.set_clim(vmin=-1, vmax=1)
            div0 = make_axes_locatable(ax0)
            cax0 = div0.append_axes('right', size='3%', pad=0.1)
            plt.colorbar(img0, cax=cax0)
            ax0.axis('off')

            ax1 = plt.subplot(122)
            ax1.set_title('Differential')
            img1 = ax1.imshow(diff_img, cmap=LinearSegmentedColormap('RedGreen', cdict_blue_red))
            img1.set_clim(vmin=-centr(diff_img), vmax=centr(diff_img))
            div1 = make_axes_locatable(ax1)
            cax1 = div1.append_axes('right', size='3%', pad=0.1)
            plt.colorbar(img1, cax=cax1)
            ax1.axis('off')

            plt.suptitle(f'{self.img_name} baseline-peak comparison, peak frame {self.stim_peak[peak_num]}', fontsize=20)
            plt.tight_layout()
            plt.savefig(f'{path}/{self.img_name}_comparison_peak_{self.stim_peak[peak_num]}.png')

        # UP REGION ENUMERATION
        up_mask_prop = measure.regionprops(self.up_diff_mask[self.best_up_mask_index])
        best_up_delta_img = self.peak_deltaF_series[self.best_up_mask_index] 
        best_up_mask_total = self.up_diff_mask[self.best_up_mask_index] != 0

        plt.figure(figsize=(8, 8))
        ax = plt.subplot()
        img = ax.imshow(ma.masked_where(~best_up_mask_total, best_up_delta_img), cmap='jet')
        img.set_clim(vmin=-1, vmax=1)
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img, cax=cax)
        for up_region in up_mask_prop:
            ax.annotate(up_region.label, xy=(up_region.centroid[1]+10, up_region.centroid[0]),
                        fontweight='heavy', fontsize=15, color='white')
        ax.axis('off')
        plt.suptitle(f'{self.img_name} up regions ΔF/F', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{path}/{self.img_name}_up_regions.png')

        # CTRL IMG OF UP/DOWN MASKS
        ctrl_img = util.img_as_ubyte(self.detection_img/np.max(np.abs(self.detection_img)))
        ctrl_img = exposure.equalize_adapthist(ctrl_img)
        plt.figure(figsize=(10, 10))
        ax = plt.subplot()
        ax.imshow(label2rgb(self.comb_diff_mask[self.best_up_mask_index],
                            image=ctrl_img,
                            colors=['green', 'red'], bg_label=1, alpha=0.5))
        ax.axis('off')
        # plt.suptitle(f'{self.img_name} up/down mask ctrl img', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{path}/{self.img_name}_up_down_ctrl.png')
        plt.close('all')

    def save_ca_gif(self, path):
        masked_ca_series = [ma.masked_where(~self.master_mask, img) for img in self.ca_series] 
        fig = plt.figure() 
        img = plt.imshow(masked_ca_series[0])
        plt_text = plt.text(10,15,'',fontsize=10)
        def ani(i):
          img.set_array(masked_ca_series[i])
          plt_text.set_text(f'{i+1}')
          return img,
        ani = anm.FuncAnimation(fig, ani, interval=200, repeat_delay=500, frames=len(masked_ca_series))
        plt.suptitle(f'{self.img_name} cytoplasm Ca2+ dynamics', fontsize=10)
        plt.axis('off')
        ani.save(f'{path}/{self.img_name}_ca_dyn.gif', writer='imagemagick', fps=5)
        plt.close('all')
    
    def save_dist_ctrl_img(self, path):
        best_up_mask = self.up_diff_mask[self.best_up_mask_index] != 0
        all_mask = self._select_larg_mask(raw_mask=(self.cell_mask + best_up_mask))
        all_rim = self._get_mask_rim(raw_mask=all_mask, rim_th=1)
        master_fp = filters.gaussian(np.mean(self.prot_series*-1, axis=0), sigma=1)

        plt.figure(figsize=(10, 10))
        ax = plt.subplot()
        ax.imshow(master_fp, interpolation='none', cmap='Greys', alpha=.6)
        ax.imshow(ma.masked_where(~all_rim, all_rim), interpolation='none', cmap='Greys', alpha=.75)
        ax.imshow(ma.masked_where(~self.nuclear_mask, np.ones_like(self.nuclear_mask)), interpolation='none', cmap='Greys', alpha=.4)
        img = ax.imshow(ma.masked_where(~best_up_mask, self.nuclear_distances), interpolation='none', cmap='jet', alpha=.6)
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img, cax=cax)
        ax.axis('off')
        plt.suptitle(f'{self.img_name} best up mask distance\nwhite - cell mask border and nuclear mask', fontsize=15)
        plt.tight_layout()
        plt.savefig(f'{path}/{self.img_name}_all_mask_ctrl.png')
        plt.close('all')

    def save_rim_profile(self, path, rim_th=2, px_size=0.25, rim_enum='line', rim_ch='FP', save_df=False):
        """ Creating of cell border rim mask for monitoring FP distribution along full cell.
        rim_th - rim thinkness in px
        px_size - img px size in um
        rim_ch - create rim for FP or Ca dye channel

        """
        if rim_ch == 'FP':
            rim_name = f'{self.img_name}_fp_rim'
            series_for_rim = self.deltaF_series
        elif rim_ch == 'Ca':
            rim_name = f'{self.img_name}_ca_rim'
            series_for_rim = self.deltaF_ca_series

        # cell rim creation
        cell_rim = self._get_mask_rim(raw_mask=self.cell_mask, rim_th=rim_th)
        rim_center = measure.regionprops(measure.label(cell_rim))[0].centroid

        # images polar transforming
        rad_rim = transform.warp_polar(cell_rim, center=rim_center)
        polar_prot_series = [transform.warp_polar(frame, center=rim_center) for frame in series_for_rim]
        polar_dist = transform.warp_polar(self.nuclear_distances, center=rim_center)

        # fig, ax = plt.subplots()
        # ax.imshow(self.nuclear_distances, cmap='jet')
        # ax.axis('off')
        # plt.show()

        # shift cell rim gap to the top
        rim_sum = np.sum(rad_rim, axis=1)
        gap_start = np.argmin(rim_sum)  # find rim gap start index
        if rim_sum[gap_start] == 0:
            logging.warning(f'Cell rim gap start index {gap_start}')
            rad_rim = np.roll(rad_rim, shift=-gap_start, axis=0)
            rad_rim = rad_rim[~np.all(rad_rim == 0, axis=1)]  # drop zeros only lines

            # drop line from images
            drop_i = np.shape(polar_prot_series[0])[0] - np.shape(rad_rim)[0] - 1
            polar_prot_series = [np.roll(frame, shift=-gap_start, axis=0)[drop_i:-1,:] for frame in polar_prot_series]
            polar_dist = np.roll(polar_dist, shift=-gap_start, axis=0)[drop_i:-1,:]
        else:
            logging.info('There is no gap in cell rim')

        if rim_enum == 'px':
            # rim pixels enumeration
            rad_rim_num = np.int_(np.copy(rad_rim))
            px_num = 1
            for i, j in np.ndindex(rad_rim_num.shape):
                if rad_rim_num[i, j] != 0:
                    rad_rim_num[i, j] = px_num
                    px_num += 1

            # loop over polar transformed FP frames
            rim_profiles = []
            for polar_frame in polar_prot_series:
                frame_row = np.array([])
                for rim_element in range(1, np.max(rad_rim_num)):
                    rim_element_mask = rad_rim_num == rim_element
                    frame_row = np.append(frame_row, polar_frame[rim_element_mask], axis=0)
                rim_profiles.append(frame_row)
            rim_profiles = np.array(rim_profiles)

            # loop over polar transformed distances image
            dist_bar = []
            for rim_element in range(1, np.max(rad_rim_num)):
                rim_element_mask = rad_rim_num == rim_element
                dist_bar.append(polar_dist[rim_element_mask])
            dist_bar = np.resize(np.array(dist_bar), (1, len(dist_bar))) * px_size

        elif rim_enum == 'line':
            rim_profiles = []
            for polar_frame in polar_prot_series:
                rim_polar_frame = ma.masked_where(~rad_rim, polar_frame)
                rim_profiles.append(np.mean(rim_polar_frame, axis=1))
            dist_bar = np.mean(ma.masked_where(~rad_rim, polar_dist), axis=1)
            dist_bar = np.resize(np.array(dist_bar), (1, len(dist_bar))) * px_size

        # data frame saving
        if save_df:
            self.rim_df = pd.DataFrame(columns=['ID',     # recording ID 
                                                'time',   # recording time           
                                                'delta',  # rim point ΔF/F
                                                'd'])     # distances from nucleus border
            for frame_i in range(len(self.time_line)):
                time_point = self.time_line[frame_i]
                rim_frame_line = rim_profiles[frame_i]
                for rim_point_i in range(len(rim_frame_line)):
                    rim_point_series = pd.Series([f'{self.img_name}{self.id_suffix}',  # recording ID
                                                  time_point,                     # recording time
                                                  rim_frame_line[rim_point_i],    # rim point ΔF/F
                                                  dist_bar[0, rim_point_i]],         # distances from nucleus border
                                                 index=self.rim_df.columns)
                    self.rim_df = self.rim_df.append(rim_point_series, ignore_index=True)
            logging.info('Rim data frame created')
            return self.rim_df

        # custom cmap for rim profiles
        if rim_ch == 'FP':
            cdict_blue_red = {
                      'red':(
                        (0.0, 0.0, 0.0),
                        (0.52, 0.0, 0.0),
                        (0.55, 0.3, 0.3),
                        (1.0, 1.0, 1.0)),
                      'blue':(
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),
                      'green':(
                        (0.0, 1.0, 1.0),
                        (0.45, 0.3, 0.3),
                        (0.48, 0.0, 0.0),
                        (1.0, 0.0, 0.0))
                        }
            rim_cmap = LinearSegmentedColormap('RedGreen', cdict_blue_red)
            vmin_val = -np.max(np.abs(rim_profiles))
            vmax_val = np.max(np.abs(rim_profiles))
        elif rim_ch == 'Ca':
            rim_cmap = 'jet'
            vmin_val = np.min(np.abs(rim_profiles))
            vmax_val = np.max(np.abs(rim_profiles))

        # plotting
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0.0, wspace=0.0,
                              height_ratios=(6, 1))

        ax0 = fig.add_subplot(gs[0]) # FP profile
        img0 = ax0.imshow(rim_profiles, vmin=vmin_val, vmax=vmax_val,
                          aspect='auto', cmap=rim_cmap)
        div0 = make_axes_locatable(ax0)
        cax0 = div0.append_axes('top', size='4%', pad=0.6)
        clb0 = plt.colorbar(img0, cax=cax0, orientation='horizontal') 
        clb0.ax.set_title('ΔF/F',fontsize=10)
        [ax0.axhline(y=i, linestyle='--', color='white') for i in self.stim_peak]
        [ax0.text(x=np.shape(rim_profiles)[1]-20, y=i, s='Stimulation', fontsize=7, color='white') for i in self.stim_peak]
        ax0.set_xticks([])
        frame_tick = np.arange(0,np.shape(rim_profiles)[0],1)
        frame_lab = (frame_tick+1) * 2
        ax0.set_yticks(frame_tick)
        ax0.set_yticklabels(frame_lab)
        ax0.set_ylabel('Time, s')
        bar_tick = np.arange(0,np.shape(dist_bar)[1],100)
        bar_lab = np.int_(bar_tick * px_size)
        ax0.set_xticks(bar_tick)
        ax0.set_xticklabels([])
        ax0.xaxis.tick_top()
        ax0.xaxis.set_label_position('top')
        ax0.set_xlabel('Distance along cell border')

        ax1 = fig.add_subplot(gs[1])  # dist bar
        img1 = ax1.imshow(dist_bar, aspect='auto', cmap='gnuplot2')
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes('bottom', size='30%', pad=0.3)
        clb1 = plt.colorbar(img1, cax=cax1, orientation='horizontal') 
        clb1.ax.set_title('Distance from nucleus, μm',fontsize=10)
        ax1.set_xticks([])
        ax1.set_yticks([])

        plt.tight_layout()
        plt.savefig(f'{path}/{rim_name}.png', dpi=300)
        plt.close('all')
        # plt.show()

    def fast_img(self, path):
        """ Some shit for fast plotting
        """
        cell_rim = self._get_mask_rim(raw_mask=self.cell_mask, rim_th=1)

        plt.figure(figsize=(4, 4))
        ax = plt.subplot()
        img = ax.imshow(ma.masked_where(~self.master_mask, self.nuclear_distances) * 0.138, cmap='jet')
        div = make_axes_locatable(ax)
        cax = div.append_axes('top', size='3%', pad=0.1)
        clb = plt.colorbar(img, cax=cax, orientation='horizontal') 
        clb.ax.set_title('Distance from nucleus, μm',fontsize=10)
        ax.imshow(ma.masked_where(~cell_rim, cell_rim), interpolation='none', cmap='magma', alpha=0.8)
        ax.plot(142, 67, marker='v', color='red', markersize=10)

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{path}/{self.img_name}_cyto_dist.png', dpi=300)
        plt.close('all')


        # ca_img_list = [self.ca_series[0], self.ca_series[17]]
        # vmin_val = np.min(ca_img_list[0])
        # vmax_val = np.max(ca_img_list[1])
        # for i in range(len(ca_img_list)):
        #     ca_img = ca_img_list[i]
        #     plt.figure(figsize=(4, 4))
        #     ax = plt.subplot()
        #     img = ax.imshow(ma.masked_where(~self.master_mask, ca_img), vmin=vmin_val, vmax=vmax_val, cmap='jet')
        #     ax.axis('off')
        #     plt.tight_layout()
        #     plt.savefig(f'{path}/{self.img_name}_ca_{i}.png', dpi=300),
        #     plt.close('all')



if __name__=="__main__":
  print('А шо ти хочеш?')


# That's all!