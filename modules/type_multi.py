#!/usr/bin/env python3
""" Copyright © 2020-2022 Borys Olifirov
Time series with simultaneous HPCA transfection and calcium dye loading.
Multiple stimulation during registration.

Does NOT requires oifpars module.

Analysis functions tree:

get_master_mask -- find_stimul_peak --|-- peak_img_diff
                                      |
                                      |-- peak_img_deltaF


Useful links:
https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/ImageProcessing-html/topic4.htm

"""

import sys
import os
import logging

import numpy as np
import numpy.ma as ma

from scipy.signal import find_peaks
from scipy.ndimage import distance_transform_edt

import yaml
import pandas as pd

from skimage import filters
from skimage import measure
from skimage import morphology
from skimage.color import label2rgb
from skimage import util
from skimage import exposure

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    def __init__(self, oif_path, img_name, meta_dict, time_scale=1, baseline_frames=5, prot_bleach_correction=True):
        self.img_name = img_name
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

        # create derevative series
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

    def get_master_mask(self, sigma=1, kernel_size=5, mask_ext=5, multi_otsu_nucleus_mask=False):
        """ Whole cell mask building by Ca dye channel data with Otsu thresholding.
        Filters greater element of draft Otsu mask and return master mask array.

        mask_ext - otsu mask extension value in px

        """
        trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for gaussian fliter according to sigma value and kernel size

        self.detection_img = filters.gaussian(np.mean(self.prot_series, axis=0), sigma=sigma, truncate=trun(kernel_size, sigma))

        otsu = filters.threshold_otsu(self.detection_img)
        draft_mask = self.detection_img > otsu
        self.element_label, self.element_num = measure.label(draft_mask, return_num=True)
        logging.info(f'{self.element_num} Otsu mask elements detected')

        detection_label = np.copy(self.element_label)
        element_area = {element.area : element.label for element in measure.regionprops(detection_label)}
        self.master_mask = detection_label == element_area[max(element_area.keys())]

        # mask expansion
        self.distances, _ = distance_transform_edt(~self.master_mask, return_indices=True)
        self.master_mask = self.distances <= mask_ext

        # multi Otsu mask for nucleus detection
        self.multi_otsu_nucleus_mask = multi_otsu_nucleus_mask
        if self.multi_otsu_nucleus_mask:
            multi_otsu = filters.threshold_multiotsu(self.detection_img, classes=3)
            self.cell_regions = np.digitize(self.detection_img, bins=multi_otsu)

    def find_stimul_peak(self, h=0.15, d=3, l_lim=5, r_lim=18):
        """ Require master_mask, results of get_master_mask!
        Find peaks on deltaF/F Fluo-4 derivate profile.
        h - minimal peaks height, in ΔF/F0
        d - minimal distance between peaks, in frames

        """
        self.stim_peak = find_peaks(edge.deltaF(self.derivate_profile), height=h, distance=d)[0]
        self.stim_peak = self.stim_peak[(self.stim_peak >= l_lim) & (self.stim_peak <= r_lim)]  # filter outer peaks
        logging.info(f'Detected peaks: {self.stim_peak}')

    # derivetive series
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
            stim_diff_img[self.distances >= 30] = 0 
            stim_diff_img = stim_diff_img/np.max(np.abs(stim_diff_img))
            self.peak_diff_series.append(stim_diff_img)

            # up regions thresholding
            frame_diff_up_mask = filters.apply_hysteresis_threshold(stim_diff_img,
                                                                    low=up_min_tolerance,
                                                                    high=up_max_tolerance)
            frame_diff_up_mask_elements = measure.label(frame_diff_up_mask)
            self.up_diff_mask.append(frame_diff_up_mask_elements)

            # down regions thresholding
            frame_diff_down_mask = filters.apply_hysteresis_threshold(stim_diff_img,
                                                                    low=down_min_tolerance,
                                                                    high=down_max_tolerance)
            self.down_diff_mask.append(frame_diff_down_mask)

            self.comb_diff_mask.append((frame_diff_up_mask*2) + (frame_diff_down_mask-2)*-1)

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
        prot_series_sigma = [filters.gaussian(i, sigma=sigma, truncate=trun(kernel_size, sigma)) for i in self.prot_series]
        baseline_prot_img = np.mean(prot_series_sigma[:baseline_win], axis=0)

        delta = lambda f, f_0: (f - f_0)/f_0 if f_0 > 0 else f_0 
        vdelta = np.vectorize(delta)
        # centr = lambda img: abs(np.max(img))*0.5 if abs(np.max(img)) > abs(np.min(img)) else abs(np.min(img))

        self.stim_mean_series = []
        for stim_position in self.stim_peak:
            peak_frames_start = stim_position + stim_shift
            peak_frames_end = stim_position + stim_shift + stim_win
            self.stim_mean_series.append(np.mean(prot_series_sigma[peak_frames_start:peak_frames_end], axis=0))

        self.peak_deltaF_series = np.asarray([ma.masked_where(~self.master_mask, vdelta(i, baseline_prot_img)) for i in self.stim_mean_series])
        self.up_delta_mask = np.copy(self.peak_deltaF_series)
        self.up_delta_mask = self.up_delta_mask >= deltaF_up
        self.up_delta_mask[~np.broadcast_to(self.master_mask, np.shape(self.up_delta_mask))] = 0

    # mask elements sub-segmentation
    def diff_mask_segment(self):
        """ Up regions mask segmentation.

        """
        best_mask = self.up_diff_mask[self.best_up_mask_index]
        demo_segment = np.copy(best_mask)  # [60:180, 100:150]
        demo_segment[demo_segment!=0] = 1

        px_num = 1
        for i, j in np.ndindex(demo_segment.shape):
            if demo_segment[i, j] != 0:
                demo_segment[i, j] = px_num
                px_num += 1
            # demo_segment[i, j] = 10

        segment_num = 4
        segment_range = px_num // segment_num
        segment_lab_dict = {segment_i:[segment_i * segment_range - segment_range + 1, segment_i * segment_range]
                            for segment_i in range(1, segment_num+1)}

        print(px_num)
        print(segment_num * segment_range)
        print(segment_lab_dict)

        demo_results = np.copy(demo_segment)
        for segment_lab in segment_lab_dict.keys():
            range_list = segment_lab_dict[segment_lab]
            demo_results[(demo_results >= range_list[0]) & (demo_results <= range_list[1])] = segment_lab

        fig, ax = plt.subplots()
        ax.imshow(demo_results, cmap='jet')
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
        

        logging.info(f'Recording profile data frame {self.area_df.shape} created')
        return self.area_df

    def save_px_df(self, id_suffix):
        """ Save up regions intensity pixel-wise, for best mask and corresponding pixel-wise ΔF/F image

        """
        self.px_df = pd.DataFrame(columns=['ID',           # recording ID 
                                           'stim',         # stimulus number
                                           'mask_region',  # mask region (1 for master or down)
                                           'int',          # px intensity
                                           'delta'])       # px ΔF/F

        best_up_mask = self.up_diff_mask[self.best_up_mask_index]
        best_up_mask_prop = measure.regionprops(best_up_mask)

        # best_delta_img = self.peak_deltaF_series[self.best_up_mask_index]
        # best_img = self.stim_mean_series[self.best_up_mask_index]

        for stim_img_num in range(len(self.stim_mean_series)):
            stim_mean_img = self.stim_mean_series[stim_img_num]
            stim_deltaF_img = self.peak_deltaF_series[stim_img_num]
            for i in best_up_mask_prop:  # calculate profiles for each up region
                best_up_mask_region = best_up_mask == i.label 
                for px_int, px_delta in zip(ma.compressed(ma.masked_where(~best_up_mask_region, stim_mean_img)),
                                            ma.compressed(ma.masked_where(~best_up_mask_region, stim_deltaF_img))): 
                    point_series = pd.Series([f'{self.img_name}{id_suffix}',  # recording ID
                                              stim_img_num+1,                 # stimulus number  
                                              i.label,                        # mask region
                                              px_int,                         # px intensity
                                              px_delta],                      # px ΔF/F
                                            index=self.px_df.columns)
                    self.px_df = self.px_df.append(point_series, ignore_index=True)

        logging.info(f'Recording profile data frame {self.px_df.shape} created')
        return self.px_df

    # image saving
    def save_ctrl_profiles(self, path, baseline_frames=3):
        """ Masks intensity profiles

        """
        # MASTER MASK PROFILES + DERIVATE PROFILE
        ca_deltaF = edge.deltaF(self.ca_profile())
        prot_deltaF = edge.deltaF(self.prot_profile(mask=self.master_mask))
        derivate_deltaF = edge.deltaF(self.derivate_profile)

        plt.figure(figsize=(15,4))
        plt.plot(self.time_line, ca_deltaF, label='Ca dye profile')
        plt.plot(self.time_line, prot_deltaF, label='FP profile')
        plt.plot(self.time_line[1:], derivate_deltaF, label='Ca dye derivate', linestyle='--')
        plt.plot(np.take(self.time_line[1:], self.stim_peak), np.take(derivate_deltaF, self.stim_peak), 'v',
                 label='stimulation peak', markersize=10, color='red')
        
        plt.grid(visible=True, linestyle=':')
        plt.xlabel('Time (s)')
        plt.ylabel('ΔF/F')
        plt.xticks(np.arange(0, np.max(self.time_line)+2, step=1/self.time_scale))
        plt.legend()
        plt.tight_layout()
        plt.suptitle(f'Master mask int profile, {self.img_name}, {self.stim_power}%', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{path}/{self.img_name}_profile_ca.png')

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
        plt.savefig(f'{path}/{self.img_name}_baseline_img.png')

        # MASTER MASK
        plt.figure(figsize=(15,8))

        ax0 = plt.subplot(121)
        ax0.set_title('Otsu mask elements')
        ax0.imshow(self.element_label, cmap='jet')
        ax0.axis('off')

        ax1 = plt.subplot(122)
        ax1.set_title('Cell master mask')
        ax1.imshow(self.master_mask, cmap='jet')
        ax1.axis('off')

        plt.suptitle(f'{self.img_name} master mask', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{path}/{self.img_name}_master_mask.png')  

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

        # # UP/DOWN REGIONS MASKS
        # for peak_num in range(0, len(self.stim_peak)):
        #     delta_up_mask = self.up_delta_mask[peak_num] 

        #     diff_up_mask = self.up_diff_mask[peak_num] 
        #     diff_down_mask = self.down_diff_mask[peak_num]
        #     diff_comb_mask = self.comb_diff_mask[peak_num]

        #     plt.figure(figsize=(15,8))

        #     ax0 = plt.subplot(121)
        #     ax0.set_title('Differential up/down masks')
        #     ax0.imshow(diff_comb_mask, cmap=LinearSegmentedColormap('RedGreen', cdict_blue_red))
        #     ax0.axis('off')

        #     ax1 = plt.subplot(122)
        #     ax1.set_title('Up mask elements')
        #     ax1.imshow(diff_up_mask, cmap='jet')
        #     ax1.axis('off')

        #     plt.suptitle(f'{self.img_name} up mask, peak frame {self.stim_peak[peak_num]}', fontsize=20)
        #     plt.tight_layout()
        #     plt.savefig(f'{path}/{self.img_name}_up_mask_{self.stim_peak[peak_num]}.png')

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
        plt.figure(figsize=(8, 8))
        ax = plt.subplot()
        ax.imshow(label2rgb(self.comb_diff_mask[self.best_up_mask_index],
                            image=ctrl_img,
                            colors=['green', 'red'], bg_label=1, alpha=0.5))
        ax.axis('off')
        plt.suptitle(f'{self.img_name} up/down mask ctrl img', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{path}/{self.img_name}_up_down_ctrl.png')


        plt.close('all')
        logging.info(f'{self.img_name} control images saved!')

if __name__=="__main__":
  pass


# That's all!