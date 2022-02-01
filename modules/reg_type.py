#!/usr/bin/env python3

""" Copyright © 2020-2022 Borys Olifirov

Registrations types.

"""

import sys
import os
import logging

import numpy as np
import numpy.ma as ma

from scipy.ndimage import distance_transform_edt

import yaml
import pandas as pd

from skimage import filters
from skimage import measure
from skimage import morphology

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

import oiffile as oif
import edge
import hyst


class FluoData():
    """ Time series of homogeneous fluoresced cells (Fluo-4,  low range of the HPCA translocation).

    Require oifpars module.

    """
    @classmethod
    def settings(cls, max_frame_num, th_method='hyst',
                 dot_on=False, dot_mode='mean', dot_sigma=1, dot_kernel_size=20, dot_return_extra=False):
        """ Global options for image series preprocessing and segmentation, initialize before it before records parsing 

        """
        # registration type options
        cls.max_frame_num = max_frame_num - 1
        cls.th_method = th_method

        # settings for dot artefact remove, mask_point_artefact function from edge
        cls.dot_on = dot_on  # preprocessing switch
        cls.dot_mode = dot_mode
        cls.dot_sigma = dot_sigma
        cls.dot_kernel_size = dot_kernel_size
        cls.dot_return_extra = dot_return_extra

    def __init__(self, oif_path, img_name, feature=False, max_frame=19,
                 background_rm=True, 
                 restrict=False,
                 img_series=False,
                 sd_area=20):
        # restricted variant for multiple file registration, next step - connect all registration to one FluoData object with fluo_ext function
        if restrict:
            pass
        # NOT READY!
        #     self.img_name = img_name
        #     self.img_series = oif.OibImread(oif_path)[0,:,:,:]                          # OIF file reading 
        #     if background_rm:                                                           # background remove option
        #         for frame in range(0, np.shape(self.img_series)[0]):
        #         self.img_series[frame] = edge.back_rm(self.img_series[frame],
        #                                               edge_lim=10,
        #                                               dim=2)
        #     self.feature = feature
        # # full variant for one-file registration
        else:
            if not img_series:
                self.img_series = oif.OibImread(oif_path)[0,:,:,:]             # first channel only (Fluo)
                if background_rm:                                              # background remove option
                    for frame in range(0, np.shape(self.img_series)[0]):
                        self.img_series[frame] = edge.back_rm(self.img_series[frame],
                                                              edge_lim=10,
                                                              dim=2)
            else:
                self.img_series = img_series
            self.img_name = img_name
            self.feature = feature                                             # feature from the YAML config file
            self.max_frame = self.img_series[self.max_frame_num,:,:]           # first frame after stimulation, maximal translocations frame

            if self.th_method == 'hyst':
                # optional image preprocessing, dot artefact removing
                if self.dot_on:
                    self.detection_img, self.element_label, self.artefact_mask = edge.mask_point_artefact(self.img_series,
                                                                                                          img_mode=self.dot_mode,
                                                                                                          sigma=self.dot_sigma,
                                                                                                          kernel_size=self.dot_kernel_size,
                                                                                                          return_extra=self.dot_return_extra)
                else:
                    self.detection_img = self.max_frame

                # apply hysteresis thresholding
                self.cell_detector = hyst.hystTool(img=self.detection_img)
                self.cell_detector.detector()
                self.max_frame_mask =  self.cell_detector.cell_mask()

            # use Otsu thresholding (for inhomogeneus cells)
            elif self.th_method == 'otsu':
                logging.info('Otsu thresholding mode')
                self.detection_img = np.mean(self.img_series, axis=0)
                otsu = filters.threshold_otsu(self.detection_img)
                draft_mask = self.detection_img > otsu
                self.element_label = measure.label(draft_mask)
                detection_label = np.copy(self.element_label)
                element_area = {element.area : element.label for element in measure.regionprops(detection_label)}
                self.max_frame_mask = detection_label == element_area[max(element_area.keys())]

    def max_mask_int(self, plot_path=False):
        """ Calculation mean intensity  in masked area along frames series.
        Mask was created by max_frame image.

        """
        # return edge.series_sum_int(self.img_series, self.max_frame_mask)
        mean_list = [round(np.sum(ma.masked_where(~self.max_frame_mask, img)) / np.sum(self.max_frame_mask), 3) for img in self.img_series]
        return np.asarray(mean_list)

    def frame_mask_int(self, plot_path=False):
        """ Calculation mean intensity  in masked area along frames series.
        Mask was created for each frame individually.

        """
        self.mask_series = [self.cell_detector.cell_mask(frame=frame) for frame in self.img_series]
        mean_list = [round(np.sum(ma.masked_where(~self.mask_series[i], self.img_series[i])) /
                     np.sum(self.mask_series[i]), 3) for i in range(len(self.img_series))]
        if plot_path:  # mask mean intensity plot saving
            plt.figure()
            ax = plt.subplot()
            img = ax.plot(mean_list)
            plt.tight_layout()
            plt.savefig(f'{plot_path}/{self.img_name}_frame_mask.png')
            plt.close('all')
        return np.asarray(mean_list)

    def custom_mask_int(self, mask, plot_path=False):
        """ Calculation mean intensity  in masked area along frames series.
        Require custom mask.

        """
        mean_list = [round(np.sum(ma.masked_where(~mask, img)) / np.sum(mask), 3) for img in self.img_series]
        if plot_path:  # mask mean intensity plot saving
            plt.figure()
            ax = plt.subplot()
            img = ax.plot(mean_list)
            plt.tight_layout()
            plt.savefig(f'{plot_path}/{self.img_name}_custom_mask.png')
            plt.close('all')
        return np.asarray(mean_list)

    def updown_mask_int(self, up_mask, down_mask, delta=False, plot_path=False):
        """ Calculation mean intensity in masked area along frames series.
        Require two masks, increasing and decreasing regions.

        """
        up_list = [round(np.sum(ma.masked_where(~up_mask, img)) / np.sum(up_mask), 3) for img in self.img_series]
        down_list = [round(np.sum(ma.masked_where(~down_mask, img)) / np.sum(down_mask), 3) for img in self.img_series]

        if delta:  # calculate ΔF/F0
            up_list = edge.deltaF(up_list)
            down_list = edge.deltaF(down_list)

        return np.asarray(up_list), np.asarray(down_list)

    def save_ctrl_img(self, path):
        """ Control images saving.

        """
        try:
            plt.figure()
            ax0 = plt.subplot(121)
            img0 = ax0.imshow(self.max_frame)
            ax0.axis('off')
            ax0.text(10,10,f'max int frame {self.max_frame_num}',fontsize=8)
            ax1 = plt.subplot(122)
            img1 = ax1.imshow(self.max_frame_mask)
            ax1.axis('off')
            ax1.text(10,10,f'binary mask',fontsize=8)
            plt.tight_layout()
            plt.savefig(f'{path}/{self.img_name}_ctrl.png')
            plt.close('all')
            
            if self.dot_on:      
                fig, axes =  plt.subplots(ncols=3, figsize=(20,10))
                ax = axes.ravel()
                ax[0].imshow(self.max_frame, vmin=self.max_frame.min(), vmax=self.max_frame.max())
                ax[0].text(10,10,'init img',fontsize=10)
                ax[0].axis('off')
                ax[1].imshow(self.detection_img, vmin=self.max_frame.min(), vmax=self.max_frame.max())
                ax[1].text(10,10,'detection img',fontsize=10)
                ax[1].axis('off')
                ax[2].imshow(self.element_label, cmap='jet')
                ax[2].text(10,10,'otsu label',fontsize=10)
                ax[2].axis('off')
                # ax[3].imshow(self.artefact_mask, cmap='jet')
                # ax[3].text(10,10,'artefact mask',fontsize=10)
                # ax[3].axis('off')

                plt.tight_layout()
                plt.savefig(f'{path}/{self.img_name}_dot_artefact.png')
                plt.close('all')

            logging.info(f'{self.img_name} control image saved!\n')
        except TypeError:
            logging.fatal(f'{self.img_name} has some trouble with cell mask, CAN`T save control image!')
            return 

    
class MembZData():
    """ Registration of static co-transferenced (HPCA-TagRFP + EYFP-Mem) cells.
    T-axis of file represent excitation laser combination (HPCA+label, HPCA, label).

    Require oifpars module.

    Multi channel time series dimensions structure:
    (ch, z-axis, t, x-axis, y-axis)

    fluo_order - order of fluo,
    if target protein label emission wavelength less then membrane label emission wavelength, HPCA-TFP + EYFP-Mem - direct order ('dir'),
    if target protein label emission wavelength greater then membrane label emission wavelength, HPCA-TagRFP + EYFP-Mem - revers order ('rev'),

    """
    def __init__(self, oif_path, img_name, feature=False,
                 background_rm=True, 
                 middle_frame=5,
                 fluo_order='dir',
                 **kwargs):
        self.img_series = oif.OibImread(oif_path)                                   # OIF file reading
        if fluo_order == 'dir':
            target_ch, target_ex = 0, 1
            label_ch, label_ex = 1, 2
        elif fluo_order == 'rev':
            target_ch, target_ex = 1, 2
            label_ch, label_ex = 0, 1
        else:
            logging.fatal('Incorrect fluo_order option!')
        self.target_series = self.img_series[target_ch,:,target_ex,:,:]             # selection of target channel and excitation series 
        self.label_series = self.img_series[label_ch,:,label_ex,:,:]                # selection of label channel and excitation series

        logging.info(f'Z-stack with {np.shape(self.target_series)[0]} slices uploaded')

        if background_rm:                                                           # background remove option
            for frame in range(0, np.shape(self.target_series)[0]):
                self.target_series[frame] = edge.back_rm(self.target_series[frame],
                                                         edge_lim=10,
                                                         dim=2)
                self.label_series[frame] = edge.back_rm(self.label_series[frame],
                                                        edge_lim=10,
                                                        dim=2)
        self.img_name = img_name
        self.middle_frame_num = middle_frame
        self.target_middle_frame = self.target_series[self.middle_frame_num,:,:]
        self.label_middle_frame = self.label_series[self.middle_frame_num,:,:]

        self.cell_detector = hyst.hystTool(self.target_middle_frame, **kwargs)  # detect all cells in max frame
        
        # hysteresis debug
        self.detection_mask = self.cell_detector.detection_mask
        self.cells_labels = self.cell_detector.cells_labels
        self.middle_mask = self.cell_detector.cell_mask(self.label_middle_frame)
        self.memb_det_masks = self.cell_detector.memb_mask(self.label_middle_frame)
        self.cells_center = self.cell_detector.cells_center
        self.custom_center = self.cell_detector.mean

        # self.mask_series = [self.cell_detector.cell_mask(frame) for frame in self.label_series]


    def __output_dir_check(output_path, output_name):
        """ Creating output directory.

        """
        save_path = f'{output_path}/memb_res'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def get_sd_surf(self, start_frame=0, fin_frame=-1):
        """ Return 3D array of SD mask for each z-stack frame.
        """
        sd_series = np.array([self.cell_detector.cell_mask(frame) for frame in self.label_series[start_frame:fin_frame]])
        logging.info(f'SD surface series with shape {np.shape(sd_series)} created')
        
        fig =plt.figure(figsize=(6,6))
        ax = fig.gca(projection='3d')
        ax.voxels(sd_series, facecolors='blue', edgecolor='none', alpha=.3)
        plt.show()


class MultiData():
    """ Time series with simultaneous HPCA transfection and calcium dye loading.
    Multiple stimulation during registration.

    Does NOT requires oifpars module.

    """
    def __init__(self, oif_path, img_name, meta_dict):
        self.img_name = img_name
        self.stim_power = meta_dict['power']

        self.baseline_frames = meta_dict['base']
        self.stim_frames = meta_dict['stimul']
        self.stim_loop_num = meta_dict['loop']
        self.tail_frames = meta_dict['tail']
        self.max_ca_frame = self.baseline_frames + self.stim_frames * self.stim_loop_num # index of frame after last stimulation

        # record OIF files combining
        base_path = f'{oif_path}/{img_name}_01.oif'
        self.img_series = oif.OibImread(base_path)  # read baseline record

        for loop_num in range(2, self.stim_loop_num+2):
            loop_path = f'{oif_path}/{img_name}_0{loop_num}.oif'
            self.img_series = np.concatenate((self.img_series, oif.OibImread(loop_path)), axis=1)  # add each stimulation loop record
        
        tail_path = f'{oif_path}/{img_name}_0{loop_num+1}.oif'
        self.img_series = np.concatenate((self.img_series, oif.OibImread(tail_path)), axis=1)  # add tail record

        # channel separation
        self.ca_series = edge.back_rm(self.img_series[0])    # calcium dye channel array
        self.prot_series = edge.back_rm(self.img_series[1])  # fluorescent labeled protein channel array

        logging.info(f'Record {self.img_name} ({self.stim_power}%, {self.baseline_frames}|{self.stim_loop_num}x {self.stim_frames}|{self.tail_frames}) uploaded')

    def get_master_mask(self, sigma=1, kernel_size=5, mask_ext=10):
        """ Whole cell mask building by Ca dye channel data with Otsu thresholding.
        Filters greater element of draft Otsu mask and return master mask array.

        mask_ext - otsu mask extension value in px

        """
        trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for gaussian fliter according to sigma value and kernel size
        self.detection_img = filters.gaussian(self.ca_series[self.max_ca_frame], sigma=sigma, truncate=trun(kernel_size, sigma))
        otsu = filters.threshold_otsu(self.detection_img)
        draft_mask = self.detection_img > otsu
        self.element_label, self.element_num = measure.label(draft_mask, return_num=True)
        logging.info(f'{self.element_num} Otsu mask elements detected')

        detection_label = np.copy(self.element_label)
        element_area = {element.area : element.label for element in measure.regionprops(detection_label)}
        self.master_mask = detection_label == element_area[max(element_area.keys())]

        # mask expansion
        distances, _ = distance_transform_edt(~self.master_mask, return_indices=True)
        self.master_mask = distances <= 10

    def get_delta_mask(self, sigma=1, kernel_size=5, baseline_win=[0, 5], stim_shift=0, loop_win_frames=3, tolerance=200, path=False):
        """ Mask for up and down regions of FP channel data.
        baseline_win - indexes of frames for baseline image creation
        stim_shift - additional value for loop_start_index
        tolerance - tolerance value in au for mask creation, down < -tolerance, up > tolerance

        """
        trun = lambda k, sd: (((k - 1)/2)-0.5)/sd  # calculate truncate value for gaussian fliter according to sigma value and kernel size
        prot_series_sigma = [filters.gaussian(i, sigma=sigma, truncate=trun(kernel_size, sigma)) for i in self.prot_series]
        baseline_prot_img = np.mean(prot_series_sigma[baseline_win[0]:baseline_win[1]], axis=0)

        self.loop_diff_img = []
        for loop_num in range(0, self.stim_loop_num):
            loop_start_index = self.baseline_frames + self.stim_frames*loop_num + stim_shift
            loop_fin_index = loop_start_index + loop_win_frames
            # logging.info(f'Loop mean frame index: {loop_start_index}-{loop_fin_index}')

            loop_prot_img = np.mean(prot_series_sigma[loop_start_index:loop_fin_index], axis=0)

            self.loop_diff_img.append(loop_prot_img - baseline_prot_img)

        centr = lambda img: abs(np.max(img)) if abs(np.max(img)) > abs(np.min(img)) else abs(np.min(img))

        _, axs = plt.subplots(1, self.stim_loop_num, figsize=(12, 12))
        axs = axs.flatten()
        for diff_img, ax in zip(self.loop_diff_img, axs):
            mask_img = ma.masked_where(~self.master_mask, diff_img)
            img = ax.imshow(mask_img, cmap='bwr')
            img.set_clim(vmin=-centr(diff_img), vmax=centr(diff_img))
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', size='3%', pad=0.1)
            ax.axis('off')
            plt.colorbar(img, cax=cax)
        plt.tight_layout()
        plt.savefig(f'{path}/{self.img_name}_loop_delta.png')


    def ca_profile(self, mask=False):
        if not mask:
            mask = self.master_mask
        self.ca_profile = [round(np.sum(ma.masked_where(~mask, img)) / np.sum(mask), 3) for img in self.ca_series]

    def prot_profile(self, mask=False):
        if not mask:
            mask = self.master_mask
        self.prot_profile = [round(np.sum(ma.masked_where(~mask, img)) / np.sum(mask), 3) for img in self.prot_series]

    def save_ctrl_img(self, path, time_scale=1):
        time_line = [i*time_scale for i in range(0,len(self.ca_profile))]

        plt.figure(figsize=(9,10))

        ax0 = plt.subplot(421)
        ax0.set_title('Ca dye base img')
        img0 = ax0.imshow(np.mean(self.ca_series[0:self.baseline_frames], axis=0))
        ax0.axis('off')

        ax1 = plt.subplot(422)
        ax1.set_title('FP base img')
        img1 = ax1.imshow(np.mean(self.prot_series[0:self.baseline_frames], axis=0))
        ax1.axis('off')

        ax2 = plt.subplot(423)
        ax2.set_title('Ca dye Otsu elements')
        img2 = ax2.imshow(self.element_label, cmap='jet')
        ax2.axis('off')

        ax4 = plt.subplot(424)
        ax4.set_title('Ca dye master mask')
        img4 = ax4.imshow(self.master_mask, cmap='jet')
        ax4.axis('off')

        ax3 = plt.subplot(413)
        ax3.set_title('Ca dye profile')
        img3 = ax3.plot(time_line, self.ca_profile)

        ax5 = plt.subplot(414)
        ax5.set_title('FP master mask profile')
        img5 = ax5.plot(time_line, self.prot_profile)

        plt.suptitle(f'{self.img_name}, {self.stim_power}%, {self.baseline_frames}|{self.stim_loop_num}x {self.stim_frames}|{self.tail_frames}')
        plt.tight_layout()
        plt.savefig(f'{path}/{self.img_name}_ctrl_img.png')

        plt.close('all')

        logging.info(f'{self.img_name} control image saved!')

if __name__=="__main__":
  pass


# That's all!
