#!/usr/bin/env python3

""" Copyright © 2020-2021 Borys Olifirov
Toolkit.
- Function for series analysis (require image series):
  back_rm
  series_sum_int
  series_point_delta
  series_derivate

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


def deltaF(int_list, f_0_win=5):
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


def alex_delta(series, mask=False, baseline_frames=5, max_frames=[10, 15], sd_tolerance=2, t_val=200, output_path=False):
    """ Detecting increasing and decreasing areas, detection limit - sd_tolerance * sd_cell.
    Framses indexes for images calc:

         stimulus
            |
    --------|--------  registration
         |--|--|
         ^  ^  ^   
         |  |  |      
         |  |  max_frames[1]
         |  max_frames[0]
         max_frames[0] - baseline_frames

    It's necessary for decreasing of the cell movement influence.

    """
    baseline_img = np.mean(series[max_frames[0]-baseline_frames:max_frames[0]-2,:,:], axis=0)
    max_img = np.mean(series[max_frames[0]:max_frames[1],:,:], axis=0)

    delta = lambda f, f_0: (f - f_0)/f_0 if f_0 > 0 else f_0  # pixel-wise ΔF/F0
    vdelta = np.vectorize(delta)

    cell_sd = np.std(ma.masked_where(~mask, series[max_frames[0],:,:]))
    logging.info(f'Cell area SD={round(cell_sd, 2)}')

    diff_img = max_img - baseline_img
    diff_img = diff_img.astype(np.int)  # convert float difference image to integer
    delta_img = vdelta(max_img, baseline_img)

    # up/down mask creating
    t_val = np.max(max_img) * 0.05
    up_mask = np.copy(diff_img) > t_val
    down_mask = np.copy(diff_img) < -t_val

    if output_path:
        plt.figure()
        ax = plt.subplot()
        img = ax.imshow(delta_img, cmap='jet')
        # img.set_clim(vmin=-1., vmax=1.) 
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img, cax=cax)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_path}/alex_mask.png')

        plt.figure()
        ax0 = plt.subplot(121)
        img0 = ax0.imshow(baseline_img)
        ax0.text(10,10,f'baseline img, frames {max_frames[0]-baseline_frames}-{max_frames[0]-2}',fontsize=8)
        ax0.axis('off')
        ax1 = plt.subplot(122)
        img1 = ax1.imshow(max_img)
        ax1.text(10,10,f'max img, frames {max_frames[0]}-{max_frames[1]}',fontsize=8)
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img1, cax=cax1)
        ax1.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_path}/alex_ctrl.png')

        ax2 = plt.subplot()
        ax2.text(10,10,f'max - baseline',fontsize=8)
        img2 = ax2.imshow(diff_img, cmap='seismic')
        img2.set_clim(vmin=-850., vmax=850.)
        div2 = make_axes_locatable(ax2)
        cax2 = div2.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(img2, cax=cax2)
        ax2.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_path}/ctrl_diff.png')

        plt.figure()
        ax0 = plt.subplot(121)
        img0 = ax0.imshow(up_mask)
        ax0.text(10,10,'up mask', fontsize=8)
        ax0.axis('off')
        ax1 = plt.subplot(122)
        img1 = ax1.imshow(down_mask)
        ax1.text(10,10,'down mask', fontsize=8)
        ax1.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_path}/masks.png')

        plt.close('all')
        logging.info('Alex F/F0 mask saved!')
        
        return up_mask, down_mask
    else:
        return up_mask, down_mask


def series_point_delta(series, mask=False, baseline_frames=3, sigma=4, kernel_size=5, output_path=False):
    """ Pixel-wise ΔF/F0 calculation.
    baseline_frames - numbers of frames for mean baseline image calculation (from first to baseline_frames value frames)

    WARNING! The function is sensitive to cell shift during long registrations!

    """
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


if __name__=="__main__":
    pass


# That's all!