#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Deconvolutio resylts vis

"""

import sys

import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from timeit import default_timer as timer
from skimage import io
from skimage.external import tifffile

sys.path.append('modules')
import oiffile as oif
import slicing as slc
import threshold as ts

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'





data_path = "/home/astria/Bio/Lab/scripts/trans_scripts/.temp/data/cell1.tif"
output_path = "/home/astria/Bio/Lab/scripts/trans_scripts/.temp/data/res.tif"


frame = 10  # 
angle = 45  # BAND SLICE ARGS
band_w = 2  #

frame_1 = 7   #
frame_2 = 10  # GRID PLOT ARGS
frame_3 = 12  #

raw = tifffile.imread(data_path)  # load imput file
dec = tifffile.imread(output_path)



# # Create tiff file from oif
# oif_raw = oif.OibImread(oif_input)
# oif_img = oif_raw[0,:,:,:]

# # print(np.shape(oif_img))

# tifffile.imsave(data_path, oif_img)
# logger.info('Create TIFF from OIF "{}"'.format(oif_input))

# 



# BAND SLICE PLOT
img = raw[frame,:,:]
img_mod = dec[frame,:,:]

cntr = ts.cellMass(img)
xy0, xy1 = slc.lineSlice(img, angle, cntr)

raw_slice = slc.bandExtract(img, xy0, xy1, band_w)
mod_slice = slc.bandExtract(img_mod, xy0, xy1, band_w)


fig, ax = plt.subplots(4, 8)

ax[0] = plt.subplot(321)
ax[0].imshow(img)  #, cmap='gray')
ax[0].set_title('Raw image')
ax[0].plot([xy0[0], xy1[0]], [xy0[1], xy1[1]], 'ro-')
ax[0].scatter(cntr[0],cntr[1],color='r')
# ax0.scatter(cntr_img[0],cntr_img[1])
# ax0.scatter(start[0]+5, start[1]+5)

ax[1] = plt.subplot(322)
ax[1].imshow(img_mod)  #, cmap='gray')
ax[1].set_title('Deconvoluted image')
ax[1].plot([xy0[0], xy1[0]], [xy0[1], xy1[1]], 'ro-')
ax[1].scatter(cntr[0],cntr[1],color='r')
# ax1.scatter(cntr_img[0],cntr_img[1])
# ax0.scatter(start[0]+5, start[1]+5)

ax[2] = plt.subplot(312)
ax[2].set_title('Rav slice')
ax[2].plot(raw_slice)

ax[3] = plt.subplot(313)
ax[3].set_title('Deconvoluted slice')
ax[3].plot(mod_slice)




# GRID PLOT
img_raw_1 = raw[frame_1,:,:]
img_dec_1 = dec[frame_1,:,:]

img_raw_2 = raw[frame_2,:,:]
img_dec_2 = dec[frame_2,:,:]

img_raw_3 = raw[frame_3,:,:]
img_dec_3 = dec[frame_3,:,:]



ax0 = plt.subplot(231)
slice_0 = ax0.imshow(img_raw_1) 
divider_0 = make_axes_locatable(ax0)
cax = divider_0.append_axes("right", size="3%", pad=0.1)
plt.colorbar(slice_0, cax=cax)
ax0.set_title('Raw, frame %s' % frame_1)

ax1 = plt.subplot(234)
slice_1 = ax1.imshow(img_dec_1)
# ax1.plot([0, psf_size[0]-1], [xy_slice, xy_slice])
divider_1 = make_axes_locatable(ax1)
cax = divider_1.append_axes("right", size="3%", pad=0.1)
plt.colorbar(slice_1, cax=cax)
ax1.set_title('Deconvoluted, frame %s' % frame_1)

ax2 = plt.subplot(232)
slice_2 = ax2.imshow(img_raw_2)
# ax1.plot([0, psf_size[0]-1], [xy_slice, xy_slice])
divider_2 = make_axes_locatable(ax2)
cax = divider_2.append_axes("right", size="3%", pad=0.1)
plt.colorbar(slice_2, cax=cax)
ax2.set_title('Raw, frame %s' % frame_2)

ax3 = plt.subplot(235)
slice_3 = ax3.imshow(img_dec_2)
# ax1.plot([0, psf_size[0]-1], [xy_slice, xy_slice])
divider_3 = make_axes_locatable(ax3)
cax = divider_3.append_axes("right", size="3%", pad=0.1)
plt.colorbar(slice_3, cax=cax)
ax3.set_title('Deconvoluted, frame %s' % frame_2)

ax4 = plt.subplot(233)
slice_4 = ax4.imshow(img_raw_3)
# ax1.plot([0, psf_size[0]-1], [xy_slice, xy_slice])
divider_4 = make_axes_locatable(ax4)
cax = divider_4.append_axes("right", size="3%", pad=0.1)
plt.colorbar(slice_4, cax=cax)
ax4.set_title('Raw, frame %s' % frame_3)

ax5 = plt.subplot(236)
slice_5 = ax5.imshow(img_dec_3)
# ax1.plot([0, psf_size[0]-1], [xy_slice, xy_slice])
divider_5 = make_axes_locatable(ax5)
cax = divider_5.append_axes("right", size="3%", pad=0.1)
plt.colorbar(slice_5, cax=cax)
ax5.set_title('Deconvoluted, frame %s' % frame_3)


plt.show()