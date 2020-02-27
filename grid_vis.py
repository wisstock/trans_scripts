#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Fast vis script

"""

import sys
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib import transforms

from skimage.external import tifffile

sys.path.append('modules')
import oiffile as oif


plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'



# oif_path = '/home/astria/Bio_data/HEK_mYFP/20180523_HEK_membYFP/cell1/20180523-1404-0003-250um.oif'
# tiff_path = "/home/astria/Bio/Lab/scripts/trans_scripts/.temp/data/cell2.tif"


# oif_raw = oif.OibImread(oif_path)
# oif_img = oif_raw[0,:,:,:]

# oif_data = oif.OifFile(oif_path)

# print(oif_data.mainfile)


# tifffile.imsave(output, tiff_path)



data_path = os.path.join(sys.path[0], '.temp/data/cell1.tif')  # "/home/astria/Bio/Lab/scripts/trans_scripts/.temp/data/cell1.tif"
output_path = os.path.join(sys.path[0], '.temp/data/dec_rw_30_rb.tif')  # "/home/astria/Bio/Lab/scripts/trans_scripts/.temp/data/dec_gl_300.tif"

print(data_path)
frame_1 = 3
frame_2 = 10
frame_3 = 17


raw = tifffile.imread(data_path)
dec = tifffile.imread(output_path)


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