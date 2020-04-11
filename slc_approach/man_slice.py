#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Manual diam slice stat calculationg

"""

import sys
import os
import logging

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.external import tifffile

sys.path.append('modules')
import slicing as slc
import threshold as ts
import membrane as memb

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)

path = os.path.join(sys.path[0], 'raw_data/3/')

angl = 150
peal_cutoff = 0.85

# cell_px = 45            #
# roi_start = [140, 200]  # cell 1 options
# roi_x_lim = 20          #
# roi_y_lim = 20          #

# cell_px = 20
# roi_start = [145, 160]  # cell 2 options
# roi_x_lim = 20          #
# roi_y_lim = 20          #

cell_px = 55
extra_px = 80
roi_start = [165, 150]  # cell 3 options
roi_x_lim = 20          #
roi_y_lim = 20          #

# roi_start = [70, 150]   # cell 4 options
# roi_x_lim = 20          #
# roi_y_lim = 20          #

# roi_start = [50, 100]   # cell 5 options
# roi_x_lim = 20          #
# roi_y_lim = 20          #


# hoi = 0.75
scaling = 6

for root, dirs, files in os.walk(path):
    for file in files:
    	if file.endswith('.tif'):

            samp = file.split('_')[0]
            file_path = os.path.join(root, file)
            img = tifffile.imread(file_path)

            if file.split('.')[0] == 'HPCATFP':
                hpca = img
                logging.info('HPCA-TFP data uploaded')
            elif file.split('.')[0] == 'membYFP':
                yfp = img
                logging.info('membYFP data uploaded')
            else:
            	logging.error('INCORRECT channels notation!')
            	sys.exit()


hpca_frame = ts.backCon(hpca, edge_lim=np.shape(hpca)[1] // scaling, dim=2)
yfp_frame = ts.backCon(yfp, edge_lim=np.shape(yfp)[1] // scaling, dim=2)


# hpca_frame = hpca_frame[:195,:]  # for cell 5
# yfp_frame = yfp_frame[:195,:]    #
# cntr = [55, 100]                 #

cntr = ts.cellMass(yfp_frame)

xy0, xy1 = slc.lineSlice(yfp_frame, angl, cntr)

yfp_band = slc.bandExtract(yfp_frame, xy0, xy1)

test_band, peak = memb.membOutDet(yfp_band,
                                   cell_mask=cell_px,
                                   outer_mask=extra_px,
                                   det_cutoff=peal_cutoff)







# if ts.badDiam(yfp_band):
#     logging.fatal('No peak detected!')
#     # sys.exit()

# coord, peak = ts.membDet(yfp_band, mode='diam', h=hoi)  # detecting membrane peak in membYFP slice

# if not coord:
#     logging.fatal('No mebrane detected!\n' % angl)
#     # sys.exit()

hpca_band = slc.bandExtract(hpca_frame, xy0, xy1)


hpca_roi = np.mean(hpca_frame[roi_start[0]:roi_start[0]+roi_x_lim, \
                   roi_start[1]:roi_start[1]+roi_y_lim])

logging.info('HPCA-TFP ROI mean value: {:.3f}'.format(hpca_roi))

cell, memb, memb_lim = memb.membExtract(slc=hpca_band,
                                        memb_loc=peak,
                                        roi_val=hpca_roi)

logging.info('Relative membrane count {:.3f}'.format(memb/(cell+memb)))

# m_yfp_l = yfp_band[coord[0][0]: coord[0][1]]
# m_yfp_r = yfp_band[coord[1][0]: coord[1][1]]
# c_yfp = yfp_band[coord[0][1]: coord[1][0]]

# m_hpca_l = hpca_band[coord[0][0]: coord[0][1]]
# m_hpca_r = hpca_band[coord[1][0]: coord[1][1]]
# c_hpca = hpca_band[coord[0][1]: coord[1][0]]

# m_yfp_total = np.sum(m_yfp_l) + np.sum(m_yfp_r)
# yfp_total = np.sum(c_yfp) + m_yfp_total

# m_hpca_total = np.sum(m_hpca_l) + np.sum(m_hpca_r)
# hpca_total = np.sum(c_hpca) + m_hpca_total


# logging.info('Sample {}'.format(samp))
# # logging.info('Frame num {}'.format(frame+1))
# logging.info('Angle {}\n'.format(angl))

# logging.info('Relative membrane mYFP {:.3f}'.format((m_yfp_total/yfp_total)*100))
# logging.info('Relative membrane mYFP {:.3f}\n'.format((m_hpca_total/hpca_total)*100))


# memb_loc = []
# for i in range(len(yfp_band)):
#     if i <= coord[0][0]:
#         memb_loc.append(0)
#     elif i > coord[0][0] and i < coord[0][1]:
#         memb_loc.append(peak[0]*hoi)
#     elif i >= coord[0][1] and i <= coord[1][0]:
#         memb_loc.append(0)
#     elif i > coord[1][0] and i < coord[1][1]:
#         memb_loc.append(peak[1]*hoi)
#     elif i >= coord[1][1]:
#         memb_loc.append(0)

roi = patches.Rectangle((roi_start[0],roi_start[1]),
                        roi_y_lim,
                        roi_x_lim,
                        linewidth=1,
                        edgecolor='b',
                        facecolor='none')


ax = plt.subplot(212)
ax.plot(yfp_band, label='membYFP')
ax.plot(hpca_band, label='HPCA-TFP')
ax.plot(test_band, linestyle='dashed')
ax.axvline(peak[0], 0, 1000)
ax.axvline(peak[1], 0, 1000)
ax.axvline(memb_lim[0], 0, 1000,
           linestyle='dashed')
ax.axvline(memb_lim[1], 0, 1000,
           linestyle='dashed')
# ax.plot(memb_loc, label='Memb. loc.', linestyle='dashed')
ax.legend(loc='upper left')

ax1 = plt.subplot(221)
ax1.plot([xy0[0], xy1[0]], [xy0[1], xy1[1]])
slc1 = ax1.imshow(yfp_frame)
div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slc1, cax=cax1)
ax1.set_title('membYFP')
ax1.add_patch(roi)

ax2 = plt.subplot(222)
ax2.plot([xy0[0], xy1[0]], [xy0[1], xy1[1]])
slc2 = ax2.imshow(hpca_frame)
div2 = make_axes_locatable(ax2)
cax2 = div2.append_axes('right', size='3%', pad=0.1)
plt.colorbar(slc2, cax=cax2)
ax2.set_title('HPCA-TFP')
# ax2.add_patch(roi)

# plt.suptitle('Samp {}_2, frame {}, angle {}'.format(samp, (frame+1), angl))
plt.show()