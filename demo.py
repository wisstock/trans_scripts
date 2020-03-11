#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Prototyping script

"""

import sys
import os
import logging

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from skimage.external import tifffile

sys.path.append('modules')
import slicing as slc
import threshold as ts

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'inferno'

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)

frame = 9
angl = 144

path = os.path.join(sys.path[0], 'dec/cell2/')

for root, dirs, files in os.walk(path):
    for file in files:
    	if file.endswith('.tif'):

            samp = file.split('_')[0]
            file_path = os.path.join(root, file)
            img = tifffile.imread(file_path)

            if file.split('_')[1] == 'ch1':
            	hpca = img
            elif file.split('_')[1] == 'ch2':
            	yfp = img
            else:
            	logging.error('INCORRECT channels notation!')
            	sys.exit()

hpca_frame = hpca[frame,:,:]
yfp_frame = yfp[frame,:,:]

cntr = ts.cellMass(hpca_frame)

xy0, xy1 = slc.lineSlice(hpca_frame, angl, cntr)

yfp_band = slc.bandExtract(yfp_frame, xy0, xy1)
hpca_band = slc.bandExtract(hpca_frame, xy0, xy1)



print(ts.badDiam(yfp_band))
# print(ts.membDet(yfp_band[0], mode='diam'))



# coord = ts.membDet(yfp_band)  # detecting membrane peak in membYFP slice
# if not coord:
#     logging.error('In slice with angle %s mebrane NOT detected!' % angl)
#     continue

# logging.info('Cytoplasm HPCA-TFP: {:.3f}, membrane HPCA-TFP: {:.3f}'.format(hpca_band[0: coord[0]], hpca_band[coord[0]: coord[1]]))
# logging.info('Cytoplasm membYFP: {:.3f}, membrane membYFP: {:.3f}'.format(yfp_band[0: coord[0]], yfp_band[coord[0]: coord[1]]))

# rel_memb_hpca.append(np.sum(m_hpca)/(np.sum(m_hpca) + np.sum(c_hpca)))
# cell_hpca.append(np.sum(c_hpca))
# memb_hpca.append(np.sum(m_hpca))

# rel_memb_yfp.append(np.sum(m_yfp)/(np.sum(m_yfp) + np.sum(c_yfp)))
# cell_yfp.append(np.sum(c_yfp))
# memb_yfp.append(np.sum(m_yfp))



ax = plt.subplot()
ax.plot(yfp_band)
ax.plot(hpca_band)
# ax.axvline(band_qual[0], ymin=0, ymax=band[band_qual[0]], linestyle='dashed')
# ax.axvline(band_qual[1], ymin=0, ymax=band[band_qual[1]], linestyle='dashed')
# ax.axvlines(band_qual[2], ymin=0, ymax=band[band_qual[2]], linestyle='dashed')


plt.show()








