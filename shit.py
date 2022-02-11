#!/usr/bin/env python3

""" Copyright © 2020-2022 Borys Olifirov

links:
https://webdevblog.ru/obyasnenie-classmethod-i-staticmethod-v-python/
https://webdevblog.ru/nasledovanie-i-kompoziciya-rukovodstvo-po-oop-python/
https://stackoverflow.com/questions/49370769/python-init-and-classmethod-do-they-have-to-have-the-same-number-of-args

"""

# Experiments with mask creation
import sys
import os
import logging

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.append('modules')
import oiffile as oif
import edge


plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'jet'

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)


# img_series = oif.OibImread('fluo_data/11_13_2021/cell14/cell14_01.oif')[0,:,:,:]
# img_mean = np.mean(img_series, axis=0)
# print(np.shape(img_series))

# img_drop, label_mask, artefact_mask = edge.mask_point_artefact(img_series, img_mode='mean', sigma=1, kernel_size=20, return_extra=True)

# fig, axes =  plt.subplots(ncols=3, figsize=(20,10))
# ax = axes.ravel()
# ax[0].imshow(img_mean, vmin=img_mean.min(), vmax=img_mean.max())
# ax[0].set_title('Initial img')
# ax[1].imshow(label_mask)
# ax[1].set_title('Artefact mask')
# ax[2].imshow(img_drop, vmin=img_mean.min(), vmax=img_mean.max())
# ax[2].set_title('Masked img')
# plt.show()

# CUSTOM COLOR MAP VISUALIZATION
cdict = {
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

# cdict1 = {
#       'red':(
#         (0.0, 0.0, 0.0),
#         (0.5, 0.0, 0.0),
#         (0.6, 0.5, 0.5),
#         (1.0, 1.0, 1.0)),

#       'green':(
#         (0.0, 1.0, 1.0),
#         (0.4, 0.5, 0.5),
#         (0.5, 0.0, 0.0),
#         (1.0, 0.0, 0.0)),

#       'blue':(
#         (0.0, 0.0, 0.0),
#         (1.0, 0.0, 0.0))
#     }

def plot_linearmap(cdict):
    newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
    rgba = newcmp(np.linspace(0, 1, 256))
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    col = ['r', 'g', 'b']
    for xx in [0.25, 0.5, 0.75]:
        ax.axvline(xx, color='0.7', linestyle='--')
    for i in range(3):
        ax.plot(np.arange(256)/256, rgba[:, i], color=col[i])
    ax.set_xlabel('index')
    ax.set_ylabel('RGB')
    plt.show()

plot_linearmap(cdict)

# # Experiments with OOS composition
# class aaa():
# 	def __init__(self):
# 		pass

# 	@classmethod
# 	def settings(cls, a=0):
# 		cls.a = a

# 	def add(self, c):
# 		self.c = c + self.a
# 		return self.c


# class bbb():
# 	def __init__(self, b=2):
# 		self.b = b
# 		self.AAA = aaa()
# 		self.det = self.AAA.add(c=self.b)

# 	def plus(self):
# 		print(self.b, self.AAA.a)
# 		print(self.det)


# aaa.settings(a=2)
# var = bbb()
# var.plus()