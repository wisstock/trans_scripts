#!/usr/bin/env python3

'''
Copyright © 2020 Borys Olifirov


'''

import sys
import os
import glob

import numpy as np
from PIL import Image

# import logging




# arr = numpy.random.randint(0,256, 100*100)
# arr.resize((100,100))
# im = Image.fromarray(numpy.uint8(arr))
# im.show()



wd_path = os.path.split(os.getcwd())  # working dir path
os.chdir(wd_path[0] + '/temp/data/')  # go to DATA dir

# /home/astria/Bio/Lab/scripts/Translocations/temp/data


for current_image in glob.glob("*.png"):
	b = current_image.split('.')[0]
	a.append(b)

print(a)