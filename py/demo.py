#!/usr/bin/env python3

'''
Copyright © 2020 Borys Olifirov


'''

import sys

import numpy
from PIL import Image

# import logging
# import glob



arr = numpy.random.randint(0,256, 100*100) #example of a 1-D array
arr.resize((100,100))
im = Image.fromarray(numpy.uint8(arr))
im.show()