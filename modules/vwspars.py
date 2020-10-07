#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Function for opening Till Photonics .vws files.

Requires Python-bioformats, a Python wrapper for Bio-Formats,
a standalone Java library for reading and writing life sciences image file formats.

"""
import logging
import numpy as np 
from xml.etree import ElementTree as ETree

import javabridge
import bioformats as bf


def readVWS(path):
	javabridge.start_vm(class_path=bf.JARS)

	try:
		logging.info('Data path: {}'.format(path))
		md = bf.get_omexml_metadata(path)
		# img = bf.ImageReader(path, perform_init=True)
	finally:
		javabridge.kill_vm()

	return md


if __name__=="__main__":
    pass


# That's all!