#!/usr/bin/env python3

""" Deconvolutio experiments

Usefull links:
https://www.researchgate.net/post/How_are_you_deconvolving_your_confocal_LSM_stacks_for_dendritic_spine_morphological_analyses
https://svi.nl/NyquistCalculator
http://bigwww.epfl.ch/algorithms/psfgenerator/


"""

import os
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import psf


FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(filename="deconvolute.log",
                    level=logging.INFO,
                    filemode="w",
                    format=FORMAT)