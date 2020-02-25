#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

PSF Gibson-Lanni model

http://kmdouglass.github.io/posts/implementing-a-fast-gibson-lanni-psf-solver-in-python/

"""

import sys

from math import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation

import scipy.special
from scipy.interpolate import interp1d


def generate(args, fb_plot=False):

	# Image properties
	# Size of the PSF array, pixels
	size_x = args['size_x']  # 128
	size_y = args['size_y']  # 128
	size_z = args['size_z']  # 128

	# Precision control
	num_basis    = args['num_basis']  # Number of rescaled Bessels that approximate the phase function, 100
	num_samples  = args['num_samples'] # Number of pupil samples along radial direction, 1000
	oversampling = args['oversampling']    # Defines the upsampling ratio on the image space grid for computations, 2

	# Microscope parameters
	NA          = args['NA']  # 0.9
	wavelength  = args['wavelength']  # microns, 0.512
	M           = args['M']   # magnification, 60
	ns          = args['ns']  # specimen refractive index (RI), 1.33
	ng0         = args['ng0']   # coverslip RI design value, 1.5
	ng          = args['ng']   # coverslip RI experimental value, 1.5
	ni0         = args['ni0']   # immersion medium RI design value, 1.33
	ni          = args['ni']   # immersion medium RI experimental value, 1.33
	ti0         = args['ti0']   # microns, working distance (immersion medium thickness) design value, 150
	tg0         = args['tg0']   # microns, coverslip thickness design value, 170
	tg          = args['tg']   # microns, coverslip thickness experimental value, 170
	res_lateral = args['res_lateral']   # microns, 0.1
	res_axial   = args['res_axial']  # microns, 0.1
	pZ          = args['pZ']     # microns, particle distance from coverslip, 2

	# Scaling factors for the Fourier-Bessel series expansion
	min_wavelength = args['min_wavelength'] # microns, 0.488

	scaling_factor = NA * (3 * np.arange(1, num_basis + 1) - 2) * min_wavelength / wavelength



	# Place the origin at the center of the final PSF array
	x0 = (size_x - 1) / 2
	y0 = (size_y - 1) / 2

	# Find the maximum possible radius coordinate of the PSF array by finding the distance
	# from the center of the array to a corner
	max_radius = round(sqrt((size_x - x0) * (size_x - x0) + (size_y - y0) * (size_y - y0))) + 1;

	# Radial coordinates, image space
	r = res_lateral * np.arange(0, oversampling * max_radius) / oversampling

	# Radial coordinates, pupil space
	a = min([NA, ns, ni, ni0, ng, ng0]) / NA
	rho = np.linspace(0, a, num_samples)

	# Stage displacements away from best focus
	z = res_axial * np.arange(-size_z / 2, size_z /2) + res_axial / 2



	# Define the wavefront aberration
	OPDs = pZ * np.sqrt(ns * ns - NA * NA * rho * rho) # OPD in the sample
	OPDi = (z.reshape(-1,1) + ti0) * np.sqrt(ni * ni - NA * NA * rho * rho) - ti0 * np.sqrt(ni0 * ni0 - NA * NA * rho * rho) # OPD in the immersion medium
	OPDg = tg * np.sqrt(ng * ng - NA * NA * rho * rho) - tg0 * np.sqrt(ng0 * ng0 - NA * NA * rho * rho) # OPD in the coverslip
	W    = 2 * np.pi / wavelength * (OPDs + OPDi + OPDg)

	# Sample the phase
	# Shape is (number of z samples by number of rho samples)
	phase = np.cos(W) + 1j * np.sin(W)

	# Define the basis of Bessel functions
	# Shape is (number of basis functions by number of rho samples)
	J = scipy.special.jv(0, scaling_factor.reshape(-1, 1) * rho)

	# Compute the approximation to the sampled pupil phase by finding the least squares
	# solution to the complex coefficients of the Fourier-Bessel expansion.
	# Shape of C is (number of basis functions by number of z samples).
	# Note the matrix transposes to get the dimensions correct.
	C, residuals, _, _ = np.linalg.lstsq(J.T, phase.T)



	# Which z-plane to compute
	z0 = 24

	# The Fourier-Bessel approximation
	if fb_plot == True:
	    est = J.T.dot(C[:,z0])

	    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(10,5))
	    ax[0].plot(rho, np.real(phase[z0, :]), label=r'$ \exp{ \left[ jW \left( \rho \right) \right] }$')
	    ax[0].plot(rho, np.real(est), '--', label='Fourier-Bessel')
	    ax[0].set_xlabel(r'$\rho$')
	    ax[0].set_title('Real')
	    ax[0].legend(loc='upper left')

	    ax[1].plot(rho, np.imag(phase[z0, :]))
	    ax[1].plot(rho, np.imag(est), '--')
	    ax[1].set_xlabel(r'$\rho$')
	    ax[1].set_title('Imaginary')
	    ax[1].set_ylim((-1.5, 1.5))

	    plt.show()



	b = 2 * np. pi * r.reshape(-1, 1) * NA / wavelength

	# Convenience functions for J0 and J1 Bessel functions
	J0 = lambda x: scipy.special.jv(0, x)
	J1 = lambda x: scipy.special.jv(1, x)

	# See equation 5 in Li, Xue, and Blu
	denom = scaling_factor * scaling_factor - b * b
	R = (scaling_factor * J1(scaling_factor * a) * J0(b * a) * a - b * J0(scaling_factor * a) * J1(b * a) * a)
	R /= denom



	# The transpose places the axial direction along the first dimension of the array, i.e. rows
	# This is only for convenience.
	PSF_rz = (np.abs(R.dot(C))**2).T

	# Normalize to the maximum value
	# PSF_rz /= np.max(PSF_rz)


	# Resample the PSF onto a rotationally-symmetric Cartesian grid #

	# Create the fleshed-out xy grid of radial distances from the center
	xy = np.mgrid[0:size_y, 0:size_x]
	r_pixel = np.sqrt((xy[1] - x0) * (xy[1] - x0) + (xy[0] - y0) * (xy[0] - y0)) * res_lateral

	PSF = np.zeros((size_y, size_x, size_z))

	for z_index in range(PSF.shape[2]):
	    # Interpolate the radial PSF function
	    PSF_interp = interp1d(r, PSF_rz[z_index, :])
	    
	    # Evaluate the PSF at each value of r_pixel
	    PSF[:,:, z_index] = PSF_interp(r_pixel.ravel()).reshape(size_y, size_x)

	# **All lines below are changes to original implementation** #

	# Transform to [z, y, x] instead of [y, x, z]
	PSF = np.moveaxis(PSF, 2, 0)

	# Re-normalize to a max of 1
	PSF /= np.max(PSF)

	return PSF

# print(np.shape(PSF))

# fig, ax = plt.subplots()

# ax.imshow(PSF_rz, extent=(r.min(), r.max(), z.max(), z.min()))
# ax.set_xlim((0,2.5))
# ax.set_ylim((-6, 0))
# ax.set_xlabel(r'r, $\mu m$')
# ax.set_ylabel(r'z, $\mu m$')

# plt.show()