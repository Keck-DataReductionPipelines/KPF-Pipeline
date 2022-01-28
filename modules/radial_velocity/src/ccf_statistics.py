"""
	This module defines class `CCFStats` which inherits from `KPF1_Primitive` and provides methods to
	extract statistics from the CCF (Bisector, BIS, FWHM, skew, etc.).
"""

import os
import numpy as np
import pandas as pd
import configparser
from scipy import stats
from scipy.optimize import fsolve, curve_fit
from scipy.interpolate import interp1d, UnivariateSpline

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

class CCFStats(object):

	def __init__(self,
				 action: Action,
				 context: ProcessingContext) -> None:
		# Initialize parent class

	def clean_ccf(ccf, ccf_fit_results, width=1):
		'''
		Prep the CCF for computing the bisector:
			1. Interpolate the CCF to estimate minimum value
			2. Define 'continuum' as first turnover point on left/right wing
			3. Normalize the minimum to 0 and 'continuum' to 1
		If `ccf` is 2D, sum into order-summed CCF and return result of 1-3

		parameters:
			ccf : 1D or 2D array
			ccf_fit_results : output of radial_vel.fit_ccf(...)
		
		returns: 
			norm_mean_ccf : Normalized CCF
		'''   
		if len(ccf.shape) == 2:
			ccf = np.nansum(ccf, axis=0) # sum all orders
		else:
			if np.all(np.diff(ccf) == np.diff(ccf)[0]):
				# Constant/linear CCF, probably fully-telluric-masked
				print(o, 'BAD ORDER')
				return np.full_like(ccf, np.nan)
		
		# Unpack CCF fit results
		gaussian_fit, x0, g_x, g_y = ccf_fit_results # x0 = mean of fitted gaussian
		fwhm = FWHM(gaussian_fit)

		# Interpolate the CCF to better estimate its minimum
		iccf = UnivariateSpline(g_x, ccf)
			
		# And compute 1st derivative to find turning points
		diccf = iccf.derivative()
		gxs = np.linspace(min(g_x), max(g_x), 1000)
		crossings = gxs[1:][np.diff(np.sign(diccf(gxs))) == -2] # -2 for maxima
		leftpeaks = sorted(crossings[np.sign(crossings - x0) == -1])
		if len(leftpeaks) == 0:
			leftpeak = np.max(ccf[g_x < x0])
		else:
			leftpeak = ccf[np.argmin(np.abs(g_x-leftpeaks[-1]))]
		rightpeaks = sorted(crossings[np.sign(crossings - x0) == 1])
		if len(rightpeaks) == 0:
			rightpeak = np.max(ccf[g_x > x0])
		else:
			rightpeak = ccf[np.argmin(np.abs(g_x-rightpeaks[0]))]
		
		# Want both wings to be at or pre-turnover for measurement
		continuum = min(leftpeak, rightpeak)
		
		# Normalize
		ccfmin = iccf(x0)
		norm_ccf = (ccf - ccfmin) / (continuum - ccfmin) # Set min to 0
		return norm_ccf

	def skew(ccf, ccf_fit_results):
		'''
		parameters:
			ccf : 1D or 2D array
			ccf_fit_results : output of radial_vel.fit_ccf(...)
		
		returns: 
			skew : skew of the CCF 
		'''
		
		return stats.skew(clean_ccf(ccf, ccf_fit_results)) 
	
	def FWHM(gaussian_fit):
		'''
		gaussian_fit : fitted Gaussian from radial_vel.fit_ccf
		
		returns: 
			FWHM : Full Width at Half Maximum of the CCF
		'''
		
		return 2*np.sqrt(2*np.log(2))*gaussian_fit.stddev.value

	def bisector(ccf, ccf_fit_results, ddepth=0.01, width=2, testing=False):
		'''
		Compute the bisector of the CCF

		ccf : 1D or 2D array, if 2D will be combined into summed CCF before computing bisector
		ccf_fit_results : output of radial_vel.fit_ccf(...)
		ddepth	: stepsize in CCF depth for bisector calculation (default 0.01)
		width	: width (in FWHM) on either side of CCF minimum to cut out wings
		testing : if True, print some useful diagnostics
		
		returns: 
			bisectorx : x axis of bisector [km/s]
			bisectory : y axis of bisector [depth]
		'''
		
		# Unpack CCF fit results
		gaussian_fit, x0, g_x, g_y = ccf_fit_results # x0 = mean of fitted gaussian
		fwhm = FWHM(gaussian_fit)
		
		# Get the normalized CCF
		norm_ccf = clean_ccf(ccf, ccf_fit_results)
		
		# Cut out wings
		whereleft  = np.where((norm_ccf >= 1) & (g_x < x0))[0]
		whereright = np.where((norm_ccf >= 1) & (g_x > x0))[0]
		if len(whereleft) == 0:
			leftbound = x0 - width*fwhm
		else:
			leftbound = g_x[whereleft[-1]]
		if len(whereright)== 0:
			rightbound = x0 + width*fwhm
		else:
			rightbound = g_x[whereright[0]]
		centered = (g_x >= leftbound) & (g_x <= rightbound)

		# Interpolate left/right halves of CCF to get RV(CCF)
		mid = np.sum(ccfx < x0)
		invccfl = interp1d(ccfy[:mid], ccfx[:mid])
		invccfr = interp1d(ccfy[mid:], ccfx[mid:])

		if testing:
			dx = min(max(g_x) - x0, x0 - min(g_x))
			xx = np.linspace(x0 - dx, x0 + dx, 201) # odd steps so x0 is included
			iccf = interp1d(g_x, norm_ccf) # to get the minimum

			# Confirm both halves are invertible
			iccfl = interp1d(ccfx[:mid], ccfy[:mid])
			iccfr = interp1d(ccfx[mid:], ccfy[mid:])
			dl = np.linspace(min(ccfy[:mid]), max(ccfy[:mid]), 101)
			dr = np.linspace(min(ccfy[mid:]), max(ccfy[mid:]), 101)
			print('Left half interpolation invertible: ', np.all(np.isclose(iccf(invccfl(dl)) - dl, 0)))
			print('Right half interpolation invertible: ', np.all(np.isclose(iccf(invccfr(dr)) - dr, 0)))

			
		dmin = max(min(ccfy[:mid]), min(ccfy[mid:]))
		dmax = min(max(ccfy[:mid]), max(ccfy[mid:]))
		depths = np.arange(max(0, dmin), min(1, dmax), ddepth)

		xl = invccfl(depths)
		xr = invccfr(depths)
		
		if testing:
			# Ensure we're drawing flat horizontal lines
			assert np.all(np.isclose(iccf(xl), iccf(xr))), 'Horizontal lines are not horizontal'
			assert np.all(np.isclose(iccfl(xl), iccfr(xr))), 'Left half =/= right half'
			assert np.all(np.isclose(iccfl(xl), iccf(xl))), 'Left half error'
			assert np.all(np.isclose(iccfr(xr), iccf(xr))), 'Right half error'
			assert np.all(np.isclose(depths, iccf(xl))), 'iccf(xl) =/= depths'
			assert np.all(np.isclose(depths, iccf(xr))), 'iccf(xr) =/= depths'
			
		bisectorx = (xl+xr)/2 # midpoints of horizontal lines (should have xl==xr)
		bisectory = depths	  # values of horizontal lines
		
		return bisectorx, bisectory

	def BIS(bisectorx, bisectory):
		'''
		Compute the bisector inverse slope (BIS), defined as the difference of average 
		velocities between 10%–40% and 55%–85% of the total CCF depth (Queloz et al. 2001)

		bisectorx: The velocity dimension of the CCF bisector
		bisectory: The depth dimension of the CCF bisector (normalized in 0 -- 1)
		
		returns: 
			BIS : Bisector Inverse Slope
			err_BIS : Uncertainty in the BIS
		'''
		
		# Calculate the BIS, defined as the difference of average velocities 
		# between 10%–40% and 55%–85% of the total CCF depth (Queloz et al. 2001)
		ccf_lo = (bisectory > 0.10) & (bisectory < 0.40) # 10-40%
		ccf_hi = (bisectory > 0.55) & (bisectory < 0.85) # 55-85%
		bvel_hi = bisectorx[ccf_hi] ; bvel_lo = bisectorx[ccf_lo]
		
		# Uncertainty in BIS from bisector
		err_mnbvelhi = np.std(bvel_hi) / np.sqrt(len(bvel_hi)) # error in mean
		err_mnbvello = np.std(bvel_lo) / np.sqrt(len(bvel_lo)) # error in mean
		
		bis = np.mean(bvel_hi) - np.mean(bvel_lo)
		err_bis = np.sqrt(err_mnbvelhi**2 + err_mnbvello**2) # propagated error in difference
		return bis, err_bis
