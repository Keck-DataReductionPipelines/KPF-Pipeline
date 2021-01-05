import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy
from scipy.signal import find_peaks as peak
from scipy.optimize import curve_fit as cv
#from lmfit.models import GaussianModel #fit type can be changed
#uses _find_peaks, gaussfit3, gaussval2 from PyReduce

from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments
from scipy.optimize.minpack import curve_fit
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.legendre import Legendre

class LFCWaveCalibration:
    """
    LFC wavelength calibration computation.

    This module defines 'LFCWaveCalibration' and methods to perform the wavelength calibration.

   Args:
        config (configparser.ConfigParser, optional): Config context. Defaults to None.
        logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.
        f0 (np.float): Frequency offset of LFC, in Hertz
        f_rep (np.float): Frequency rate of LFC, in Hertz

    Attributes:
        f0 (np.float): From parameter 'f0'
        f_rep (np.float): From parameter 'f_rep'
        
    Raises:
        

    """
    #possible raises: lfcdata isn't in right format
    def __init__(self, f0, f_rep, config=None, logger=None): #maybe add row as well
        """
        Inits LFCWaveCalibration class with LFC data, config, logger.

        Args:
            config (configparser.ConfigParser, optional): Config context. Defaults to None.
            logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.
            f0 (np.float): Frequency offset of LFC, in Hertz
            f_rep (np.float): Frequency rate of LFC, in Hertz

        """
        self.f0=f0
        self.f_rep=f_rep
        self.config=config
        self.logger=logger

    def peak_detect(self,LFCData):
        #algorithm from PyReduce
        #PyReduce notes:
            # Find peaks in the data spectrum
            # Run find_peak twice
            # once to find the average distance between peaks
            # once for real (disregarding close peaks)
        """Detects peaks in LFC Data. First runs peak finding to find average distance between peaks, then a second to 
        find peak x and y-coordinates, disregarding close-together peaks. 
        Then, fits peaks to Gaussian to get accurate peak positions. Takes i-th peak x-value as center of gaussian, subtracts and adds 
        average width around it to get x data for gaussian fit. y data is simply the corresponding LFC data. Performs gaussian fitting 
        on x and y data arrays to get coefficients.

        Args:
            LFCData (np.ndarray): The FITS LFCData

        Returns:
            new_peaks (np.ndarray): X-coordinate of each found peak
            props(dict): Further peak properties, i.e. peak heights
        """
        c=LFCData-np.ma.min(LFCData)
        height = np.ma.median(c)
        peaks, props = peak(c, height=height)
        distance=np.median(np.diff(peaks))//4
        peaks, props = peak(c, height=height,distance=distance)
    
        # Fit peaks with gaussian to get accurate position
        new_peaks = peaks.astype(float)
        width = np.mean(np.diff(peaks)) // 2
        for j, p in enumerate(peaks):
            idx = p + np.arange(-width, width + 1, 1)
            idx = np.clip(idx, 0, len(c) - 1).astype(int)

            coef= gauss_fit(np.arange(len(idx)), c[idx])
            new_peaks[j] = coef[1] + p - width

        n = np.arange(len(peaks))

        return new_peaks, props

    def gauss_fit(self,x, y):
        """ 
        Simple gaussian fit: gauss = A * exp(-(x-mu)**2/(2*sig**2)) + offset

        Args:
            x (np.ndarray): array of shape (n,) of x data
            y (np.ndarray): array of shape (n,) of y data

        Returns:
            popt (list) : list of shape (4,) including parameters A, mu, sigma**2, offset
        """
        x = np.ma.compressed(x)
        y = np.ma.compressed(y)
        gauss = gauss_value
        i = np.argmax(y[len(y) // 4 : len(y) * 3 // 4]) + len(y) // 4
        p0 = [y[i], x[i], 1, np.min(y)]

        with np.warnings.catch_warnings():
            np.warnings.simplefilter("ignore")
            popt, _ = curve_fit(gauss, x, y, p0=p0)

        return popt

    def gauss_value(self,x, a, mu, sig, const):
        """Gaussian function.

        Args:
            x (np.ndarray): Peak x-coordinates
            a (np.float): Constant, i-th value of peak y-coordinates (height of peak curve)
            mu (np.float): Constant, i-th value of peak x-coordinates (peak position)
            sig (np.int): Standard deviation
            const (np.float): Offset

        Returns:
            a * np.exp(-((x - mu) ** 2) / (2 * sig)) + const []: Gauss function values
        """
        return a * np.exp(-((x - mu) ** 2) / (2 * sig)) + const

    def comb_gen(self,mode_nos):
        """Generates comb wavelengths.

        Args:
            mode_nos (np.ndarray): Evenly spaced mode lines

        Returns:
            comb_lines_ang (np.ndarray): Comb wavelengths, in angstroms
        """

        fn=self.f0+(mode_nos*self.f_rep)
        ln=scipy.constants.c/fn
        ln_ang=ln/(1e-10)

        return ln_ang

    def mode_nos(self,min_wave,max_wave):
        """Generates comb lines.

        Returns:
            comb_lines_ang(np.ndarray): Comb wavelengths, in angstroms
            mode_nos(np.ndarray): Evenly spaced mode lines
        """
        mode_start=np.int((((scipy.constants.c*1e10)/min_wave)-self.f0)/self.f_rep)
        mode_end=np.int((((scipy.constants.c*1e10)/max_wave)-self.f0)/self.f_rep)

        mode_nos=np.arange(mode_start,mode_end,-1)
        comb_lines_ang=comb_gen(mode_nos,self.f0,self.f_rep)

        return comb_lines_ang


    def mode_match(self,comb_lines_ang,peaks):
        #peak_hght from find_peaks
        """Finds corresponding mode numbers to peaks.

        Args:
            comb_lines_ang(np.ndarray): Comb wavelengths, in angstroms
            peaks (np.ndarray): Peak x-coordinates

        Returns:
            idx(np.int): Corresponding index
        """
        idx=(np.abs(comb_lines_ang-peaks[0])).argmin()
        return idx

    def poly_fit(self,comb_lines_ang,peaks,idx,fit_order):
        """Fits order to polynomial.

        Args:
            comb_lines_ang(np.ndarray): Comb wavelengths, in angstroms
            peaks (np.ndarray): Peak x-coordinates
            idx(np.int): Corresponding index
            
        Returns:
            wave_soln_leg(np.polynomial): Legendre polynomial-fit wavelength solution
            wave_soln_poly(np.Polynomial): Regular polynomial-fit wavelength solution
        """
        wavelengths=comb_lines_ang[idx:(idx+len(peaks))]
        #polynomial
        polyfit=Polynomial.fit(peaks,wavelengths,fit_order)
        #legendre
        legfit=Legendre.fit(peaks,wavelengths,fit_order)

        x_coords=np.arange(len(peaks))
        wave_soln_leg=legfit(x_coords)
        #wave_soln_poly=polyfit(x_coords)
        return wave_soln_leg
    

    def residuals(self,comb_lines_ang,idx,wave_soln,peaks):
        """Calculates residuals.

        Args:
            wave_soln (np.Polynomial): Polynomial-fit wavelength solution
            peaks (np.ndarray): Peak x-coordinates

        Returns:
            std_resid(): Standard deviation of residuals
        """
        wavelengths=comb_lines_ang[idx:(idx+len(peaks))]
        new_pos=wave_soln[peaks]
        residual =((new_pos-wavelengths)*scipy.constants.c)/wavelengths
        std_resid=np.std(residual)
        return std_resid

