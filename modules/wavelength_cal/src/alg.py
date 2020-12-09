import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as peak
from scipy.optimize import curve_fit as cv
#from lmfit.models import GaussianModel #fit type can be changed
#uses _find_peaks, gaussfit3, gaussval2 from PyReduce

from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments

class LFCWaveCalibration:
    """
    LFC wavelength calibration computation.

    This module defines 'LFCWaveCalibration' and methods to perform the wavelength calibration.

   Args:
        config (configparser.ConfigParser, optional): Config context. Defaults to None.
        logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.
        LFCData (array): FITS laser frequency comb data

    Attributes:
        logger (logging.Logger): Instance of logging.Logger.

    Raises:
        

    """
    def __init__(self, LFCData, f0, f_rep,config=None, logger=None): #maybe add row as well
        """
        Inits LFCWaveCalibration class with LFC data, config, logger.

        Args:
            LFCData (np.ndarray): The FITS LFCData
            config (configparser.ConfigParser, optional): Config context. Defaults to None.
            logger (logging.Lobber, optional): Instance of logging.Logger. Defaults to None.
        """
        self.LFCData=LFCData
        self.f0=f0
        self.f_rep=f_rep
        self.config=config
        self.logger=logger

    #file loaded in
    #load in specifically calflux?

    def peak_detect(self):
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
            n (np.int): Number of peaks found
            new_peaks (np.ndarray): X-coordinate of each found peak
            peakhts (np.ndarray): Y-coordinate of each found peak (peak height)
        """
        width = lfc_peak_width
        height = np.ma.median(self.LFCdata)
        peaks, _ = peak(self.LFCdata, height=height, width=width)
        distance = np.median(np.diff(peaks)) // 4
        peaks, props = peak(self.LFCdata, height=height, distance=distance, width=width)
        peakhts=props["peak_heights"]

        # Fit peaks with gaussian to get accurate position
        new_peaks = peaks.astype(float)
        width = np.mean(np.diff(peaks)) // 2
        for j, p in enumerate(peaks):
            idx = p + np.arange(-width, width + 1, 1)
            idx = np.clip(idx, 0, len(data) - 1).astype(int)

            coef= gauss_fit(np.arange(len(idx)), self.LFCdata[idx])
            new_peaks[j] = coef[1] + p - width

        n = np.arange(len(peaks))

        return n, new_peaks, peakhts

    def gauss_fit(x, y):
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
        gauss = gaussvalue
        i = np.argmax(y[len(y) // 4 : len(y) * 3 // 4]) + len(y) // 4
        p0 = [y[i], x[i], 1, np.min(y)]

        with np.warnings.catch_warnings():
            np.warnings.simplefilter("ignore")
            popt, _ = curve_fit(gauss, x, y, p0=p0)

        return popt

    def gauss_value(x, a, mu, sig, const):
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

    # def calc_LFC_modes(self):
    #     #need f0 and f_rep fed in in recipe?
    #     #c = speed of light
    #     nlines=
    #     fn=self.f0+(nlines*f_rep)
    #     ln=c*1e13/fn


    # def poly_fit(self,peakxs,peakys):
    #     """Fits order to polynomial.

    #     Args:
    #         peakxs (np.ndarray): X-coordinates of peaks
    #         peakys (np.ndarray): Y-coordinates of peaks (peak heights)

    #     Returns:
    #     """
        
    # def get(self):
    #     """Returns LFC wavelength solution.

    #     Returns:
    #         wavesoln: LFC wavelength solution.
    #     """
    #     return wavesoln


