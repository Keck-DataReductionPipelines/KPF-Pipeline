import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as peak
from scipy.optimize import curve_fit as cv
from lmfit.models import GaussianModel #fit type can be changed

class LFCWaveCalibration:
    """
    LFC wavelength calibration computation.

    This module defines 'LFCWaveCalibration' and methods to perform the wavelength calibration.

   Args:
        config (configparser.ConfigParser, optional): Config context. Defaults to None.
        logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.
        LFcomb (array): FITS laser frequency comb data

    Attributes:
        logger (logging.Logger): Instance of logging.Logger.

    Raises:

    """
    def __init__(self, LFCData, row, config=None, logger=None): #maybe add row as well
        """
        Inits LFCWaveCalibration class with LFC data, row, config, logger.

        Args:
            LFCData (np.ndarray): The FITS LFCData
            row (integer): Row of FITS data to examine.
            config (configparser.ConfigParser, optional): Config context. Defaults to None.
            logger (logging.Lobber, optional): Instance of logging.Logger. Defaults to None.
        """
        self.LFCData=LFCData
        self.config=config
        self.logger=logger
        self.row=row

    #file loaded in

    def peak_detect(self):
        """Finds peaks given specific row of data and various thresholds peaks must surpass. 

        Returns:
            peakxcoords: X-coordinates of found peaks
            peakxhts: Y-coordinates (heights) of found peaks
        """
        peakxcoords,properties=peak(calflux[self.row],distance=10,threshold=50,height=0) #row of data, tester uses calflux[40]
        peakhts=properties["peak_heights"]
        return peakxcoords,peakhts

    def approx_fit(self,peakxs,peakys): #currently gaussian
        """Approximates gaussians to peaks.

        Args:
            peakxs (array): X-coordinates of peaks
            peakys (array): Y-coordinates of peaks (peak heights)
        """
        func=GaussianModel()
        #choosing fit type
        wavewid=np.mean(np.diff(peakxs))
        #getting average difference between peaks, will change to iterate through soon
        wavehalf=int(wavewid/2)
        x=np.arange(0,int(wavewid),1)
        #x components
        for i in range(wavehalf,len(calflux[self.row]),wavehalf):
            pars=func.guess(calflux[self.row][i-wavehalf:i+wavehalf],x=x)
            out=func.fit(calflux[self.row][i-wavehalf:i+wavehalf],pars,x=x)
            #iterating through every curve to get overlapping fit

    #     def poly_fit(self):
    #     """ Fits polynomial to spectrum.
    #     Args:
    #     """

