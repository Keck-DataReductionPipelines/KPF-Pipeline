import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import find_peaks as peak
from scipy.optimize import curve_fit as cv
#from lmfit.models import GaussianModel #fit type can be changed
#uses _find_peaks, gaussfit3, gaussval2 from PyReduce
#import get_config_value once it is util primitve
from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments
from scipy.optimize.minpack import curve_fit
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.legendre import Legendre

class LFCWaveCalibration:
    """
    LFC wavelength calibration computation. Algorithm is called to repeat under perform in wavelength_cal.py,
    for each order between min_order and max_order. 

    This module defines 'LFCWaveCalibration' and methods to perform the wavelength calibration.

   Args:
        config (configparser.ConfigParser, optional): Config context. Defaults to None.
        logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.

    Attributes:
        config_param(ConfigHandler): Instance representing pull from config file.
        
    Raises:

    """
    #possible raises: lfcdata isn't in right format
    def __init__(self, config=None, logger=None): #maybe add row as well
        """
        Inits LFCWaveCalibration class with LFC data, config, logger.

        Args:
            config (configparser.ConfigParser, optional): Config context. Defaults to None.
            logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.
        
        Attributes:
            f0 (np.int): Offset frequency of comb, in Hertz. Pulled from config file.
            f_rep (np.int): Repetition frequency of comb, in Hertz. Pulled from config file.
            max_wave (np.int): Maximum wavelength of wavelength range, in Angstroms. Pulled from config file.
            min_wave (np.int): Minimum wavelength of wavelength range, in Angstroms. Pulled from config file.
            fit_order (np.int): Order of fitting polynomial. Pulled from config file.
            min_order (np.int): Minimum order with coherent light/flux in flux extension. Pulled from config file.
            max_order (np.int): Maximum order with coherent light/flux in flux extension. Pulled from config file.
        """
        configpull=ConfigHandler(config,'PARAM')
        self.f0=configpull.get_config_value('f0',-5e9)
        self.f_rep=configpull.get_config_value('f_rep',20e9)
        self.max_wave=configpull.get_config_value('max_wave',9300)
        self.min_wave=configpull.get_config_value('min_wave',3800)
        self.fit_order=configpull.get_config_value('fit_order',6)
        self.min_order=configpull.get_config_value('min_order',25)
        self.max_order=configpull.get_config_value('max_order',100)
        self.config=config
        self.logger=logger

    def get_master_data(self,master_path):
        """Temporary function to pull master data from master calibration file - will be removed once L1 is updated
        and permanent master file is created.

        Args:
            master_path (str): Path to master file name

        Returns:
            master_data: Master calibration data
        """
        m_file=fits.open(master_path)
        if len(m_file)>2:
            print ("Cannot find data extension when there is more than one image HDU")
        else:
            master_data=m_file[1].data
            
        return master_data

    def run_wave_cal(self,flux,master):
        """Runs wavelength calibration algorithm with necessary repetitions for looping through orders: 
        Begins with assembling list of orders to run algorithm on; 
        generates comb lines for eventual wavelength mapping; detects peaks in each order spectrum; generates 
        corresponding indeces to fit wavelengths to comb lines; fits results to polynomial/Legendre; calculates standard error.

        Args:
            flux (np.ndarray): Flux spectrum data
            master (np.ndarray): Master calibration data

        Returns:
            all_wls(np.ndarray): Legendre-fit wavelength solution
        """
        orders=self.order_list()

        comb_lines_ang=self.comb_gen()

        comb_len=self.comb_len(flux)

        ns,all_peaks_exact,all_peaks_approx=[],[],[]
        for order in orders:
            n,peaks_exact,peaks_approx=self.peak_detect(flux,order)
            ns.append(n)
            all_peaks_exact.append(peaks_exact);all_peaks_approx.append(peaks_approx)

        all_idx=[]
        for order,peaks in zip(orders,all_peaks_exact):
            idx=self.mode_match(comb_lines_ang,peaks,comb_len,master,order)
            all_idx.append(idx)

        all_leg,all_wls=[],[]
        for idx,peaks in zip(all_idx,all_peaks_exact):
            leg,wavelengths=self.poly_fit(comb_len,comb_lines_ang,peaks,idx)
            all_leg.append(leg);all_wls.append(wavelengths)

        # errors=[]
        # for wavelengths,idx,peaks,leg in zip(all_wls,all_idx,all_peaks_approx,all_leg):
        #     print(f"wls: {len(wavelengths)}, idx: {idx}, peaks: {len(peaks)}, leg: {len(leg)}")
        #     std_error=self.error_calc(wavelengths,idx,leg,peaks)
        #     errors.append(std_error)

        #padding to match original dimensions which is order by length: (order,length)
        flux_shape=flux.shape
        zeros=np.linspace(0,0,flux_shape[1])
        for i in range(0,min_order):
            all_leg.insert(i,zeros)
        for i in range(max_order,flux_shape[0]):
            all_leg.insert(i,zeros)

        return all_leg

    def order_list(self):
        """Creates list of orders with light for algorithm to iterate through.

        Returns:
            order_list(np.ndarray): List of orders with coherent light/flux.
        """
        order_list=np.arange(self.min_order,self.max_order,1)
        return order_list

    def comb_len(self,flux):
        """Generates number of comb lines, to be used by other functions.

        Args:
            flux (np.ndarray): Flux spectrum data

        Returns:
            comb_len(np.int): Number of comb lines.
        """
        comb=flux[self.min_order]
        comb_len=len(comb)
        return comb_len

    def peak_detect(self,flux,order):
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
            flux(np.ndarray): Flux data.
            order(np.int): Specific flux data order.

        Returns:
            n (np.ndarray): Array of number of peaks. 
            new_peaks (np.ndarray): X-coordinate of each found peak.
            peaks (np.ndarray): Approximate x-coordinate of each found peak.
            
        """
        #for NEID - temporary until linelist creation
        flux[:,435:455] = 0
        flux[48,1933:1938] = 0
        flux[48,48:56] = 0
        #end of - for NEID

        comb=flux[order] #loop through orders

        c = comb - np.ma.min(comb)
        height = np.ma.median(c)
        peaks, properties = signal.find_peaks(c, height=height)
        distance = np.median(np.diff(peaks)) // 4
        peaks, properties = signal.find_peaks(c, height=height, distance=distance)

        # Fit peaks with gaussian to get accurate position -- should really return all coeff 
        new_peaks = peaks.astype(float)
        width = np.mean(np.diff(peaks)) // 2
        for j, p in enumerate(peaks):
            idx = p + np.arange(-width, width + 1, 1)
            idx = np.clip(idx, 0, len(c) - 1).astype(int)
            
            x = np.ma.compressed(np.arange(len(idx)))
            y = np.ma.compressed(c[idx])
            
            def gauss_value(x, a, mu, sig, const):
                return a * np.exp(-((x - mu) ** 2) / (2 * sig)) + const
        
            i = np.argmax(y[len(y) // 4 : len(y) * 3 // 4]) + len(y) // 4
            p0 = [y[i], x[i], 1, np.min(y)]

            with np.warnings.catch_warnings():
                np.warnings.simplefilter("ignore")
                popt, _ = curve_fit(gauss_value, x, y, p0=p0)
            
            coef=popt
            
            new_peaks[j] = coef[1] + p - width

        n = np.arange(len(peaks))
        return n, new_peaks, peaks

    def comb_gen(self):
        """Generates comb lines for mapping flux.

        Returns:
            comb_lines_ang(np.ndarray): Array of comb lines, in Angstroms.
        """
        mode_start=np.int((((scipy.constants.c*1e10)/self.min_wave)-self.f0)/self.f_rep)
        mode_end=np.int((((scipy.constants.c*1e10)/self.max_wave)-self.f0)/self.f_rep)
        mode_nos=np.arange(mode_start,mode_end,-1)

        fxn=self.f0+(mode_nos*self.f_rep)
        ln=scipy.constants.c/fxn
        comb_lines_ang=ln/(1e-10)

        return comb_lines_ang

    def mode_match(self,comb_lines_ang,peaks,comb_len,thar,order):
        #peak_hght from find_peaks
        """Finds corresponding mode numbers to peaks.

        Args:
            comb_lines_ang(np.ndarray): Comb wavelengths, in angstroms
            peaks (np.ndarray): Peak x-coordinates
            comb_len(np.int): Number of comb lines
            thar(np.ndarray): Thorium-Argon solution.
            order(np.int): Specific flux data order.

        Returns:
            idx(np.int): Calibration index
        """
        thar_wavesoln=thar[order]
        approx_peaks_lambda = np.interp(peaks,np.arange(comb_len),thar_wavesoln)
        idx=(np.abs(comb_lines_ang-approx_peaks_lambda[0])).argmin()
        return idx

    def poly_fit(self,comb_len,comb_lines_ang,peaks,idx):
        """Fits order to polynomial.

        Args:
            comb_len(np.int): Number of comb lines
            comb_lines_ang(np.ndarray): Comb wavelengths, in angstroms
            peaks (np.ndarray): Peak x-coordinates
            idx(np.int): Calibration index
            
        Returns:
            wave_soln_leg(np.polynomial): Legendre polynomial-fit wavelength solution
            wavelengths(np.ndarray): Wavelengths of comb lines, correlated with peaks 
        """
        wavelengths=comb_lines_ang[idx:(idx+len(peaks))]
        #polynomial
        #polyfit=Polynomial.fit(peaks,wavelengths,self.fit_order)
        #legendre
        legfit=Legendre.fit(peaks,wavelengths,self.fit_order)

        x_coords=np.arange(len(peaks))
        wave_soln_leg=legfit(x_coords)
        #wave_soln_poly=polyfit(x_coords)
        return wave_soln_leg,wavelengths
    

    def error_calc(self,wavelengths,idx,wave_soln,peaks):
        """Calculates standard error of order.

        Args:
            wavelengths(np.ndarray): Wavelengths of comb lines, correlated with peaks 
            idx(np.int): Calibration index
            wave_soln (np.Polynomial): Polynomial-fit wavelength solution
            peaks (np.ndarray): Peak x-coordinates (requires approximate, integer peaks)

        Returns:
            std_error(): Standard error of polynomial fit per order
        """
        new_pos=wave_soln[peaks]
        residual = ((new_pos - wavelengths)*scipy.constants.c)/wavelengths
        std_resid=np.std(residual)
        std_error=std_resid/np.sqrt(len(peaks))
        return std_error