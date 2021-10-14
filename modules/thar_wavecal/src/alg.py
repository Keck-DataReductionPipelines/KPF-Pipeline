#imports
import numpy as np
import os
from scipy import signal, constants
from astropy.io import fits
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre
from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0

from modules.wavelength_cal.src.alg import LFCWaveCalibration

class TharWaveCalAlg:
    """Performs Thorium Argon (ThAr) wavelength calibration. 
    
    Args:
        config (configparser.ConfigParser, optional): Config context. Defaults to None.
        logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.

    Attributes:
        config_param(ConfigHandler): Instance representing pull from config file.
        
    """

    def __init__(self, config=None, logger=None):
        configpull=ConfigHandler(config,'PARAM')
        self.search_within = configpull.get_config_value('peak_search_range', 1)
        self.fit_order = configpull.get_config_value('fit_order', 6)
        self.gaussian_fit_width = configpull.get_config_value('gaussian_fit_width', 10)
        self.min_order = configpull.get_config_value('min_order', 3)
        self.max_order = configpull.get_config_value('max_order', 117)
        self.n_zoom_sections = configpull.get_config_value('n_zoom_sections', 20)
        self.end_pixels_to_clip = configpull.get_config_value('clip_end_pxls', 1500)
        self.subplot_size = configpull.get_config_value('subplot_size', (6,20))
        #self.saveplots = configpull.get_config_value('saveplots', 'ThAr_plots')
        self.run = LFCWaveCalibration()

    def find_and_fit_peaks(self,flux,linelist,line_pixels_expected,savefig=None):
        """Identifies and fits peaks to gaussian.

        Args:
            flux (np.ndarray): Flux data
            linelist (list): Wavelength line list
            line_pixels_expected (list): Line positions
            savefig (str): Directory for plots
        """
        peaks,_ = signal.find_peaks(flux)
        print('{} peaks detected ({} lines in input list).'.format(len(peaks),len(linelist)))

        num_input_lines = len(linelist)  
        num_pixels = len(flux)

        missed_lines = 0
        coefs = np.zeros((4,num_input_lines))
        for i in np.arange(num_input_lines):
            line_location = line_pixels_expected[i]
            potential_matches = peaks[(
                (peaks > line_location - self.search_within ) &
                (peaks < line_location + self.search_within)
            )]
            num_matches = len(potential_matches)
            if num_matches == 0:
                missed_lines += 1
            elif num_matches > 0:

                if num_matches > 1:
                    raise ValueError('Multiple peaks detected in a 2-pixel radius. Something went wrong with the fitting.')
                else:
                    peak_pixel = potential_matches[0]

                if peak_pixel < self.gaussian_fit_width:
                    first_fit_pixel = 0
                else:
                    first_fit_pixel = peak_pixel - self.gaussian_fit_width
                
                if peak_pixel > num_pixels:
                    last_fit_pixel = num_pixels
                else:
                    last_fit_pixel = peak_pixel + self.gaussian_fit_width

                #fit gaussian to matched peak location
                coefs[:,i] = self.run.fit_gaussian(
                    np.arange(first_fit_pixel,last_fit_pixel),
                    flux[first_fit_pixel:last_fit_pixel])

                amp = coefs[0,i]
                if amp < 0:
                    missed_lines += 1

        return coefs

    def fit_polynomial(self,peak_pixels,vacuum_wavelens,n_pixels):
        """Fits polynomial to peaks.

        Args:
            peak_pixels (np.ndarray): Peak pixel locations
            vacuum_wavelens (np.ndarray): Vaccuum wavelength values
            n_pixels (int): Number of pixels
        """
        leg_out = Legendre.fit(
            peak_pixels[peak_pixels > 0], vacuum_wavelens[peak_pixels >0], self.fit_order
        )
        wl_soln_for_order = leg_out(np.arange(n_pixels))
        
        return wl_soln_for_order


    def calculate_precision(self,our_wls,other_wls,num_lines_fit):
        """Calculates precision of generated wavelengths versus wavelengths key

        Args:
            our_wls (np.ndarray): Wavelengths resulting from ThAr wavelength calibration
            other_wls (np.ndarray): External wavelengths data
            num_lines_fit (int): Number of fit lines
            savefig (str): Directory for plot
        """
        residuals = ((our_wls - other_wls) * constants.c) / other_wls
        precision = np.std(residuals)/np.sqrt(num_lines_fit)

        return precision, residuals

    def run_on_all_orders(self,flux,redman_w,redman_i,linelist_sub,other_wls):
        """Runs ThAr wavelength calibration steps on list of orders.

        Args:
            flux (np.ndarray): Flux data
            redman_w (list): Redman linelist wavelengths
            redman_i (list): Redman linelist identifications (?)
            linelist_sub (list): Linelist subset 
            other_wls (np.ndarray): External wavelengths key data
            plot_toggle (bool): Whether or not to create plots
        """
    
        num_pixels = len(flux[0])

        wls_soln = []

        for order_num in np.arange(self.min_order,self.max_order+1):
            print('\nRunning Order {}!'.format(order_num))

            min_order_wl = np.min(linelist_sub[order_num]['known_wavelengths_vac'])
            max_order_wl = np.max(linelist_sub[order_num]['known_wavelengths_vac'])
            in_order_mask = ((redman_w > min_order_wl) & (redman_w < max_order_wl))
            
            gauss_fit_coefs = self.find_and_fit_peaks(
                flux[order_num],
                linelist_sub[order_num]['known_wavelengths_vac'],
                linelist_sub[order_num]['line_positions'],
            )

            wls = self.fit_polynomial(
                gauss_fit_coefs[1,:],
                linelist_sub[order_num]['known_wavelengths_vac'],
                len(flux[order_num])
            )
            wls_soln.append(wls)

            num_lines_fit = len(gauss_fit_coefs[1,:][gauss_fit_coefs[1,:] > 0])
            precision,residuals = self.calculate_precision(wls,other_wls[order_num],
                num_lines_fit)

            print('Order {} Precision: {:.2f} m/s.'.format(order_num, precision))

        return wls_soln
        








    
                
        