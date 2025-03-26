from astropy import units as u, constants as cst
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import Legendre
from modules.Utils.utils import DummyLogger
import os
import time
import pandas as pd
import scipy
from scipy import signal
from scipy.special import erf
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline, interp1d
from scipy.optimize.minpack import curve_fit
from datetime import datetime, timedelta
from dateutil import parser
from modules.Utils.config_parser import ConfigHandler
import modules.Utils.utils
import warnings
from numpy.polynomial.polynomial import Polynomial

class WaveCalibration:
    """
    This module defines 'WaveCalibration' and methods to perform the 
    wavelength calibration.
    
    Wavelength calibration computation. Algorithm is called under _perform() 
    in wavelength_cal.py. Algorithm itself iterates over orders.
    """
    
    def __init__(
        self, cal_type, clip_peaks_toggle, quicklook, min_order, max_order, save_diagnostics=None, 
        config=None, logger=None
    ):
        """Initializes WaveCalibration class.
        Args:
            clip_peaks_toggle (bool): Whether or not to clip any peaks. True to clip, false to not clip.          
            quicklook (bool): Whether or not to run quicklook-specific algorithmic steps. False runs non-quicklook, full pipeline version.
            min_order (int): minimum order to fit
            max_order (int): maximum order to fit
            save_diagnostics (str) : Directory in which to save diagnostic plots and information. Defaults to None, which results 
                in no saved diagnostics info.
            config (configparser.ConfigParser, optional): Config context. 
                Defaults to None.
            logger (logging.Logger, optional): Instance of logging.Logger. 
                Defaults to None.        
        """
        self.cal_type = cal_type
        self.clip_peaks_toggle = clip_peaks_toggle
        self.quicklook = quicklook
        self.min_order = min_order
        self.max_order = max_order
        self.save_diagnostics_dir = save_diagnostics
        configpull = ConfigHandler(config,'PARAM')
        self.figsave_name = configpull.get_config_value('drift_figsave_name','instrument_drift')
        self.red_skip_orders = configpull.get_config_value('red_skip_orders')
        self.green_skip_orders = configpull.get_config_value('green_skip_orders')
        self.chi_2_threshold = configpull.get_config_value('chi_2_threshold')
        self.skip_orders = configpull.get_config_value('skip_orders',None)
        self.quicklook_steps = configpull.get_config_value('quicklook_steps',10)
        self.min_wave = configpull.get_config_value('min_wave',3800)
        self.max_wave = configpull.get_config_value('max_wave',9300)
        self.fit_order = configpull.get_config_value('fit_order',9)
        self.fit_type = configpull.get_config_value('fit_type', 'Legendre')
        self.n_sections = configpull.get_config_value('n_sections',1)
        self.clip_peaks_toggle = configpull.get_config_value('clip_peaks',False)
        self.clip_below_median  = configpull.get_config_value('clip_below_median',True)
        self.peak_height_threshold = configpull.get_config_value('peak_height_threshold',1.5)
        self.sigma_clip = configpull.get_config_value('sigma_clip',2.1)
        self.fit_iterations = configpull.get_config_value('fit_iterations',5)
        self.logger = logger
        self.etalon_mask_in = configpull.get_config_value('master_etalon_file',None)
 
    def run_wavelength_cal(
        self, calflux, rough_wls=None, our_wavelength_solution_for_order=None,
        peak_wavelengths_ang=None, lfc_allowed_wls=None,input_filename=None):
        """ Runs all wavelength calibration algorithm steps in order.
        Args:
            calflux (np.array): (N_orders x N_pixels) array of L1 flux data of a 
                calibration source
            rough_wls (np.array): (N_orders x N_pixels) array of wavelength 
                values describing a "rough" wavelength solution. Always None for
                lamps. For LFC, this is generally a lamp-derived solution.
                For Etalon, this is generally an LFC-derived solution. Default None.
            peak_wavelengths_ang (dict of dicts): dictionary of order number-dict
                pairs. Each order number corresponds to a dict containing 
                an array of expected line positions (pixel values) under the key 
                "line_positions" and an array of corresponding wavelengths in 
                Angstroms under the key "known_wavelengths_vac". This value must be
                set for lamps. Can be set or not set for LFC and Etalon. If set to None,
                then peak finding is not run. Defaults to None. Ex:
                    {51: {
                            "line_positions" : array([500.2, ... 8000.3]),
                            "known_wavelengths_vac" : array([3633.1, ... 3570.1])
                        }
                    }
        
            lfc_allowed_wls (np.array): array of all allowed wavelengths for the 
                LFC, computed using the order_flux equation. Should be None unless we
                are calibrating an LFC frame. Defaults to None.
        Examples:
            1: Calibrating an LFC frame using a rough ThAr solution,
               with no previous LFC frames to inform this one:
                rough_wls -> ThAr-derived wavelength solution
                lfc_allowed_wls -> wavelengths computed from comb eq
            2: Calibrating an LFC frame using a rough ThAr solution,
               given information about expected mode position:
                rough_wls -> ThAr-derived wavelength solution
                lfc_allowed_wls -> wavelengths computed from comb eq
                peak_wavelengths_ang -> LFC mode wavelengths and their
                    expected pixel locations
            3: Calibrating a lamp frame:
                peak_wavelengths_ang -> lamp line wavelengths in vacuum and their
                    expected rough pixel locations
            4: Calibrating an Etalon frame using an LFC-derived solution, with
               no previous Etalon frames to inform this one:
                rough_wls -> LFC-derived wavelength solution
            5: Calibrating an Etalon frame using an LFC-derived solution and
               at least one other Etalon frame to inform this one:
                rough_wls -> LFC-derived wavelength solution
                peak_wavelengths_ang -> Etalon peak wavelengths and their
                    expected pixel locations
        Returns:
            tuple of:
                np.array: Calculated polynomial solution 
                np.array: (N_orders x N_pixels) array of the computed wavelength
                    for each pixel.
                dictionary: information about the fits for each line and order (orderlet_dict)
        """
        self.filename=input_filename
        # create directories for diagnostic plots
        if type(self.save_diagnostics_dir) == str:
            if not os.path.isdir(self.save_diagnostics_dir):
                os.makedirs(self.save_diagnostics_dir)
            if not os.path.isdir(self.save_diagnostics_dir + '/order_diagnostics'):
                os.makedirs(self.save_diagnostics_dir + '/order_diagnostics')

        if self.quicklook == False:
            order_list = self.remove_orders(step=1)
            n_orders = len(order_list)

            # masked_calflux = self.mask_array_neid(calflux, n_orders)
            masked_calflux = calflux # TODO: fix

            # perform wavelength calibration
            poly_soln, wls_and_pixels, orderlet_dict = self.fit_many_orders(
                masked_calflux, order_list, rough_wls=rough_wls, 
                comb_lines_angstrom=lfc_allowed_wls,
                expected_peak_locs=peak_wavelengths_ang, peak_wavelengths_ang=peak_wavelengths_ang,
                our_wavelength_solution_for_order=our_wavelength_solution_for_order, print_update=True, plt_path=self.save_diagnostics_dir   
            )

            # make a plot of all of the precise new wls minus the rough input  wls
            if self.save_diagnostics_dir is not None and rough_wls is not None:
                # don't do this for etalon exposures, where we're either not 
                # deriving a new wls or using drift to do so
                if self.cal_type != 'Etalon':
                    fig, ax = plt.subplots(2,1, figsize=(12,5))
                    for i in order_list:
                        wls_i = poly_soln[i, :]
                        rough_wls_i = rough_wls[i,:]
                        ax[0].plot(wls_i - rough_wls_i, color='grey', alpha=0.5)

                        pixel_sizes = rough_wls_i[1:] - rough_wls_i[:-1]
                        ax[1].plot(
                            (wls_i[:-1] - rough_wls_i[:-1]) / pixel_sizes, 
                            color='grey', alpha=0.5
                        )

                    ax[0].set_title('Derived WLS - Approx WLS')
                    ax[0].set_xlabel('Pixel')
                    ax[0].set_ylabel('[$\\rm \AA$]')
                    ax[1].set_xlabel('Pixel')
                    ax[1].set_ylabel('[Pixel]')
                    plt.tight_layout()
                    plt.savefig(
                        '{}/all_wls.png'.format(self.save_diagnostics_dir), 
                        dpi=250
                    )

        if self.quicklook == True:
            #TODO
            order_list = self.remove_orders(step = self.quicklook_steps)
            n_orders = len(order_list)
            
            #masked_calflux = self.mask_array_neid(calflux,n_orders)
            masked_calflux = calflux
            
            poly_soln, wls_and_pixels, orderlet_dict = self.fit_many_orders(
                masked_calflux, order_list, rough_wls=rough_wls, 
                comb_lines_angstrom=lfc_allowed_wls,
                expected_peak_locs=peak_wavelengths_ang, peak_wavelengths_ang=peak_wavelengths_ang,
                print_update=True, plt_path=self.save_diagnostics_dir ###CHECK THIS TODO
            )

        return poly_soln, wls_and_pixels, orderlet_dict    

    def find_etalon_peaks(self,flux,wave,etalon_mask):
        """
        Fit peaks of etalon calibration with a gaussian function
        Args: 
            Wavelengths of one order
            Flux of one order
            Full etalon mask from master file.
        Returns
            Original an new etalon peak positions for one order
        """
        mask1 = etalon_mask[(etalon_mask['wave'] > min(wave)) & (etalon_mask['wave'] < max(wave))] 
        mask = np.sort(mask1['wave'].values) # This may be causing problems on edges of orderw, where they overlap.
        mask = mask[::-1]#reverse order
        params=[]
        new_peaks = []

        # Next loop over the peaks in the mask, extacting a wavelength section on each side, how many pixels?
        for i,item in enumerate(mask[:]): # remove the first element of mask, too close to edge
            if item < 6000: # Green CCD  # green: 0.15:54 bad peaks.
                incr = 0.15
            else:
                incr = 0.29  # Red CCD  0.29:278 missed peaks
                        
            w_lw = item-incr
            w_hi  = item+incr
            fit_indx = (wave > w_lw) & (wave < w_hi) # may need to catch exceptions here.
            wave_clp = wave[fit_indx]
            flux_clp = flux[fit_indx] # Sometimes flux_clp is always false. Avoid this.
            #Quality check:
            no_flux_index = not any(flux_clp) # True if no flux values are found for this wavelength
            if not no_flux_index:
                popt = self.fit_gaussian(wave_clp,flux_clp)
                if np.abs(popt[1] - item) < 0.5: # wavelength sections are much smaller than 0.1
                    new_peaks.append(popt[1]) # if fit is okay.
                else:
                    new_peaks.append(item) # If new fit is way off, keep initial guess
            else:
                new_peaks.append(item) # Fill with initial guess. This happens due to edge trimming

        return mask, new_peaks

    def fit_many_orders(
        self, cal_flux, order_list, rough_wls=None, comb_lines_angstrom=None,
        expected_peak_locs=None,peak_wavelengths_ang=None, our_wavelength_solution_for_order=None, plt_path=None, print_update=False):
        """
        Iteratively performs wavelength calibration for all orders.
        Args:
            cal_flux (np.array): (n_orders x n_pixels) array of calibrator fluxes
                for which to derive a wavelength solution
            order_list (list of int): list order to compute wls for
            rough_wls (np.array): (N_orders x N_pixels) array of wavelength 
                values describing a "rough" wavelength solution. Always None for
                lamps. For LFC, this is generally a lamp-derived solution.
                For Etalon, this is generally an LFC-derived solution. Default None.
            comb_lines_angstrom (np.array): array of all allowed wavelengths for the 
                LFC, computed using the order_flux equation. Should be None unless we
                are calibrating an LFC frame. Default None.
            expected_peak_locs (dict): dictionary of order number-dict
                pairs. See description in run_wavelength_cal().
            plt_path (str): if set, all diagnostic plots will be saved in this
                directory. If None, no plots will be made.
            print_update (bool): whether subfunctions should print updates.
        Returns:
            tuple of:
                np.array of float: (N_orders x N_pixels) derived wavelength 
                    solution for each pixel
                dict: the peaks and wavelengths used for wavelength cal. Keys 
                    are ints representing order numbers, values are 2-tuples of:
                        - lists of wavelengths corresponding to peaks
                        - the corresponding (fractional) pixels on which the 
                          peaks fall
                dict: the orderlet dictionary, that is folded into wls_dict at a higher level
        """    
        
        # Construct dictionary for each order in wlsdict 
        orderlet_dict = {}
        for order_num in order_list:
            orderlet_dict[order_num] = {"ordernum" : order_num}

        # Plot 2D extracted spectra
        if plt_path is not None:
            plt.figure(figsize=(20,10), tight_layout=True)
            im = plt.imshow(cal_flux, aspect='auto')
            im.set_clim(0, 20000)
            plt.xlabel('Pixel')
            plt.ylabel('Order Number')
            plt.savefig('{}/extracted_spectra.png'.format(plt_path), dpi=250)
            plt.close()

        # Define variables to be used later
        order_precisions = []
        num_detected_peaks = []
        wavelengths_and_pixels = {}
        poly_soln_final_array = np.zeros(np.shape(cal_flux))

        # Iterate over orders
        for order_num in order_list:
            if print_update:
                print('\nRunning order # {}'.format(order_num))

            if plt_path is not None:
                order_plt_path = '{}/order_diagnostics/order{}'.format(plt_path, order_num)
                if not os.path.isdir(order_plt_path):
                    os.makedirs(order_plt_path)

                plt.figure(figsize=(20,10), tight_layout=True)
                #plt.plot(cal_flux[order_num,:], color='k', alpha=0.5)
                plt.plot(cal_flux[order_num,:], color='k', linewidth = 0.5)
                plt.title('Order # {}'.format(order_num), fontsize=36)
                plt.xlabel('Pixel', fontsize=28)
                plt.ylabel('Flux', fontsize=28)
                plt.yscale('symlog')
                plt.tick_params(axis='both', direction='inout', length=6, width=3, colors='k', labelsize=24)
                plt.savefig('{}/order_spectrum.png'.format(order_plt_path), dpi=250)
                plt.close()
            else:
                order_plt_path = None

            order_flux = cal_flux[order_num,:]
            rough_wls_order = rough_wls[order_num,:]
            n_pixels = len(order_flux)
            
            # Add information for this order to the orderlet dictionary
            orderlet_dict[order_num]['flux'] = order_flux
            orderlet_dict[order_num]['initial_wls'] = rough_wls_order
            orderlet_dict[order_num]['echelle_order'] = \
                modules.Utils.utils.get_kpf_echelle_order(np.median(rough_wls_order))
            orderlet_dict[order_num]['n_pixels'] = n_pixels
            orderlet_dict[order_num]['lines'] = {}

            # check if there's flux in the orderlet (e.g., SKY order 0 is off of the GREEN CCD)
            npixels_wflux = len([x for x in order_flux if x != 0])
            if npixels_wflux == 0: 
                self.logger.warn('This order has no flux, defaulting to rough WLS')
                continue

            if self.cal_type == 'Etalon':  # For etalon
                etalon_mask = pd.read_csv(self.etalon_mask_in, names=['wave','weight'], sep='\s+')
                wls, fitted_peak_pixels = self.find_etalon_peaks(order_flux,rough_wls_order,etalon_mask) # returns original mask and new mask positions for one order.
                wls=wls.tolist()

            # find, clip, and compute precise wavelengths for peaks.
            # this code snippet will only execute for Etalon and LFC frames.
            elif expected_peak_locs is None:
                skip_orders_wls = None
                if self.red_skip_orders and max(order_list) == 31:  # KPF max order for red chip (update if changed in KPF.cfg)
                    skip_orders_wls = np.fromstring(self.red_skip_orders, dtype=int, sep=',')
                elif self.green_skip_orders and max(order_list) == 34:  # KPF max order for green chip (update if changed in KPF.cfg)
                    skip_orders_wls = np.fromstring(self.green_skip_orders, dtype=int, sep=',')

                if skip_orders_wls is not None:
                    try:
                        if order_num in skip_orders_wls:
                            raise Exception(f'Order {order_num} is skipped in the config, defaulting to rough WLS')
                    except Exception as e:
                        print(e)
                        poly_soln_final_array[order_num, :] = rough_wls_order
                        wavelengths_and_pixels[order_num] = {
                            'known_wavelengths_vac': rough_wls_order,
                            'line_positions': []
                        }
                        continue

                try:
                    fitted_peak_pixels, detected_peak_pixels, \
                        detected_peak_heights, gauss_coeffs, lines_dict = self.find_peaks_in_order(
                        order_flux, plot_path=order_plt_path
                    )
                    orderlet_dict[order_num]['lines'] = lines_dict
                    
                except TypeError as e:
                    self.logger.warn('Not enough peaks found in order, defaulting to rough WLS')
                    self.logger.warn('TypeError = ' + str(e))
                    poly_soln_final_array[order_num,:] = rough_wls_order
                    wavelengths_and_pixels[order_num] = {
                        'known_wavelengths_vac': rough_wls_order, 
                        'line_positions':[]
                    }
                    order_dict = {}
                    continue

                if self.clip_peaks_toggle:
                    good_peak_idx = self.clip_peaks(
                        order_flux, fitted_peak_pixels, detected_peak_pixels,
                        gauss_coeffs, detected_peak_heights, 
                        clip_below_median=self.clip_below_median,
                        plot_path=order_plt_path, print_update=print_update
                    )
                else:
                    good_peak_idx = np.arange(len(detected_peak_pixels))

                if self.cal_type == 'LFC':
                    try:
                        wls, _, good_peak_idx = self.mode_match(
                            order_flux, fitted_peak_pixels, good_peak_idx, 
                            rough_wls_order, comb_lines_angstrom, 
                            print_update=print_update, plot_path=order_plt_path
                        )
                    except:
                        poly_soln_final_array[order_num,:] = rough_wls_order
                        wavelengths_and_pixels[order_num] = {
                            'known_wavelengths_vac': rough_wls_order, 
                            'line_positions':[]
                        }
                        order_dict = {}
                        continue
                elif self.cal_type == 'Etalon':

                    assert comb_lines_angstrom is None, '`comb_lines_angstrom` \
                        should not be set for Etalon frames.'

                    wls = np.interp(
                        fitted_peak_pixels[good_peak_idx], np.arange(n_pixels)[rough_wls_order>0], 
                        rough_wls_order[rough_wls_order>0]
                    )

                fitted_peak_pixels = fitted_peak_pixels[good_peak_idx]

                # Mark lines with bad fits and lambda_fit for each line in dictionary:
                '''
                good_line_ind = 0
                for l in np.arange(len(lines_dict)):
                    if l not in good_peak_idx:
                        orderlet_dict[order_num]['lines'][l]['quality'] = 'bad' #TODO: add this functionality to ThAr dictionaries
                    else:
                        orderlet_dict[order_num]['lines'][l]['lambda_fit'] = wls[good_line_ind]
                        good_line_ind += 1
                '''
            # use expected peak locations to compute updated precise wavelengths for each pixel
            # (only ThAr)
            else:
                if order_plt_path is not None:
                    plot_toggle = True
                else:
                    plot_toggle = False

                min_order_wave = np.min(rough_wls_order)
                max_order_wave = np.max(rough_wls_order)
                line_wavelengths = expected_peak_locs.query(f'{min_order_wave} < wave < {max_order_wave}')['wave'].values
                
                pixels_order = np.arange(0, len(rough_wls_order))
                wave_to_pix = interp1d(rough_wls_order, pixels_order,
                                       assume_sorted=False)
                line_pixels_expected = wave_to_pix(line_wavelengths)

                sorted_indices = np.argsort(line_pixels_expected)
                line_wavelengths = line_wavelengths[sorted_indices]


                line_pixels_expected = line_pixels_expected[sorted_indices]

                line_wavelengths = np.array([
                    line_wavelengths[i] for i in 
                    np.arange(1, len(line_pixels_expected)) if 
                    line_pixels_expected[i] != line_pixels_expected[i-1]
                ])
                line_pixels_expected = np.array([
                    line_pixels_expected[i] for i in 
                    np.arange(1, len(line_pixels_expected)) if 
                    line_pixels_expected[i] != line_pixels_expected[i-1]
                ])
                wls, gauss_coeffs, lines_dict = self.line_match(
                    order_flux, line_wavelengths, line_pixels_expected, 
                    plot_toggle, order_plt_path
                )
                
                orderlet_dict[order_num]['lines'] = lines_dict
                
                fitted_peak_pixels = gauss_coeffs[1,:]

            # if we don't have an etalon frame, we won't use drift to calculate the wls
            # To-do for Etalon: add line_dicts
            if self.cal_type != 'Etalon':
                if expected_peak_locs is None:
                    peak_heights = detected_peak_heights[good_peak_idx]
                else:
                    peak_heights = fitted_peak_pixels

                # calculate the wavelength solution for the order
                polynomial_wls, leg_out = self.fit_polynomial(
                    wls, rough_wls_order, peak_wavelengths_ang, order_list, n_pixels, fitted_peak_pixels, peak_heights=peak_heights,
                    plot_path=order_plt_path, fit_iterations=self.fit_iterations,
                    sigma_clip=self.sigma_clip)
                poly_soln_final_array[order_num,:] = polynomial_wls

                if plt_path is not None:
                    fig, ax = plt.subplots(2, 1, figsize=(12,5))
                    ax[0].set_title('Precise WLS - Rough WLS')
                    ax[0].plot(np.arange(n_pixels), leg_out(np.arange(n_pixels)) - rough_wls_order, color='k')
                    ax[0].set_ylabel('[$\\rm \AA$]')
                    pixel_sizes = rough_wls_order[1:] - rough_wls_order[:-1]
                    ax[1].plot(np.arange(n_pixels - 1),   
                              (leg_out(np.arange(n_pixels - 1)) - rough_wls_order[:-1]) / pixel_sizes, color='k')
                    ax[1].set_ylabel('[Pixels]')
                    ax[1].set_xlabel('Pixel')
                    plt.tight_layout()
                    plt.savefig('{}/precise_vs_rough.png'.format(order_plt_path), dpi=250)
                    plt.close()

                # compute various RV precision values for order
                rel_precision, abs_precision = self.calculate_rv_precision(
                    fitted_peak_pixels, wls, leg_out, rough_wls_order, our_wavelength_solution_for_order, rough_wls_order, plot_path=order_plt_path, 
                    print_update=print_update
                )
                order_precisions.append(abs_precision)
                num_detected_peaks.append(len(fitted_peak_pixels))

                # Add to dictionary for this order
                orderlet_dict[order_num]['fitted_wls'] = polynomial_wls 
                orderlet_dict[order_num]['rel_precision_cms'] = rel_precision 
                orderlet_dict[order_num]['abs_precision_cms'] = abs_precision 
                orderlet_dict[order_num]['num_detected_peaks'] = len(fitted_peak_pixels) 
                orderlet_dict[order_num]['known_wavelengths_vac'] = wls 
                orderlet_dict[order_num]['line_positions'] = fitted_peak_pixels 

            # compute drift, and use this to update the wavelength solution
            else:
                pass
                
            wavelengths_and_pixels[order_num] = {
                'known_wavelengths_vac':wls, 
                'line_positions':fitted_peak_pixels
            }

        # for lamps and LFC, we can compute absolute precision across all orders
        if self.cal_type != 'Etalon':
            squared_resids = (np.array(order_precisions) * num_detected_peaks)**2
            sum_of_squared_resids = np.sum(squared_resids)
            overall_std_error = (np.sqrt(sum_of_squared_resids) / np.sum(num_detected_peaks))
            #orderlet_dict['overall_std_error_cms'] = overall_std_error
            print('\n\n\nOverall absolute precision (all orders): {:2.2f} cm/s\n\n\n'.format(overall_std_error))

        return poly_soln_final_array, wavelengths_and_pixels, orderlet_dict

    def remove_orders(self,step=1):
        """Removes bad orders from order list if between min and max orders to test.
        Args:
            step (int): Interval at which to test orders. Used to skip orders 
                for QLP. Defaults to 1, which means every order will be tested 
                on and none will be removed.
        Returns:
            list: List of orders to run wavelength calibration on.
        """

        order_list = [*range(self.min_order, self.max_order + 1, step)]
    
        if self.skip_orders:
            self.skip_orders = np.array(self.skip_orders.split(',')).astype('int')
            for i in self.skip_orders:
                if i in order_list:
                    order_list.remove(i)
                else:
                    continue
                
        return order_list

    def find_peaks_in_order(self, order_flux, plot_path=None):
        """
        Runs find_peaks on successive subsections of the order_flux lines and concatenates
        the output. The difference between adjacent peaks changes as a function
        of position on the detector, so this results in more accurate peak-finding.
        Based on pyreduce.
        Args:
            order_flux (np.array): flux values. Their indices correspond to
                their pixel numbers. Generally the entire order.
            plot_path (str): Path for diagnostic plots. If None, plots are not made.
        Returns:
            tuple of:
                np.array: array of true peak locations as determined by Gaussian fitting
                np.array: array of detected peak locations (pre-Gaussian fitting)
                np.array: array of detected peak heights (pre-Gaussian fitting)
                np.array: array of size (4, n_peaks) 
                    containing best-fit Gaussian parameters [a, mu, sigma**2, const]
                    for each detected peak
                dict: dictionary of information about each line in the order
        """

        lines_dict = {}
    
        n_pixels = len(order_flux)
        fitted_peak_pixels = np.array([])
        detected_peak_pixels = np.array([])
        detected_peak_heights = np.array([])
        gauss_coeffs = np.zeros((4,0))
        ind_dict = 0

        try:
            for i in np.arange(self.n_sections):
    
                if i == self.n_sections - 1:
                    indices = np.arange(i * n_pixels // self.n_sections, n_pixels)
                else:
                    indices = np.arange(i * n_pixels // self.n_sections, (i+1) * n_pixels // self.n_sections)
                    
                fitted_peaks_section, detected_peaks_section, peak_heights_section, \
                    gauss_coeffs_section, this_lines_dict = self.find_peaks(order_flux[indices], peak_height_threshold=self.peak_height_threshold)
    
                for ii, row in enumerate(this_lines_dict):
                    lines_dict[ind_dict] = this_lines_dict[ii]
                    ind_dict += 1
                
                detected_peak_heights = np.append(detected_peak_heights, peak_heights_section)
                gauss_coeffs = np.append(gauss_coeffs, gauss_coeffs_section, axis=1)
                if i == 0:
                    fitted_peak_pixels = np.append(fitted_peak_pixels, fitted_peaks_section)
                    detected_peak_pixels = np.append(detected_peak_pixels, detected_peaks_section)
    
                else:
                    fitted_peak_pixels = np.append(
                        fitted_peak_pixels, 
                        fitted_peaks_section + i * n_pixels // self.n_sections
                    )
                    detected_peak_pixels = np.append(
                        detected_peak_pixels, 
                        detected_peaks_section + i * n_pixels // self.n_sections
                    )
        
        except Exception as e:
            print('Exception: ' + str(e))
            print('self.n_sections = ', str(self.n_sections))
        
        if plot_path is not None:
            plt.figure(figsize=(20,10), tight_layout=True)
            #plt.plot(order_flux, color='k', lw=0.1)   
            plt.plot(order_flux, color='k', lw=0.5)   
            plt.scatter(detected_peak_pixels, detected_peak_heights, s=2, color='r')
            plt.xlabel('Pixel', fontsize=28)
            plt.ylabel('Flux', fontsize=28)
            plt.yscale('symlog')
            plt.tick_params(axis='both', direction='inout', length=6, width=3, colors='k', labelsize=24)
            plt.savefig('{}/detected_peaks.png'.format(plot_path), dpi=250)
            plt.close()

            n_zoom_sections = 5
            zoom_section_pixels = n_pixels // n_zoom_sections

            _, ax_list = plt.subplots(n_zoom_sections, 1, figsize=(12,6))
            for i, ax in enumerate(ax_list):
                ax.plot(order_flux,color='k', lw=0.5)
                ax.scatter(detected_peak_pixels,detected_peak_heights,s=1,color='r')
                ax.set_xlim(zoom_section_pixels * i, zoom_section_pixels * (i+1))
                ax.set_ylim(
                    0,
                    np.max(
                        order_flux[zoom_section_pixels * i : zoom_section_pixels * (i+1)]
                    )
                )
                ax.set_ylabel('Flux', fontsize=14)
                if i == n_zoom_sections-1:
                    ax.set_xlabel('Pixel', fontsize=14)

            plt.tight_layout()
            plt.savefig('{}/detected_peaks_zoom.png'.format(plot_path),dpi=250)
            plt.close()
                  
        return fitted_peak_pixels, detected_peak_pixels, detected_peak_heights, gauss_coeffs, lines_dict

    def find_peaks(self, order_flux, peak_height_threshold=1.5):
        """
        Finds all order_flux peaks in an array. This runs scipy.signal.find_peaks 
            twice: once to find the average distance between peaks, and once
            for real, disregarding close peaks.
        Args:
            order_flux (np.array): flux values. Their indices correspond to
                their pixel numbers. Generally a subset of the full order.
            peak_height_threshold (float): only detect peaks above this num * sigma
                above the chip median.
            
        Returns:
            tuple of:
                np.array: array of true peak locations as determined by Gaussian fitting
                np.array: array of detected peak locations (pre-Gaussian fitting)
                np.array: array of detected peak heights (pre-Gaussian fitting)
                np.array: array of size (4, n_peaks) containing best-fit Gaussian 
                    parameters [a, mu, sigma**2, const] for each detected peak
        """

        lines_dict = {} # dictionary of lines and their parameters
        
        c = order_flux - np.ma.min(order_flux)

        # TODO: make this more indep of order_flux flux
        height = peak_height_threshold * np.ma.median(c)
        detected_peaks, properties = signal.find_peaks(c, height=height)

        distance = np.median(np.diff(detected_peaks)) // 2
        detected_peaks, properties = signal.find_peaks(c, distance=distance, height=height)
        peak_heights = np.array(properties['peak_heights'])

        # Only consider peaks with height greater than 500
        valid_peak_indices = np.where(peak_heights > 500)[0]
        detected_peaks = detected_peaks[valid_peak_indices]
        peak_heights = peak_heights[valid_peak_indices]

        # fit peaks with Gaussian to get accurate position
        fitted_peaks = detected_peaks.astype(float)
        gauss_coeffs = np.empty((4, len(detected_peaks)))
        width = np.mean(np.diff(detected_peaks)) // 2

        # Create mask initially set to True for all detected peaks
        mask = np.ones(len(detected_peaks), dtype=bool)

        for j, p in enumerate(detected_peaks):
            idx = p + np.arange(-width, width + 1, 1)
            idx = np.clip(idx, 0, len(c) - 1).astype(int)
            coef, line_dict = self.fit_gaussian_integral(np.arange(len(idx)), c[idx])
            if coef is None:
                mask[j] = False # mask out bad fits
            else: # Only update the coefficients and peaks if fit_gaussian did not return None
                gauss_coeffs[:, j] = coef
                fitted_peaks[j] = coef[1] + p - width
                lines_dict[j] = line_dict
                
        # Remove the peaks where fit_gaussian returned None
        fitted_peaks = fitted_peaks[mask]
        detected_peaks = detected_peaks[mask]
        peak_heights = peak_heights[mask]
        gauss_coeffs = gauss_coeffs[:, mask]
                
        return fitted_peaks, detected_peaks, peak_heights, gauss_coeffs, lines_dict
        
    def clip_peaks(
        self, order_flux, fitted_peak_pixels, detected_peak_pixels, gauss_coeffs, 
        detected_peak_heights, clip_below_median=True,
        print_update=True, plot_path=None #print_update should be false TESTING TODO
    ):
        """
        Clips peaks that have detected and Gaussian-fitted central pixels values 
        more than 1 pixel apart. Also clip peaks that are immediately next to a
        flux value of 0 (this prevents peak detection near masked areas). There
        is also an option to clip detected peaks that are less than the median of 
        the overall chip flux.
        Args:
            order_flux (np.array): array of order_flux data
            fitted_peak_pixels (np.array): array of true peak locations as 
                determined by Gaussian fitting
            detected_peak_pixels (np.array of float): array of detected peak locations 
                (pre-Gaussian fitting)
            gauss_coeffs (np.array): array of size (4, n_peaks) containing best-fit 
                Gaussian parameters [a, mu, sigma**2, const] for each detected peak            
            detected_peak_heights (np.array): array of detected peak heights (pre-Gaussian fitting)
            clip_below_median (bool): if True, clip all peaks below the overall
                chip median.
            print_update (bool): if True, print how many peaks were clipped
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.
        Returns: 
            np.array: indices of surviving peaks
        """

        n_pixels = len(order_flux)

        # clip peaks that have Gaussian-fitted centers more than 1 pixel from
        # their detected centers & have detected heights below the chip median value
        if clip_below_median:
            good_peak_idx = np.where(
                (np.abs(fitted_peak_pixels - detected_peak_pixels) < 1) &
                (detected_peak_heights > np.median(order_flux[np.nonzero(order_flux)]))
            ) [0]
        else:
            good_peak_idx = np.where(
                (np.abs(fitted_peak_pixels - detected_peak_pixels) < 1)
            ) [0]

        # clip peaks that are a factor of 3 greater than the 8 adjacent peaks on both sides
        new_good_peak_idx = []
        for i in range(len(good_peak_idx)):
            peak_idx = good_peak_idx[i]
            if peak_idx >= 8 and peak_idx < n_pixels - 8:
                peak_flux = detected_peak_heights[peak_idx]
                adjacent_fluxes = np.concatenate((detected_peak_heights[peak_idx-9:peak_idx-1], detected_peak_heights[peak_idx+1:peak_idx+9]))
                max_adjacent_flux = np.max(adjacent_fluxes)
                if peak_flux <= 3 * max_adjacent_flux:
                    new_good_peak_idx.append(peak_idx)
            elif peak_idx < 8:
                # Handle peaks near the beginning
                peak_flux = detected_peak_heights[peak_idx]
                adjacent_fluxes = detected_peak_heights[peak_idx+1:peak_idx+9]
                max_adjacent_flux = np.max(adjacent_fluxes)
                if peak_flux <= 3 * max_adjacent_flux:
                    new_good_peak_idx.append(peak_idx)
            else:
                # Handle peaks near the end
                peak_flux = detected_peak_heights[peak_idx]
                adjacent_fluxes = detected_peak_heights[peak_idx-9:peak_idx-1]
                max_adjacent_flux = np.max(adjacent_fluxes)
                if peak_flux <= 3 * max_adjacent_flux:
                    new_good_peak_idx.append(peak_idx)

        good_peak_idx = np.array(new_good_peak_idx)

        # clip peaks with heights less than a third of previous peak
        final_good_peak_idx = []
        prev_peak_height = 0
        for i in range(len(good_peak_idx)):
            if detected_peak_heights[good_peak_idx[i]] >= (prev_peak_height / 3):
                final_good_peak_idx.append(good_peak_idx[i])
                prev_peak_height = detected_peak_heights[good_peak_idx[i]]
                missed_peak_count = 0
            else:
                missed_peak_count += 1
                if missed_peak_count == 5:
                    print('Warning: Check for outliers. 5 peaks in a row with heights < 1/3 of previous peak.')

        good_peak_idx = np.array(final_good_peak_idx)

        # # if we know the wavelengths of the peaks (i.e. if dealing with LFC),
        # # then we can clip peaks with derived wavelengths far from the location
        # # of a comb line
        # if comb_lines_angstrom is not None:

        #     # compute an approx wavelength solution that we'll use to find
        #     # the nearest LFC mode
        #     n_pixels = len(rough_wls_order)
        #     s = InterpolatedUnivariateSpline(np.arange(n_pixels), rough_wls_order)
        #     approx_peaks_lambda = s(fitted_peak_pixels)

            # # iterate through all modes and save only those that are less than ~1 pixel from an 
            # # LFC mode
            # peaks_nearby_lfcmodes = []
            # for i, lamb in enumerate(approx_peaks_lambda):

            #     # delta lambda between adjacent pixels, as measured by rough wls
            #     approx_pixel_size = (approx_peaks_lambda[i] - s(fitted_peak_pixels[i] - 1))

            #     best_mode_idx = (
            #         np.abs(comb_lines_angstrom - lamb)
            #     ).argmin()
            #     if np.abs(comb_lines_angstrom[best_mode_idx] - lamb) < approx_pixel_size:
            #         peaks_nearby_lfcmodes.append(i)
            
            # good_peak_idx = np.intersect1d(peaks_nearby_lfcmodes, good_peak_idx)

        # clip peaks that are immediately next to zero pixels (indicating 
        # they're next to a masked section, eg, and therefore unreliable
        notnearmask_peaks = []
        for i, lamb in enumerate(detected_peak_pixels):
            if order_flux[int(lamb) + 1] != 0 and order_flux[int(lamb) - 1] != 0:
                notnearmask_peaks.append(i)
        
        good_peak_idx = np.intersect1d(notnearmask_peaks, good_peak_idx)
        
        # TODO: remove bad peaks from the line dictionary

        if print_update:
            print('{} peaks fit'.format(len(detected_peak_pixels)))
            print('{} peaks clipped'.format(len(detected_peak_pixels) - len(good_peak_idx)))

        if plot_path is not None:
            '''
            n = np.arange(len(fitted_peak_pixels))

            plt.figure()
            plt.scatter(
                n[good_peak_idx], 
                gauss_coeffs[0,:][good_peak_idx] - fitted_peak_pixels[good_peak_idx],
                color='k'
            )
            plt.savefig(
                '{}/peak_heights_after_clipping.png'.format(plot_path), dpi=250
            )
            plt.close()

            plt.figure()
            plt.scatter(
                n[good_peak_idx], 
                fitted_peak_pixels[good_peak_idx] - detected_peak_pixels[good_peak_idx],
                color='k'
            )
            plt.savefig(
                '{}/peak_locs_after_clipping.png'.format(plot_path), dpi=250
            )
            plt.close()

            plt.figure()
            plt.plot(order_flux, color='k', lw=0.1)   
            plt.scatter(
                detected_peak_pixels[good_peak_idx], detected_peak_heights[good_peak_idx], s=1, color='r'
            )
            plt.scatter(
                np.delete(detected_peak_pixels, good_peak_idx), 
                np.delete(detected_peak_heights, good_peak_idx), s=10, color='k'
            )
            plt.savefig('{}/unclipped_peaks.png'.format(plot_path), dpi=250)
            plt.close()
            '''
            n_zoom_sections = 10
            zoom_section_pixels = n_pixels // n_zoom_sections

            _, ax_list = plt.subplots(n_zoom_sections, 1, figsize=(6, 12))
            for i, ax in enumerate(ax_list):
                ax.plot(order_flux, color='k', lw=0.1)   
                ax.scatter(
                    detected_peak_pixels[good_peak_idx], detected_peak_heights[good_peak_idx], 
                    s=1, color='r'
                )
                ax.scatter(
                    np.delete(detected_peak_pixels, good_peak_idx), 
                    np.delete(detected_peak_heights, good_peak_idx), s=10, color='k'
                )
                ax.set_xlim(
                    zoom_section_pixels * i, zoom_section_pixels * (i + 1)
                )
                ax.set_ylim(
                    0, 
                    np.max(
                        order_flux[zoom_section_pixels * i : zoom_section_pixels * (i + 1)]
                    )
                )

                ax.set_xlabel('Pixel')
                ax.set_ylabel('Counts')

            plt.tight_layout()
            plt.savefig('{}/unclipped_peaks_zoom.png'.format(plot_path), dpi=250)
            plt.close()

        return good_peak_idx
    
    def line_match(self, flux, linelist, line_pixels_expected, plot_toggle, savefig, gaussian_fit_width=10):
        """
        Given a linelist of known wavelengths of peaks and expected pixel locations
        (from a previous wavelength solution), returns precise, updated pixel locations 
        for each known peak wavelength.
        Args:
            flux (np.array): flux of order
            linelist (np.array of float): wavelengths of lines to be fit (Angstroms)
            line_pixels_expected (np.array of float): expected pixels for each wavelength
                (Angstroms); must be same length as `linelist`
            plot_toggle (bool): if True, make and save plots.
            savefig (str): path to directory where plots will be saved
            gaussian_fit_width (int): pixel +/- range to use for Gaussian fitting
        Retuns:
            tuple of:
                np.array: same input linelist, with unfit lines removed
                np.array: array of size (4, n_peaks) containing best-fit 
                    Gaussian parameters [a, mu, sigma**2, const] for each detected peak
                dictionary: a dictionary of information about the lines fit within this order 
        """        
        if self.cal_type == 'ThAr':
            gaussian_fit_width = 5
        num_input_lines = len(linelist)  
        num_pixels = len(flux)
        successful_fits = []
        lines_dict = {}

        missed_lines = 0
        coefs = np.zeros((4,num_input_lines))
        for i in np.arange(num_input_lines):
            line_location = line_pixels_expected[i]
            peak_pixel = np.floor(line_location).astype(int)
            # don't fit saturated lines
            if peak_pixel < len(flux) and flux[peak_pixel] <= 1e6:
                if peak_pixel < gaussian_fit_width:
                    first_fit_pixel = 0
                else:
                    first_fit_pixel = peak_pixel - gaussian_fit_width
                
                if peak_pixel + gaussian_fit_width > num_pixels:
                    last_fit_pixel = num_pixels
                else:
                    last_fit_pixel = peak_pixel + gaussian_fit_width

                # fit gaussian to matched peak location
                result, line_dict = self.fit_gaussian_integral(
                    np.arange(first_fit_pixel,last_fit_pixel),
                    flux[first_fit_pixel:last_fit_pixel]
                )

                #add_to_line_dict = False
                if result is not None:
                    coefs[:, i] = result
                    successful_fits.append(i)  # Append index of successful fit
                    line_dict['lambda_fit'] = linelist[i]
                    lines_dict[str(i)] = line_dict  # Add line dictionary to lines dictionary
                else:
                    missed_lines += 1

                amp = coefs[0,i]
                if amp < 0:
                    missed_lines += 1
                    coefs[:,i] = np.nan

            else:
                coefs[:,i] = np.nan
                missed_lines += 1

        linelist = linelist[successful_fits]
        coefs = coefs[:, successful_fits]
        linelist = linelist[np.isfinite(coefs[0,:])]
        coefs = coefs[:, np.isfinite(coefs[0,:])]
        
        print('{}/{} lines not fit.'.format(missed_lines, num_input_lines))
        if plot_toggle:

            n_zoom_sections = 10
            zoom_section_pixels = num_pixels // n_zoom_sections

            zoom_section_pixels = (num_pixels // n_zoom_sections)
            _, ax_list = plt.subplots(n_zoom_sections,1,figsize=(6, 20))
            ax_list[0].set_title('({} missed lines)'.format(missed_lines))
            for i, ax in enumerate(ax_list):

                # plot the flux
                ax.plot(
                    np.arange(num_pixels)[i*zoom_section_pixels:(i+1)*zoom_section_pixels],
                    flux[i*zoom_section_pixels:(i+1)*zoom_section_pixels],color='k'
                )

                # #  plot the fitted peak maxima as points
                # ax.scatter(
                #     coefs[1,:][
                #         (coefs[1,:] > i * zoom_section_pixels) & 
                #         (coefs[1,:] < (i+1) * zoom_section_pixels)
                #     ], 
                #     coefs[0,:][
                #         (coefs[1,:] > i * zoom_section_pixels) & 
                #         (coefs[1,:] < (i+1) * zoom_section_pixels)
                #     ] + 
                #     coefs[3,:][
                #         (coefs[1,:] > i * zoom_section_pixels) & 
                #         (coefs[1,:] < (i+1) * zoom_section_pixels)
                #     ],
                #     color='red'
                # )

                # overplot the Gaussian fits
                for j in np.arange(num_input_lines-missed_lines):

                    # if peak in range:
                    if (
                        (coefs[1,j] > i * zoom_section_pixels) & 
                        (coefs[1,j] < (i+1) * zoom_section_pixels)
                    ):

                        xs = np.floor(coefs[1,j]) - gaussian_fit_width + \
                            np.linspace(
                                0, 
                                2 * gaussian_fit_width, 
                                2 * gaussian_fit_width
                            )
                        gaussian_fit = self.integrate_gaussian(
                            xs, coefs[0,j], coefs[1,j], coefs[2,j], coefs[3,j]
                        )

                        ax.plot(xs, gaussian_fit, alpha=0.5, color='red')

            plt.tight_layout()
            plt.savefig('{}/spectrum_and_gaussian_fits.png'.format(savefig), dpi=250)
            plt.close()

        return linelist, coefs, lines_dict

    def mode_match(
        self, order_flux, fitted_peak_pixels, good_peak_idx, rough_wls_order, 
        comb_lines_angstrom, print_update=False, plot_path=None, start_check=True,
    ):
        """
        Matches detected order_flux peaks to the theoretical locations of LFC wavelengths
        and returns the derived wavelength solution.
        Given detected peak locations in data, a preexisting coarse wavelength solution
        (e.g. from ThAr), and theoretical line wavelengths from physics, returns 
        precise wavelengths for peaks.
        Args:
            order_flux (np.array of float): flux values for an order. Their indices 
                correspond to their pixel numbers. 
            fitted_peak_pixels (np.array): array of true peak locations as 
                determined by Gaussian fitting.
            good_peak_idx (np.array): indices (of ``new_peaks``) of detected 
                and unclipped peaks
            rough_wls_order (np.array): a rough (generally ThAr-based) wavelength 
                solution. Each entry in the array is the wavelength (in Angstroms) 
                corresponding to a pixel (indicated by its index)
            comb_lines_angstrom (np.array): theoretical LFC wavelengths
                as computed by fundamental physics (in Angstroms)
            print_update (bool): if True, print total number of LFC modes in
                the order that were not detected (n_clipped + n_never_detected)
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.
            
        Returns:
            tuple of:
                np.array: the precise wavelengths of detected `order_flux` peaks. Each
                    entry in the array is the wavelength (in Angstroms) corresponding
                    to a pixel (indicated by its index)
                np.array: the mode numbers of the LFC modes to be used for 
                    wavelength calibration
        """
        
        # Calculate the peak differences
        peak_diffs = np.diff(fitted_peak_pixels[good_peak_idx])
        peaks_to_keep = [good_peak_idx[0]]  # Always keep the first peak

        # Iterate over the peak differences, starting from the second peak
        for i in range(1, len(good_peak_idx) - 1):
            if i < 8:
                nearest_peak_diffs = peak_diffs[i+1:i+9]
            elif i > len(peak_diffs) - 8:
                nearest_peak_diffs = peak_diffs[i-9:i-1]
            else:
                nearest_peak_diffs_before = peak_diffs[i-7:i-3]
                nearest_peak_diffs_after = peak_diffs[i+3:i+7]
                nearest_peak_diffs = np.concatenate((nearest_peak_diffs_before, nearest_peak_diffs_after))
    
            # Find the minimum of the nearest peak differences
            min_nearest_peak_diff = np.min(nearest_peak_diffs)           
            # If the current peak difference is not less than 0.9 times the minimum, keep it
            if peak_diffs[i - 1] >= 0.9 * min_nearest_peak_diff:
                peaks_to_keep.append(good_peak_idx[i])
        
        # Always keep the last peak
        peaks_to_keep.append(good_peak_idx[-1])
        good_peak_idx = np.array(peaks_to_keep)
        
        n_pixels = len(order_flux)
        s = InterpolatedUnivariateSpline(np.arange(n_pixels)[rough_wls_order>0], rough_wls_order[rough_wls_order>0])
        approx_peaks_lambda = s(fitted_peak_pixels[good_peak_idx])

        # approx_peaks_lambda = np.interp(
        #     new_peaks[good_peak_idx], np.arange(n_pixels), rough_wls_order)

        # Now figure what mode numbers the peaks correspond to
        n_clipped_peaks = len(fitted_peak_pixels[good_peak_idx])
        mode_nums = np.empty(n_clipped_peaks)

        def increment_mode_num(mode_num, backwards=True):
            if backwards:
                return mode_num - 1
            else:
                 return mode_num + 1

        if len(np.where((rough_wls_order[:-1] <= rough_wls_order[1:])  == 1)[0]) > 10:
            backwards=False
        else:
            backwards=True

        peak_mode_num = 0
        
	    # Find peak spacing (peak_diff) for all adjacent peaks, remove outliers with median filter
        peak_diff = fitted_peak_pixels[good_peak_idx][1:] - fitted_peak_pixels[good_peak_idx][:-1]

        # Calculate the difference between the peak indices
        peak_indices_difference = good_peak_idx[3:] - good_peak_idx[:-3]

        # Check for large gaps in peak indices
        large_gaps_detected = np.any(peak_indices_difference > 9)
        
        # Adjust kernel size if large gaps are detected
        kernel_size = 7
        if large_gaps_detected:
            kernel_size += 10  # Increase by 10. Adjust as needed.

        recursive_peak_diff = peak_diff.copy()
    
        for iteration in range(5):
            filtered = signal.medfilt(recursive_peak_diff, kernel_size=kernel_size)
    
            if np.array_equal(filtered, recursive_peak_diff):
                break  # Stop if there's no change after filtering
            recursive_peak_diff = filtered
    
        else: 
            print('Medfilt iterations > 5')

        # Identify and remove outlier peak spacings not removed by recursive median filter
        # This process primarily removes peak spacing aliases (2x, 3x, etc) 
        # Removes outlier peak spacings discrepant by > 50% relative to adjacent peak spacings
        spline_peak_pixels = fitted_peak_pixels[good_peak_idx][1:]
        spline_peak_diff = recursive_peak_diff # getting peak spacing for SPLINE fit
        spline_peak_diff_min = np.minimum(spline_peak_diff[1:], spline_peak_diff[:-1]) # lesser of each adjacent value pair in spline_peak_diff
        spline_peak_diff_diff = spline_peak_diff[1:] - spline_peak_diff[:-1] # diff in peak spacings used to identify peak spacing drift
        spline_peak_diff_bool0 = spline_peak_diff_diff/spline_peak_diff_min >  0.5 # bool mask to identify half of bad peak spacings
        spline_peak_diff_bool1 = spline_peak_diff_diff/spline_peak_diff_min < -0.5 # bool mask to identify half of bad peak spacings
        spline_peak_diff_bool0 = np.insert(spline_peak_diff_bool0, 0, False) # bool padding to select correct peak
        spline_peak_diff_bool1 = np.append(False, spline_peak_diff_bool1) # bool padding to select correct peak
        index = np.where((spline_peak_diff_bool0 | spline_peak_diff_bool1) == False)[0] # combine bool arrays to remove bad peaks on LHS and RHS of peak spacing
        spline_peak_diff_new = spline_peak_diff[index]
        
        # Now, recurse the above process
        counter_spline = 1 
        while len(spline_peak_diff_new) != len(spline_peak_diff):
            spline_peak_diff = spline_peak_diff_new
            spline_peak_diff_min = np.minimum(spline_peak_diff[1:], spline_peak_diff[:-1])
            spline_peak_diff_diff = spline_peak_diff[1:] - spline_peak_diff[:-1]
            spline_peak_diff_bool0 = spline_peak_diff_diff/spline_peak_diff_min >  0.5 # bool mask to identify half of bad peak spacings
            spline_peak_diff_bool1 = spline_peak_diff_diff/spline_peak_diff_min < -0.5 # bool mask to identify half of bad peak spacings
            spline_peak_diff_bool0 = np.insert(spline_peak_diff_bool0, 0, False) # bool padding to select correct peak
            spline_peak_diff_bool1 = np.append(False, spline_peak_diff_bool1) # bool padding to select correct peak
            index = np.where((spline_peak_diff_bool0 | spline_peak_diff_bool1) == False)[0]
            spline_peak_diff_new = spline_peak_diff[index]

            counter_spline += 1     
            if counter_spline == 15:
                print('Warning: Outlier Removal Iterations = 15')
                break

        # SPLINE fit on peak spacings to create function that estimates peak spacing as a function of pixel
        peak_diff_spline = UnivariateSpline(fitted_peak_pixels[good_peak_idx][1:][index], spline_peak_diff_new, k = 2)
        
        if plot_path is not None:
            plt.figure(tight_layout=True)
            plt.plot(fitted_peak_pixels[good_peak_idx][:-1], peak_diff, 
                     'ko', alpha = 0.5, label = 'Peak Difference', markersize = 8)
            plt.plot(fitted_peak_pixels[good_peak_idx][:-1][index], spline_peak_diff_new, 
                     'bo', alpha = 0.3, label = 'Filtered Peak Difference', markersize = 6)
            plt.plot(np.arange(n_pixels), peak_diff_spline(np.arange(n_pixels)), 
                     'r-', label = 'SPLINE Fit', lw =2)
            plt.xlabel('Pixel Location', fontsize=14)
            plt.ylabel('Peak Spacing (to subsequent peak)', fontsize=14)
            plt.tick_params(axis='both', direction='inout', length=6, width=3, colors='k', labelsize=12)
            plt.legend()
            plt.savefig('{}/peak_diff.png'.format(plot_path), dpi=250)
            plt.close()
                    
        for i in range(len(good_peak_idx)):
            # estimate local peak diff from SPLINE fit function
            running_peak_diff = peak_diff_spline(fitted_peak_pixels[good_peak_idx][i])

            if i==0:
                for j in np.arange(50):
                    if fitted_peak_pixels[good_peak_idx][i] > (j + 1.5) * running_peak_diff:
                        peak_mode_num = increment_mode_num(peak_mode_num, backwards=backwards)
                # if fitted_peak_pixels[good_peak_idx][i] > 50.5 * running_peak_diff:                        
                #     assert False, 'More than 50 peaks in a row at the start of the chip not detected!'
        
            # if current peak location is greater than (n + 0.5) * sigma of 
            # previous peak diffs, then skip over n modes
            if i > 0:
                for j in np.arange(8):
                    if (
                        fitted_peak_pixels[good_peak_idx][i] - 
                        fitted_peak_pixels[good_peak_idx][i - 1] > 
                        (j + 1.5) * running_peak_diff
                    ):
                        peak_mode_num = increment_mode_num(peak_mode_num, backwards=backwards)
                if (
                    fitted_peak_pixels[good_peak_idx][i] - 
                    fitted_peak_pixels[good_peak_idx][i - 1] > 
                    8.5 * running_peak_diff
                ):
                    print('Warning: more than 8 peaks in a row not detected!')

            # set mode_nums
            mode_nums[i] = peak_mode_num
            peak_mode_num = increment_mode_num(peak_mode_num, backwards=backwards)

        idx = (np.abs(comb_lines_angstrom - 
            approx_peaks_lambda[len(approx_peaks_lambda) // 2])).argmin()

        n_skipped_modes_in_chip_first_half = mode_nums[
            (len(approx_peaks_lambda) // 2)] - (len(approx_peaks_lambda) // 2)
        mode_nums += (idx - (len(approx_peaks_lambda) // 2) - 
            n_skipped_modes_in_chip_first_half)
        
        if plot_path is not None:
            plt.figure(tight_layout=True)
            plt.plot(rough_wls_order, order_flux, alpha=0.2, label='Flux')
            plt.vlines(comb_lines_angstrom, ymin=0, ymax=5000, color='r', label='Comb Lines')
            plt.xlim(np.nanmin(rough_wls_order), np.nanmin(rough_wls_order) + 6)
            plt.yscale('symlog')
            plt.xlabel('Wavelength [$\\rm \AA$]', fontsize=14)
            plt.ylabel('Flux', fontsize=14)
            plt.title('Rough Solution and LFC Lines', fontsize=18)
            plt.savefig('{}/rough_sol_and_lfc_lines.png'.format(plot_path), dpi=250)
            plt.legend()
            plt.close()

            n_zoom_sections = 10
            zoom_section_wavelen = (
                (np.nanmax(rough_wls_order) - np.nanmin(rough_wls_order)) // 
                n_zoom_sections
            )
            zoom_section_pixels = n_pixels // n_zoom_sections

            _, ax_list = plt.subplots(n_zoom_sections, 1, figsize=(12, 10))
            for i, ax in enumerate(ax_list):
                ax.plot(rough_wls_order, order_flux, color='k', alpha=0.1)
                for mode_num in mode_nums:
                    if (
                        (
                            comb_lines_angstrom[mode_num.astype(int)] > 
                            zoom_section_wavelen * i + np.nanmin(rough_wls_order)
                        ) and (
                            comb_lines_angstrom[mode_num.astype(int)] < 
                            zoom_section_wavelen * (i + 1) + np.nanmin(rough_wls_order)
                        )
                    ):
                        ax.text(
                            comb_lines_angstrom[mode_num.astype(int)], 0, 
                            str(int(mode_num)), fontsize=4
                        )
                ax.set_xlim(
                    zoom_section_wavelen * i + np.nanmin(rough_wls_order), 
                    zoom_section_wavelen * (i + 1) + np.nanmin(rough_wls_order)
                )
                ax.set_ylim(
                    0, 
                    np.max(
                        order_flux[zoom_section_pixels * i: zoom_section_pixels * (i + 1)]
                    )
                )
                ax.set_yticks([])
                ax.set_ylabel('Flux')
                if i == n_zoom_sections-1:
                    ax.set_xlabel(r'Wavelength ($\AA$)')
            plt.tight_layout()
            plt.savefig('{}/labeled_line_locs.png'.format(plot_path), dpi=250)
            plt.close()
        wls = comb_lines_angstrom[mode_nums.astype(int)]
        return wls, mode_nums, peaks_to_keep
    
    def fit_gaussian(self,x,y):
        """
        Fits a continous Gaussian in wavelength space for an input flux

        Args:
            x (np.array): wavelength segment to fit
            y (np.array): Flux data to be fit
        Returns:
            Height of Gaussian
            Center of Gaussian
            Width  of Gaussian
        """
        x = np.ma.compressed(x)
        y = np.ma.compressed(y)

        i = np.argmax(y)# or use previous peak position
        p0 = [y[i], x[i], 0.015*3, np.min(y)] #0.015 Ang/pix. Args are heigh, center,width, zero-pt
        #print("Initial guess:",p0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:   
                popt, _ = curve_fit(self.calculate_gaussian, x, y, p0=p0, maxfev=1000000)
            except RuntimeError:
                print("Runtime Error")
                return p0
        return popt

    def calculate_gaussian(self,x, y, height, center, width):
        """
        Fits a continous Gaussian in wavelength space for

        Args:
            x (np.array): wavelength segment to fit
            y (np.array): Flux data to be fit
        Returns:
            Height of Gaussian
            Center of Gaussian
            Width  of Gaussian
        """
        x = np.ma.compressed(x)
        y = np.ma.compressed(y)

        #i = np.argmax(y) # index of maximum flux
        #p0 = [y[i],x[i],0.015*3] # ~0.015 Ang/pix
        output = height * np.exp(-(x - center)**2 / (2 * width**2))
        return output

    def integrate_gaussian(self, x, a, mu, sig, const, int_width=0.5):
        """
        Returns the integral of a Gaussian over a specified symmetric range. 
        Gaussian given by:
        g(x) = a * exp(-(x - mu)**2 / (2 * sig**2)) + const
        Args:
            x (float): the central value over which the integral will be calculated
            a (float): the amplitude of the Gaussian
            mu (float): the mean of the Gaussian
            sig (float): the standard deviation of the Gaussian
            const (float): the Gaussian's offset from zero (i.e. the value of
                the Gaussian at infinity).
            int_width (float): the width of the range over which the integral will 
                be calculated (i.e. if I want to calculate from 0.5 to 1, I'd set
                x = 0.75 and int_width = 0.25).
        Returns:
            float: the integrated value
        """

        integrated_gaussian_val = a * 0.5 * (
            erf((x - mu + int_width) / (np.sqrt(2) * sig)) - 
            erf((x - mu - int_width) / (np.sqrt(2) * sig))
            ) + (const * 2 * int_width)
        
        return integrated_gaussian_val
    
    def fit_gaussian_integral(self, x, y):
        """
        Fits a continuous Gaussian to a discrete set of x and y datapoints
        using scipy.curve_fit
        
        Args:
            x (np.array): x data to be fit
            y (np.array): y data to be fit
        Returns a tuple of:
            list: best-fit parameters [a, mu, sigma**2, const]
            line_dict: dictionary of best-fit parameters, wav, flux, model, etc.
        """
        
        line_dict = {} # initialize dictionary to store fit parameters, etc.

        x = np.ma.compressed(x)
        y = np.ma.compressed(y)
        i = np.argmax(y[len(y) // 4 : len(y) * 3 // 4]) + len(y) // 4
        
        p0 = [y[i], x[i], 1.5, np.min(y)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(self.integrate_gaussian, x, y, p0=p0, maxfev=1000000)
            pcov[np.isinf(pcov)] = 0 # convert inf to zero
            pcov[np.isnan(pcov)] = 0 # convert nan to zero
            line_dict['amp']   = popt[0] # optimized parameters
            line_dict['mu']    = popt[1] # "
            line_dict['sig']   = popt[2] # "
            line_dict['const'] = popt[3] # ""
            line_dict['covar'] = pcov    # covariance
            line_dict['data']  = y
            line_dict['model'] = self.integrate_gaussian(x, *popt)
            line_dict['quality'] = 'good' # fits are assumed good until marked bad elsewhere
            

        if self.cal_type == 'ThAr':          
            # Quality Checks for Gaussian Fits
            
            if max(y) == 0:
                print('Amplitude is 0')
                return(None, line_dict)
            
            chi_squared_threshold = int(self.chi_2_threshold)

            # Calculate chi^2
            predicted_y = self.integrate_gaussian(x, *popt)
            chi_squared = np.sum(((y - predicted_y) ** 2) / np.var(y))
            line_dict['chi2'] = chi_squared

            # Calculate RMS of residuals for Gaussian fit
            rms_residual = np.sqrt(np.mean(np.square(y - predicted_y)))
            line_dict['rms'] = np.sqrt(np.mean(np.square(rms_residual)))

            #rms_threshold = 1000 # RMS Quality threshold
            #disagreement_threshold = 1000 # Disagreement with initial guess threshold
            #asymmetry_threshold = 1000 # Asymmetry in residuals threshold

            # Calculate disagreement between Gaussian fit and initial guess
            #disagreement = np.abs(popt[1] - p0[1])
            line_dict['mu_diff'] = popt[1] - p0[1] # disagreement between Gaussian fit and initial guess

            ## Check for asymmetry in residuals
            #residuals = y - predicted_y
            #left_residuals = residuals[:len(residuals)//2]
            #right_residuals = residuals[len(residuals)//2:]
            #asymmetry = np.abs(np.mean(left_residuals) - np.mean(right_residuals))
            
            # Run checks against defined quality thresholds
            if (chi_squared > chi_squared_threshold):
                print("Chi squared exceeded the threshold for this line. Line skipped")
                return None, line_dict

            # Check if the Gaussian amplitude is positive, the peak is higher than the wings, or the peak is too high
            if popt[0] <= 0 or popt[0] <= popt[3] or popt[0] >= 500*max(y):
                line_dict['quality'] = 'bad_amplitude'  # Mark the fit as bad due to bad amplitude or U shaped gaussian
                print('Bad amplitude detected')
                return None, line_dict

        return (popt, line_dict)
    
    def fit_polynomial(self, wls, rough_wls_order, peak_wavelengths_ang, order_list, n_pixels, fitted_peak_pixels, fit_iterations=5, sigma_clip=2.1, peak_heights=None, plot_path=None):
        """
        Given precise wavelengths of detected LFC order_flux lines, fits a 
        polynomial wavelength solution.
        Args:
            wls (np.array): the known, precise wavelengths of the detected peaks,
                either from fundamental physics or a previous wavelength solution.
            n_pixels (int): number of pixels in the order
            fitted_peak_pixels (np.array): array of true detected peak locations as 
                determined by Gaussian fitting.
            fit_iterations (int): number of sigma-clipping iterations in the polynomial fit
            sigma_clip (float): clip outliers in fit with residuals greater than sigma_clip away from fit
            peak_heights (np.array): heights of peaks (either detected heights or 
                fitted heights). We use this to weight the peaks in the polynomial 
                fit, assuming Poisson errors. 
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.
        Returns:
            tuple of:
                np.array: calculated wavelength solution for the order (i.e. 
                    wavelength value for each pixel in the order)
                func: a Python function that, given an array of pixel locations, 
                    returns the Legendre polynomial wavelength solutions
        """
        weights = 1 / np.sqrt(peak_heights)
        if self.fit_type.lower() not in ['legendre', 'spline']:
            raise NotImplementedError("Fit type must be either legendre or spline")
        
        if self.fit_type.lower() == 'legendre' or self.fit_type.lower() == 'spline': 

            _, unique_idx, count = np.unique(fitted_peak_pixels, return_index=True, return_counts=True)
            unclipped_idx = np.where(
                (fitted_peak_pixels > 0)
            )[0]
            unclipped_idx = np.intersect1d(unclipped_idx, unique_idx[count < 2])
            
            sorted_idx = np.argsort(fitted_peak_pixels[unclipped_idx])
            x, y, w = fitted_peak_pixels[unclipped_idx][sorted_idx], wls[unclipped_idx][sorted_idx], weights[unclipped_idx][sorted_idx]           

            for i in range(fit_iterations):
                if self.fit_type.lower() == 'legendre':
                    if self.cal_type == 'ThAr':
                        # fit ThAr based on 4/30 WLS
                        rough_wls_int = interp1d(np.arange(n_pixels), rough_wls_order, kind='linear', fill_value="extrapolate")
                        
                        def polynomial_func(x, c0, c1, c2):
                            """
                            Polynomial function to fit.
                            Args:
                                x (np.array): Pixel values.
                                c0, c1, c2 (float): Coefficients of the polynomial.
                            Returns:
                                np.array: Evaluated polynomial.
                            """
                            return rough_wls_int(x) + c0 + c1 * x + c2 * x**2
                        
                        # Using curve_fit to find the best-fit values of {c0, c1}
                        popt, _ = curve_fit(polynomial_func, x, y)

                        # Create the wavelength solution for the order
                        our_wavelength_solution_for_order = polynomial_func(np.arange(len(rough_wls_order)), *popt)
                        leg_out = Legendre.fit(np.arange(n_pixels), our_wavelength_solution_for_order, 9)
                    
                    if self.cal_type == 'LFC':
                        leg_out = Legendre.fit(x, y, self.fit_order, w=w)
                        our_wavelength_solution_for_order = leg_out(np.arange(n_pixels))
                if self.fit_type == 'spline':
                    leg_out = UnivariateSpline(x, y, w, k=5)
                    our_wavelength_solution_for_order = leg_out(np.arange(n_pixels))
                
                res = y - leg_out(x)
                good = np.where(np.abs(res) <= sigma_clip*np.std(res))
                x = x[good]
                y = y[good]
                w = w[good]
                res = res[good]
            
            plt.plot(x, res, 'k.')
            plt.axhline(0, color='b', lw=2)
            plt.xlabel('Pixel')
            plt.ylabel('Fit residuals [$\AA$]')
            plt.tight_layout()
            #plt.savefig('{}/polyfit.png'.format(plot_path))
            plt.close()
            
            if plot_path is not None and self.cal_type =='ThAr':
                approx_dispersion = (our_wavelength_solution_for_order[2000] - our_wavelength_solution_for_order[2100])/100
                fig, ax1 = plt.subplots(tight_layout=True, figsize=(8, 4))
                
                # Range of interest b/c CCF chops off first/last 500 pixels
                pixel_range = np.arange(500, 3500)
                rough_wls_int_range = rough_wls_int(pixel_range)
                wavelength_solution_range = our_wavelength_solution_for_order[500:3500]

                # Create the plot
                fig, ax1 = plt.subplots(tight_layout=True, figsize=(8, 4))
                ax1.plot(
                    pixel_range, 
                    rough_wls_int_range - wavelength_solution_range, 
                    color='k'
                )
                ax1.set_xlabel('Pixel')
                ax1.set_ylabel(r'Wavelength Difference ($\AA$)')
                ax2 = ax1.twinx()
                ax2.set_ylabel("Difference (pixels) \nusing dispersion " + r'$\approx$' + '{0:.2}'.format(approx_dispersion) + r' $\AA$/pixel')
                ax2.set_ylim(ax1.get_ylim())
                ax1_ticks = ax1.get_yticks()
                ax2.set_yticklabels([str(round(tick / approx_dispersion, 2)) for tick in ax1_ticks])
                plt.savefig('{}/interp_vs_our_wls.png'.format(plot_path))
                plt.close()
        else:
            raise ValueError('Only set up to perform Legendre fits currently! Please set fit_type to "Legendre"')

        return our_wavelength_solution_for_order, leg_out

    def calculate_rv_precision(
        self, fitted_peak_pixels, wls, leg_out, rough_wls, our_wavelength_solution_for_order, rough_wls_order, 
        print_update=True, plot_path=None
    ):
        """
        Calculates 1) RV precision from the difference between the known (from 
        physics) wavelengths of pixels containing peak flux values and the 
        fitted wavelengths of the same pixels, generated using a polynomial 
        wavelength solution ("absolute RV precision") and 2) RV precision from
        the difference between the "master" wavelength solution and our 
        fitted wavelength solution ("relative RV precision")
        Args:
            fitted_peak_pixels (np.array of float): array of true detected peak locations as 
                determined by Gaussian fitting (already clipped)
            wls (np.array of float): precise wavelengths of `fitted_peak_pixels`,
                from fundamental physics or another wavelength solution.
            leg_out (func): a Python function that, given an array of pixel 
                locations, returns the Legendre polynomial wavelength solutions
            rough_wls (np.array of float): rough wavelength values for each
                pixel in the order [Angstroms]
            print_update (bool): If true, prints standard error per order.
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.
        Returns:
            tuple of:
                float: absolute RV precision in cm/s
                float: relative RV precision in cm/s
        """
        our_wls_peak_pos = leg_out(fitted_peak_pixels) 
        # absolute/polynomial precision of order = difference between fundemental wavelengths
        # and our wavelength solution wavelengths for (fractional) peak pixels
        abs_residual = ((our_wls_peak_pos - wls) * scipy.constants.c) / wls
        abs_precision_cm_s = 100 * np.nanstd(abs_residual)/np.sqrt(len(fitted_peak_pixels))
        # the above line should use RMS not STD

        # relative RV precision of order = difference between rough wls wavelengths
        # and our wavelength solution wavelengths for all pixels
        n_pixels = len(rough_wls)
        our_wavelength_solution_for_order = leg_out(np.arange(n_pixels))
        rel_residual = (our_wavelength_solution_for_order[rough_wls>0] -  rough_wls[rough_wls>0]) * scipy.constants.c /rough_wls[rough_wls>0]
        rel_precision_cm_s = 100 * np.std(rel_residual)/np.sqrt(len(rough_wls[rough_wls>0]))
        if print_update:
            print('Absolute standard error (this order): {:.2f} cm/s'.format(abs_precision_cm_s))
            print('Relative standard error (this order): {:.2f} cm/s'.format(rel_precision_cm_s))
        
        if plot_path is not None:
            fig, ax = plt.subplots(2,1) #figsize=(20,16), tight_layout=True
            ax[0].plot(abs_residual)
            ax[0].set_xlabel('Pixel')
            ax[0].set_ylabel('Absolute Error [m/s]')
            ax[1].plot(rel_residual)
            ax[1].set_xlabel('Pixel')
            ax[1].set_ylabel('Relative Error [m/s]')
            plt.savefig('{}/rv_precision.png'.format(plot_path), dpi=250)
            plt.close()

        return rel_precision_cm_s, abs_precision_cm_s

    def mask_array_neid(self, calflux, n_orders):
        """ Creates ad-hoc mask to remove bad pixel regions specific to order. 
        For NEID testing. 
        Args:
            calflux (np.array): (N_orders x N_pixels) flux array to be masked
            n_orders (np.array): number of orders to be masked
        Returns:
            np.array: masked flux array
        """
        mask = np.zeros((2,n_orders),dtype=int)
        
        mask_order_lims = {
        50: (430, 457),
        51: (432, 459),
        52: (434, 461),
        53: (435, 463),
        54: (437, 466),
        55: (432, 468),
        56: (432, 471),
        57: (433, 464),
        58: (434, 464),
        59: (436, 466),
        60: (437, 470),
        61: (430, 470),
        62: (430, 472),
        63: (433, 474),
        64: (433, 464),
        65: (435, 468),
        66: (437, 468),
        67: (432, 465),
        68: (432, 463),
        69: (436, 466),
        70: (437, 470),
        71: (433, 460),
        72: (435, 458),
        73: (437, 457),
        74: (437, 455),
        75: (434, 459),
        76: (433, 463),
        77: (437, 457),
        78: (437, 457),
        79: (430, 461),
        80: (430, 461),
        81: (430, 465),
        82: (433, 456),
        83: (435, 458),
        84: (433, 458),
        85: (435, 458),
        86: (437, 458),
        87: (437, 458),
        88: (429, 461),
        89: (429, 462),
        90: (429, 468),
        91: (429, 468),
        92: (433, 478),
        93: (433, 475),
        94: (437, 480),
        95: (437, 480),
        96: (437, 482),
        97: (425, 485),
        98: (425, 485),
        99: (425, 485),
        100: (425, 485),
        101: (425, 485),
        102: (425, 485),
        103: (425, 490),
        104: (425, 490),
        }
        
        for i in np.arange(n_orders):
            mask[0, i] = mask_order_lims[i + self.min_order][0]
            mask[1, i] = mask_order_lims[i + self.min_order][1]

        # zero out bad pixels
            j = mask[0,i]
            k = mask[1,i]
            calflux[i + self.min_order, j:k] = 0

        # orders 71, 75 & 86 have some additional weird stuff going on
        calflux[71, 1550:1560] = 0
        calflux[75, 1930:1940] = 0
        calflux[75, 6360:6366] = 0
        calflux[86, 1930:1940] = 0
        
        return calflux

    
    def comb_gen(self, f0, f_rep):
        """ Computes wavelengths of LFC modes using the comb equation
        Args:
            f0 (float): initial comb frequency [Hz]
            f_rep (float): comb repitition frequency [Hz]
        Returns:
            np.array: array of comb lines [Angstroms]
        """
        mode_start = int((((scipy.constants.c * 1e10) / self.min_wave) - f0) / f_rep)
        
        mode_end = int((((scipy.constants.c * 1e10) / self.max_wave) - f0) / f_rep)

        mode_numbers = np.arange(mode_start, mode_end, -1)

        frequencies = f0 + (mode_numbers * f_rep)
        comb_lines_ang = scipy.constants.c * 1e10 / frequencies

        return comb_lines_ang

    def save_wl_pixel_info(self,file_name,wave_pxl_data):
        """
        Saves wavelength pixel reference file.
        
        Args: 
            file_name (str): Filename including date and time from original science file
            wave_pxl_data (np.array): Wavelength per pixel reference information output by 
                function 'run_wavelength_cal'.
                
        Returns:
            str: Full wavelength pixel reference filename
        """

        np.save(file_name,wave_pxl_data,allow_pickle=True)

    def save_etalon_mask_update(self,file_name,wave_pxl_data):
        """
        Saves nightly etalon mask
        
        Args: 
            file_name (str): Filename including date and time from original science file
            new_mask (np.array): Wavlengths of updated etalon mask
                function 'run_wavelength_cal'.
                
        Returns:
            str: Updated mask in two column, csv file
        """
        df_out = pd.DataFrame()     
        for i,item in enumerate(wave_pxl_data):
            dic1 = wave_pxl_data[i]  # Assuming you want the first dictionary in the values list
            known_wavelengths_vac = dic1['known_wavelengths_vac']
            line_positions = dic1['line_positions']

            # Keep the old and new values in to check results against original mask.
            # data = {'known_wavelengths_vac': known_wavelengths_vac, 'line_positions': line_positions} #test
            data = {'line_positions': line_positions,'weight': np.ones_like(line_positions)}
            df_one = pd.DataFrame(data)
            df_out = pd.concat([df_out, df_one])   
        #df_out.drop_duplicates(subset='known_wavelengths_vac',keep='first',inplace=True)    
        df_out.drop_duplicates(subset='line_positions',keep='first',inplace=True)
        df_out.sort_values(df_out.columns[0],inplace=True)
        df_out.to_csv(file_name,index=False,header=False,sep=' ')

def calcdrift_polysolution(wlpixelfile1, wlpixelfile2):
    
    peak_wavelengths_ang1 = np.load(
        wlpixelfile1, allow_pickle=True
    ).tolist()

    peak_wavelengths_ang2 = np.load(
        wlpixelfile2, allow_pickle=True
    ).tolist()

    orders1 = list(peak_wavelengths_ang1.keys())
    orders2 = list(peak_wavelengths_ang2.keys())

    orders = np.intersect1d(orders1, orders2)

    drift_all_orders = np.empty((len(orders),2))

    # make a dataframe and join on wavelength
    for i, order_num in enumerate(orders):
        order_wls1 = pd.DataFrame(
            data = np.transpose([
                peak_wavelengths_ang1[order_num]['known_wavelengths_vac'],
                peak_wavelengths_ang1[order_num]['line_positions']
            ]),
            columns=['wl', 'pixel1']
        )

        order_wls2 = pd.DataFrame(
            data = np.transpose([
                peak_wavelengths_ang2[order_num]['known_wavelengths_vac'],
                peak_wavelengths_ang2[order_num]['line_positions']
            ]),
            columns=['wl', 'pixel2']
        )

        order_wls = order_wls1.set_index('wl').join(order_wls2.set_index('wl'))

        delta_lambda = order_wls.index.values[1:] - order_wls.index.values[:-1]
        delta_pixel = order_wls.pixel1.values[1:] - order_wls.pixel1.values[:-1]

        drift_pixels = order_wls['pixel2'] - order_wls['pixel1']

        drift_wl = drift_pixels.values[1:] / delta_pixel * delta_lambda

        alpha = (drift_wl / order_wls.index.values[1:])

        drifts_cms = (alpha**2 + 2 * alpha) / (alpha**2 + 2 * alpha + 2) * cst.c.to(u.cm/u.s).value

        drift_all_orders[i,0] = order_num
        drift_all_orders[i,1] = np.nanmedian(drifts_cms)

    return drift_all_orders

def plot_drift(wlpixelfile1,wlpixelfile2, figsave_name):
    """Overall RV of cal data vs time for array of input files.
    Args:
        wlpixelfile1 (str): Path to first wavelength solution file
        wlpixelfile2 (str): Path to second wavelength solution file
    """
    drift = calcdrift_polysolution(wlpixelfile1,wlpixelfile2)
    obsname1 = wlpixelfile1.split('_')[1]
    obsname2 = wlpixelfile2.split('_')[1]

    fig,ax = plt.subplots()
    ax.axhline(0,color='grey',ls='--')

    plt.plot(
        drift[:,0],drift[:,1],'ko',ls='-'
    )
    plt.title('Inst. drift: {} to {}'.format(obsname1,obsname2))
    plt.xlabel('Order')
    plt.ylabel('Drift [cm s$^{-1}$]')
    plt.savefig(figsave_name, dpi=250)
    plt.close()

class WaveInterpolation:
    """
    This module defines 'WaveInterpolation' and methods to perform interpolation 
    between different wavelength solutions.
    
    Wavelength interpolation computation. Algorithm is called under _perform() 
    in wavelength_cal.py. Algorithm itself iterates over orders.
    """
    
    def __init__(
        self, l1_timestamp, wls_timestamps, wls1_arrays, wls2_arrays, config=None, logger=None
    ):
        """Initializes WaveCalibration class.
        Args:
            l1_timestamp (float): Datetime of the input L1 file. WLS will be interpolated to this point in time         
            wls_timestamps (list): List of timestamps for each of the input WLS arrays
            wls1_arrays (dict): Dictionary of the input WLS arrays for the first WLS. Keys should match the extension names in the L1 obj
            wls2_arrays (dict): Dictionary of the input WLS arrays for the second WLS. Keys should match the extension names in the L1 obj
            config (configparser.ConfigParser, optional): Config context. 
                Defaults to None.
            logger (logging.Logger, optional): Instance of logging.Logger. 
                Defaults to None, which involves DummyLogger (print statements).        

        """
        self.logger = logger if logger is not None else DummyLogger()
        self.l1_timestamp = l1_timestamp
        self.wls_timestamps = wls_timestamps
        self.wls1_arrays = wls1_arrays
        self.wls2_arrays = wls2_arrays
        self.config = config

    def wave_interpolation(self, method='linear'):
        msg = "Performing wavelength interpolation."
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
      
        if method == 'linear':
            # Determine how the dates are formatted
            isJD = isinstance(self.l1_timestamp, float)
            isDatetime = isinstance(self.l1_timestamp, datetime)
            try:
                # If the string is successfully parsed, it's a datetime string
                foo = parser.parse(self.l1_timestamp)
                isDateStr = True
            except:
                isDateStr = False
            
            # Compute differences between timestamps
            if isDateStr:
                self.l1_timestamp_obj = datetime.strptime(self.l1_timestamp, "%Y-%m-%dT%H:%M:%S.%f")
                self.wls_timestamp_objs = [datetime.strptime(self.wls_timestamps[0], "%Y-%m-%dT%H:%M:%S.%f"), 
                                           datetime.strptime(self.wls_timestamps[1], "%Y-%m-%dT%H:%M:%S.%f")]
                tdiff = (self.wls_timestamp_objs[1] - self.wls_timestamp_objs[0]).total_seconds()
                deltat = (self.l1_timestamp_obj - self.wls_timestamp_objs[0]).total_seconds()
            elif isJD:
                tdiff = self.wls_timestamp[1] - self.wls_timestamp[0]
                deltat = self.l1_timestamp - self.wls_timestamp[0]
            elif isDatetime:
                tdiff = (self.wls_timestamp[1] - self.wls_timestamp[0]).total_seconds()
                deltat = (self.l1_timestamp - self.wls_timestamp[0]).total_seconds()
            else:
                 self.logger.error("l1_timestamp not in a recognized format")
            if tdiff == 0:
                frac = 0.0
            else:
                frac = deltat / tdiff
    
            # Perform linear interpolation between wls1 and wls2
            new_wls_arrays = {}
            for ext, arr in self.wls1_arrays.items():
                new_wls_arrays[ext] = self.wls1_arrays[ext] + frac * (self.wls2_arrays[ext] - self.wls1_arrays[ext])
    
            return new_wls_arrays
        
        else:
            self.logger.error('Unsupported method specified in wave_interpolation')
            return None
