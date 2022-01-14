from astropy import units as u, constants as cst
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import Legendre
import os
import pandas as pd
import scipy
from scipy import signal
from scipy.special import erf
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize.minpack import curve_fit
from modules.Utils.config_parser import ConfigHandler

class WaveCalibration:
    """
    This module defines 'WaveCalibration' and methods to perform the 
    wavelength calibration.
    
    Wavelength calibration computation. Algorithm is called under _perform() 
    in wavelength_cal.py. Algorithm itself iterates over orders.
    """
    
    def __init__(self, cal_type, quicklook, data_type, config=None, logger=None):
        """Initializes WaveCalibration class.

        Args:
            save_wl_pixel_toggle: TODO
            quicklook: TODO
            config (configparser.ConfigParser, optional): Config context. 
                Defaults to None.
            logger (logging.Logger, optional): Instance of logging.Logger. 
                Defaults to None.        
                
        Attributes:        
            quicklook (bool): Whether or not to run quicklook pipeline. Defaults to False.

        """
        self.cal_type = cal_type
        self.quicklook = quicklook
        self.data_type = data_type
        configpull = ConfigHandler(config,'PARAM')
        self.figsave_name = configpull.get_config_value('drift_figsave_name','instrument_drift')
        self.skip_orders = configpull.get_config_value('skip_orders',None)
        self.quicklook_steps = configpull.get_config_value('quicklook_steps',10)
        self.min_wave = configpull.get_config_value('min_wave_lfc',3800)
        self.max_wave = configpull.get_config_value('max_wave_lfc',9300)
        self.fit_order = configpull.get_config_value('fit_order',9)
        self.fit_type = configpull.get_config_value('fit_type', 'Legendre')
        self.min_order = configpull.get_config_value('min_order',None)
        self.max_order = configpull.get_config_value('max_order',None)
        self.n_sections = configpull.get_config_value('n_sections',1)
        self.skip_orders = configpull.get_config_value('skip_orders',None)
        self.save_diagnostics_dir = configpull.get_config_value('save_diagnostics','outputs/')
        self.linelist_path = configpull.get_config_value('linelist_path_etalon',None)
        self.clip_peaks_toggle = configpull.get_config_value('clip_peaks','False')
 
## wavecal fxns ## -run_wavelength_cal, -remove_orders, -fit_many_orders,
#-find_peaks_in_order, -find_peaks, -integrate_gaussian, -fit_gaussian, -clip_peaks,
#-line_match, -mode_match,-fit_polynomial,-calculate_rv_precision, -mask_array_neid
       
    
    def run_wavelength_cal(
        self, calflux, rough_wls=None, 
        peak_wavelengths_ang=None, lfc_allowed_wls=None):
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
            np.array: (N_orders x N_pixels) array of the computed wavelength
                for each pixel.
            TODO: second return     
        """

        if self.min_order is None:
            self.min_order = 10
        if self.max_order is None:
            self.max_order = len(calflux)

        # create directories for diagnostic plots
        if type(self.save_diagnostics_dir) == str:
            self.save_diagnostics_dir = self.save_diagnostics_dir+'{}_diagnostics'.format(self.cal_type)
            SAVEPLOTS = ('{}/%s' % self.save_diagnostics_dir).format(os.getcwd())
            if not os.path.isdir(SAVEPLOTS):
                os.makedirs(SAVEPLOTS)
        if self.save_diagnostics_dir == 'False':
            SAVEPLOTS = None

        if self.quicklook == False:
            order_list = self.remove_orders(step=1)
            n_orders = len(order_list)

            if self.data_type == 'NEID':
                calflux = self.mask_array_neid(calflux,n_orders)

            # perform wavelength calibration
            poly_soln, wls_and_pixels = self.fit_many_orders(
                calflux, order_list, rough_wls=rough_wls, 
                comb_lines_angstrom=lfc_allowed_wls, 
                expected_peak_locs=peak_wavelengths_ang, 
                print_update=True, plt_path=SAVEPLOTS
            )

            # make a plot of all of the precise new wls minus the rough input  wls
            if SAVEPLOTS is not None and rough_wls is not None:
                plt.figure()
                for i in np.arange(len(poly_soln)):
                    wls_i = poly_soln[i, :]
                    rough_wls_i = rough_wls[i,:]
                    plt.plot(wls_i - rough_wls_i, color='grey', alpha=0.5)

                plt.xlabel('pixel')
                plt.ylabel('Derived WLS - Approx WLS [$\\rm \AA$]')
                plt.saveplot('{}/all_wls.png'.format(SAVEPLOTS), dpi=250)


        if self.quicklook == True:
            #TODO
            order_list = self.remove_orders(step = self.quicklook_steps)
            n_orders = len(order_list)
            
            masked_calflux = self.mask_array_neid(calflux,n_orders)
            
            poly_soln, _, wls_and_pixels = self.fit_many_orders(
                masked_calflux, order_list, rough_wls=rough_wls, 
                comb_lines_angstrom=lfc_allowed_wls, 
                expected_peak_locs=peak_wavelengths_ang, 
                print_update=True, plt_path=SAVEPLOTS ###CHECK THIS TODO
            )

        return poly_soln, wls_and_pixels    
        
    def fit_many_orders(
        self, cal_flux, order_list, rough_wls=None, comb_lines_angstrom=None,
        expected_peak_locs=None, plt_path=None, print_update=False):
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
                dict: the peaks and wavelengths used for wavelength cal. Keys are ints
                    representing order numbers, values are 2-tuples of:
                        - lists of wavelengths corresponding to peaks
                        - the corresponding (fractional) pixels on which the peaks fall
        """    

        # 2D extracted spectra
        if plt_path is not None:
            plt.figure(figsize=(20,10))
            im = plt.imshow(cal_flux, aspect='auto')
            im.set_clim(0, 20000)
            plt.xlabel('pixel')
            plt.ylabel('order number')
            plt.savefig('{}/extracted_spectra.png'.format(plt_path), dpi=250)
            plt.close()

        order_precisions = []
        num_detected_peaks = []
        wavelengths_and_pixels = {}

        poly_soln_final_array = np.zeros(np.shape(cal_flux))

        for order_num in order_list:
            if print_update:
                print('\nRunning order # {}'.format(order_num))

            if plt_path is not None:
                order_plt_path = '{}/order{}'.format(plt_path, order_num)
                if not os.path.isdir(order_plt_path):
                    os.makedirs(order_plt_path)

                plt.figure(figsize=(20,10))
                plt.plot(cal_flux[order_num,:], color='k', alpha=0.5)
                plt.title('Order # {}'.format(order_num))
                plt.xlabel('pixel')
                plt.ylabel('flux')
                plt.savefig('{}/order_spectrum.png'.format(order_plt_path), dpi=250)
                plt.close()
            else:
                order_plt_path = None

            order_flux = cal_flux[order_num,:]
            rough_wls_order = rough_wls[order_num,:]
            n_pixels = len(order_flux)

            # find, clip, and compute precise wavelengths for peaks.
            # this code snippet will only execute for Etalon and LFC frames.
            if expected_peak_locs is None:

                fitted_peak_pixels, detected_peak_pixels, detected_peak_heights, gauss_coeffs = self.find_peaks_in_order(
                    order_flux, plot_path=order_plt_path
                )

                if self.clip_peaks_toggle == 'True':
                    good_peak_idx = self.clip_peaks(
                        order_flux, fitted_peak_pixels, detected_peak_pixels,
                        gauss_coeffs, detected_peak_heights, rough_wls_order,
                        comb_lines_angstrom=comb_lines_angstrom, 
                        plot_path=order_plt_path, print_update=print_update
                    )
                else:
                    good_peak_idx = np.arange(len(detected_peak_pixels))

                if self.cal_type == 'LFC' or self.cal_type == 'Simulated':
                    wls, _ = self.mode_match(
                        order_flux, fitted_peak_pixels, good_peak_idx, rough_wls_order, comb_lines_angstrom, 
                        print_update=print_update, plot_path=order_plt_path
                    )
                    # TODO: - save pixel-wavelength pairs of LFC peaks
                elif self.cal_type == 'Etalon':

                    assert comb_lines_angstrom is None, '`comb_lines_angstrom` should not be set for Etalon frames.'

                    wls = np.interp(fitted_peak_pixels[good_peak_idx], np.arange(n_pixels), rough_wls_order)
                    
                    # save pixel-wavelength mapping for detected etalon peaks and return
                    # TODO: save LFC-derived wls as "official" wavelength solution for this image
                    # have: - precise pixel locations of (potentially clipped) etalon peaks (fitted_peak_pixels[good_peak_idx])
                    #       - precise LFC-derived pixel-wavelength sol

                fitted_peak_pixels = fitted_peak_pixels[good_peak_idx]

            # use expected peak locations to compute updated precise wavelengths for each pixel
            else:

                if order_plt_path is not None:
                    plot_toggle = True
                else:
                    plot_toggle = False

                line_wavelengths = expected_peak_locs[order_num]['known_wavelengths_vac']
                line_pixels_expected = expected_peak_locs[order_num]['line_positions']

                wls, gauss_coeffs = self.line_match(
                    order_flux, line_wavelengths, line_pixels_expected, 
                    plot_toggle, order_plt_path
                )

                fitted_peak_pixels = gauss_coeffs[1,:]
            
            # Only calculate a new wavelength solution if we aren't using an Etalon frame.
            # Computing a drift value and using it to update the wavelength solution
            # is handled by the inst drift module.
            if self.cal_type != 'Etalon':

                # calculate the wavelength solution for the order
                polynomial_wls, leg_out = self.fit_polynomial(
                    wls, n_pixels, fitted_peak_pixels, 
                    plot_path=order_plt_path
                )
                poly_soln_final_array[order_num,:] = polynomial_wls

                if plt_path is not None:
                    plt.figure(figsize=(12,5))
                    plt.plot(
                        np.arange(n_pixels), 
                        leg_out(np.arange(n_pixels)) - rough_wls_order, 
                        color='k'
                    )
                    plt.xlabel('pixel')
                    plt.ylabel('Precise WLS - Rough WLS [$\\rm \AA$]')
                    plt.savefig(
                        '{}/precise_vs_rough.png'.format(order_plt_path),
                        dpi=250
                    )
                    plt.tight_layout()
                    plt.close()

                # compute various RV precision values for order
                poly_precision, abs_precision = self.calculate_rv_precision(
                    fitted_peak_pixels, good_peak_idx, wls, leg_out, plot_path=order_plt_path, 
                    print_update=print_update
                )

                order_precisions.append(abs_precision)
                num_detected_peaks.append(len(good_peak_idx))

                squared_resids = (np.array(order_precisions) * num_detected_peaks)**2
                sum_of_squared_resids = np.sum(squared_resids)
                overall_std_error = (
                    np.sqrt(sum_of_squared_resids) / 
                    np.sum(num_detected_peaks)
                )
                print('Overall absolute precision: {:2.2f} cm/s'.format(overall_std_error))

            wavelengths_and_pixels[order_num] = {'known_wavelengths_vac':wls, 'line_positions':fitted_peak_pixels}

        return poly_soln_final_array, wavelengths_and_pixels

    def remove_orders(self,step=1):
        """Removes bad orders from order list if between min and max orders to test.

        Args:
            step (int): Interval at which to test orders. Used to skip orders for QLP.
                Defaults to 1, which means every order will be tested on and none will be removed.

        Returns:
            list: List of orders to run wavelength calibration on.
        """
        order_list = [*range(self.min_order, self.max_order + 1,step)]
        if self.skip_orders:
            #self.skip_orders = self.skip_orders.split(',')
            for i in self.skip_orders:
                #i = int(i)
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
                np.array: array of true peak locations as 
                    determined by Gaussian fitting
                np.array: array of detected peak locations (pre-
                    Gaussian fitting)
                np.array: array of detected peak heights (pre-Gaussian fitting)
                np.array: array of size (4, n_peaks) 
                    containing best-fit Gaussian parameters [a, mu, sigma**2, const]
                    for each detected peak
        """
    
        n_pixels = len(order_flux)
        fitted_peak_pixels = np.array([])
        detected_peak_pixels = np.array([])
        detected_peak_heights = np.array([])
        gauss_coeffs = np.zeros((4,0))

        for i in np.arange(self.n_sections):

            if i == self.n_sections - 1:
                indices = np.arange(i * n_pixels // self.n_sections, n_pixels)
            else:
                indices = np.arange(i * n_pixels // self.n_sections, (i+1) * n_pixels // self.n_sections)

            fitted_peaks_section, detected_peaks_section, peak_heights_section, gauss_coeffs_section = self.find_peaks(order_flux[indices])

            detected_peak_heights = np.append(detected_peak_heights, peak_heights_section)
            gauss_coeffs = np.append(gauss_coeffs, gauss_coeffs_section, axis=1)

            if i == 0:
                fitted_peak_pixels = np.append(fitted_peak_pixels, fitted_peaks_section)
                detected_peak_pixels = np.append(detected_peak_pixels, detected_peaks_section)

            else:
                fitted_peak_pixels = np.append(fitted_peak_pixels, fitted_peak_pixels + i * n_pixels // self.n_sections)
                detected_peak_pixels = np.append(detected_peak_pixels, detected_peak_pixels + i * n_pixels // self.n_sections)
        
        if plot_path is not None:
            plt.figure()
            plt.plot(order_flux, color='k', lw=0.1)   
            plt.scatter(detected_peak_pixels, detected_peak_heights, s=1, color='r')
            plt.savefig('{}/detected_peaks.png'.format(plot_path), dpi=250)
            plt.close()

            n_zoom_sections = 10
            zoom_section_pixels = n_pixels // n_zoom_sections

            _, ax_list = plt.subplots(n_zoom_sections, 1, figsize=(6,12))
            for i, ax in enumerate(ax_list):
                ax.plot(order_flux,color='k', lw=0.1)
                ax.scatter(detected_peak_pixels,detected_peak_heights,s=1,color='r')
                ax.set_xlim(zoom_section_pixels * i, zoom_section_pixels * (i+1))
                ax.set_ylim(0,np.max(order_flux[zoom_section_pixels * i : zoom_section_pixels * (i+1)]))

            plt.tight_layout()
            plt.savefig('{}/detected_peaks_zoom.png'.format(plot_path),dpi=250)
            plt.close()

        return fitted_peak_pixels, detected_peak_pixels, detected_peak_heights, gauss_coeffs
        
    def find_peaks(self, order_flux):
        """
        Finds all order_flux peaks in an array. This runs scipy.signal.find_peaks 
            twice: once to find the average distance between peaks, and once
            for real, disregarding close peaks.

        Args:
            order_flux (np.array): flux values. Their indices correspond to
                their pixel numbers. Generally a subset of the full order.
            
        Returns:
            tuple of:
                np.array: array of true peak locations as 
                    determined by Gaussian fitting
                np.array: array of detected peak locations (pre-
                    Gaussian fitting)
                np.array: array of detected peak heights 
                    (pre-Gaussian fitting)
                np.array: array of size (4, n_peaks) 
                    containing best-fit Gaussian parameters [a, mu, sigma**2, const]
                    for each detected peak
        """

        c = order_flux - np.ma.min(order_flux)

        # TODO: make this more indep of order_flux flux
        height = 3 * np.ma.median(c) # 0.5 * np.ma.median(c) works for whole chip
        detected_peaks, properties = signal.find_peaks(c, height=height)

        distance = np.median(np.diff(detected_peaks)) // 2
        detected_peaks, properties = signal.find_peaks(c, distance=distance, height=height)
        peak_heights = np.array(properties['peak_heights'])

        # fit peaks with Gaussian to get accurate position
        fitted_peaks = detected_peaks.astype(float)
        gauss_coeffs = np.empty((4, len(detected_peaks)))
        width = np.mean(np.diff(detected_peaks)) // 2

        for j, p in enumerate(detected_peaks):
            idx = p + np.arange(-width, width + 1, 1)
            idx = np.clip(idx, 0, len(c) - 1).astype(int)
            coef = self.fit_gaussian(np.arange(len(idx)), c[idx])
            gauss_coeffs[:,j] = coef
            fitted_peaks[j] = coef[1] + p - width

        return fitted_peaks, detected_peaks, peak_heights, gauss_coeffs
        
    def clip_peaks(
        self, order_flux, fitted_peak_pixels, detected_peak_pixels, gauss_coeffs, 
        detected_peak_heights, rough_wls_order, comb_lines_angstrom=None,
        print_update=False, plot_path=None
    ):
        """
        Clips peaks that have detected and Gaussian-fitted central pixels values 
        more than 1 pixel apart. If clipping peaks for an LFC frame with known
        comb wavelengths, then also clips detected peaks which are more than one
        pixel (delta lambda/lambda) away from an LFC mode wavelength.

        Args:
            order_flux (np.array): array of order_flux data
            fitted_peak_pixels (np.array): array of true peak locations as 
                determined by Gaussian fitting
            detected_peak_pixels (np.array of float): array of detected peak locations 
                (pre-Gaussian fitting)
            gauss_coeffs (np.array): array of size (4, n_peaks) containing best-fit 
                Gaussian parameters [a, mu, sigma**2, const] for each detected peak            
            detected_peak_heights (np.array): array of detected peak heights (pre-Gaussian fitting)
            rough_wls_order (np.array): array of ThAr solution data
            comb_lines_angstrom (np.array): theoretical LFC wavelengths
                as computed by fundamental physics (in Angstroms). If peak
                clipping is being performed on an etalon frame, this should
                be set to None. Default None.
            print_update (bool): if True, print how many peaks were clipped
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.

        Returns: 
            np.array: indices of surviving peaks

        """

        # clip peaks that have Gaussian-fitted centers more than 1 pixel from
        # their detected centers
        good_peak_idx = np.where(np.abs(fitted_peak_pixels - detected_peak_pixels) < 1) [0]

        # if we know the wavelengths of the peaks (i.e. if dealing with LFC),
        # then we can clip peaks with derived wavelengths far from the location
        # of a comb line
        if comb_lines_angstrom is not None:

            # compute an approx wavelength solution that we'll use to find
            # the nearest LFC mode
            n_pixels = len(rough_wls_order)
            s = InterpolatedUnivariateSpline(np.arange(n_pixels), rough_wls_order)
            approx_peaks_lambda = s(fitted_peak_pixels)

            # iterate through all modes and save only those that are near 
            # LFC modes
            peaks_nearby_lfcmodes = []
            for i, lamb in enumerate(approx_peaks_lambda):

                # delta lambda / lambda
                approx_pixel_size = (approx_peaks_lambda[i] - s(fitted_peak_pixels[i] - 1)) / approx_peaks_lambda[i]

                best_mode_idx = (
                    np.abs(comb_lines_angstrom - lamb)
                ).argmin()
                if np.abs(comb_lines_angstrom[best_mode_idx] - lamb) < approx_pixel_size:
                    peaks_nearby_lfcmodes.append(i)
            
            good_peak_idx = np.intersect1d(peaks_nearby_lfcmodes, good_peak_idx)

        if print_update:
            print('{} peaks clipped'.format(len(detected_peak_pixels) - len(good_peak_idx)))

        if plot_path is not None:

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

            plt.tight_layout()
            plt.savefig('{}/unclipped_peaks_zoom.png'.format(plot_path), dpi=250)
            plt.close()

        return good_peak_idx
    
    def line_match(
        self, flux, linelist, line_pixels_expected, plot_toggle, savefig,
        gaussian_fit_width = 10):
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
                - np.array: same input linelist
                - np.array: array of size (4, n_peaks) containing best-fit 
                  Gaussian parameters [a, mu, sigma**2, const] for each detected peak  


        """
        num_input_lines = len(linelist)  
        num_pixels = len(flux)

        missed_lines = 0
        coefs = np.zeros((4,num_input_lines))
        for i in np.arange(num_input_lines):
            line_location = line_pixels_expected[i]
            peak_pixel = np.floor(line_location).astype(int)

            if peak_pixel < gaussian_fit_width:
                first_fit_pixel = 0
            else:
                first_fit_pixel = peak_pixel - gaussian_fit_width
            
            if peak_pixel + gaussian_fit_width > num_pixels:
                last_fit_pixel = num_pixels
            else:
                last_fit_pixel = peak_pixel + gaussian_fit_width

            # fit gaussian to matched peak location
            coefs[:,i] = self.fit_gaussian(
                np.arange(first_fit_pixel,last_fit_pixel),
                flux[first_fit_pixel:last_fit_pixel]
            )

            amp = coefs[0,i]
            if amp < 0:
                missed_lines += 1
        
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
                    flux[i*zoom_section_pixels:(i+1)*zoom_section_pixels],color='k',alpha=.1
                )

                # plot the fitted peak maxima as points
                ax.scatter(
                    coefs[1,:][
                        (coefs[1,:] > i * zoom_section_pixels) & 
                        (coefs[1,:] < (i+1) * zoom_section_pixels)
                    ], 
                    coefs[0,:][
                        (coefs[1,:] > i * zoom_section_pixels) & 
                        (coefs[1,:] < (i+1) * zoom_section_pixels)
                    ] + 
                    coefs[3,:][
                        (coefs[1,:] > i * zoom_section_pixels) & 
                        (coefs[1,:] < (i+1) * zoom_section_pixels)
                    ],
                    color='red'
                )

                # overplot the Gaussian fits
                for j in np.arange(num_input_lines):

                    # if peak in range:
                    if (coefs[1,j] > i * zoom_section_pixels) & (coefs[1,j] < (i+1) * zoom_section_pixels):

                        xs = np.floor(coefs[1,j]) - gaussian_fit_width + np.linspace(0, 2 * gaussian_fit_width, 2 * gaussian_fit_width)
                        gaussian_fit = self.integrate_gaussian(
                            xs, coefs[0,j], coefs[1,j], coefs[2,j], coefs[3,j]
                        )

                        ax.plot(xs, gaussian_fit, alpha=0.5, color='grey')

                ax.set_yscale('log')

            plt.tight_layout()
            plt.savefig('{}/spectrum_and_peaks.png'.format(savefig), dpi=250)
            plt.close()

        return linelist, coefs

    def mode_match(
        self, order_flux, fitted_peak_pixels, good_peak_idx, rough_wls_order, 
        comb_lines_angstrom, print_update=False, plot_path=None
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

        n_pixels = len(order_flux)

        s = InterpolatedUnivariateSpline(np.arange(n_pixels), rough_wls_order)
        approx_peaks_lambda = s(fitted_peak_pixels[good_peak_idx])

        # approx_peaks_lambda = np.interp(
        #     new_peaks[good_peak_idx], np.arange(n_pixels), rough_wls_order)

        # Now figure what mode numbers the peaks correspond to
        n_clipped_peaks = len(fitted_peak_pixels[good_peak_idx])
        mode_nums = np.empty(n_clipped_peaks)

        peak_mode_num = 0
        for i in range(n_clipped_peaks):

            # calculate difference in peak locs of last several peaks
            num_peaks_kernel = 100
            if (i < num_peaks_kernel):
                running_peak_diff = np.median(
                    np.diff(fitted_peak_pixels[good_peak_idx][0:num_peaks_kernel])
                )
            else:
                running_peak_diff = np.median(
                    np.diff(fitted_peak_pixels[good_peak_idx][i-num_peaks_kernel:i])
                )

            if i==0:
                for j in np.arange(15):
                    if fitted_peak_pixels[good_peak_idx][i] > (j + 1.5) * running_peak_diff:
                        peak_mode_num += 1
                if fitted_peak_pixels[good_peak_idx][i] > 15.5 * running_peak_diff:
                    assert False, 'More than 15 peaks in a row not detected!'
        
            # if current peak location is greater than (n + 0.5) * sigma of 
            # previous peak diffs, then skip over n modes
            if i > 0:
                for j in np.arange(15):
                    if (
                        fitted_peak_pixels[good_peak_idx][i] - 
                        fitted_peak_pixels[good_peak_idx][i - 1] > 
                        (j + 1.5) * running_peak_diff
                    ):
                        peak_mode_num += 1
                if (
                    fitted_peak_pixels[good_peak_idx][i] - 
                    fitted_peak_pixels[good_peak_idx][i - 1] > 
                    15.5 * running_peak_diff
                ):
                    assert False, 'More than 15 peaks in a row not detected!'

            # set mode_nums
            mode_nums[i] = peak_mode_num
            peak_mode_num += 1

        idx = (np.abs(comb_lines_angstrom - 
            approx_peaks_lambda[len(approx_peaks_lambda) // 2])).argmin()

        n_skipped_modes_in_chip_first_half = mode_nums[
            (len(approx_peaks_lambda) // 2)] - (len(approx_peaks_lambda) // 2)
        mode_nums += (idx - (len(approx_peaks_lambda) // 2) - 
            n_skipped_modes_in_chip_first_half)
        
        if plot_path is not None:
            plt.figure()
            plt.plot(rough_wls_order, order_flux, alpha=0.2)
            plt.vlines(comb_lines_angstrom, ymin=0, ymax=5000, color='r')
            plt.xlim(rough_wls_order[200], rough_wls_order[700])
            plt.xlabel('wavelength [$\\rm \AA$]')
            plt.savefig('{}/rough_sol_and_lfc_lines.png'.format(plot_path), dpi=250)
            plt.close()

            n_zoom_sections = 20
            zoom_section_wavelen = (
                (np.max(rough_wls_order) - np.min(rough_wls_order)) // 
                n_zoom_sections
            )
            zoom_section_pixels = n_pixels // n_zoom_sections

            _, ax_list = plt.subplots(n_zoom_sections, 1, figsize=(6, 20))
            for i, ax in enumerate(ax_list):
                ax.plot(rough_wls_order, order_flux, color='k', alpha=0.1)
                for mode_num in mode_nums:
                    if (
                        (
                            comb_lines_angstrom[mode_num.astype(int)] > 
                            zoom_section_wavelen * i + np.min(rough_wls_order)
                        ) and (
                            comb_lines_angstrom[mode_num.astype(int)] < 
                            zoom_section_wavelen * (i + 1) + np.min(rough_wls_order)
                        )
                    ):
                        ax.text(
                            comb_lines_angstrom[mode_num.astype(int)], 0, 
                            str(int(mode_num)), fontsize=4
                        )
                ax.set_xlim(
                    zoom_section_wavelen * i + np.min(rough_wls_order), 
                    zoom_section_wavelen * (i + 1) + np.min(rough_wls_order)
                )
                ax.set_ylim(
                    0, 
                    np.max(
                        order_flux[zoom_section_pixels * i: zoom_section_pixels * (i + 1)]
                    )
                )
                ax.set_yticks([])
            plt.tight_layout()
            plt.savefig('{}/labeled_line_locs.png'.format(plot_path), dpi=250)
            plt.close()

        if print_update:
            print(
                '{} LFC modes not detected'.format(peak_mode_num - n_clipped_peaks)
            )
        wls = comb_lines_angstrom[mode_nums.astype(int)]

        return wls, mode_nums
    
    def integrate_gaussian(self, x, a, mu, sig, const, int_width=0.5):
        """
        Returns the integral of a Gaussian over a specified symamtric range. 
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

    def fit_gaussian(self, x, y):
        """
        Fits a continuous Gaussian to a discrete set of x and y datapoints
        using scipy.curve_fit
        
        Args:
            x (np.array): x data to be fit
            y (np.array): y data to be fit

        Returns:
            list: best-fit parameters [a, mu, sigma**2, const]
        """
        x = np.ma.compressed(x)
        y = np.ma.compressed(y)

        i = np.argmax(y[len(y) // 4 : len(y) * 3 // 4]) + len(y) // 4
        p0 = [y[i], x[i], 1, np.min(y)]

        with np.warnings.catch_warnings():
            np.warnings.simplefilter("ignore")
            popt, _ = curve_fit(self.integrate_gaussian, x, y, p0=p0, maxfev=100000)

        return popt  
          
    def fit_polynomial(self, wls, n_pixels, fitted_peak_pixels, plot_path=None):
        """
        Given precise wavelengths of detected LFC order_flux lines, fits a 
        polynomial wavelength solution.

        Args:
            wls (np.array): the known, precise wavelengths of the detected peaks,
                either from fundamental physics or a previous wavelength solution.
            n_pixels (int): number of pixels in the order
            fitted_peak_pixels (np.array): array of true detected peak locations as 
                determined by Gaussian fitting.
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.

        Returns:
            tuple of:
                np.array: calculated wavelength solution for the order (i.e. 
                    wavelength value for each pixel in the order)
                func: a Python function that, given an array of pixel locations, 
                    returns the Legendre polynomial wavelength solutions
        """

        # fitted_heights = gauss_coeffs[0,:]
        # weights = np.sqrt(fitted_heights)
        if self.fit_type == 'Legendre': 
            leg_out = Legendre.fit(fitted_peak_pixels, wls, self.fit_order)
            our_wavelength_solution_for_order = leg_out(np.arange(n_pixels))

            if plot_path is not None:

                s = InterpolatedUnivariateSpline(fitted_peak_pixels, wls)
                interpolated_ground_truth = s(np.arange(n_pixels))

                # plot ground truth wls vs our wls
                plt.figure()
                plt.plot(
                    np.arange(n_pixels), 
                    interpolated_ground_truth - our_wavelength_solution_for_order, 
                    color='k'
                )

                plt.xlabel('pixel')
                plt.ylabel('wavelength diff (A)')
                plt.savefig('{}/interp_vs_our_wls.png'.format(plot_path))
                plt.close()
        else:
            raise ValueError('Only set up to perform Legendre fits currently! Please set fit_type to "Legendre"')

        return our_wavelength_solution_for_order, leg_out

    def calculate_rv_precision(
        self, fitted_peak_pixels, good_peak_idx, wls, leg_out, 
        print_update=True, plot_path=None
    ):
        """
        Calculates 1) RV precision from the difference between the known wavelengths of pixels
        containing peak flux values and the fitted wavelengths of the same
        pixels, generated using a polynomial wavelength solution, and 2)
        absolute RV precision from the difference between contiguous peaks in
        RV space.

        Args:
            fitted_peak_pixels (np.array of float): array of true detected peak locations as 
                determined by Gaussian fitting.
            good_peak_idx (np.array of int): indices of unclipped peaks
            wls (np.array of float): precise wavelengths of `fitted_peak_pixels`,
                from fundamental physics or another wavelength solution.
            leg_out (func): a Python function that, given an array of pixel 
                locations, returns the Legendre polynomial wavelength solutions
            print_update (bool): If true, prints standard error per order.
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.

        Returns:
            tuple of:
                float: polynomial RV precision in cm/s
                float: absolute RV precision in cm/s
        """
        our_wls_peak_pos = leg_out(fitted_peak_pixels[good_peak_idx]) 

        # polynomial precision of order
        residual = ((our_wls_peak_pos - wls) * scipy.constants.c) / wls
        poly_precision_cm_s = 100 * np.std(residual)/np.sqrt(len(good_peak_idx))

        # absolute RV precision of order, i.e. std(delta wavelength of peaks/lambda) / sqrt(num peaks))
        abs_residual = (wls[1:] - wls[:-1]) * scipy.constants.c / wls[1:]
        abs_precision_cm_s = 100 * np.std(abs_residual)/np.sqrt(len(good_peak_idx))

        if print_update:
            print('Polynomial standard error (this order): {:.2f} cm/s'.format(poly_precision_cm_s))

        if plot_path is not None:
            plt.figure()
            plt.plot(residual)
            plt.xlabel('pixel')
            plt.ylabel('error [m/s]')
            plt.savefig('{}/rv_precision.png'.format(plot_path), dpi=250)
            plt.close()

        return poly_precision_cm_s, abs_precision_cm_s

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
        60: (437, 468),
        61: (430, 470),
        62: (430, 472),
        63: (433, 474),
        64: (433, 464),
        65: (435, 466),
        66: (437, 468),
        67: (432, 463),
        68: (432, 463),
        69: (436, 466),
        70: (437, 470),
        71: (433, 460),
        72: (433, 460),
        73: (437, 457),
        74: (437, 457),
        75: (434, 459),
        76: (433, 463),
        77: (437, 457),
        78: (437, 457),
        79: (430, 461),
        80: (430, 461),
        81: (430, 465),
        82: (433, 458),
        83: (433, 458),
        84: (433, 458),
        85: (435, 458),
        86: (437, 458),
        87: (437, 458),
        88: (429, 461),
        89: (429, 462),
        90: (429, 468),
        91: (429, 468),
        92: (433, 475),
        93: (433, 475),
        94: (437, 480),
        95: (437, 480),
        96: (437, 485),
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

        # orders 75 & 86 have some additional weird stuff going on
        calflux[75, 1930:1940] = 0
        calflux[75, 6360:6366] = 0
        calflux[86, 1930:1940] = 0
        
        return calflux

    ## instrument drift fxns ## plot_drift, calcdrift_polysolution
    
    def plot_drift(self,wlpixelfile1,wlpixelfile2):
        """Overall RV of cal data vs time for array of input files.

        Args:
            wlpixelfile1 ([type]): [description]
            wlpixelfile2 ([type]): [description]
        """
        drift = self.calcdrift_polysolution(wlpixelfile1,wlpixelfile2)
        obsname1 = wlpixelfile1.split('_')[1]
        obsname2 = wlpixelfile2.split('_')[1]
        
        fig,ax = plt.subplots()
        ax.axhline(0,color='grey',ls='--')
        
        plt.plot(
            drift[:,0],drift[:,1],'ko',ls='-'
        )
        plt.title('Inst. drift: {} to {}'.format(obsname1,obsname2))
        plt.xlabel('order')
        plt.ylabel('drift [cm s$^{-1}$]')
        plt.savefig(self.figsave_name, dpi=250)

    def calcdrift_polysolution(self,wlpixelfile1,wlpixelfile2):
        peak_wavelengths_ang1 = np.load(
            wlpixelfile1, allow_pickle=True
        ).tolist()

        peak_wavelengths_ang2 = np.load(
            wlpixelfile2, allow_pickle=True
        ).tolist()

        orders = peak_wavelengths_ang1.keys()

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
            delta_pixel = order_wls.pixel2.values[1:] - order_wls.pixel1.values[:-1]

            drift_pixels = order_wls['pixel2'] - order_wls['pixel1']

            drift_wl = drift_pixels.values[1:] / delta_pixel * delta_lambda # TODO: is this the correct way to compute drift?

            drifts_cms = (drift_wl / order_wls.index.values[1:] * cst.c).to(u.cm/u.s).value

            drift_all_orders[i,0] = order_num
            drift_all_orders[i,1] = np.mean(drifts_cms)

        return drift_all_orders
    
    ## plotting/unused? ## plot_poly_coefs, calcdrift_ccf, 
    def plot_poly_coefs(self,coef_num,order_num):
        """*** to implement in the future: polynomail coeffs vs time for specific order (inputs: poly num, order num)

        Args:
            coef_num ([type]): [description]
            order_num ([type]): [description]
        """
        pass
    
    def calcdrift_ccf(self,obstime,calfile1,calfile2):
        pass
    
    ## quicklook ## order_flux_gen
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

    def save_wl_pixel_info(self,file_suffix,wave_pxl_data):
        """
        Saves wavelength pixel reference file.
        
        Args: 
            file_suffix (str): String including date and time from original science file
            wave_pxl_data (np.array): Wavelength per pixel reference information output by 
                function 'run_wavelength_cal'.
                
        Returns:
            str: Full wavelength pixel reference filename
        """
        full_filename = self.wl_pixel_file_prefix + file_suffix
        np.save(full_filename,wave_pxl_data,allow_pickle=True)
        
        return full_filename
        ## self.wl_pixel_file_prefix in module config will need to include full path as well for now
