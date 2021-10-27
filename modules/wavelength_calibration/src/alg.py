import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy
import os
from scipy import signal
# from scipy.signal import find_peaks as peak
# from scipy.optimize import curve_fit as cv
from scipy.special import erf
from scipy.interpolate import InterpolatedUnivariateSpline
#from lmfit.models import GaussianModel #fit type can be changed
#uses _find_peaks, gaussfit3, gaussval2 from PyReduce
#import get_config_value once it is util primitve
from modules.Utils.config_parser import ConfigHandler
# from kpfpipe.models.level0 import KPF0
# from keckdrpframework.models.arguments import Arguments
from scipy.optimize.minpack import curve_fit
# from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.legendre import Legendre

class WaveCalibrationAlg:
    """
    Wavelength calibration computation. Algorithm is called to repeat under perform in wavelength_calibration.py,
    for each order between min_order and max_order. 

    This module defines 'WavelengthCalibration' and methods to perform the wavelength calibration.

   Args:
        config (configparser.ConfigParser, optional): Config context. Defaults to None.
        logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.

    Attributes:
        config_param(ConfigHandler): Instance representing pull from config file.
        
    """
    def __init__(self, config=None, logger=None): 
        """
        Inits LFCWaveCalibration class with LFC data, config, logger.

        Args:
            config (configparser.ConfigParser, optional): Config context. Defaults to None.
            logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.
        
        Attributes:
            max_wave (int): Maximum wavelength of wavelength range, in Angstroms. Pulled from config file.
            min_wave (int): Minimum wavelength of wavelength range, in Angstroms. Pulled from config file.
            fit_order (int): Order of fitting polynomial. Pulled from config file.
            min_order (int): Minimum order with coherent light/flux in flux extension. Pulled from config file.
            max_order (int): Maximum order with coherent light/flux in flux extension. Pulled from config file.
            n_sections (int): Number of sections to divide the comb into. Pulled from config file.
            skip_orders ():
            save_diagnostics ():
            quicklook_ord_s ():
        """

        configpull=ConfigHandler(config,'PARAM')
        self.max_wave=configpull.get_config_value('max_wave',9300)
        self.min_wave=configpull.get_config_value('min_wave',3800)
        self.fit_order=configpull.get_config_value('fit_order',9)
        self.lfc_min_order=configpull.get_config_value('lfc_min_order',50)
        self.lfc_max_order=configpull.get_config_value('lfc_max_order',100)
        self.etalon_min_order=configpull.get_config_value('etalon_min_order',40)
        self.etalon_max_order=configpull.get_config_value('etalon_max_order',45)
        self.thar_min_order=configpull.get_config_value('thar_min_order',3)
        self.thar_max_order=configpull.get_config_value('thar_max_order',117)
        self.n_sections=configpull.get_config_value('n_sections',20)
        #self.clip_peaks_opt=configpull.get_config_value('clip_peaks',False)
        self.lfc_skip_orders=configpull.get_config_value('lfc_skip_orders',None)
        self.etalon_skip_orders=configpull.get_config_value('etalon_skip_orders',None)
        self.thar_skip_orders=configpull.get_config_value('thar_skip_orders',None)
        self.save_diagnostics=configpull.get_config_value('save_diagnostics', 'False')
        self.quicklook_ord_steps=configpull.get_config_value('quicklook_ord_steps',5)
        self.height_sigma=configpull.get_config_value('height_sigma',.5)
        self.config=config
        self.logger=logger

    def remove_orders(self,min_order,max_order,skip_orders,step=1):
        """Removes bad orders from order list if between min and max orders to test.

        Args:
            step (int): Number of orders to skip in order removal. Used to skip orders for QLP.

        Returns:
            order_list (list): List of orders to run wavelength calibration on.
        """
        order_list = [*range(min_order, max_order + 1,step)]
        if skip_orders:
            #self.skip_orders = self.skip_orders.split(',')
            for i in skip_orders:
                #i = int(i)
                if i in order_list:
                    order_list.remove(i)
                else:
                    continue
        
        return order_list

    def get_master_data(self,master_path):
        """Temporary function to pull master data from master calibration file - will be removed once L1 is updated
        and permanent master file is created.

        Args:
            master_path (str): Path to master file name

        Returns:
            master_data (np.ndarray): Master calibration data
        """
        # m_file=fits.open(master_path)
        # if len(m_file)>2:
        #     print ("Cannot find data extension when there is more than one image HDU")
        # else:
        #     master_data=m_file[1].data

        m_file = fits.open(master_path)
        master_data = m_file['SCIWAVE'].data
            
        return master_data

    def find_peaks_in_order(self,comb, plot_path=None):
        """
        Runs find_peaks on successive subsections of the comb lines and concatenates
        the output. The difference between adjacent peaks changes as a function
        of position on the detector, so this results in more accurate peak-finding.

        Based on pyreduce.

        Args:
            comb (np.array): flux values. Their indices correspond to
                their pixel numbers. Generally the entire order.
            plot_path (str): Path for diagnostic plots. If None, plots are not made.

        Returns:
            new_peaks (np.array): array of true peak locations as 
                determined by Gaussian fitting
            peaks (np.array): array of detected peak locations (pre-
                Gaussian fitting)
            peak_heights (np.array): array of detected peak heights 
                (pre-Gaussian fitting)
            gauss_coeffs (np.array): array of size (4, n_peaks) 
                containing best-fit Gaussian parameters [a, mu, sigma**2, const]
                for each detected peak
        """
    
        n_pixels = len(comb)
        new_peaks = np.array([])
        peaks = np.array([])
        peak_heights = np.array([])
        gauss_coeffs = np.zeros((4,0))

        for i in np.arange(self.n_sections):

            if i == self.n_sections - 1:
                indices = np.arange(i * n_pixels // self.n_sections, n_pixels)
            else:
                indices = np.arange(i * n_pixels // self.n_sections, (i+1) * n_pixels // self.n_sections)

            new_peaks_section, peaks_section, peak_heights_section, gauss_coeffs_section = self.find_peaks_lfc(comb[indices])

            peak_heights = np.append(peak_heights, peak_heights_section)

            gauss_coeffs = np.append(gauss_coeffs, gauss_coeffs_section, axis=1)

            if i == 0:
                new_peaks = np.append(new_peaks, new_peaks_section)
                peaks = np.append(peaks, peaks_section)

            else:
                new_peaks = np.append(new_peaks, new_peaks_section + i * n_pixels // self.n_sections)
                peaks = np.append(peaks, peaks_section + i * n_pixels // self.n_sections)
        
        if plot_path is not None:
            plt.figure()
            plt.plot(comb, color='k', lw=0.1)   
            plt.scatter(peaks, peak_heights, s=1, color='r')
            plt.savefig('{}/detected_peaks.png'.format(plot_path), dpi=250)
            plt.close()

            n_zoom_sections = 10
            zoom_section_pixels = n_pixels // n_zoom_sections

            _, ax_list = plt.subplots(n_zoom_sections, 1, figsize=(6,12))
            for i, ax in enumerate(ax_list):
                ax.plot(comb,color='k', lw=0.1)
                ax.scatter(peaks,peak_heights,s=1,color='r')
                ax.set_xlim(zoom_section_pixels * i, zoom_section_pixels * (i+1))
                ax.set_ylim(0,np.max(comb[zoom_section_pixels * i : zoom_section_pixels * (i+1)]))

            plt.tight_layout()
            plt.savefig('{}/detected_peaks_zoom.png'.format(plot_path),dpi=250)
            plt.close()

        return new_peaks, peaks, peak_heights, gauss_coeffs

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
            integrated_gaussian_val (float): the integrated value
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
            popt (list): best-fit parameters [a, mu, sigma**2, const]
        """
        x = np.ma.compressed(x)
        y = np.ma.compressed(y)

        i = np.argmax(y[len(y) // 4 : len(y) * 3 // 4]) + len(y) // 4
        p0 = [y[i], x[i], 1, np.min(y)]

        with np.warnings.catch_warnings():
            np.warnings.simplefilter("ignore")
            popt, _ = curve_fit(self.integrate_gaussian, x, y, p0=p0, maxfev=100000)

        return popt

    def find_peaks_thar(self,flux,linelist,line_pixels_expected,plot_toggle,savefig):
        """
        Args:
            flux ([type]): [description]
            linelist ([type]): [description]
            line_pixels_expected ([type]): [description]
            plot_toggle ([type]): [description]
            savefig ([type]): [description]
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

                if peak_pixel < gaussian_fit_width:
                    first_fit_pixel = 0
                else:
                    first_fit_pixel = peak_pixel - gaussian_fit_width
                
                if peak_pixel > num_pixels:
                    last_fit_pixel = num_pixels
                else:
                    last_fit_pixel = peak_pixel + gaussian_fit_width

                #fit gaussian to matched peak location
                coefs[:,i] = LFC_wavelength_cal.fit_gaussian(
                    np.arange(first_fit_pixel,last_fit_pixel),
                    flux[first_fit_pixel:last_fit_pixel])

                amp = coefs[0,i]
                if amp < 0:
                    missed_lines += 1
        
        if plot_toggle == True:
            zoom_section_pixels = (num_pixels // self.n_zoom_sections)
            _, ax_list = plt.subplots(self.n_zoom_sections,1,figsize=(self.subplot_size))
            ax_list[0].set_title('({} missed lines)'.format(missed_lines))
            for i, ax in enumerate(ax_list):
                ax.plot(
                    np.arange(num_pixels[i*zoom_section_pixels:(i+1)*zoom_section_pixels]
                    flux[i*zoom_section_pixels:(i+1)*zoom_section_pixels],color='k',alpha=.1)
                    )

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
                ax.set_yscale('log')

            plt.tight_layout()
            plt.savefig('{}/spectrum_and_peaks.png'.format(savefig), dpi=250)
            plt.close()

        return coefs


    def find_peaks_lfc(self, comb):
        """
        Finds all comb peaks in an array. This runs scipy.signal.find_peaks 
            twice: once to find the average distance between peaks, and once
            for real, disregarding close peaks.

        Args:
            comb (np.array): flux values. Their indices correspond to
                their pixel numbers. Generally a subset of the full order.
            
        Returns:
            new_peaks (np.array): array of true peak locations as 
                determined by Gaussian fitting
            peaks (np.array): array of detected peak locations (pre-
                Gaussian fitting)
            peak_heights (np.array): array of detected peak heights 
                (pre-Gaussian fitting)
            gauss_coeffs (np.array): array of size (4, n_peaks) 
                containing best-fit Gaussian parameters [a, mu, sigma**2, const]
                for each detected peak
        """

        c = comb - np.ma.min(comb)

        # #todo: try to make this more indep of comb flux
        height = 3 * np.ma.median(c) # 0.5 * np.ma.median(c) works for whole chip
        peaks, properties = signal.find_peaks(c, height=height)

        distance = np.median(np.diff(peaks)) // 2
        peaks, properties = signal.find_peaks(c, distance=distance, height=height)
        peak_heights = np.array(properties['peak_heights'])

        # fit peaks with Gaussian to get accurate position
        new_peaks = peaks.astype(float)
        gauss_coeffs = np.empty((4, len(peaks)))
        width = np.mean(np.diff(peaks)) // 2

        for j, p in enumerate(peaks):
            idx = p + np.arange(-width, width + 1, 1)
            idx = np.clip(idx, 0, len(c) - 1).astype(int)
            coef = self.fit_gaussian(np.arange(len(idx)), c[idx])
            gauss_coeffs[:,j] = coef
            new_peaks[j] = coef[1] + p - width

        return new_peaks, peaks, peak_heights, gauss_coeffs

    def clip_peaks(self, comb, new_peaks, peaks, gauss_coeffs, peak_heights, reference_wavecal, comb_lines_angstrom,print_update=False, plot_path=None):
        """
        If fitted peak locations are move than 1 Angstrom from detected locations,
        remove them.

        Args:
            comb (np.array): array of comb data
            new_peaks (np.array): array of true peak locations as 
                determined by Gaussian fitting
            peaks (np.array of float): array of detected peak locations 
                (pre-Gaussian fitting)
            gauss_coeffs (np.array): array of size (4, n_peaks) containing best-fit 
                Gaussian parameters [a, mu, sigma**2, const] for each detected peak            
            peak_heights (np.array): array of detected peak heights (pre-Gaussian fitting)
            reference_wavecal (np.array): array of reference solution data
            comb_lines_angstrom (np.array): theoretical LFC wavelengths
                as computed by fundamental physics (in Angstroms)
            print_update (bool): if True, print how many peaks were clipped
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.

        Returns: 
            good_peak_idx(np.array): indices of surviving peaks
        """
        approx_pixel_size = 0.01
        good_peak_idx =np.where(np.abs(new_peaks - peaks) < 1) [0]
        n_pixels = len(reference_wavecal)

        s = InterpolatedUnivariateSpline(np.arange(n_pixels),reference_wavecal)
        approx_peaks_lambda = s(new_peaks)
        good_peak_idx_modes = []

        for i, lamb in enumerate(approx_peaks_lambda):
            best_mode_idx = (
                np.abs(comb_lines_angstrom - lamb)
            ).argmin()
            if np.abs(comb_lines_angstrom[best_mode_idx] - lamb) < approx_pixel_size:
                good_peak_idx_modes.append(i)
        
        good_peak_idx = np.intersect1d(good_peak_idx_modes,good_peak_idx)

        if print_update:
            print('{} peaks clipped'.format(len(peaks) - len(good_peak_idx)))

        if plot_path is not None:

            n = np.arange(len(new_peaks))

            plt.figure()
            plt.scatter(
                n[good_peak_idx], 
                gauss_coeffs[0,:][good_peak_idx] - peak_heights[good_peak_idx],
                color='k'
            )
            plt.savefig(
                '{}/peak_heights_after_clipping.png'.format(plot_path), dpi=250
            )
            plt.close()

            plt.figure()
            plt.scatter(
                n[good_peak_idx], 
                new_peaks[good_peak_idx] - peaks[good_peak_idx],
                color='k'
            )
            plt.savefig(
                '{}/peak_locs_after_clipping.png'.format(plot_path), dpi=250
            )
            plt.close()

            plt.figure()
            plt.plot(comb, color='k', lw=0.1)   
            plt.scatter(
                peaks[good_peak_idx], peak_heights[good_peak_idx], s=1, color='r'
            )
            plt.scatter(
                np.delete(peaks, good_peak_idx), 
                np.delete(peak_heights, good_peak_idx), s=10, color='k'
            )
            plt.savefig('{}/unclipped_peaks.png'.format(plot_path), dpi=250)
            plt.close()

            n_zoom_sections = 10
            zoom_section_pixels = n_pixels // n_zoom_sections

            _, ax_list = plt.subplots(n_zoom_sections, 1, figsize=(6, 12))
            for i, ax in enumerate(ax_list):
                ax.plot(comb, color='k', lw=0.1)   
                ax.scatter(
                    peaks[good_peak_idx], peak_heights[good_peak_idx], 
                    s=1, color='r'
                )
                ax.scatter(
                    np.delete(peaks, good_peak_idx), 
                    np.delete(peak_heights, good_peak_idx), s=10, color='k'
                )
                ax.set_xlim(
                    zoom_section_pixels * i, zoom_section_pixels * (i + 1)
                )
                ax.set_ylim(
                    0, 
                    np.max(
                        comb[zoom_section_pixels * i : zoom_section_pixels * (i + 1)]
                    )
                )

            plt.tight_layout()
            plt.savefig('{}/unclipped_peaks_zoom.png'.format(plot_path), dpi=250)
            plt.close()
        return good_peak_idx

    def mode_match(self,comb, new_peaks, good_peak_idx, reference_wavecal, 
    comb_lines_angstrom, print_update=False, plot_path=None):
        """
        Matches detected comb peaks to the theoretical locations of LFC wavelengths
        and returns the derived wavelength solution.

        Args:
            comb (np.array of float): flux values for an order. Their indices 
                correspond to their pixel numbers. 
            new_peaks (np.array): array of true detected peak locations as 
                determined by Gaussian fitting.
            good_peak_idx (np.array): indices (of ``new_peaks``) of detected 
                and unclipped peaks
            reference_wavecal (np.array): ThAr-based wavelength solution. Each
                entry in the array is the wavelength (in Angstroms) corresponding
                to a pixel (indicated by its index)
            comb_lines_angstrom (np.array): theoretical LFC wavelengths
                as computed by fundamental physics (in Angstroms)
            print_update (bool): if True, print total number of LFC modes in
                the order that were not detected (n_clipped + n_never_detected)
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.
            
        Returns:
            wls (np.array): the precise wavelengths of detected comb peaks. Each
                entry in the array is the wavelength (in Angstroms) corresponding
                to a pixel (indicated by its index)
            mode_nums (np.array): the mode numbers of the LFC modes to be used for 
                wavelength calibration
        """

        n_pixels = len(comb)

        s = InterpolatedUnivariateSpline(np.arange(n_pixels), reference_wavecal)
        approx_peaks_lambda = s(new_peaks[good_peak_idx])

        # Now figure what mode numbers the peaks correspond to
        n_clipped_peaks = len(new_peaks[good_peak_idx])
        mode_nums = np.empty(n_clipped_peaks)

        peak_mode_num = 0
        for i in range(n_clipped_peaks):

            # calculate difference in peak locs of last several peaks
            num_peaks_kernel = 100
            if (i < num_peaks_kernel):
                running_peak_diff = np.median(
                    np.diff(new_peaks[good_peak_idx][0:num_peaks_kernel])
                )
            else:
                running_peak_diff = np.median(
                    np.diff(new_peaks[good_peak_idx][i-num_peaks_kernel:i])
                )

            if i==0:
                for j in np.arange(15):
                    if new_peaks[good_peak_idx][i] > (j + 1.5) * running_peak_diff:
                        peak_mode_num += 1
                if new_peaks[good_peak_idx][i] > 15.5 * running_peak_diff:
                    assert False, 'More than 15 peaks in a row not detected!'
        
            # if current peak location is greater than (n + 0.5) * sigma of 
            # previous peak diffs, then skip over n modes
            if i > 0:
                for j in np.arange(15):
                    if (
                        new_peaks[good_peak_idx][i] - 
                        new_peaks[good_peak_idx][i - 1] > 
                        (j + 1.5) * running_peak_diff
                    ):
                        peak_mode_num += 1
                if (
                    new_peaks[good_peak_idx][i] - 
                    new_peaks[good_peak_idx][i - 1] > 
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
            plt.plot(reference_wavecal, comb, alpha=0.2)
            plt.vlines(comb_lines_angstrom, ymin=0, ymax=5000, color='r')
            plt.xlim(reference_wavecal[200], reference_wavecal[700])
            plt.xlabel('wavelength [$\\rm \AA$]')
            plt.savefig('{}/thar_sol_and_lfc_lines.png'.format(plot_path), dpi=250)
            plt.close()

            n_zoom_sections = 20
            zoom_section_wavelen = (
                (np.max(reference_wavecal) - np.min(reference_wavecal)) // 
                n_zoom_sections
            )
            zoom_section_pixels = n_pixels // n_zoom_sections

            _, ax_list = plt.subplots(n_zoom_sections, 1, figsize=(6, 20))
            for i, ax in enumerate(ax_list):
                ax.plot(reference_wavecal, comb, color='k', alpha=0.1)
                for mode_num in mode_nums:
                    if (
                        (
                            comb_lines_angstrom[mode_num.astype(int)] > 
                            zoom_section_wavelen * i + np.min(reference_wavecal)
                        ) and (
                            comb_lines_angstrom[mode_num.astype(int)] < 
                            zoom_section_wavelen * (i + 1) + np.min(reference_wavecal)
                        )
                    ):
                        ax.text(
                            comb_lines_angstrom[mode_num.astype(int)], 0, 
                            str(int(mode_num)), fontsize=4
                        )
                ax.set_xlim(
                    zoom_section_wavelen * i + np.min(reference_wavecal), 
                    zoom_section_wavelen * (i + 1) + np.min(reference_wavecal)
                )
                ax.set_ylim(
                    0, 
                    np.max(
                        comb[zoom_section_pixels * i: zoom_section_pixels * (i + 1)]
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

    def fit_polynomial(self, wls, gauss_coeffs, good_peak_idx, n_pixels, new_peaks, plot_path=None):
        """
        Given precise wavelengths of detected LFC comb lines, fits a Legendre 
        polynomial wavelength solution.

        Args:
            wls (np.array): the precise wavelengths of detected comb peaks,
                from fundamental physics.
            gauss_coeffs (np.array): array of size (4, n_peaks) 
                containing best-fit Gaussian parameters [a, mu, sigma**2, const]
                for each detected peak
            good_peak_idx (np.array of int): indices of unclipped peaks
            n_pixels (int): number of pixels in the order
            new_peaks (np.array): array of true detected peak locations as 
                determined by Gaussian fitting.
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.

        Returns:
                our_wavelength_solution_for_order (np.array): calculated wavelength solution for the order (i.e. 
                    wavelength value for each pixel in the order)
                leg_out (func): a Python function that, given an array of pixel locations, 
                    returns the Legendre polynomial wavelength solutions
        """

        # fitted_heights = gauss_coeffs[0,:][good_peak_idx]
        # weights = np.sqrt(fitted_heights)

        leg_out = Legendre.fit(new_peaks[good_peak_idx], wls, self.fit_order)

        our_wavelength_solution_for_order = leg_out(np.arange(n_pixels))

        if plot_path is not None:

            s = InterpolatedUnivariateSpline(new_peaks[good_peak_idx], wls)
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

        return our_wavelength_solution_for_order, leg_out

    def calculate_precision(self, new_peaks, good_peak_idx, wls, leg_out, 
    print_update=True, plot_path=None):
        """
        [summary]

        Args:
            new_peaks (np.array of float): array of true detected peak locations as 
                determined by Gaussian fitting.
            good_peak_idx (np.array of int): indices of unclipped peaks
            wls (np.array of float): the precise wavelengths of detected comb peaks,
                from fundamental physics.
            leg_out (func): a Python function that, given an array of pixel 
                locations, returns the Legendre polynomial wavelength solutions
            print_update (bool): If true, prints standard error per order.
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.

        Returns:
            precision_cm_s (float): RV precision in cm/s
        """
        our_wls_peak_pos = leg_out(new_peaks[good_peak_idx]) 

        residual = ((our_wls_peak_pos - wls) * scipy.constants.c) / wls

        precision_cm_s = 100 * np.std(residual)/np.sqrt(len(good_peak_idx))

        if print_update:
            print('Standard error (this order): {:.2f} cm/s'.format(precision_cm_s))

        if plot_path is not None:
            plt.figure()
            plt.plot(residual)
            plt.xlabel('pixel')
            plt.ylabel('error [m/s]')
            plt.savefig('{}/rv_precision.png'.format(plot_path), dpi=250)
            plt.close()

        return precision_cm_s

    def mask_array_neid(self,flux,n_orders):
        """Creates mask to remove bad pixel regions specific to order. For NEID testing.

        Args:
            flux (np.array): [description]
            n_orders (np.array): [description]

        Returns:
            flux [type]: [description]
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
            flux[i + self.min_order, j:k] = 0

        # orders 75 & 86 have some additional weird stuff going on
        flux[75, 1930:1940] = 0
        flux[75, 6360:6366] = 0
        flux[86, 1930:1940] = 0
        
        return flux

    def fit_many_orders(self, comb_all, reference_wavecal_all, comb_lines_angstrom, order_list, plt_path=None, print_update=False):
        """
        Iteratively performs LFC wavelength calibration for all orders.

        Args:
            comb_all (np.array): (n_orders x n_pixels) array of LFC fluxes
            reference_wavecal_all (np.array): (n_orders x n_pixels) array of    
                reference wavelength solution values for each pixel on the detector
            comb_lines_angstrom (np.array): theoretical LFC wavelengths
                as computed by fundamental physics (in Angstroms)
            order_list (list): Order list
            plt_path (str): if set, all diagnostic plots will be saved in this
                directory. If None, no plots will be made.
            print_update (bool): whether subfunctions should print updates.

        Returns:
            dict: the LFC mode numbers used for wavelength cal. Keys are ints
                representing order numbers, values are 2-tuples of:
                    - lists of mode numbers
                    - the corresponding pixels on which the LFC mode maxima fall

            poly_soln_final_array (np.array): Polynomial solution for each tested order
            thars (np.array): ThAr solutions for each tested order
        """    
        # 2D extracted spectra
        if plt_path is not None:
            plt.figure(figsize=(20,10))
            plt.title('LFC Spectra')
            im = plt.imshow(comb_all, aspect='auto')
            im.set_clim(0, 20000)
            plt.xlabel('pixel')
            plt.ylabel('order number')
            plt.savefig('{}/extracted_spectra.png'.format(plt_path), dpi=250)
            plt.close()

        order_precisions = []
        num_detected_peaks = []
        poly_solns = []
        references = []
        modenums_and_pixels = {}

        #print("orderlist",order_list)
        total_order_no = np.shape(comb_all)
        #total_order_list = np.arange(total_order_no[0])
        poly_soln_final_array = np.zeros(np.shape(comb_all))
        references = np.zeros(np.shape(comb_all))

        for order_num in order_list:
            if print_update:
                print('\nRunning order # {}'.format(order_num))

            if plt_path is not None:
                order_plt_path = '{}/order{}'.format(plt_path, order_num)
                if not os.path.isdir(order_plt_path):
                    os.makedirs(order_plt_path)

                plt.figure(figsize=(20,10))
                plt.plot(comb_all[order_num,:], color='k', alpha=0.5)
                plt.title('Order # {}'.format(order_num))
                plt.xlabel('pixel')
                plt.ylabel('flux')
                plt.savefig('{}/order_spectrum.png'.format(order_plt_path), dpi=250)
                plt.close()
            else:
                order_plt_path = None

            comb = comb_all[order_num,:]
            reference_wavecal = reference_wavecal_all[order_num,:]
            n_pixels = len(comb)

            # calculate, clip, and mode-match peaks
            new_peaks, peaks, peak_heights, gauss_coeffs = self.find_peaks_in_order(
                comb, plot_path=order_plt_path
            )
            #if self.clip_peaks_opt == True:
            good_peak_idx = self.clip_peaks(
                comb, new_peaks, peaks, gauss_coeffs, peak_heights,reference_wavecal,
                comb_lines_angstrom,plot_path=order_plt_path, print_update=print_update
            )
            wls, lfc_modes = self.mode_match(
                comb, new_peaks, good_peak_idx, reference_wavecal, comb_lines_angstrom, 
                print_update=print_update, plot_path=order_plt_path
            )
            # calculate the wavelength solution for the order
            polynomial_wls, leg_out = self.fit_polynomial(
                wls, gauss_coeffs, good_peak_idx, n_pixels, new_peaks, 
                plot_path=order_plt_path
            )
            poly_soln_final_array[order_num,:] = polynomial_wls
            #poly_solns.append(polynomial_wls)
            if plt_path is not None:
                plt.figure(figsize=(12,5))
                plt.plot(
                    np.arange(n_pixels), 
                    leg_out(np.arange(n_pixels)) - reference_wavecal, 
                    color='k'
                )
                plt.xlabel('pixel')
                plt.ylabel('Our LFC WLS - ThAr WLS [$\\rm \AA$]')
                plt.savefig(
                    '{}/lfc_wls_sub_thar.png'.format(order_plt_path),
                    dpi=250
                )
                plt.tight_layout()
                plt.close()

            # compute RV precision for order
            print(poly_soln_final_array[order_num])
            precision = self.calculate_rv_precision(
                new_peaks, good_peak_idx, wls, leg_out, plot_path=order_plt_path, 
                print_update=print_update
            )

            order_precisions.append(precision)
            num_detected_peaks.append(len(good_peak_idx))

            modenums_and_pixels[order_num] = (lfc_modes, np.floor(new_peaks))

        squared_resids = (np.array(order_precisions) * num_detected_peaks)**2
        sum_of_squared_resids = np.sum(squared_resids)
        overall_std_error = (
            np.sqrt(sum_of_squared_resids) / 
            np.sum(num_detected_peaks)
        )
        print('Overall precision: {:2.2f} cm/s'.format(overall_std_error))

        return poly_soln_final_array,references

    def comb_gen(self, f0, f_rep):
        """Generates comb lines for mapping flux.

        Args:
            f0 (float): Initial comb frequency
            f_rep (float): Comb repitition frequency

        Returns:
            comb_lines_ang (np.array): Array of comb lines, in Angstroms.
        """
        mode_start=np.int((((scipy.constants.c*1e10)/self.min_wave)-f0)/f_rep)
        mode_end=np.int((((scipy.constants.c*1e10)/self.max_wave)-f0)/f_rep)
        mode_nos=np.arange(mode_start,mode_end,-1)

        fxn=f0+(mode_nos*f_rep)
        ln=scipy.constants.c/fxn
        comb_lines_ang=ln/(1e-10)

        return comb_lines_ang

    def open_and_run_etalon(self, flux, quicklook):
        """Runs all Etalon Wavecal alg steps in order.

        Args:
            flux ([type]): [description]
            quicklook ([type]): [description]
        """
        new_peaks,peaks,peak_heights,gauss_coeffs = self.find_peaks_in_order(
            flux[order],self.height_sigma, plot_path=plot_dir))

    def open_and_run_thar(self,flux,redman_w,redman_i,linelist_sub,other_wls,plot_toggle,quicklook):
        """[summary]

        Args:
            flux ([type]): [description]
            redman_w ([type]): [description]
            redman_i ([type]): [description]
            linelist_sub ([type]): [description]
            other_wls ([type]): [description]
            plot_toggle ([type]): [description]
            quicklook ([type]): [description]
        """
        if plot_toggle == True:
            _, summary_ax = plt.subplots()
            summary_ax.set_xlabel('pixel')
            summary_ax.set_ylabel('Our WLS - Their WLS [m/s]')
            summary_ax.set_ylim(-500, 500)

        num_pixels = len(flux[0])

        for order_num in np.arange(first_order,last_order+1):
            print('\nRunning Order {}!'.format(order_num))

            if plot_toggle == True:
                order_plt_path = '{}/order{}'.format(self.saveplots,order_num)
                if not os.path.exists(order_plt_path):
                    os.mkdir(order_plt_path)

            min_order_wl = np.min(linelist_sub[order_num]['known_wavelengths_vac'])
            max_order_wl = np.max(linelist_sub[order_num]['known_wavelengths_vac'])
            in_order_mask = ((redman_w > min_order_wl) & (redman_w < max_order_wl))

            if plot_toggle == True:
                _, ax = plt.subplots(figsize=(12,6))
                plt.plot(
                    redman_w[in_order_mask], 
                    redman_i[in_order_mask],
                    color='purple'
                )
                ax.vlines(
                    linelist_sub[order_num]['known_wavelengths_vac'], 
                    ymin=0, 
                    ymax=0.5 * np.max(redman_i[in_order_mask]),
                    color='grey', alpha=0.5
                )
                plt.title('Order {} Linelist'.format(order_num))
                plt.xlabel('wavelength [$\\rm \AA$]')
                plt.ylabel('intensity')
                plt.savefig('{}/linelist.png'.format(order_plt_path), dpi=250)
                plt.close()

                # plot the full spectrum
                plt.figure(figsize=(20,10))
                plt.title('ThAr Spectra')
                im = plt.imshow(flux, aspect='auto')
                im.set_clim(0, 20000)
                plt.xlabel('pixel')
                plt.ylabel('order number')
                plt.savefig('{}/extracted_spectra.png'.format(SAVEPLOTS), dpi=250)
                plt.close()

                # plot the spectrum for the order of interest
                plt.figure()
                plt.plot(flux[order_num])  
                plt.title('Order {} Spectrum'.format(order_num))
                plt.xlabel('pixel')
                plt.ylabel('flux')
                plt.tight_layout()
                plt.savefig('{}/extracted_spectra.png'.format(order_plt_path), dpi=250)
                plt.close()
            
            gauss_fit_coefs = self.find_and_fit_peaks(
                flux[order_num],
                linelist_sub[order_num]['known_wavelengths_vac'],
                linelist_sub[order_num]['line_positions'],
                order_plt_path
            )

            wls = self.fit_polynomial(
                gauss_fit_coefs[1,:],
                linelist_sub[order_num]['known_wavelengths_vac'],
                len(flux[order_num])
            )

            num_lines_fit = len(gauss_fit_coefs[1,:][gauss_fit_coefs[1,:] > 0])
            precision,residuals = self.calculate_precision(wls,other_wls[order_num],
                num_lines_fit,order_plt_path)

            print('Order {} Precision: {:.2f} m/s.'.format(order_num, precision))

            if plot_toggle == True:
                summary_ax.plot(
                    np.arange(self.end_pixels_to_clip, num_pixels - self.end_pixels_to clip),
                    residuals[self.end_pixels_to_clip:num_pixels - self.end_pixels_to_clip]
                )
        
        if plot_toggle == True:
            plt.savefig('{}/wls_comp.png'.format(self.saveplots), dpi=250)
            plt.close()

    def open_and_run_LFC(self, flux, master_data, f0, f_rep, quicklook):
        """Runs all LFC Wavecal alg steps in order.

        Args:
            flux (np.array): [description]
            master_data (np.array): [description]
            f0 (float): Initial comb frequency
            f_rep (float): Comb repition frequency
            quicklook (bool): Whether or not to run quicklook pipeline
            wavecal_type (str): Choice between ThAr, Etalon, and LFC

        Returns:
            poly_soln (np.array): [description]
        """

        if quicklook == False:
            if type(self.save_diagnostics) == str:
                SAVEPLOTS = ('{}/%s' % self.save_diagnostics).format(os.getcwd())
                if not os.path.isdir(SAVEPLOTS):
                    os.makedirs(SAVEPLOTS)
            if self.save_diagnostics == 'False':
                SAVEPLOTS = None

            cl_ang = self.comb_gen(f0, f_rep)
            order_list = self.remove_orders(self.lfc_min_order,self.lfc_max_order,self.lfc_skip_orders,step=1)
            n_orders = len(order_list)
            new_flux = self.mask_array_neid(flux,n_orders)

            # perform wavelength calibration
            poly_soln,thars = self.fit_many_orders(new_flux, master_data, cl_ang, order_list, print_update=True, plt_path=SAVEPLOTS)

            return poly_soln
    
        if quicklook == True:
            if type(self.save_diagnostics) == str:
                SAVEPLOTS = ('{}/%s' % self.save_diagnostics).format(os.getcwd())
                if not os.path.isdir(SAVEPLOTS):
                    os.makedirs(SAVEPLOTS)
            if self.save_diagnostics == False:
                SAVEPLOTS = None

            cl_ang = self.comb_gen(f0, f_rep)
            order_list = self.remove_orders(self.lfc_min_order,self.lfc_max_order,self.lfc_skip_orders,step=self.quicklook_ord_steps)
            n_orders = len(order_list)
            new_flux = self.mask_array_neid(flux,n_orders)
            # perform wavelength calibration
            poly_soln,thars = self.fit_many_orders(new_flux, master_data, cl_ang, order_list, print_update=True, plt_path=SAVEPLOTS)

            ##### #draft of quicklook drift plot
            # def plot_drift(thars,poly_soln,order_list):
            #     dev = np.array(thars) - np.array(poly_soln)
            #     mean = np.mean(dev,1)
            #     if np.max(mean) > .01:
            #         plt.axhspan(.01,np.max(mean), facecolor='grey', alpha=0.5,label='Warning: Large deviation region')
            #     if np.max(mean) < .01:
            #         plt.axhspan(.01,.012,facecolor='grey',alpha=.5,label='Warning: Large deviation region')

            #     if np.min(mean) < -.01:
            #         plt.axhspan(-.01,np.min(mean),facecolor='grey',alpha=.5,label='Warning: Large deviation region')
            #     if np.min(mean) > -.01:
            #         plt.axhspan(-.01,-.012,facecolor='grey',alpha=.5,label='Warning: Large deviation region')

            #     #print(order_list,mean,np.shape(order_list),np.shape(mean))
            #     plt.scatter(order_list,mean,marker='.',color='orange')
            #     plt.xlabel('Order')
            #     plt.ylabel('Drift (?km/s)')
            #     h, l = plt.gca().get_legend_handles_labels()
            #     newh=[h[0]]
            #     newl=[l[0]]
            #     plt.legend(newh,newl)
            #     plt.savefig('master_soln_deviation.pdf')
            #     plt.close()
            
            # print(np.shape(master_data),np.shape(poly_soln),np.shape(thars))
            # plot_drift(master_data,poly_soln)
            #####

            def plot_orders(wav,flux):
                from matplotlib import gridspec
                n = 10 # number of sub panels
                m = 6 #number of panel per panel
                gs = gridspec.GridSpec(n, 1, height_ratios=np.ones(n))

                plt.rcParams.update({'font.size': 8})
                #fig = plt.figure(figsize=(6, 12))
                fig, ax = plt.subplots(n,1, sharey=False,figsize=(24,18))

                plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1)
                fig.subplots_adjust(hspace=0.3)

                for i in range(55,115,1):#np.shape(wav)[0]
                    low, high = np.nanpercentile(flux[i,:],[0.5,99.5])
                    flux[i,:][(flux[i,:]>high) | (flux[i,:]<low)] = np.nan
                    j = int((i-55)/m)
                    ax[j].plot(wav[i,:],flux[i,:], linewidth =  0.1)

                #for j in range(n):
                    #low, high = np.nanpercentile(flux[j*m:(j+1)*m,:],[.5,99.5])
                    #print(j,high*1.5)
                    #ax[j].set_ylim(-high*0.1, high*1.5)

                low, high = np.nanpercentile(flux,[0.5,99.5])
                #plt.ylim(-high*0.1, high*1.2)
                ax[int(n/2)].set_ylabel('Counts',fontsize = 20)
                plt.xlabel('Wavelength (Ang)',fontsize = 20)
                plt.savefig('quicklook_LFC_orderplot.pdf') #name should have pre/suffix with og file number
                plt.close()

            plot_orders(master_data,flux)
            #####
            return poly_soln