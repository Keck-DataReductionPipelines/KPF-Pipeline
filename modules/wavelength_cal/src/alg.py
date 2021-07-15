import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy
import os
from scipy import signal
from scipy.signal import find_peaks as peak
from scipy.optimize import curve_fit as cv
from scipy.special import erf
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
            n_sections (np.int): Number of sections to divide the comb into. Pulled from config file.
        """
        configpull=ConfigHandler(config,'PARAM')
        self.f0=configpull.get_config_value('f0',-5e9)
        self.f_rep=configpull.get_config_value('f_rep',20e9)
        self.max_wave=configpull.get_config_value('max_wave',9300)
        self.min_wave=configpull.get_config_value('min_wave',3800)
        self.fit_order=configpull.get_config_value('fit_order',9)
        self.min_order=configpull.get_config_value('min_order',50)
        self.max_order=configpull.get_config_value('max_order',100)
        self.n_sections=configpull.get_config_value('n_sections',20)
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

    def find_peaks_in_order(self,comb, plot_path=None):
        """
        Runs find_peaks on successive subsections of the comb lines and concatenates
        the output. The difference between adjacent peaks changes as a function
        of position on the detector, so this results in more accurate peak-finding.

        Based on pyreduce.

        Args:
            comb (np.array of float): flux values. Their indices correspond to
                their pixel numbers. Generally the entire order.
            n_sections (int): number of sections to split the comb into
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.

        Returns:
            tuple of:
                new_peaks (np.array of float): array of true peak locations as 
                    determined by Gaussian fitting
                peaks (np.array of float): array of detected peak locations (pre-
                    Gaussian fitting)
                peak_heights (np.array of float): array of detected peak heights 
                    (pre-Gaussian fitting)
                gauss_coeffs (np.array of float): array of size (4, n_peaks) 
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

            new_peaks_section, peaks_section, peak_heights_section, gauss_coeffs_section = self.find_peaks(comb[indices])

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
            plt.savefig('{}/detected_peaks.png'.format(plot_path))
            plt.close()

            plt.figure()
            plt.plot(comb, color='k', lw=0.1)   
            plt.scatter(peaks, peak_heights, s=1, color='r')
            plt.xlim(0,1000)

            plt.savefig('{}/detected_peaks_zoom.png'.format(plot_path), dpi=250)
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
            float: the integrated value
        """

        integrated_gaussian_val = a * 0.5 * (erf((x - mu + int_width) / (np.sqrt(2) * sig)) - erf((x - mu - int_width) / (np.sqrt(2) * sig))) + (const * 2 * int_width)
        
        return integrated_gaussian_val

    def fit_gaussian(self, x, y):
        """
        Fits a continuous Gaussian to a discrete set of x and y datapoints
        using scipy.curve_fit
        
        Args:
            x (np.array of float): x data to be fit
            y (np.array of float): y data to be fit

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

    def find_peaks(self, comb):
        """
        Finds all comb peaks in an array. This runs scipy.signal.find_peaks 
            twice: once to find the average distance between peaks, and once
            for real, disregarding close peaks.

        Args:
            comb (np.array of float): flux values. Their indices correspond to
                their pixel numbers. Generally a subset of the full order.
            
        Returns:
            tuple of:
                new_peaks (np.array of float): array of true peak locations as 
                    determined by Gaussian fitting
                peaks (np.array of float): array of detected peak locations (pre-
                    Gaussian fitting)
                peak_heights (np.array of float): array of detected peak heights 
                    (pre-Gaussian fitting)
                gauss_coeffs (np.array of float): array of size (4, n_peaks) 
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

    def clip_peaks(self, new_peaks, peaks, gauss_coeffs, peak_heights, print_update=False, plot_path=None):
        """
        If fitted peak locations are move than 1 Angstrom from detected locations,
        remove them.

        Args:
            new_peaks (np.array of float): array of true peak locations as 
                determined by Gaussian fitting
            peaks (np.array of float): array of detected peak locations (pre-
                Gaussian fitting)
            peak_heights (np.array of float): array of detected peak heights 
                (pre-Gaussian fitting)
            gauss_coeffs (np.array of float): array of size (4, n_peaks) 
                containing best-fit Gaussian parameters [a, mu, sigma**2, const]
                for each detected peak
            print_update (bool): if True, print how many peaks were clipped
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.

        Returns: 
            np.array of int: indices of surviving peaks
        """
        good_peak_idx =np.where(np.abs(new_peaks - peaks) < 1)[0]

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

        return good_peak_idx

    def generate_comb_wavelengths(self, mode_numbers):
        """ 
        Compute theoretical wavelengths for the LFC using the frequency comb
        equation:
        
        f_n = f_0 + n * f_r

        Based on NEID pipeline.
        
        Args:
            mode_numbers (np.array of int): mode numbers at which wavelengths are
                desires

        Returns:
            np.array of float: comb wavelengths [Ang]
        """

        frequencies = self.f0 + self.f_rep * mode_numbers
        wavelengths_vacuum = scipy.constants.c / frequencies
        wavelengths_vac_angstrom = wavelengths_vacuum / 1e-10

        return wavelengths_vac_angstrom

    def mode_match(self,comb, new_peaks, good_peak_idx, thar_wavecal, 
    comb_lines_angstrom, print_update=False):
        """
        Matches detected comb peaks to the theoretical locations of LFC wavelengths
        and returns the derived wavelength solution.

        Args:
            comb (np.array of float): flux values for an order. Their indices 
            correspond to their pixel numbers. 
            new_peaks (np.array of float): array of true detected peak locations as 
                determined by Gaussian fitting.
            good_peak_idx (np.array of int): indices (of ``new_peaks``) of detected 
                and unclipped peaks
            thar_wavecal (np.array of float): ThAr-based wavelength solution. Each
                entry in the array is the wavelength (in Angstroms) corresponding
                to a pixel (indicated by its index)
            comb_lines_angstrom (np.array of float): theoretical LFC wavelengths
                as computed by fundamental physics (in Angstroms)
            print_update (bool): if True, print total number of LFC modes in
                the order that were not detected (n_clipped + n_never_detected)
            
        Returns:
            tuple of:
                np.array of float: the precise wavelengths of detected comb peaks. Each
                    entry in the array is the wavelength (in Angstroms) corresponding
                    to a pixel (indicated by its index)
                np.array of int: the mode numbers of the LFC modes to be used for 
                    wavelength calibration
        """

        n_pixels = len(comb)

        approx_peaks_lambda = np.interp(
            new_peaks[good_peak_idx], np.arange(n_pixels), thar_wavecal
        )

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
                    assert False
        
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
                    assert False

            # set mode_nums
            mode_nums[i] = peak_mode_num
            peak_mode_num += 1

        idx = (np.abs(comb_lines_angstrom - 
            approx_peaks_lambda[len(approx_peaks_lambda) // 2])).argmin()

        n_skipped_modes_in_chip_first_half = mode_nums[
            (len(approx_peaks_lambda) // 2)] - (len(approx_peaks_lambda) // 2)
        mode_nums += (idx - (len(approx_peaks_lambda) // 2) - 
            n_skipped_modes_in_chip_first_half)
        
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
            wls (np.array of float): the precise wavelengths of detected comb peaks,
                from fundamental physics.
            gauss_coeffs (np.array of float): array of size (4, n_peaks) 
                containing best-fit Gaussian parameters [a, mu, sigma**2, const]
                for each detected peak
            good_peak_idx (np.array of int): indices of unclipped peaks
            n_pixels (int): number of pixels in the order
            new_peaks (np.array of float): array of true detected peak locations as 
                determined by Gaussian fitting.
            print_update (bool): if True, print the RV precision.
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.

        Returns:
            tuple of:
                np.array of float: calculated wavelength solution for the order (i.e. 
                    wavelength value for each pixel in the order)
                func: a Python function that, given an array of pixel locations, 
                    returns the Legendre polynomial wavelength solutions
        """

        # fitted_heights = gauss_coeffs[0,:][good_peak_idx]
        # weights = np.sqrt(fitted_heights)

        leg_out = Legendre.fit(
            new_peaks[good_peak_idx], wls, self.fit_order
        )#, w=weights)

        our_wavelength_solution_for_order = leg_out(np.arange(n_pixels))

        if plot_path is not None:

            # interpolate and plot difference
            interpolated_ground_truth = np.interp(
                np.arange(n_pixels), new_peaks[good_peak_idx], wls
            )

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

    def calculate_rv_precision( self, new_peaks, good_peak_idx, wls, leg_out, 
    print_update=True, plot_path=None):
        """
        Calculates the difference between the LFC modes of detected peaks from
        fundamental physics and the polynomial WLS for a given order.

        Args:
            new_peaks (np.array of float): array of true detected peak locations as 
                determined by Gaussian fitting.
            good_peak_idx (np.array of int): indices of unclipped peaks
            wls (np.array of float): the precise wavelengths of detected comb peaks,
                from fundamental physics.
            leg_out (func): a Python function that, given an array of pixel 
                locations, returns the Legendre polynomial wavelength solutions
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.

        Returns:
            np.float: RV precision in cm/s
        """
        our_wls_peak_pos = leg_out(new_peaks[good_peak_idx]) 

        residual = ((our_wls_peak_pos - wls) * scipy.constants.c) / wls

        precision_cm_s = 100 * np.std(residual)/np.sqrt(len(good_peak_idx))

        if print_update:
            print('Standard error (this order): {:.2f} cm/s'.format(precision_cm_s))

        if plot_path is not None:
            plt.figure()
            plt.plot(residual)
            plt.xlabel('pixel]')
            plt.ylabel('error [m/s]')
            plt.savefig('{}/rv_precision.png'.format(plot_path), dpi=250)
            plt.close()

        return precision_cm_s

    def mask_array_neid(self, calflux,n_orders):
    
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

    def fit_many_orders(self, comb_all, thar_wavecal_all, comb_lines_angstrom, plt_path=None, print_update=False):
        """
        Iteratively performs LFC wavelength calibration for all orders.

        Args:
            comb_all (np.array of floar): (n_orders x n_pixels) array of LFC fluxes
            thar_wavecal_all (np.array of float): (n_orders x n_pixels) array of    
                ThAr-derived wavelength soluton values for each pixel on the 
                detector
            plt_path (str): if set, all diagnostic plots will be saved in this
                directory. If None, no plots will be made.
            print_update (bool): whether subfunctions should print updates.

        Returns:
            dict: the LFC mode numbers used for wavelength cal. Keys are ints
                representing order numbers, values are 2-tuples of:
                    lists of mode numbers
                    the corresponding pixels on which the LFC mode maxima fall
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

        modenums_and_pixels = {}
        for order_num in np.arange(self.min_order, self.max_order):

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
            thar_wavecal = thar_wavecal_all[order_num,:]
            n_pixels = len(comb)

            # calculate, clip, and mode-match peaks
            new_peaks, peaks, peak_heights, gauss_coeffs = self.find_peaks_in_order(
                comb, plot_path=order_plt_path
            )
            good_peak_idx = self.clip_peaks(
                new_peaks, peaks, gauss_coeffs, peak_heights, 
                plot_path=order_plt_path, print_update=print_update
            )
            wls, lfc_modes = self.mode_match(
                comb, new_peaks, good_peak_idx, thar_wavecal, comb_lines_angstrom, 
                print_update=print_update
            )

            # calculate the wavelength solution for the order
            polynomial_wls, leg_out = self.fit_polynomial(
                wls, gauss_coeffs, good_peak_idx, n_pixels, new_peaks, 
                plot_path=order_plt_path
            )

            if plt_path is not None:
                plt.plot(
                    np.arange(n_pixels), 
                    leg_out(np.arange(n_pixels)) - thar_wavecal, 
                    color='k'
                )
                plt.xlabel('pixel')
                plt.ylabel('Our LFC WLS - ThAr WLS [$\\rm \AA$]')
                plt.savefig(
                    '{}/lfc_wls_sub_thar.png'.format(order_plt_path),
                    dpi=250
                )
                plt.close()

            # compute RV precision for order
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

        return modenums_and_pixels

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

    def open_and_run(self, calflux, master_data):
    
        SAVEPLTS = '{}/diagnostic_plots'.format(os.getcwd())
        if not os.path.isdir(SAVEPLTS):
            os.makedirs(SAVEPLTS)

        # open data file
        # hdul = fits.open(file)
        # calflux = hdul['CALFLUX'].data

        cl_ang = self.comb_gen()
        
        n_orders = self.max_order - self.min_order + 1
        new_calflux = self.mask_array_neid(calflux,n_orders)
        
        # perform wavelength calibration
        modenums_and_pixels = self.fit_many_orders(new_calflux, master_data, 
        cl_ang, print_update=True, plt_path=SAVEPLTS)