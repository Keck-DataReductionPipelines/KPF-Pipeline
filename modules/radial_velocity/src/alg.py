from astropy.io import fits
import numpy as np
from astropy.modeling import models, fitting
from astropy import constants as const
import warnings
import datetime
import pandas as pd
import math
from modules.radial_velocity.src.alg_rv_init import RadialVelocityAlgInit
from modules.radial_velocity.src.alg_rv_base import RadialVelocityBase
from modules.barycentric_correction.src.alg_barycentric_corr import BarycentricCorrectionAlg

LIGHT_SPEED = const.c.to('km/s').value  # light speed in km/s
LIGHT_SPEED_M = const.c.value  # light speed in m/s
SEC_TO_JD = 1.0 / 86400.0
FIT_G = fitting.LevMarLSQFitter()


class RadialVelocityAlg(RadialVelocityBase):
    """Radial velocity calculation using cross correlation method.

    This module defines class 'RadialVelocityAlg' and methods to perform radial velocity calculation by using
    cross correlation to convert level 1 data to level 2 data.

    Args:
        spectrum_data (numpy.ndarray): 2D data containing reduced 1D Data for all orders from optimal extraction.
        header (fits.header.Header): Header of HDU associated with `spectrum_data`.
        init_rv (dict): A dict instance, created by ``RadialVelocityAlgInit``, containing the init values
            based on the settings in the configuration file for radial velocity computation.
        wave_cal (numpy.ndarray): Wavelength calibration for each order of `spectrum_data`.
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger.

    Attributes:
        spectrum_data (numpy.ndarray): From parameter `spectrum_data`.
        wave_cal (numpy.ndarray): From parameter `wave_cal`.
        header (fits.header.Header): From parameter `header`.
        rv_config (dict): A dict instance, created by ``RadialVelocityAlgInit``,  containing the values defined in
            radial velocity related configuration.
        init_data (dict): A dict instance, created by ``RadialVelocityAlgInit``,  containing the values defined in and
            derived from radial velocity related configuration.
        velocity_loop (numpy.ndarray): Velocity steps for cross correlation computation.
        velocity_steps (int): Total number in `velocity_loop`.
        mask_line (dict): Mask line information created by ``RadialVelocityMaskLine``.
        obj_jd (float): Observation time of Julian Date format.
        start_order (int): First order to be processed.
        end_order (int): Last order to be processed.
        spectrum_order (int): Total spectrum order to be processed.
        start_x_pos (int): First position on x axis to be processed.
        end_x_pos (int): Last position on x axis to be processed.
        spectro (str): Name of instrument or spectrograph.
        reweighting_ccf_method (str): Method of reweighting ccf orders. Method of `ratio` or `ccf` is to scale ccf
            of an order based on a number from a ratio table or the mean of the ratio of a template ccf and current
            ccf of the same order.

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
        TypeError: If there is type error for `spectrum_data`.
        TypeEoor: If there is type error for `header`.
        TypeError: If there is type error for `wave_cal`.
        Exception: If there is init error for radial velocity calculation.

    """
    ROWS_FOR_ANALYSIS = 3
    """int: Extra rows added to the 2D result in which each row contains the cross correlation result for one order. 
    
    The first extra row is left blank.
    The second extra row contains the velocity steps. 
    The third extra row contains the summation of cross correlation results over all orders.
    """

    def __init__(self, spectrum_data, header, init_rv, wave_cal=None, config=None, logger=None):

        if not isinstance(spectrum_data, np.ndarray):
            raise TypeError('results of optimal extraction type error')
        if header is None:
            raise TypeError('data header type error')
        if wave_cal is not None and not isinstance(wave_cal, np.ndarray):
            raise TypeError('wave calibration data type error')
        if 'data' not in init_rv or not init_rv['status']:
            raise Exception('radial velocity init error: '+init_rv['msg'])

        init_data = init_rv['data']

        RadialVelocityBase.__init__(self, config, logger)
        self.spectrum_data = spectrum_data
        self.wave_cal = wave_cal
        self.header = header
        self.init_data = init_data

        # ra, dec, pm_ra, pm_dec, parallax, def_mask, obslon, obslan, obsalt, star_rv, step
        # air_to_vacuum, step_range, mask_width
        self.rv_config = init_data[RadialVelocityAlgInit.RV_CONFIG]
        self.velocity_loop = init_data[RadialVelocityAlgInit.VELOCITY_LOOP]    # loop of velocities for rv finding
        self.velocity_steps = init_data[RadialVelocityAlgInit.VELOCITY_STEPS]  # total steps in velocity_loop
        self.mask_line = init_data[RadialVelocityAlgInit.MASK_LINE]       # def_mask,
        self.reweighting_ccf_method = init_data[RadialVelocityAlgInit.REWEIGHTING_CCF]

        self.obs_jd = None
        ny, nx = np.shape(self.spectrum_data)

        self.start_order = 0
        self.end_order = ny-1
        self.spectrum_order = ny
        self.start_x_pos = 0
        self.end_x_pos = nx-1
        self.spectro = self.rv_config[RadialVelocityAlgInit.SPEC].lower() \
            if RadialVelocityAlgInit.SPEC in self.rv_config else 'neid'

    def get_spectrum(self):
        """Get spectrum information.

        Returns:
            tuple: Spectrum data and the dimension.

                * (*numpy.ndarray*): Spectrum data 2d array
                * **nx** (*int*): Horizontal size.
                * **ny** (*int*): Vertical size.

        """
        ny, nx = np.shape(self.spectrum_data)
        return self.spectrum_data, nx, ny

    def set_order_range(self, lower_order=None, upper_order=None):
        """Set the order range for radial velocity calculation.

        Args:
            lower_order (int, optional): Start order to be processed. Defaults to None, meaning no change.
            upper_order (int, optional): End order to be processed. Defaults to None, meaning no change.

        """

        _, _, ny = self.get_spectrum()

        if lower_order is not None:
            self.start_order = lower_order if lower_order >= 0 else (ny + lower_order)
        if upper_order is not None:
            self.end_order = upper_order if upper_order >= 0 else (ny + upper_order)

        if self.end_order < self.start_order:
            self.start_order, self.end_order = self.end_order, self.start_order

        self.spectrum_order = self.end_order - self.start_order + 1

    def set_x_range(self, x1=None, x2=None):
        """Set the x range for radial velocity calculation.

        Args:
            x1 (int, optional): Start x position. Defaults to None, meaning no change.
            x2 (int, optional): End x position. Defaults to None, meaning no change.

        """

        _, nx, _ = self.get_spectrum()

        if x1 is not None:
            self.start_x_pos = x1 if x1 >= 0 else (nx + x1)

        if x2 is not None:
            self.end_x_pos = x2 if x2 >= 0 else (nx + x2)

        if self.end_x_pos < self.start_x_pos:
            self.end_x_pos, self.start_x_pos = self.start_x_pos, self.end_x_pos

    def get_obs_time(self, default=None):
        """Get Observation time in Julian Date format.

        Args:
            default (float, optional): Default observation time. Defaults to None.

        Returns:
            float: Observation time in Julian Date format.

        """
        if self.spectro == 'neid':
            return self.get_obs_time_neid(default=default)
        elif self.spectro == 'harps':
            return self.get_obs_time_harps(default=default)
        else:
            return default

    def get_obs_time_neid(self, default=None):
        if 'SSBJD100' in self.header:  # jd format
            obs_time = float(self.header['SSBJD100'])
        else:
            obs_time = default

        return obs_time

    def get_obs_time_harps(self, default=None):
        if 'MJD-OBS' in self.header and 'EXPTIME' in self.header:
            obs_time = self.header['MJD-OBS'] + 2400000.5 + self.header['EXPTIME'] * SEC_TO_JD / 2
        else:
            obs_time = default

        return obs_time

    def get_redshift(self, default=None):
        """Get redshift value.

        Args:
            default (float, optional): Default redshift value. Defaults to None.

        Returns:
            float: redshift at observation time.

        """
        if self.spectro == 'neid' and 'SSBZ100' in self.header:
            """
            # recompute zb and compare the value in header
            rv_config_bc_key = [RadialVelocityAlgInit.RA, RadialVelocityAlgInit.DEC,
                                RadialVelocityAlgInit.PMRA, RadialVelocityAlgInit.PMDEC,
                                RadialVelocityAlgInit.PARALLAX, RadialVelocityAlgInit.OBSLAT,
                                RadialVelocityAlgInit.OBSLON,
                                RadialVelocityAlgInit.OBSALT, RadialVelocityAlgInit.STAR_RV, 
                                RadialVelocityAlgInit.SPEC]
            rv_config_bc = {k: self.rv_config[k] for k in rv_config_bc_key}
            obs_time_jd = self.get_obs_time()
            bc_corr = BarycentricCorrectionAlg.get_zb_from_bc_corr(rv_config_bc, obs_time_jd)
            tmp_zb_from_corr = bc_corr[0]

            print("from barycorrpy: ", tmp_zb_from_corr, " from header:", self.header['SSBZ100'])
            """
            return float(self.header['SSBZ100'])

        obs_time_jd = self.get_obs_time()
        if obs_time_jd is None:
            return default

        rv_config_bc_key = [RadialVelocityAlgInit.RA, RadialVelocityAlgInit.DEC,
                            RadialVelocityAlgInit.PMRA, RadialVelocityAlgInit.PMDEC,
                            RadialVelocityAlgInit.PARALLAX, RadialVelocityAlgInit.OBSLAT, RadialVelocityAlgInit.OBSLON,
                            RadialVelocityAlgInit.OBSALT, RadialVelocityAlgInit.STAR_RV,
                            RadialVelocityAlgInit.SPEC]
        rv_config_bc = {k: self.rv_config[k] for k in rv_config_bc_key}

        bc_corr = BarycentricCorrectionAlg.get_zb_from_bc_corr(rv_config_bc, obs_time_jd)
        return bc_corr[0]

    def wavelength_calibration(self, spectrum_x):
        """Wavelength calibration extraction.

        Extract wavelength calibration based on spectrum order range and horizontal range to be analyzed.

        Args:
            spectrum_x (numpy.ndarray): Horizontal pixel range to be processed.

        Returns:
            numpy.ndarray: 2D array containing wavelength calibration for the pixel range and order range
            to be processed.

        """
        if self.spectro == 'harps':
            return self.wavelength_calibration_harps(spectrum_x)
        elif self.spectro == 'neid':
            return self.wavelength_calibration_neid(spectrum_x)
        else:
            return None

    def wavelength_calibration_neid(self, spectrum_x: np.ndarray):
        new_calibs = self.wave_cal[self.start_order:self.start_order+self.spectrum_order, spectrum_x]

        return new_calibs

    def wavelength_calibration_harps(self, spectrum_x: np.ndarray):
        calib_hdr = self.header
        CAL_TH_COEFF = 'HIERARCH ESO DRS CAL TH COEFF LL'
        p_degree = calib_hdr['HIERARCH ESO DRS CAL TH DEG LL']

        calib_coeffs_orders = []
        for ord_idx in range(self.start_order, self.spectrum_order + self.start_order):
            coeff_base = int(p_degree + 1) * ord_idx
            ll_coeffs_order = [calib_hdr[CAL_TH_COEFF + str(coeff_base + i)] for i in range(p_degree, -1, -1)]
            calib_coeffs_orders.append(ll_coeffs_order)

        # calibrate pixel to wavelength
        wave_cals = np.zeros((self.spectrum_order, np.size(spectrum_x)))
        for ord_idx in range(self.spectrum_order):
            wave_cals[ord_idx, :] = np.polyval(np.poly1d(calib_coeffs_orders[ord_idx]), spectrum_x)

        return wave_cals

    def get_rv_on_spectrum(self, ref_ccf=None, start_x=None, end_x=None, start_order=None, end_order=None,
                           order_diff=0):
        """Radial velocity analysis.

        Compute radial velocity of all orders based on level 1 data, wavelength calibration,
        and horizontal pixel and spectrum order range, and scale the result based on a cross correlation reference if
        there is. 

        Args:
            ref_ccf (array, optional): Reference to scale the cross correlation results. Defaults to None. 
            start_x (int, optional): Start horizontal position of the data to be processed. Defaults to None.
                                    The number means the position relative to the first pixel of the same order
                                    if it is greater than or equal to 0, otherwise it means the position relative to
                                    the last pixel.
            end_x (int, optional): End horizontal position of the data to be processed. Defaults to None.
                                The number has the same meaning as that of `start_x`.
            start_order (int, optional): Start order of the data to be processed. Defaults to Noe.
                                The number means the order relative to the first one if it is greater than or equal
                                to 0, otherwise it means the order relative to the last one.
            end_order (int, optional): End order of the data to be processed. Defaults to None.
                                The number has the same meaning as that of `start_order`.
            order_diff (int, optional): The offset alignment between the spectrum data and reference data,
                    i.e. <order in `ref_ccf`> = `order_diff` + <order in spectrum>. Defaults to 0.

        Returns:
            numpy.ndarray: 2D array containing the cross correlation result of all orders at each velocity step.
            Here are some information for the array result,

                * size: (`spectrum_order` + `ROWS_FOR_ANALYSIS`) x `total velocity steps`, where
                  `spectrum_order` means the total orders of spectrum data for radial velocity computation. 
                * row 0 ~ `spectrum_order`-1:   storing cross correlation results of all orders.
                * row `spectrum_order`: blank
                * row `spectrum_order` + 1: reserved to store velocity steps.
                * row `spectrum_order` + 2: reserved to store the summation result by the use of
                  :func:`~alg.RadialVelocityAlg.analyze_ccf()`.

        """

        self.obs_jd = self.get_obs_time()
        if not self.obs_jd:
            return None, 'observation jd time error'

        zb = self.get_redshift()
        if not zb:
            return None, 'redshift value error'

        self.set_order_range(start_order, end_order)
        self.set_x_range(start_x, end_x)

        s_x = self.start_x_pos
        e_x = self.end_x_pos+1

        spectrum, nx, ny = self.get_spectrum()
        spectrum_x = np.arange(nx)[s_x:e_x]

        s_order = self.start_order
        e_order = self.end_order+1

        new_spectrum = spectrum[s_order:e_order, s_x:e_x]
        result_ccf = np.zeros([self.spectrum_order + self.ROWS_FOR_ANALYSIS, self.velocity_steps])
        wavecal_all_orders = self.wavelength_calibration(spectrum_x)

        for ord_idx in range(self.spectrum_order):
            # self.d_print(ord_idx, ' ', end="")
            self.d_print('order ', ord_idx, info=True)
            wavecal = wavecal_all_orders[ord_idx, :]

            if np.any(wavecal != 0.0):
                result_ccf[ord_idx, :] = \
                    self.cross_correlate_by_mask_shift(wavecal, new_spectrum[ord_idx, :], zb)
            else:
                self.d_print("all wavelength zero")

        self.d_print("\n")
        result_ccf[~np.isfinite(result_ccf)] = 0.
        return result_ccf, ''

    def cross_correlate_by_mask_shift(self, wave_cal, spectrum, zb):
        """Cross correlation by the shifted mask line and the spectrum data of one order for each velocity step.

        Args:
            wave_cal (numpy.ndarray): Wavelength calibration associated with `spectrum`.
            spectrum (numpy.ndarray): Reduced 1D spectrum data of one order from optimal extraction computation.
            zb (float): Redshift at the observation time.

        Returns:
            numpy.ndarray: Cross correlation result of one order at all velocity steps. Please refer to `Returns` of
            function :func:`~alg.RadialVelocityAlg.get_rv_on_spectrum()` for cross correlation results of
            all orders.

        """
        line = self.mask_line

        # made some fix on line_index. the original calculation may miss some pixels at the edges while
        # finding the overlap between the wavelength range of the pixels and the maximum wavelength range of
        # the mask line
        # from the original
        line_index = np.where((line.get('bc_corr_start') > np.min(wave_cal)) &
                              (line.get('bc_corr_end') < np.max(wave_cal)))[0]
        # line_index = np.where((line.get('bc_corr_end') > np.min(wave_cal)) &
        #                       (line.get('bc_corr_start') < np.max(wave_cal)))[0]
        n_line_index = len(line_index)
        v_steps = self.velocity_steps
        ccf = np.zeros(v_steps)
        if n_line_index == 0:
            return ccf

        n_pixel = np.shape(wave_cal)[0]
        pix1, pix2 = 10, n_pixel - 11

        new_line_start = line['start'][line_index]
        new_line_end = line['end'][line_index]
        # new_line_center = line['center'][line_index]
        new_line_weight = line['weight'][line_index]

        x_pixel_wave_start = (wave_cal + np.roll(wave_cal, 1)) / 2.0  # w[0]-(w[1]-w[0])/2, (w[0]+w[1]).....
        x_pixel_wave_end = np.roll(x_pixel_wave_start, -1)            # (w[0]+w[1])/2,      (w[1]+w[2])/2....

        # pixel_wave_end = (wave_cal + np.roll(wave_cal,-1))/2.0      # from the original
        # pixel_wave_start[0] = wave_cal[0]
        # pixel_wave_end[-1] = wave_cal[-1]

        # fix
        x_pixel_wave_start[0] = wave_cal[0] - (wave_cal[1] - wave_cal[0]) / 2.0
        x_pixel_wave_end[-1] = wave_cal[-1] + (wave_cal[-1] - wave_cal[-2]) / 2.0

        shift_lines_by = (1.0 + (self.velocity_loop / LIGHT_SPEED)) / (1.0 + zb)  # Shifting mask in redshift space

        for c in range(v_steps):
            line_doppler_shifted_start = new_line_start * shift_lines_by[c]
            line_doppler_shifted_end = new_line_end * shift_lines_by[c]
            # line_doppler_shifted_center =  new_line_center * shift_lines_by[c]

            # from the original:
            # closest_match = np.sum((x_pixel_wave_start - line_doppler_shifted_center[:,np.newaxis] <= 0.), axis=1)
            closest_match = np.sum((x_pixel_wave_end - line_doppler_shifted_start[:, np.newaxis] < 0.), axis=1)
            closest_match_next = np.sum((x_pixel_wave_start - line_doppler_shifted_end[:, np.newaxis] <= 0.), axis=1)
            mask_spectra_doppler_shifted = np.zeros(n_pixel)

            # this is from the original code, it may miss some pixels at the ends or work on more pixels than needed
            """
            for k in range(n_line_index):
                closest_x_pixel = closest_match[k] - 1    # fix: closest index before line_doppler_shifted_center
                # closest_x_pixel = closest_match[k]      # before fix

                line_start_wave = line_doppler_shifted_start[k]
                line_end_wave = line_doppler_shifted_end[k]
                line_weight = new_line_weight[k]

                if pix1 < closest_x_pixel < pix2:
                    for n in range(closest_x_pixel - 5, closest_x_pixel + 5):
                        # if there is overlap
                        if x_pixel_wave_start[n] <= line_end_wave and x_pixel_wave_end[n] >= line_start_wave:
                            wave_start = max(x_pixel_wave_start[n], line_start_wave)
                            wave_end = min(x_pixel_wave_end[n], line_end_wave)
                            mask_spectra_doppler_shifted[n] = line_weight * (wave_end - wave_start) / \
                                                              (x_pixel_wave_end[n] - x_pixel_wave_start[n])

            """
            idx_collection = list()
            for k in range(n_line_index):
                closest_x_pixel = closest_match[k]  # closest index starting before line_dopplershifted_start
                closest_x_pixel_next = closest_match_next[k]  # closest index starting after line_dopplershifted_end
                line_start_wave = line_doppler_shifted_start[k]
                line_end_wave = line_doppler_shifted_end[k]
                line_weight = new_line_weight[k]

                if closest_x_pixel_next <= pix1 or closest_x_pixel >= pix2:
                    continue
                else:
                    for n in range(closest_x_pixel, closest_x_pixel_next):
                        if n >= pix2:
                            break
                        if n <= pix1:
                            continue
                        # if there is overlap
                        if x_pixel_wave_start[n] <= line_end_wave and x_pixel_wave_end[n] >= line_start_wave:
                            wave_start = max(x_pixel_wave_start[n], line_start_wave)
                            wave_end = min(x_pixel_wave_end[n], line_end_wave)
                            mask_spectra_doppler_shifted[n] = line_weight * (wave_end - wave_start) / \
                                (x_pixel_wave_end[n] - x_pixel_wave_start[n])
                            if n in idx_collection:
                                print(str(n), ' already taken')
                                # import pdb;pdb.set_trace()
                            else:
                                idx_collection.append(n)

            ccf[c] = np.nansum(spectrum * mask_spectra_doppler_shifted)

        return ccf

    def analyze_ccf(self, ccf, row_for_analysis=None):
        """Analyze cross correlation results.

        Do summation on cross correlation values over specified orders.

        Args:
            ccf (numpy.ndarray): A container storing cross correlation values of all orders at each velocity step.
            total_order (int): Total order to analyze.
            row_for_analysis (numpy.ndarray): Rows for analysis. Defaults to None, meaning skipping the first row.

        Returns:
            numpy.ndarray: 2D array containing the cross correlation results of all orders plus
            a blank row, a row with velocity steps and a row with the summation of cross correlation values over orders.

        """
        ccf[self.spectrum_order + 1, :] = self.velocity_loop
        if row_for_analysis is None:
            row_for_analysis = np.arange(1, self.spectrum_order, dtype=int)
        # skip order 0
        ccf[self.spectrum_order + self.ROWS_FOR_ANALYSIS - 1, :] = np.nansum(ccf[row_for_analysis, :], axis=0)
        return ccf

    def get_rv_guess(self):
        return self.get_rv_estimation(self.header, self.init_data)

    @staticmethod
    def get_rv_estimation(hdu_header, init_data):
        rv_guess = hdu_header['QRV'] if 'QRV' in hdu_header\
            else init_data[RadialVelocityAlgInit.RV_CONFIG][RadialVelocityAlgInit.STAR_RV]
        return rv_guess

    @staticmethod
    def fit_ccf(result_ccf, rv_guess, velocities, velocity_cut=100.0):
        """Gaussian fitting to the values of cross correlation vs. velocity steps.

        Find the radial velocity from the summation of cross correlation values over orders by the use of
        Gaussian fitting and starting from a guessed value.

        Args:
            result_ccf (numpy.ndarray): 1D array containing summation the summation of cross correlation data over
                orders. Please refer to `Returns` of :func:`~alg.RadialVelocityAlg.get_rv_on_spectrum()`.
            rv_guess (float): Approximation of radial velocity.
            velocities (np.array): An array of velocity steps.
            velocity_cut (float, optional): Range limit around the guessed radial velocity. Defaults to 100.0 (km/s).
        Returns:
            tuple: Gaussian fitting mean and values for the fitting,

                * **gaussian_fit** (*fitting.LevMarLSQFitter*): Instance for doing Gussian fitting based on
                  Levenberg-Marquardt algorithm and least squares statistics.
                * **mean** (*float*): Mean value from Gaussian fitting.
                * **g_x** (*numpy.ndarray*): Collection of velocity steps for Gaussian fitting.
                * **g_y** (*numpy.ndarray*): Collection of cross correlation summation offset to the mean of
                  cross correlation summation values along *g_x*.

        """
        g_init = models.Gaussian1D(amplitude=-1e7, mean=rv_guess, stddev=5.0)
        ccf = result_ccf
        i_cut = (velocities >= rv_guess - velocity_cut) & (velocities <= rv_guess + velocity_cut)
        g_x = velocities[i_cut]
        g_y = ccf[i_cut] - np.nanmedian(ccf)
        gaussian_fit = FIT_G(g_init, g_x, g_y)
        return gaussian_fit, gaussian_fit.mean.value, g_x, g_y

    def output_ccf_to_dataframe(self, ccf, ref_head=None):
        """Convert cross correlation data to be in the form of Pandas DataFrame.

        Args:
            ccf (numpy.ndarray): Result of cross correlation computation and analysis.
            ref_head (fits.header.Header, optional): Reference fits header. Defaults to None.

        Returns:
            Pandas.DataFrame: Result of cross correlation in form of Pandas DataFrame and the resultant
            radial velocity is stored as the value of attribute `CCF-RVC`.
            
        """

        ccf_table = {}
        for i in range(self.velocity_steps):
            ccf_table['vel-'+str(i)] = ccf[:, i]
        results = pd.DataFrame(ccf_table)
        # results = pd.DataFrame(ccf)
        _, rv_result, _, _ = self.fit_ccf(
            ccf[self.spectrum_order+RadialVelocityAlg.ROWS_FOR_ANALYSIS-1, :],
            self.get_rv_guess(), self.init_data[RadialVelocityAlgInit.VELOCITY_LOOP])

        def f_decimal(num):
            return "{:.10f}".format(num)

        if self.spectro == 'harps':
            if ref_head is not None:  # mainly for PARAS data
                for key in ref_head:
                    if key in results.attrs or key == 'COMMENT':
                        continue
                    else:
                        if key == 'Date':
                            results.attrs[key] = str(datetime.datetime.now())
                        elif 'ESO' in key:
                            if 'ESO DRS CCF RVC' in key:
                                results.attrs['CCF-RVC'] = f_decimal(rv_result)+' Baryc RV (km/s)'
                        else:
                            results.attrs[key] = ref_head[key]
        else:
            results.attrs['CCFJDSUM'] = self.get_obs_time()
            results.attrs['CCF-RVC'] = (f_decimal(rv_result), 'BaryC RV (km/s)')
            results.attrs['CCFSTART'] = str(self.rv_config[RadialVelocityAlgInit.STAR_RV])
            results.attrs['CCFSTEP'] = str(self.rv_config[RadialVelocityAlgInit.STEP])
            results.attrs['STARTORD'] = str(self.start_order)
            results.attrs['ENDORDER'] = str(self.end_order)

        return results

    def compute_rv_by_cc(self, start_x=None, end_x=None, start_order=None, end_order=None,
                         order_diff=0, ref_ccf=None, print_progress=None):
        """Compute radial velocity by using cross correlation method.

        Compute and analyze radial velocity on level 1 data based on the specified pixel positions and the order range
        and output the result in both numpy array and Pandas DataFrame styles.

        Args:
            start_x (int, optional): Start horizontal (x) position of the data. Defaults to None.
                            The number means the position relative to the first pixel of the same order
                            if it is greater than or equal to 0, otherwise it means the position relative to the last
                            pixel.
            end_x (int, optional): End horizontal (x) position of the data. Defaults to None.
                            The number has the same meaning as that of `start_x`.
            start_order (int, optional): Start order of the data. Defaults to None.
                            The number means the order relative to the first one if it is greater than or equal to 0,
                            otherwise it means the order relative to the last one.
            end_order (int, optional): End order of the data. Defaults to None.
                            The number has the same meaning as that of `start_order`.
            order_diff (int, optional): Order difference between spectrum data and the
                reference data, i.e. <order in ref> = `order_diff` + <order in spectrum>.
                Defaults to 0.
            ref_ccf (numpy.ndarray, optional): Reference of cross correlation values or ratio table for scaling the
                computation of cross correlation. The dimension of ref_ccf is the same as that of the computed ccf.
            print_progress (str, optional):  Print debug information to stdout if it is provided as empty string
                or to a file path, `print_progress`,  if it is non empty string, or no print is made if it is None.
                Defaults to None.

        Returns:
            dict: Instance containing cross correction results in type of numpy.ndarray and Pandas DataFrame, like::

                {
                    'ccf_df': Pandas.DataFrame
                                        # cross correlation result in DataFrame type.
                    'ccf_ary': numpy.ndarray
                                        # cross correlation result in numpy.ndarray type.
                    'jd': float         # observation time in Julian Date format.
                }

            Please refer to `Returns` of :func:`~alg.RadialVelocityAlg.get_rv_on_spectrum()` for the content of
            the cross correlation results.

        Raises:
            AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
            Exception: If there is error from radial velocity computation.

        """
        self.add_file_logger(print_progress)
        self.d_print('computing radial velocity ... ')

        ccf, msg = self.get_rv_on_spectrum(ref_ccf, start_x, end_x, start_order, end_order, order_diff)
        if ccf is None:
            raise Exception(msg)

        if ref_ccf is not None:
            ccf = self.reweight_ccf(ccf, self.spectrum_order, ref_ccf, self.reweighting_ccf_method,
                                    s_order=start_order)

        analyzed_ccf = self.analyze_ccf(ccf)
        df = self.output_ccf_to_dataframe(analyzed_ccf)
        return {'ccf_df': df, 'ccf_ary': analyzed_ccf, 'jd': self.obs_jd}

    @staticmethod
    def is_good_reweighting_method(method, for_ratio=False):
        return method in ['ccf_max', 'ccf_mean'] if for_ratio else method in ['ccf_max', 'ccf_mean', 'ccf_steps']

    @staticmethod
    def make_reweighting_ratio_table(order_val, s_idx, e_idx, reweighting_method, max_ratio=1.0, output_csv=''):
        """Make the ratio table from the given CCF orders

        Args:
            order_val (numpy.ndarray): CCF orders.
            s_idx (int): The starting index that the first row of `order_val` is associated with.
            e_idx (int):  The last index of the CCF order to be collected from `order_val`.
            reweighting_method (str): Reweight methods for making the ratio table, **ccf_max** is to make the ratio
                based on 95 percentile CCF value of each order, and **ccf_mean** is to make the ratio based on the mean
                of the CCF value of each order.
            max_ratio (float, optional): Maximum ratio number in the ratio table. Defaults to 1.0.
            output_csv (str, optional): Output the ratio table into a .csv file. Default to no output.

        Returns:
            pandas.Dataframe: ratio table in DataFrame format containing two columns. The first column is the order
            index from `s_idx` to `e_idx` and the second column is the ratio for the order of the first column.

        Raises:
            Exception: invalid reweighting method to build the ratio table.
        """
        if not RadialVelocityAlg.is_good_reweighting_method(reweighting_method, for_ratio=True):
            raise Exception('invalid reweighting method to build the ratio table')

        row_val = order_val[np.arange(0, e_idx-s_idx+1, dtype=int), :]
        row_val = np.where((row_val < 0.0), 0.0, row_val)

        if reweighting_method == 'ccf_max':
            t_val = np.nanpercentile(row_val, 95, axis=1)
        elif reweighting_method == 'ccf_mean':
            t_val = np.nanmean(row_val, axis=1)

        t_val = np.where(np.isnan(t_val), 0.0, t_val)

        if max_ratio is not None:
            max_t_val = np.max(t_val)
            if max_t_val != 0.0:
                t_val = (t_val/max_t_val) * max_ratio
        ratio_table = {'order': np.arange(s_idx, e_idx + 1, dtype=int),
                       'ratio': t_val}

        df = pd.DataFrame(ratio_table)
        if output_csv:
            df.to_csv(output_csv, index=False)

        return df

    @staticmethod
    def reweight_ccf(crt_rv, total_order, reweighting_table_or_ccf, reweighting_method, s_order=0,
                     do_analysis=False, velocities=None):
        """Reweighting ccf orders.

        Reweight the CCF ordres based on the given CCF ratios or CCF orders from the observation template.

        Args:
            crt_rv (numpy.ndarray): CCF orders.
            total_order (int): Total orders for reweighting. It is in default from the first row of `crt_rv`.
            reweighting_table_or_ccf (numpy.ndarray): Ratios among CCF orders or CCF data from the observation template.
            reweighting_method (str): Reweighting methods, **ccf_max**, **ccf_mean**, or **ccf_steps**.
            s_order (int, optional): The start order index for reweighting. This is used to select the row from `crt_rv`
                in case the order index column is included in `reweighting_table_or_ccf`. Defaults to 0.
            do_analysis (bool, optional): Do summation on the weighted ccf orders as what
                :func:`~alg.RadialVelocityAlg.analysis_ccf()` dose on CCF orders. Defaults to False.
            velocities (np.ndarray, optional): 1D array consisting of the velocity loop for cross-correlation
                computation. Used when `do_analysis` is **True**. Defaults to None.

         Returns:
             numpy.ndarray: 2D array containing Reweighted CCF orders and the velocity loop and the CCF summation
                from CCF orders at the last two rows.

        Raises:
            Exception: no valid reference data from observation template
            Exception: invalid reweighting method
        """

        # the order index of crt_rv and reweighting_table_or_ccf are aligned

        if reweighting_table_or_ccf is None:
            raise Exception("no valid data from observation template")
        if not reweighting_method or not RadialVelocityAlg.is_good_reweighting_method(reweighting_method):
            raise Exception("invalid reweighting method")

        ny, nx = np.shape(crt_rv)

        total_order = min(total_order, ny)

        if reweighting_method == 'ccf_max' or reweighting_method == 'ccf_mean':
            # if the ratio table containing a column of order index, using s_order to select the ratio with
            # order index from s_order to s_order+total_order-1
            if np.shape(reweighting_table_or_ccf)[1] >= 2:
                s_order = 0 if s_order is None else s_order
                e_order = s_order + total_order - 1
                c_idx = np.where((reweighting_table_or_ccf[:, 0] >= s_order)&
                                 (reweighting_table_or_ccf[:, 0] <= e_order))[0]
                tval = reweighting_table_or_ccf[c_idx, -1]
                crt_rv = crt_rv[c_idx, :]
                total_order = np.size(tval)
            else:
                tval = reweighting_table_or_ccf[0:total_order, -1]

            new_crt_rv = np.zeros((total_order + RadialVelocityAlg.ROWS_FOR_ANALYSIS, nx))
            max_index = np.where(tval == np.max(tval))[0]       # the max from ratio table, 1.0 if ratio max is 1.0
            oval = np.nanpercentile(crt_rv[0:total_order], 95, axis=1) if reweighting_method == 'ccf_max' \
                else np.nanmean(crt_rv[0:total_order], axis=1)  # max or mean from each order

            oval_at_index = oval[max_index]                     # value from oder of max_index
            if oval_at_index == 0.0:      # order of max_index has value 0.0, skip reweighting, returns all zeros
                return new_crt_rv
            oval = oval/oval_at_index     # ratio of orders before reweighting, order of max_index is 1.0
            for order in range(total_order):
                if oval[order] != 0.0:
                    new_crt_rv[order, :] = crt_rv[order, :] * tval[order]/oval[order]
        elif reweighting_method == 'ccf_steps':             # assume crt_rv and reweighting ccf cover the same orders
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                new_crt_rv = np.zeros((total_order + RadialVelocityAlg.ROWS_FOR_ANALYSIS, nx))
                for order in range(total_order):
                    if np.size(np.where(crt_rv[order, :] != 0.0)[0]) > 0:
                        new_crt_rv[order, :] = crt_rv[order, :] * \
                                           np.nanmean(reweighting_table_or_ccf[order, :]/crt_rv[order, :])
        if do_analysis:
            row_for_analysis = np.arange(1, total_order, dtype=int)
            new_crt_rv[total_order + RadialVelocityAlg.ROWS_FOR_ANALYSIS -1, :] = \
                            np.nansum(new_crt_rv[row_for_analysis, :], axis=0)
            if velocities is not None and np.size(velocities) == nx:
                new_crt_rv[total_order + RadialVelocityAlg.ROWS_FOR_ANALYSIS - 2, :] = velocities
        return new_crt_rv


