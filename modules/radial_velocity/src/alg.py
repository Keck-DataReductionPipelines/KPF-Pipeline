from astropy.io import fits
import numpy as np
from astropy.modeling import models, fitting
from astropy import constants as const
import warnings
import datetime
import pandas as pd
import os
import math

from modules.radial_velocity.src.alg_rv_init import RadialVelocityAlgInit
from modules.radial_velocity.src.alg_rv_base import RadialVelocityBase
from modules.barycentric_correction.src.alg_barycentric_corr import BarycentricCorrectionAlg
from modules.CLib.CCF import CCF_3d_cpython
from modules.Utils.config_parser import ConfigHandler

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
        ccf_engine (string): CCF engine to use, 'c' or 'python'. Defaults to None,
        reweighting_method (string): reweighting method, ccf_max or ccf_mean, of ccf_steps. Defaults to None.
        segment_list (pandas.DataFrame): Table containing segment list containing segment index, start wavelength,
            and end wavelength. Defaults to None.
        order_limits_mask (pandas.DataFrame): Table containing order index and the left and right limits of the
            order. Defaults to None.
        bary_corr_table (pd.ndarray): table from L1 BARY_CORR table. Defaults to None.
        start_bary_index (int): starting index in BARY_CORR table.

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
        bary_corr_table (pd.ndarray): table from L1 BARY_CORR table.
        start_bary_index (int): starting index in BARY_CORR table.

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
        TypeError: If there is type error for `spectrum_data`.
        TypeEoor: If there is type error for `header`.
        TypeError: If there is type error for `wave_cal`.
        Exception: If there is init error for radial velocity calculation.

    """
    ROWS_FOR_ANALYSIS = 3
    SEGMENT_IDX = 0
    SEGMENT_X1 = 1
    SEGMENT_X2 = 2
    SEGMENT_W1 = 3
    SEGMENT_W2 = 4
    SEGMENT_ORD = 5
    CCF_Methods = ['ccf_max', 'ccf_mean', 'ccf_static', 'ccf_steps']
    vel_range_per_mask = {'lfc': 5.0, 'thar': 5.0, 'etalon': 5.0}

    map_fibers_mask = {'kpf':
                           {'sci': RadialVelocityAlgInit.KEY_SCI_OBJ,
                            'sky': RadialVelocityAlgInit.KEY_SKY_OBJ,
                            'cal': RadialVelocityAlgInit.KEY_CAL_OBJ}}

    """int: Extra rows added to the 2D result in which each row contains the cross correlation result for one order. 
    
    The first extra row is left blank.
    The second extra row contains the velocity steps. 
    The third extra row contains the summation of cross correlation results over all orders.
    """

    def __init__(self, spectrum_data, header, init_rv, wave_cal=None, config=None, logger=None, ccf_engine=None,
                 reweighting_method=None, segment_limits=None, order_limits=None, area_limits=None,
                 bary_corr_table=None, start_bary_index=0, orderlet=None, vel_span_pixel=None):

        if spectrum_data is not None and not isinstance(spectrum_data, np.ndarray):
            raise TypeError('results of optimal extraction type error')
        if header is None:
            raise TypeError('data header type error')
        if wave_cal is not None and not isinstance(wave_cal, np.ndarray):
            raise TypeError('wave calibration data type error')
        if 'data' not in init_rv or not init_rv['status']:
            raise Exception('radial velocity init error: '+init_rv['msg'])
        if area_limits is not None and not isinstance(area_limits, list) and len(area_limits) != 4:
            raise Exception('area definition error')

        init_data = init_rv['data']

        RadialVelocityBase.__init__(self, config, logger)
        self.spectrum_data = spectrum_data
        self.wave_cal = wave_cal
        self.header = header
        self.init_data = init_data

        # ra, dec, pm_ra, pm_dec, parallax, def_mask, obslon, obslan, obsalt, star_rv, step
        # step_range, mask_width
        self.rv_config = init_data[RadialVelocityAlgInit.RV_CONFIG]
        self.velocity_loop = init_data[RadialVelocityAlgInit.VELOCITY_LOOP]    # loop of velocities for rv finding
        self.velocity_steps = init_data[RadialVelocityAlgInit.VELOCITY_STEPS]  # total steps in velocity_loop
        self.mask_line = init_data[RadialVelocityAlgInit.MASK_LINE]             # def_mask
        self.reweighting_ccf_method = init_data[RadialVelocityAlgInit.REWEIGHTING_CCF] \
            if reweighting_method is None or not self.is_good_reweighting_method(reweighting_method) \
            else reweighting_method
        self.ccf_code = ccf_engine if (ccf_engine and ccf_engine in ['c', 'python']) else \
            init_data[RadialVelocityAlgInit.CCF_CODE]

        if self.spectrum_data is not None and self.spectrum_data.size != 0:
            ny, nx = np.shape(self.spectrum_data)
        else:
            ny = nx = 0

        self.start_order = 0
        self.end_order = ny-1
        self.spectrum_order = ny
        self.start_x_pos = 0
        self.end_x_pos = nx-1
        self.spectro = self.rv_config[RadialVelocityAlgInit.SPEC].lower() \
            if RadialVelocityAlgInit.SPEC in self.rv_config else 'neid'
        self.segment_limits = segment_limits
        self.order_limits_mask = order_limits
        self.area_limits = area_limits
        self.segment_limits_table = None
        self.total_segments = None
        self.total_rv_segment = None
        self.zb = None
        self.obs_jd = None
        self.bary_corr_table = bary_corr_table
        self.start_bary_index = start_bary_index
        self.orderletname = orderlet
        self.orderlet_mask_line = None
        self.vel_span_pixel = vel_span_pixel if vel_span_pixel is not None else \
            RadialVelocityAlg.comp_velocity_span_pixel(init_data, config, self.spectro)

    @staticmethod
    def comp_velocity_span_pixel(init_data, config, instrument):
        vel_span_pixel = None
        if init_data is not None:
            vel_span_pixel = init_data[RadialVelocityAlgInit.VEL_SPAN_PIXEL]
        else:
            vel_span_pixel = \
                RadialVelocityAlg.get_config_value(config, instrument, RadialVelocityAlgInit.VEL_SPAN_PIXEL, 0.87)
        return vel_span_pixel

    @staticmethod
    def compute_vel_range(config, instrument, mask, default_v):
        c_key = 'velocity_range_'+mask

        return RadialVelocityAlg.get_config_value(config, instrument, c_key, default_v)

    @staticmethod
    def get_config_value(config, instrument, c_key, default_v):
        val = None
        if config is not None:
            config_param = None
            if instrument is not None and config.has_section(instrument):
                config_param = config[instrument.upper()]
            elif config.has_section('PARAM'):
                config_param = config['PARAM']
            if config_param is not None:
                val = config_param.getfloat(c_key, default_v)
        return val

    def get_vel_span_pixel(self):
        return self.vel_span_pixel

    def reset_spectrum(self, spec_data, header, wave_cal, orderlet=None):
        if spec_data is not None and not isinstance(spec_data, np.ndarray):
            raise TypeError('results of optimal extraction type error')
        if header is None:
            raise TypeError('data header type error')
        if wave_cal is not None and not isinstance(wave_cal, np.ndarray):
            raise TypeError('wave calibration data type error')
        self.spectrum_data = spec_data
        self.wave_cal = wave_cal
        self.header = header
        self.orderletname = orderlet
        return

    def get_spectrum(self):
        """Get spectrum information.

        Returns:
            tuple: Spectrum data and the dimension.

                * (*numpy.ndarray*): Spectrum data 2d array
                * **nx** (*int*): Horizontal size.
                * **ny** (*int*): Vertical size.

        """

        if self.spectrum_data is not None and self.spectrum_data.size != 0:
            ny, nx = np.shape(self.spectrum_data)
        else:
            nx = ny = 0
        return self.spectrum_data, nx, ny

    def get_segment_info(self):
        return self.segment_limits_table

    def get_segment_limits(self, seg_idx=0):
        if self.segment_limits_table is None:
            segment_list = []   # segment_index,  start_x, end_x, start_wavelength, end_wavelength, order_index

            seg_idx = 0
            if self.segment_limits is not None:   # per what segment limits define
                seg_total, col_num = np.shape(self.segment_limits)
                col_num = min(col_num-2, 2)
                for s in range(seg_total):
                    wlen = [self.segment_limits[s, 1], self.segment_limits[s, col_num]]
                    sel_w = np.where((self.wave_cal >= wlen[0]) & (self.wave_cal <= wlen[1]))
                    if np.size(sel_w[0]) > 0:
                        sel_order = self.segment_limits[s, -1] if self.segment_limits[s, -1] in sel_w[0] else sel_w[0][0]
                        sel_pixel = sel_w[1][np.where(sel_w[0] == sel_order)[0]]
                        sel_pixel.sort()
                        segment_list.append([seg_idx, sel_pixel[0], sel_pixel[-1], wlen[0], wlen[1], int(sel_order)])
                        seg_idx += 1
            elif self.order_limits_mask is not None:  # 1-1 between segments and orders
                order_total, col_num = np.shape(self.order_limits_mask)
                num_limits = min(col_num - 1, 2)  # 1 or 2 limit columns
                seg_list = np.arange(0, self.end_order+1, dtype=int)       # list with full orders
                order_list = self.order_limits_mask[:, 0]                  # order_index in order limits mask
                for ord_idx in range(seg_list):

                    if ord_idx in order_list:
                        r = np.where(order_list == ord_idx)[0][0]
                        limits = [self.order_limits_mask[r, 1], self.end_x_pos - self.order_limits_mask[r, num_limits]]
                    else:
                        limits = [self.start_x_pos, self.end_x_pos]

                    segment_list.append([ord_idx, limits[0], limits[1],
                                         self.wave_cal[ord_idx, limits[0]],
                                         self.wave_cal[ord_idx, limits[1]], ord_idx])
            elif self.area_limits is not None:     # all orders are counted, 1-1 between segments and orders
                order_range = [self.area_limits[i] if self.area_limits[i] >= 0
                               else self.end_order + self.area_limits[i] for i in [0, 1]]

                for r in range(self.end_order+1):
                    if order_range[0] <= r <= order_range[1]:
                        x_range = [self.area_limits[i]
                                   if self.area_limits[i] >= 0 else (self.end_x_pos + self.area_limits[i])
                                   for i in [2, 3]]
                        segment_list.append([r, x_range[0], x_range[1],
                                             self.wave_cal[r, x_range[0]],
                                             self.wave_cal[r, x_range[1]],  r])
                    else:
                        segment_list.append([r, self.start_x_pos, self.end_x_pos,
                                             self.wave_cal[r, 0], self.wave_cal[r, -1], r])
            else:         # order between start_order and end_order are counted, 1-1 between segments and orders
                for r in range(self.start_order, self.end_order+1):
                    segment_list.append([r, self.start_x_pos, self.end_x_pos,
                                         self.wave_cal[r, self.start_x_pos],
                                         self.wave_cal[r, self.end_x_pos], r])
            self.total_segments = len(segment_list)
            self.segment_limits_table = np.array(segment_list)

        idx = np.where(self.segment_limits_table[:, self.SEGMENT_IDX] == seg_idx)[0][0]
        return self.segment_limits_table[idx]

    def get_total_segments(self):
        if self.total_segments is None:
            self.get_segment_limits()
        return self.total_segments

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

    def get_obs_time(self, default=None, seg=0):
        """Get Observation time in Julian Date format.

        Args:
            default (float, optional): Default observation time. Defaults to None.
            seg (int): segment index. Defaults to 0. Get exposure time from header if seg is -1.

        Returns:
            float: Observation time in Julian Date format.

        """
        if self.obs_jd is None:

            self.obs_jd = np.zeros(self.get_total_segments(), dtype=float)

            if self.spectro == 'neid':
                self.obs_jd[:] = self.get_obs_time_neid(default=default)
            elif self.spectro == 'harps':
                self.obs_jd[:]= self.get_obs_time_harps(default=default)
            elif self.spectro == 'kpf':
                for s in range(self.get_total_segments()):
                    self.obs_jd[s] = self.get_obs_time_kpf(default=default, seg=s)

        if seg < 0:
            if self.spectro == 'kpf':
                return self.get_obs_time_kpf(seg=-1)
            else:
                seg = 0

        return self.obs_jd[seg]

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

    # get obs time for kpf on specific segment.
    def get_obs_time_kpf(self, default=None, seg=0):
        if seg >= 0:
            seg_limits = self.get_segment_limits(seg_idx=seg)
            ord_idx = int(seg_limits[self.SEGMENT_ORD])
        else:
            ord_idx = 0

        if seg >= 0 and self.bary_corr_table is not None and not self.bary_corr_table.empty and \
                np.shape(self.bary_corr_table.values)[0] > ord_idx:
            obs_time = np.array(self.bary_corr_table['PHOTON_BJD'])[ord_idx+self.start_bary_index]
        elif 'MJD-OBS' in self.header and 'EXPTIME' in self.header:
            obs_time = self.header['MJD-OBS'] + 2400000.5
        else:
            obs_time = default

        return obs_time

    # get redshift for kpf on specific segment
    def get_redshift_kpf(self, seg=0, default=None):
        seg_limits = self.get_segment_limits(seg_idx=seg)
        ord_idx = int(seg_limits[self.SEGMENT_ORD])
        if 'MASK' in self.header and self.header['MASK'] in ['lfc', 'thar', 'etalon'] \
                or ('IMTYPE' in self.header and self.header['IMTYPE'].lower() != 'object'):
        # if self.init_data['mask_type'] in ['lfc', 'thar'] or \
            bc = 0.0
        elif self.bary_corr_table is not None and np.shape(self.bary_corr_table.values)[0] > ord_idx:
            bc = self.bary_corr_table['BARYVEL'][ord_idx + self.start_bary_index]
        else:
            bc = self.get_redshift_gen(default=default, seg=seg)
        return bc

    def get_redshift_gen(self, default=None, seg=0):

        obs_time = self.get_obs_time(default=default, seg=seg)
        if obs_time is None:
            return default
        else:
            rv_config_bc_key = [RadialVelocityAlgInit.RA, RadialVelocityAlgInit.DEC,
                            RadialVelocityAlgInit.PMRA, RadialVelocityAlgInit.PMDEC, RadialVelocityAlgInit.EPOCH,
                            RadialVelocityAlgInit.PARALLAX, RadialVelocityAlgInit.OBSLAT,
                            RadialVelocityAlgInit.OBSLON, RadialVelocityAlgInit.OBSALT,
                            RadialVelocityAlgInit.STAR_RV,
                            RadialVelocityAlgInit.SPEC, RadialVelocityAlgInit.STARNAME]
            rv_config_bc = {k: self.rv_config[k] for k in rv_config_bc_key}
            if RadialVelocityAlgInit.is_unknown_target(self.spectro, rv_config_bc[RadialVelocityAlgInit.STARNAME],
                                                       rv_config_bc[RadialVelocityAlgInit.EPOCH]):
                return 0.0
            bc = BarycentricCorrectionAlg.get_zb_from_bc_corr(rv_config_bc, obs_time)[0]

            return bc

    def get_redshift(self, default=None, seg=0):
        """Get redshift value.

        Args:
            default (float, optional): Default redshift value. Defaults to None.
            seg (int, optional): redshift value for specific segment in case the redshift varies per segment.

        Returns:
            float: redshift at observation time.

        """

        # zb is computed for each segment for kpf

        if self.zb is None:
            self.zb = np.zeros(self.get_total_segments(), dtype=float)

            if self.spectro == 'neid' and 'SSBZ100' in self.header:
                bc = float(self.header['SSBZ100'])
                self.zb[:] = bc
            elif self.spectro == 'kpf':
                for s in range(self.get_total_segments()):
                    self.zb[s] = self.get_redshift_kpf(seg=s, default=default)
            else:
                bc = self.get_redshift_gen(default, seg=seg)
                self.zb[:] = bc

        return self.zb[seg]

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
        elif self.spectro == 'kpf':
            return self.wavelength_calibration_kpf(spectrum_x)
        else:
            return None

    def wavelength_calibration_neid(self, spectrum_x: np.ndarray):
        new_calibs = self.wave_cal[self.start_order:self.start_order+self.spectrum_order, spectrum_x]

        return new_calibs

    def wavelength_calibration_kpf(self, spectrum_x: np.ndarray):
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

    def get_rv_on_spectrum(self, start_seg=None, end_seg=None):
        """Radial velocity analysis.

        Compute radial velocity of all orders based on level 1 data, wavelength calibration,
        and horizontal pixel and spectrum order range, and scale the result based on a cross correlation reference if
        there is. 

        Args:
            start_seg (int, optional): First segment of the data to be processed. Defaults to None.
                                The number means the order relative to the first segment if it is greater than or equal
                                to 0, otherwise it means the order relative to the last one.
            end_seg (int, optional): Last segment of the data to be processed. Defaults to None.
                                The number has the same meaning as that of `start_segment`.


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

        obs_jd = self.get_obs_time()
        if obs_jd is None or not obs_jd:
            return None, 'observation jd time error'

        zb = self.get_redshift()
        if zb is None:
            return None, 'redshift value error'

        self.set_x_range()
        self.set_order_range()

        s_x = self.start_x_pos
        e_x = self.end_x_pos+1
        s_order = self.start_order
        e_order = self.end_order + 1

        spectrum, nx, ny = self.get_spectrum()
        spectrum_x = np.arange(nx)[s_x:e_x]

        new_spectrum = spectrum[s_order:e_order, s_x:e_x]
        total_seg = self.get_total_segments()

        if start_seg is not None:
            s_seg_idx = start_seg if start_seg >= 0 else (total_seg+start_seg)
        else:
            s_seg_idx = 0
        if end_seg is not None:
            e_seg_idx = end_seg if end_seg >= 0 else (total_seg+end_seg)
        else:
            e_seg_idx = total_seg-1

        result_ccf = np.zeros([total_seg + self.ROWS_FOR_ANALYSIS, self.velocity_steps])
        # result_ccf = np.zeros([(e_seg_idx - s_seg_idx + 1) + self.ROWS_FOR_ANALYSIS, self.velocity_steps])
        wavecal_all_orders = self.wavelength_calibration(spectrum_x)     # from s_order to e_order, s_x to e_x

        seg_ary = np.arange(total_seg)[s_seg_idx:e_seg_idx+1]
        for idx, seg_idx in np.ndenumerate(seg_ary):
            seg_limits = self.get_segment_limits(seg_idx=seg_idx)
            ord_idx = int(seg_limits[self.SEGMENT_ORD])
            self.d_print("RadialVelocityAlg: ", "segment", ord_idx, ' ',
                         [int(seg_limits[self.SEGMENT_X1]), int(seg_limits[self.SEGMENT_X2]),
                          seg_limits[self.SEGMENT_W1], seg_limits[self.SEGMENT_W2], int(seg_limits[self.SEGMENT_ORD])],
                         ' ')
            wavecal = wavecal_all_orders[ord_idx]
            left_x = int(seg_limits[self.SEGMENT_X1])
            right_x = int(seg_limits[self.SEGMENT_X2])

            if np.any(wavecal != 0.0):
                if wavecal[-1] < wavecal[0]:
                    ordered_spec = self.fix_nan_spectrum(np.flip(new_spectrum[ord_idx][left_x:right_x]))   # check??
                    ordered_wavecal = np.flip(wavecal[left_x:right_x])
                else:
                    ordered_spec = self.fix_nan_spectrum(new_spectrum[ord_idx][left_x:right_x])
                    ordered_wavecal = wavecal[left_x:right_x]
                zb = self.get_redshift(seg=seg_idx)
                result_ccf[seg_idx, :] = \
                        self.cross_correlate_by_mask_shift(ordered_wavecal, ordered_spec, zb)
            else:
                self.d_print("RadialVelocityAlg: all wavelength zero")
        result_ccf[~np.isfinite(result_ccf)] = 0.
        return result_ccf, ''

    @staticmethod
    def fix_nan_spectrum(spec_vals):
        """ Fix NaN in spectrum.

        Args:
            spec_vals (numpy.ndarray): Spectrum data

        Returns:
            numpy.ndarray: Spectrum with fixed NaN values

        """

        # convert NaN to zero
        def nan_to_zero(s_vals):
            new_spec = np.where(np.isnan(s_vals), 0.0, s_vals)
            return new_spec

        # linear NaN
        def linear_nan(s_vals):
            nan_locs = np.where(np.isnan(s_vals))[0]
            if np.size(nan_locs) == 0:
                return s_vals

            new_spec = np.where(np.isnan(s_vals), 0.0, s_vals)
            last_idx = np.shape(spec_vals)[0] - 1
            last_nan_idx = np.shape(nan_locs)[0] - 1
            e_nan_loc = -1
            s_nan_loc = -1

            for idx in range(last_nan_idx+1):
                nan_loc = nan_locs[idx]
                if idx != 0 and (nan_loc - e_nan_loc) == 1:
                    e_nan_loc = nan_loc
                else:
                    s_nan_loc = nan_loc   # s_nan_loc, e_lan_loc starts to be assigned when idx == 0
                    e_nan_loc = nan_loc

                # not the last loc and the next one is right next to the current loc.
                if idx != last_nan_idx and (nan_locs[idx+1] - nan_loc) == 1:
                    continue

                s_idx = s_nan_loc - 1   # surrounding non-nan location
                e_idx = e_nan_loc + 1
                if s_idx >= 0 and e_idx <= last_idx:
                    delta = (spec_vals[e_idx] - spec_vals[s_idx]) / (e_idx - s_idx)
                    for n_idx in range(s_nan_loc, e_nan_loc+1):
                        new_spec[n_idx] = new_spec[n_idx-1] + delta
                elif e_idx <= last_idx:
                    for n_idx in range(s_nan_loc, e_nan_loc+1):
                        new_spec[n_idx] = new_spec[e_idx]
                elif s_idx >= 0:
                    for n_idx in range(s_nan_loc, e_nan_loc+1):
                        new_spec[n_idx] = new_spec[s_idx]

            return new_spec

        return linear_nan(spec_vals)

    def get_fiber_key_in_maskline(self):
        key_in_maskline = self.get_fiber_object_in_header(self.spectro, self.orderletname)
        if key_in_maskline is not None and key_in_maskline in self.mask_line:
            return key_in_maskline
        return None

    @staticmethod
    def get_orderlet_masktype(ins, orderletname, rv_init_data):
        if not rv_init_data[RadialVelocityAlgInit.MASK_ORDERLET]:   # no need ins and orderletname
            return rv_init_data[RadialVelocityAlgInit.MASK_TYPE]
        else:
            fiber_key = RadialVelocityAlg.get_fiber_object_in_header(ins, orderletname)
            return rv_init_data[RadialVelocityAlgInit.MASK_ORDERLET][fiber_key][RadialVelocityAlgInit.MASK_TYPE]

    @staticmethod
    def get_fiber_object_in_header(ins, sciname):
        if ins.lower() in RadialVelocityAlg.map_fibers_mask:
            fibers = RadialVelocityAlg.map_fibers_mask[ins.lower()]
            for fb in fibers.keys():
                if sciname is not None and fb in sciname.lower():
                    return fibers[fb]

        return None

    def get_mask_line(self):
        """Get mask line info. based on orderlet name if there is.

        Returns:
            dict: mask line info from RadialVelocityMaskLine.

        """
        if self.orderlet_mask_line is None:
            if 'center' in self.mask_line:
                self.orderlet_mask_line = self.mask_line
            else:
                fiber_in_maskline = self.get_fiber_key_in_maskline()
                if fiber_in_maskline is not None:
                    self.orderlet_mask_line = self.mask_line[fiber_in_maskline]

        return self.orderlet_mask_line


    def cross_correlate_by_mask_shift(self, wave_cal, spectrum, zb):
        """Cross correlation by the shifted mask line and the spectrum data of one order for each velocity step.

        Args:
            wave_cal (numpy.ndarray): Wavelength calibration associated with `spectrum`.
            spectrum (numpy.ndarray): Reduced 1D spectrum data of one order from optimal extraction computation.
            zb (float): BC velocity (m/sec) at the observation time.

        Returns:
            numpy.ndarray: Cross correlation result of one order at all velocity steps. Please refer to `Returns` of
            function :func:`~alg.RadialVelocityAlg.get_rv_on_spectrum()` for cross correlation results of
            all orders.

        """

        v_steps = self.velocity_steps
        ccf = np.zeros(v_steps)

        line = self.get_mask_line()
        if line is None:
            return ccf

        # made some fix on line_index. the original calculation may miss some pixels at the edges while
        # finding the overlap between the wavelength range of the pixels and the maximum wavelength range of
        # the mask line
        # from the original
        line_index = np.where((line.get('bc_corr_start') > np.min(wave_cal)) &
                              (line.get('bc_corr_end') < np.max(wave_cal)))[0]


        # line_index = np.where((line.get('bc_corr_end') > np.min(wave_cal)) &
        #                       (line.get('bc_corr_start') < np.max(wave_cal)))[0]
        n_line_index = len(line_index)
        if n_line_index == 0 or wave_cal.size <= 2:
            return ccf

        n_pixel = np.shape(wave_cal)[0]

        new_line_start = line['start'][line_index]
        new_line_end = line['end'][line_index]
        # new_line_center = line['center'][line_index]
        new_line_weight = line['weight'][line_index]

        x_pixel_wave_start = (wave_cal + np.roll(wave_cal, 1)) / 2.0  # w[0]-(w[1]-w[0])/2, (w[0]+w[1])/2.....
        x_pixel_wave_end = np.roll(x_pixel_wave_start, -1)            # (w[0]+w[1])/2,      (w[1]+w[2])/2....

        # pixel_wave_end = (wave_cal + np.roll(wave_cal,-1))/2.0      # from the original
        # pixel_wave_start[0] = wave_cal[0]
        # pixel_wave_end[-1] = wave_cal[-1]

        # fix
        x_pixel_wave_start[0] = wave_cal[0] - (wave_cal[1] - wave_cal[0]) / 2.0
        x_pixel_wave_end[-1] = wave_cal[-1] + (wave_cal[-1] - wave_cal[-2]) / 2.0

        x_pixel_wave = np.zeros(n_pixel + 1)
        x_pixel_wave[1:n_pixel] = x_pixel_wave_start[1:n_pixel]
        x_pixel_wave[n_pixel] = x_pixel_wave_end[-1]
        x_pixel_wave[0] = x_pixel_wave_start[0]

        # shift_lines_by = (1.0 + (self.velocity_loop / LIGHT_SPEED)) / (1.0 + zb)  # Shifting mask in redshift space
        if self.ccf_code == 'c':
            # ccf_pixels_c = np.zeros([v_steps, n_pixel])

            # update redshift

            zb = zb/LIGHT_SPEED_M              # zb in (m/s)/3*10e8 m/s
            v_b = ((1.0/(1+zb)) - 1.0) * LIGHT_SPEED

            for c in range(v_steps):
                # add one pixel before and after the original array in order to uniform the calculation between c code
                # and python code
                new_wave_cal = np.pad(wave_cal, (1, 1), 'constant')
                new_wave_cal[0] = 2 * wave_cal[0] - wave_cal[1]     # w[0] - (w[1]-w[0])
                new_wave_cal[-1] = 2 * wave_cal[-1] - wave_cal[-2]  # w[n-1] + (w[n-1] - w[n-2])

                new_spec = np.pad(spectrum, (1, 1), 'constant')
                new_spec[1:n_pixel+1] = spectrum
                sn = np.ones(n_pixel+2)

                ccf[c] = CCF_3d_cpython.calc_ccf(new_line_start.astype('float64'), new_line_end.astype('float64'),
                                                 new_wave_cal.astype('float64'), new_spec.astype('float64'),
                                                 new_line_weight.astype('float64'), sn.astype('float64'),
                                                 self.velocity_loop[c], -v_b)    # need check??
                """
                ccf_pixels = CCF_3d_cpython.calc_ccf_pixels(new_line_start.astype('float64'),
                                                            new_line_end.astype('float64'),
                                                            new_wave_cal.astype('float64'),
                                                            new_spec.astype('float64'),
                                                            new_line_weight.astype('float64'), sn.astype('float64'),
                                                            self.velocity_loop[c], v_b)
                ccf_pixels_c[c, :] = ccf_pixels
                """
            """
            sn_p = np.ones(n_pixel)
            ccf_python, ccf_pixels_python = self.calc_ccf(v_steps, new_line_start.astype('float64'),
                                                       new_line_end.astype('float64'),
                                                       x_pixel_wave.astype('float64'),
                                                       spectrum.astype('float64'),
                                                       new_line_weight.astype('float64'),
                                                       sn_p, zb)
            ccf_diff_pixel = ccf_pixels_python - ccf_pixels_c
            max_pixel_diff = np.amax(abs(ccf_diff_pixel))
            total_pixel_size = np.size(ccf_diff_pixel)
            total_pixel_diff = np.size(np.where(ccf_diff_pixel != 0.0)[0])

            ccf_diff = ccf_python-ccf
            max_ccf_diff = np.amax(abs(ccf_diff))
            total_ccf_size = np.size(ccf_diff)
            total_ccf_diff = np.size(np.where(ccf_diff != 0.0)[0])
            self.d_print("max diff of ccf in pixels: ", max_pixel_diff, '(', total_pixel_diff, '/', total_pixel_size, ')', info=True)
            self.d_print("max diff of ccf in v-steps: ", max_ccf_diff, '(', total_ccf_diff, '/', total_ccf_size, ')', info=True)
            """
        else:
            sn_p = np.ones(n_pixel)
            ccf, ccf_pixels_python = self.calc_ccf(v_steps, new_line_start.astype('float64'),
                                                   new_line_end.astype('float64'),
                                                   x_pixel_wave.astype('float64'),
                                                   spectrum.astype('float64'),
                                                   new_line_weight.astype('float64'),
                                                   sn_p, zb/LIGHT_SPEED_M)

        return ccf

    def calc_ccf(self, v_steps, new_line_start, new_line_end, x_pixel_wave, spectrum, new_line_weight, sn, zb):
        """ Cross correlation by the shifted mask line and the spectrum data of one order for each velocity step.

        Args:
            v_steps (int): Total velocity steps.
            new_line_start (numpy.ndarray): Start of the mask line.
            new_line_end (numpy.ndarray): End of the mask line.
            x_pixel_wave (numpy.ndarray): Wavelength calibration of the pixels.
            spectrum (numpy.ndarray): 1D Spectrum data.
            new_line_weight (numpy.ndarray): Mask weight
            sn (numpy.ndarray): Additional SNR scaling factor (comply with the implementation of CCF of C version)
            zb (float): Redshift at the observation time.

        Returns:
            numpy.ndarray: ccf at velocity steps.
            numpy.ndarray: Intermediate CCF numbers at pixels.
        """

        ccf = np.zeros(v_steps)
        shift_lines_by = (1.0 + (self.velocity_loop / LIGHT_SPEED)) / (1.0 + zb)

        n_pixel = np.shape(x_pixel_wave)[0] - 1                # total size in  x_pixel_wave_start
        n_line_index = np.shape(new_line_start)[0]

        # pix1, pix2 = 10, n_pixel - 11
        pix1, pix2 = 0, n_pixel-1
        x_pixel_wave_end = x_pixel_wave[1: n_pixel+1]            # total size: n_pixel
        x_pixel_wave_start = x_pixel_wave[0: n_pixel]
        ccf_pixels = np.zeros([v_steps, n_pixel])

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
                        if n > pix2:
                            break
                        if n < pix1:
                            continue
                        # if there is overlap
                        if x_pixel_wave_start[n] <= line_end_wave and x_pixel_wave_end[n] >= line_start_wave:
                            wave_start = max(x_pixel_wave_start[n], line_start_wave)
                            wave_end = min(x_pixel_wave_end[n], line_end_wave)
                            mask_spectra_doppler_shifted[n] = line_weight * (wave_end - wave_start) / \
                                (x_pixel_wave_end[n] - x_pixel_wave_start[n])

                            if n in idx_collection:
                                pass
                                # print(str(n), ' already taken')
                            else:
                                idx_collection.append(n)
            ccf_pixels[c, :] = spectrum * mask_spectra_doppler_shifted * sn
            ccf[c] = np.nansum(ccf_pixels[c, :])
        return ccf, ccf_pixels

    def analyze_ccf(self, ccf, row_for_analysis=None):
        """Analyze cross correlation results.

        Do summation on cross correlation values over specified orders.

        Args:
            ccf (numpy.ndarray): A container storing cross correlation values of all orders at each velocity step.
            row_for_analysis (numpy.ndarray): Rows for analysis. Defaults to None, meaning skipping the first row.

        Returns:
            numpy.ndarray: 2D array containing the cross correlation results of all orders plus
            a blank row, a row with velocity steps and a row with the summation of cross correlation values over orders.

        """
        total_seg_rv = np.shape(ccf)[0] - self.ROWS_FOR_ANALYSIS
        ccf[total_seg_rv+1, :] = self.velocity_loop
        if row_for_analysis is None:
            row_for_analysis = np.arange(0, total_seg_rv, dtype=int)
        # skip order 0
        ccf[total_seg_rv + self.ROWS_FOR_ANALYSIS - 1, :] = np.nansum(ccf[row_for_analysis, :], axis=0)
        return ccf

    def get_rv_guess(self):
        # rv guess from the peak or valley
        return self.get_rv_estimation(self.header, self.init_data)

    @staticmethod
    def get_rv_estimation(hdu_header, init_data=None):
        rv_guess = 0.0
        if 'QRV' in hdu_header:
            rv_guess = hdu_header['QRV']
        elif 'STAR_RV' in hdu_header:
            rv_guess = hdu_header['STAR_RV']
        elif 'TARGRADV' in hdu_header:              #kpf
            rv_guess = hdu_header['TARGRADV']
        elif init_data != None:
            rv_guess = init_data[RadialVelocityAlgInit.RV_CONFIG][RadialVelocityAlgInit.STAR_RV]

        return rv_guess

    @staticmethod
    def rv_estimation_from_ccf_order(ccf_v, velocities, first_guess, mask_method=None):
        abs_min_idx = np.argmin(np.absolute(ccf_v))
        abs_max_idx = np.argmax(np.absolute(ccf_v))
        ccf_dir = 1 if (mask_method is not None) and (mask_method in ['thar', 'lfc', 'etalon']) else -1

        if abs_min_idx == abs_max_idx:         # no ccf result (all zero)
            vel_order = 0.0
            ccf_order = ccf_v[abs_min_idx]
            ccf_dir = 0
        elif ccf_dir > 0:                      # pointing upwards
            vel_order = 0.0
            ccf_order = ccf_v[abs_max_idx]
        else:                                   # pointing downwards
            vel_order = velocities[abs_min_idx] # if first_guess != 0.0 else first_guess
            ccf_order = ccf_v[abs_min_idx]

        return vel_order, ccf_order, ccf_dir

    @staticmethod
    def fit_ccf(result_ccf, rv_guess, velocities, mask_method=None, velocity_cut=500.0, rv_guess_on_ccf=False,
                vel_span_pixel=None):
        """Gaussian fitting to the values of cross correlation vs. velocity steps.

        Find the radial velocity from the summation of cross correlation values over orders by the use of
        Gaussian fitting and starting from a guessed value.

        Args:
            result_ccf (numpy.ndarray): 1D array containing summation the summation of cross correlation data over
                orders. Please refer to `Returns` of :func:`~alg.RadialVelocityAlg.get_rv_on_spectrum()`.
            rv_guess (float): Approximation of radial velocity.
            velocities (np.array): An array of velocity steps.
            mask_method (str): mask method for ccf, default to None.
            velocity_cut (float, optional): Range limit around the guessed radial velocity. Defaults to 100.0 (km/s).
            rv_guss_on_ccf (bool, optional): If doing rv guess per ccf values and mask method.
            vel_span_pixel (float, optional) Velocity width per pixel for rv error calculation.
        Returns:
            tuple: Gaussian fitting mean and values for the fitting,

                * **gaussian_fit** (*fitting.LevMarLSQFitter*): Instance for doing Gussian fitting based on
                  Levenberg-Marquardt algorithm and least squares statistics.
                * **mean** (*float*): Mean value from Gaussian fitting.
                * **g_x** (*numpy.ndarray*): Collection of velocity steps for Gaussian fitting.
                * **g_y** (*numpy.ndarray*): Collection of cross correlation summation offset to the mean of
                  cross correlation summation values along *g_x*.

        """
        if mask_method is not None:
            mask_method = mask_method.lower()
        if rv_guess_on_ccf:  # kpf get rv_guess from the ccf values
            # print('first guess: ', rv_guess)
            rv_guess, ccf_guess, ccf_dir = RadialVelocityAlg.rv_estimation_from_ccf_order(result_ccf, velocities,
                        rv_guess, mask_method)
            # print('second guess: ', rv_guess, ' mask: ', mask_method)
        else:
            ccf_dir = -1

        rv_error = 0.0
        if ccf_dir == 0 and rv_guess == 0.0:
            return None, rv_guess, None, None, rv_error

        def gaussian_rv(v_cut, rv_mean, sd):
            # amp = -1e7 if ccf_dir < 0 else 1e7
            ccf = result_ccf
            i_cut = (velocities >= rv_mean - v_cut) & (velocities <= rv_mean + v_cut)
            if not i_cut.any():
                return None, None, None
            g_x = velocities[i_cut]
            g_y = ccf[i_cut] - np.nanmedian(ccf[i_cut])
            y_dist = abs(np.nanmax(g_y) - np.nanmin(g_y)) * 100
            amp = max(-1e7, np.nanmin(g_y) - y_dist) if ccf_dir < 0 else min(1e7, np.nanmax(g_y) + y_dist)

            if sd is None:
                g_init = models.Gaussian1D(amplitude=amp, mean=rv_mean)
            else:
                g_init = models.Gaussian1D(amplitude=amp, mean=rv_mean, stddev=sd)

            gaussian_fit = FIT_G(g_init, g_x, g_y)
            return gaussian_fit, g_x, g_y

        two_fitting = True

        # first gaussian fitting
        if mask_method in RadialVelocityAlg.vel_range_per_mask.keys():
            velocity_cut = RadialVelocityAlg.vel_range_per_mask[mask_method]
            sd = 0.5            # for narrower velocity range
            two_fitting = False
        else:
            sd = 5.0
        g_fit, g_x, g_y = gaussian_rv(velocity_cut, rv_guess, sd)
        if g_fit is not None \
                and g_x[0] <= g_fit.mean.value <= g_x[-1] \
                and two_fitting:
            v_cut = 25.0
            # print('mean before 2nd fitting: ', g_fit.mean.value)

            g_fit2, g_x2, g_y2 = gaussian_rv(v_cut, g_fit.mean.value, sd)
            if g_fit2 is not None and \
                    not (g_x2[0] <= g_fit2.mean.value <= g_x2[-1]):
                # print('mean after 2nd fitting (out of range): ', g_fit2.mean.value)
                g_fit2 = None
        else:
            g_fit2 = None

        if vel_span_pixel is not None and g_fit is not None:
            if g_fit2 is None or math.isnan(g_fit.mean.value):
                g_mean = rv_guess               # use the 1st guess if the 2nd fitting fails
                f_wid = velocity_cut            # use the 1st vel range if the 2nd fitting fails
            else:
                g_mean = g_fit.mean.value
                f_wid = v_cut

            rv_error = RadialVelocityAlg.ccf_error_calc(velocities, result_ccf, f_wid*2, vel_span_pixel, g_mean)

        if g_fit2 is not None and not math.isnan(g_fit2.mean.value):
            return g_fit2, g_fit2.mean.value, g_x2, g_y2, rv_error
        elif g_fit is not None:
            return g_fit, (0.0 if math.isnan(g_fit.mean.value) else g_fit.mean.value), g_x, g_y, rv_error
        else:
            return None, 0.0, None, None, 0.0

    def compute_segments_ccf(self, ccf):
        total_segments = np.shape(ccf)[0] - RadialVelocityAlg.ROWS_FOR_ANALYSIS
        rv_segments = np.zeros(total_segments)
        erv_segments = np.zeros(total_segments)
        v_span = self.get_vel_span_pixel()
        mask_type =  self.get_orderlet_masktype(self.spectro, self.orderletname, self.init_data)
        for i in range(total_segments):
            _, rv_segments[i], _, _, erv_segments[i] = self.fit_ccf(
                ccf[i, :], self.get_rv_guess(), self.init_data[RadialVelocityAlgInit.VELOCITY_LOOP],
                mask_type,
                rv_guess_on_ccf=(self.spectro == 'kpf'),
                vel_span_pixel=v_span
            )
        return rv_segments, erv_segments

    def output_ccf_to_dataframe(self, ccf, ref_head=None, ervs=None, ref_ccf=None):
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

        # calculate rv on ccfs summation
        _, rv_result, _, _, rv_error = self.fit_ccf(
            ccf[-1, :], self.get_rv_guess(), self.init_data[RadialVelocityAlgInit.VELOCITY_LOOP],
            self.get_orderlet_masktype(self.spectro, self.orderletname, self.init_data),
            rv_guess_on_ccf=(self.spectro == 'kpf'),
            vel_span_pixel=self.get_vel_span_pixel())

        # compute rv and erv on each segment and overwrite erv which is computed before reweighting if needed
        rv_segments, erv_segments = self.compute_segments_ccf(ccf)
        if ervs is not None:
            erv_segments = ervs

        ratio_ccf = ref_ccf if ref_ccf is not None else None
        def f_decimal(num):
            return float("{:.10f}".format(num))

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
            results.attrs['CCFJDSUM'] = self.get_obs_time(seg=-1)   # exposure time from the header
            results.attrs['CCFJDSEG'] = self.obs_jd                 # an array for all segments
            results.attrs['CCF-RVC'] = (f_decimal(rv_result), 'BaryC RV (km/s)')
            results.attrs['CCFSTART'] = self.init_data[RadialVelocityAlgInit.VELOCITY_LOOP][0]
            results.attrs['CCFSTEP'] = self.rv_config[RadialVelocityAlgInit.STEP]
            results.attrs['TOTALSTP'] = self.velocity_steps
            results.attrs['STARTORD'] = self.start_order
            results.attrs['ENDORDER'] = self.end_order
            results.attrs['TOTALORD'] = self.end_order - self.start_order+1
            results.attrs['BARY'] = self.zb * 1.0e-3  # from m/sec to km/sec, an array for all segments
            results.attrs['STARRV'] = self.init_data[RadialVelocityAlgInit.RV_CONFIG][RadialVelocityAlgInit.STAR_RV]
            results.attrs['CCF-ERV'] = f_decimal(rv_error)   # error on ccf summation
            results.attrs['ERV_SEGMS'] = erv_segments
            results.attrs['RV_SEGMS'] = rv_segments
            results.attrs['RV_MEAN'] = self.weighted_rv(rv_segments, rv_segments.size, ratio_ccf)
            results.attrs['ERV_MEAN'] = self.weighted_rv_error(erv_segments, erv_segments.size, ratio_ccf) # error on mean erv

        return results

    def is_none_fiberobject(self, fiberobj_key):
        """ Check if fiber object is None

        Args:
            fiber_key (str): SCI-OBJ, SKY-OBJ, or CAL-OBJ for kpf

        Returns:
            bool: is None or not

        """

        if fiberobj_key is None:
            return False
        try:
            is_none = (self.header[fiberobj_key] == 'None')
        except KeyError:
            is_none = False

        return is_none

    def compute_rv_by_cc(self, start_seg=None, end_seg=None, ref_ccf=None, print_progress=None):
        """Compute radial velocity by using cross correlation method.

        Compute and analyze radial velocity on level 1 data based on the specified pixel positions and the order range
        and output the result in both numpy array and Pandas DataFrame styles.

        Args:
            start_seg (int, optional): Start segment of the data. Defaults to None.
                            The number means the position relative to the first defined segment
                            if it is greater than or equal to 0, otherwise it means the position relative to the last
                            segment.
            end_seg (int, optional): End segment of the data. Defaults to None.
                            The number has the same meaning as that of `star_seg`.
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
        self.d_print('RadialVelocityAlg: computing radial velocity ... ')

        if self.spectrum_data is None or self.spectrum_data.size == 0:
            return {'ccf_df': None, 'ccf_ary': None, 'jd': self.obs_jd, 'msg': 'no spectral data'}
        elif self.is_none_fiberobject(self.get_fiber_object_in_header(self.spectro, self.orderletname)):
            return {'ccf_df': None, 'ccf_ary': None, 'jd': self.obs_jd, 'msg': 'fiber object is None'}

        self.get_segment_limits()

        ccf, msg = self.get_rv_on_spectrum(start_seg=start_seg, end_seg=end_seg)
        if ccf is None:
            return {'ccf_df': None, 'ccf_ary': None, 'jd': self.obs_jd, 'msg': msg}

        total_seg_rv = np.shape(ccf)[0] - self.ROWS_FOR_ANALYSIS
        erv_segments = None
        if ref_ccf is not None:
            rv_segs, erv_segments = self.compute_segments_ccf(ccf)

            ccf, _ = self.reweight_ccf(ccf[0:total_seg_rv, :], end_seg-start_seg+1, ref_ccf, self.reweighting_ccf_method,
                                    s_seg=start_seg)

        analyzed_ccf = self.analyze_ccf(ccf)

        df = self.output_ccf_to_dataframe(analyzed_ccf, ervs=erv_segments, ref_ccf=ref_ccf )
        return {'ccf_df': df, 'ccf_ary': analyzed_ccf, 'jd': self.obs_jd}

    @staticmethod
    def ccf_error_calc(velocities, ccfs, fit_wid, vel_span_pixel, rv_guess = 0.0):
        """Estimate photon-limited RV uncertainty of computed CCF.

           Calculate weighted slope information of CCF and convert to approximate RV uncertainty based on
           photon noise alone.

        Args:
            velocities (np.ndarray): velocity steps for CCF computation
            ccfs (np.ndarray): cross correlation results on velocities
            fit_wid (float): velocity width of the CCF.
            vel_span_pixel (float): approximate velocity span per CCD pixel.

        Returns:
            float: Estimated photon-limited uncertainty of RV measurement using specified ccf

        """
        vel_step = np.mean(np.diff(velocities))    # km/s,  velocity coverage per step
        n_scale_pix = vel_step / vel_span_pixel    # number of spectral pixels per ccf velocity step

        inds_fit = np.where((velocities >= (rv_guess - fit_wid / 2.)) & (velocities <= (rv_guess + fit_wid / 2.)))
        vels_fit = velocities[inds_fit]
        ccfs_fit  = ccfs[inds_fit]

        # the cases causing crashes
        if not ccfs_fit.any() or np.size(np.where(ccfs_fit < 0)[0]) > 0:
            return 0.0

        noise_ccf = (ccfs_fit) ** 0.5
        deriv_ccf = np.gradient(ccfs_fit, vels_fit)

        weighted_slopes = (deriv_ccf) ** 2. / (noise_ccf) ** 2.

        top = (np.sum(weighted_slopes)) ** 0.5
        bottom = (np.sum(ccfs_fit)) ** 0.5
        qccf = (top / bottom) * (n_scale_pix ** 0.5)
        sigma_ccf = 1. / (qccf * ((np.sum(ccfs_fit)) ** 0.5))  # km/s

        return sigma_ccf


    @staticmethod
    def is_good_reweighting_method(method, for_ratio=False):
        return method in ['ccf_max', 'ccf_mean'] if for_ratio else method in RadialVelocityAlg.CCF_Methods

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

        #row_val = order_val[np.arange(0, e_idx-s_idx+1, dtype=int), :]
        row_val = order_val[np.arange(s_idx, e_idx+1, dtype=int), :]
        row_val = np.where((row_val < 0.0), 0.0, row_val)

        # t_val = np.zeros(np.shape(row_val)[0])  # check if this setting is needed
        if reweighting_method == 'ccf_max':
            t_val = np.nanpercentile(row_val, 95, axis=1)
        elif reweighting_method == 'ccf_mean':
            t_val = np.nanmean(row_val, axis=1)

        t_val = np.where(np.isnan(t_val), 0.0, t_val)

        if max_ratio is not None:
            max_t_val = np.max(t_val)
            if max_t_val != 0.0:
                t_val = (t_val/max_t_val) * max_ratio
        t_seg = np.shape(order_val)[0]
        result_ratio_table = np.zeros(t_seg)
        result_ratio_table[s_idx:e_idx+1] = t_val

        ratio_table = {'segment': np.arange(0, t_seg, dtype=int),
                       'ratio': result_ratio_table}

        df = pd.DataFrame(ratio_table)
        if output_csv:
            if not os.path.isdir(os.path.dirname(output_csv)):
                os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df.to_csv(output_csv, index=False)

        return df

    @staticmethod
    def weighted_avg_and_std(values, weights):
        """
        Return the weighted average and standard deviation.

        Args:
            values (array): values to average
            weights (array): weights with same shape as values

        Returns:
            (average, standard deviation)

        """
        average = np.average(values, weights=weights)

        variance = np.average((values - average) ** 2, weights=weights)
        return (average, variance ** 0.5)

    @staticmethod
    def weighted_sigma_clipped_mean(values, weights, max_iters=7, sigma=3):
        """
        Return a weighted sigma-clipped mean

        Args:
            values (array): array of values to average
            weigths (array): weights for each value
            max_iters (int): (default=7) maximum number of iterations, will halt when no more outliers are clipped
            sigma (float): (default=3.0) clip outlier greater than sigma away from the mean

        Returns:
            float
        """
        good_idx = np.where(values != 0.0)[0]
        if good_idx.size == 0:
            return 0.0
        values = values[good_idx]
        weights = weights[good_idx]

        mean, std = RadialVelocityAlg.weighted_avg_and_std(values, weights)

        for i in range(max_iters):
            good_idx = np.where(np.abs(values - mean) <= sigma * std)[0]
            # bad_idx = np.where(np.abs(values - mean) > sigma * std)[0]

            if len(good_idx) == len(values):
                break
            else:
                values = values[good_idx]
                weights = weights[good_idx]
                mean, std = RadialVelocityAlg.weighted_avg_and_std(values, weights)

        return mean

    @staticmethod
    def weighted_rv(rv_arr, total_segment, reweighting_table_or_ccf, s_seg=0):
        if reweighting_table_or_ccf is None:
            reweighting_table_or_ccf = np.ones((total_segment, 1))
        if np.shape(reweighting_table_or_ccf)[1] >= 2:
            s_seg = 0 if s_seg is None else s_seg
            e_seg = s_seg + total_segment - 1
            c_idx = np.where((reweighting_table_or_ccf[:, 0] >= s_seg) &
                             (reweighting_table_or_ccf[:, 0] <= e_seg))[0]
            tval = reweighting_table_or_ccf[c_idx, -1]  # selected weighting on selected segment
            sval = reweighting_table_or_ccf[c_idx, 0].astype(int)  # selected original segment index
        else:
            tval = reweighting_table_or_ccf[0:total_segment, -1]
            sval = np.arange(0, total_segment, dtype=int)

        new_rv_arr = rv_arr[sval]
        w_rv = RadialVelocityAlg.weighted_sigma_clipped_mean(new_rv_arr, tval)
        return w_rv

    @staticmethod
    def weighted_rv_error(sigma_arr, total_segment, reweighting_table_or_ccf, s_seg=0):
        if reweighting_table_or_ccf is None:
            reweighting_table_or_ccf = np.ones((total_segment, 1))
        if np.shape(reweighting_table_or_ccf)[1] >= 2:
            s_seg = 0 if s_seg is None else s_seg
            e_seg = s_seg + total_segment - 1
            c_idx = np.where((reweighting_table_or_ccf[:, 0] >= s_seg) &
                             (reweighting_table_or_ccf[:, 0] <= e_seg))[0]
            tval = reweighting_table_or_ccf[c_idx, -1]  # selected weighting on selected segment
            sval = reweighting_table_or_ccf[c_idx, 0].astype(int)  # selected original segment index
        else:
            tval = reweighting_table_or_ccf[0:total_segment, -1]
            sval = np.arange(0, total_segment, dtype=int)

        sigma_arr2 = sigma_arr[sval]

        r_sigma_arr = [ tval[s]/sigma_arr2[s]**2 if sigma_arr2[s] != 0.0 else 0.0 for s in range(np.shape(sigma_arr2)[0])]
        sum_r_err = sum(r_sigma_arr)
        res = math.sqrt(1/sum_r_err) if sum_r_err != 0.0 else 0.0
        return res

    @staticmethod
    def reweight_ccf(crt_rv, total_segment, reweighting_table_or_ccf, reweighting_method, s_seg=0,
                     do_analysis=False, velocities=None):
        """Reweighting ccf orders.

        Reweight the CCF ordres based on the given CCF ratios or CCF orders from the observation template.

        Args:
            crt_rv (numpy.ndarray): CCF orders.
            total_segment (int): Total segments for reweighting. It is in default from the first row of `crt_rv`.
            reweighting_table_or_ccf (numpy.ndarray): Ratios among CCF orders or CCF data from the observation template.
            reweighting_method (str): Reweighting methods, **ccf_max**, **ccf_mean**, or **ccf_steps**.
            s_seg (int, optional): The start order index for reweighting. This is used to select the row from `crt_rv`
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

        total_segment = min(total_segment, ny)

        if reweighting_method == 'ccf_max' or reweighting_method == 'ccf_mean' or reweighting_method == 'ccf_static':
            # if the ratio table containing a column of order index, using s_order to select the ratio with
            # order index from s_order to s_order+total_order-1
            if np.shape(reweighting_table_or_ccf)[1] >= 2:
                s_seg = 0 if s_seg is None else s_seg
                e_seg = s_seg + total_segment - 1
                c_idx = np.where((reweighting_table_or_ccf[:, 0] >= s_seg) &
                                 (reweighting_table_or_ccf[:, 0] <= e_seg))[0]
                tval = reweighting_table_or_ccf[c_idx, -1]                   # selected weighting on selected segment
                sval = reweighting_table_or_ccf[c_idx, 0].astype(int)        # selected original segment index
                crt_rv = crt_rv[c_idx, :]
                total_segment = np.size(tval)
            else:
                tval = reweighting_table_or_ccf[0:total_segment, -1]
                sval = np.arange(0, total_segment, dtype=int)

            new_crt_rv = np.zeros([ny + RadialVelocityAlg.ROWS_FOR_ANALYSIS, nx])

            if reweighting_method == 'ccf_static':
                ccf_sums = np.nansum(crt_rv, axis=1)     # summation along each order

                for idx in range(total_segment):
                    if ccf_sums[idx] > 0. and tval[idx] != 0.0:
                        new_crt_rv[sval[idx], :] = (crt_rv[idx, :] / ccf_sums[idx]) * tval[idx]
            else:
                max_index = np.where(tval == np.max(tval))[0][0]     # the max from ratio table, 1.0 if ratio max is 1.0
                oval = np.nanpercentile(crt_rv[0:total_segment], 95, axis=1) if reweighting_method == 'ccf_max' \
                    else np.nanmean(crt_rv[0:total_segment], axis=1) # max or mean from each order

                oval_at_index = oval[max_index]                      # value from oder of max_index
                if oval_at_index == 0.0:      # order of max_index has value 0.0, skip reweighting, returns all zeros
                    return new_crt_rv
                oval = oval/oval_at_index     # ratio of orders before reweighting, value at order of max_index is 1.0

                for order in range(total_segment):
                    if oval[order] != 0.0:
                        new_crt_rv[sval[order], :] = crt_rv[order, :] * tval[order]/oval[order]
        elif reweighting_method == 'ccf_steps':             # assume crt_rv and reweighting ccf cover the same orders
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                new_crt_rv = np.zeros((total_segment + RadialVelocityAlg.ROWS_FOR_ANALYSIS, nx))
                for order in range(total_segment):
                    if np.size(np.where(crt_rv[order, :] != 0.0)[0]) > 0:
                        new_crt_rv[order, :] = crt_rv[order, :] * \
                                           np.nanmean(reweighting_table_or_ccf[order, :]/crt_rv[order, :])

        if do_analysis:
            row_for_analysis = np.arange(0, ny, dtype=int)
            new_crt_rv[ny + RadialVelocityAlg.ROWS_FOR_ANALYSIS - 1, :] = \
                np.nansum(new_crt_rv[row_for_analysis, :], axis=0)
            if velocities is not None and np.size(velocities) == nx:
                new_crt_rv[ny + RadialVelocityAlg.ROWS_FOR_ANALYSIS - 2, :] = velocities
        return new_crt_rv, ny
