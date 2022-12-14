# Standard dependencies
"""
    This module defines class `RadialVelocity` which inherits from `KPF1_Primitive` and provides methods to perform
    the event on radial velocity in the recipe.

    Attributes:
        RadialVelocity

    Description:
        * Method `__init__`:

            RadialVelocity constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `RadialVelocity` event issued in the recipe:

                    - `action.args[0] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing data constructed by
                      optimal extraction.
                    - `action.args[1] (dict)`: Result from the init work made by `RadialVelocityInit` which makes
                      mask lines and velocity steps based on star and other module associated configuration for
                      radial velocity computation.
                    - `action.args[2] (kpfpipe.models.level2.KPF2)`: Instance of `KPF2` containing radial velocity
                      results. If not existing, it is None.
                    - `action.args[3] (str)`: Extension name associated with the level 1 science data.
                    - `action.args['area_def'] (list, optional)`: pixel area, [start_y, end_y, start_x, end_x],
                      to be processed. Defaults to None.
                    - `action.args['segment_def'] (str)`: csv file defining segment wavelength information.
                    - `action.args['order_def'] (str)`: csv file defining order limits information.
                    - `action.args['ccf_ext'] (str)`: Extension name containing the ccf results.
                    - `action.args['rv_ext'] (str)`: Extension name containing rv results.
                    - `action.args['input_ref'] (np.ndarray|str|pd.DataFrame, optional)`: Reference for
                      reweighting ccf orders. Defaults to None.
                    - `action.args['reweighting_method'] (str, optional)`: reweighting method. Defaults to None.
                    - `action.args['start_seg'] (int)`: Index of first segment to be processed. Defaults to None.
                    - `action.args['end_seg'] (int)`: Index of last segment to be processed. Defaults to None.

                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of radial velocity in the master
                  config file associated with the recipe.

            and the following attributes are defined to initialize the object,

                - `input (kpfpipe.models.level1.KPF1)`: Instance of `KPF1`, assigned by `action.args[0]`.
                - `rv_init (dict)`: Result from radial velocity init.
                - `output_level2 (kpfpipe.models.level2.KPF2)`: Instance of `KPF2`, assigned by `action.args[2]`.
                - `sci_names (list)`: Name of the order to be processed, assigned by `action.args[2]`.
                - `area_def (list)`: Pixel area to be processed.
                - `segment_def (np.ndarray)`: segments to be processed.
                - `order_def (np.ndarray)`: orders to be processed.
                - `start_seg (int)`: Index of first segment to be processed.
                - `end_seg (int)`: Index of last segment to be processed.
                - `reweighting_method (str)`: reweighting method associated with `action.args['reweighting_method']`.
                - `config_path (str)`: Path of config file for radial velocity.
                - `config (configparser.ConfigParser)`: Config context.
                - `logger (logging.Logger)`: Instance of logging.Logger.
                - `spectrum_data (numpy.ndarray)`: Reduced 1D data of all orders from optimal extraction, associated
                  with `action.args[0]`.
                - `wave_cal (numpy.ndarray)`: Wavelength calibration data, associated with `action.args[0]`.
                - `header (fits.header.Header)`: Fits header of HDU associated with `spectrum_data`.
                - `ref_ccf (numpy.ndarray)`: Reference or ratio of cross correlation values for scaling the computation
                  of cross correlation, associated with `action.args['input_ref']`.
                - `alg (RadialVelocityAlg)`: Instance of RadialVelocityAlg which has operation codes for the
                  computation of radial velocity.

        * Method `__perform`:

            RadialVelocity returns the result in `Arguments` object which contains the original input
            level 1 data object (`KPF1`) plus an extension with the radial velocity result.
            (the result will be put into a level 2 data object after level 2 data model is implemented.)

    Usage:
        For the recipe, the optimal extraction event is issued like::

            rv_init = RadialVelocityInit()
            :
            lev1_data = kpf1_from_fits(input_L1_file, data_type='KPF')
            rv_data = RadialVelocity(lev1_data, rv_init, order_name=order_name)
            :
"""

import configparser
import numpy as np
import os
import os.path
import pandas as pd
# Pipeline dependencies
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.models.level1 import KPF1
from kpfpipe.models.level2 import KPF2

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

from modules.radial_velocity.src.alg import RadialVelocityAlg
from modules.radial_velocity.src.alg_rv_init import RadialVelocityAlgInit
from astropy.time import Time

DEFAULT_CFG_PATH = 'modules/radial_velocity/configs/default.cfg'


class RadialVelocity(KPF1_Primitive):

    default_args_val = {
        'order_names': ['SCIFLUX'],
        'output_level2': None,
        'ccf_engine': 'c',
        'ccf_ext': 'CCF',
        'rv_ext': 'RV',
        'rv_set': 0,
        'obstime': None,
        'exptime': None
    }

    RV_COL_ORDERLET = 'orderlet'
    RV_COL_START_W = 's_wavelength'
    RV_COL_END_W = 'e_wavelength'
    RV_COL_SEG_NO = 'segment no.'
    RV_COL_ORD_NO = 'order no.'
    RV_COL_RV = 'RV'
    RV_COL_RV_ERR = 'RV error'
    RV_COL_CCFJD = 'CCFJD'
    RV_COL_BARY = 'Bary_RVC'
    RV_COL_SOURCE = 'source'
    rv_col_names = [RV_COL_ORDERLET, RV_COL_START_W, RV_COL_END_W, RV_COL_SEG_NO, RV_COL_ORD_NO,
                    RV_COL_RV, RV_COL_RV_ERR, RV_COL_CCFJD, RV_COL_BARY, RV_COL_SOURCE]
    rv_col_on_orderlet = [RV_COL_ORDERLET, RV_COL_SOURCE]

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        # Initialize parent class
        KPF1_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        self.input = action.args[0]
        self.rv_init = action.args[1]
        self.output_level2 = action.args[2]
        self.sci_names = action.args[3] if isinstance(action.args[3], list) else [action.args[3]]
        self.ref_ccf = None

        if 'input_ref' in args_keys:
            if isinstance(action.args['input_ref'], np.ndarray):
                self.ref_ccf = action.args['input_ref']
            elif isinstance(action.args['input_ref'], pd.DataFrame):
                self.ref_ccf = action.args['input_ref'].values
            elif isinstance(action.args['input_ref'], str) and os.path.exists(action.args['input_ref']):
                ratio_df = pd.read_csv(action.args['input_ref'])
                self.ref_ccf = ratio_df.values

        self.ccf_engine = action.args['ccf_engine'].lower() \
            if 'ccf_engine' in args_keys and action.args['ccf_engine'] is not None \
            else self.default_args_val['ccf_engine']
        self.area_def = action.args['area_def'] if 'area_def' in args_keys else None
        self.segment_limits = self.load_csv(action.args['segment_def']) if 'segment_def' in args_keys else None
        self.order_limits = self.load_csv(action.args['order_def']) if 'order_def' in args_keys else None
        self.start_seg = action.args['start_seg'] if 'start_seg' in args_keys else 0
        self.end_seg = action.args['end_seg'] if 'end_seg' in args_keys else None
        self.reweighting_method = action.args['reweighting_method'] if 'reweighting_method' in args_keys else None
        self.ccf_ext = action.args['ccf_ext'] if 'ccf_ext' in args_keys else self.default_args_val['ccf_ext']
        self.rv_ext = action.args['rv_ext'] if 'rv_ext' in args_keys else self.default_args_val['rv_ext']
        self.rv_set_idx = action.args['rv_set'] if 'rv_set' in args_keys else self.default_args_val['rv_set']

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['radial_velocity']
        except:
            self.config_path = DEFAULT_CFG_PATH

        self.config.read(self.config_path)

        # start a logger
        self.logger = None
        # self.logger = start_logger(self.__class__.__name__, self.config_path)
        if not self.logger:
            self.logger = self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        self.spectrum_data_set = list()
        self.wave_cal_set = list()
        self.header_set = list()

        exptime_v = action.args['exptime'] if 'exptime' in args_keys else self.default_args_val['exptime']
        obstime_v = action.args['obstime'] if 'obstime' in args_keys else self.default_args_val['obstime']

        for sci in self.sci_names:
            self.spectrum_data_set.append(getattr(self.input, sci) if hasattr(self.input, sci) else None)

            wave = sci.replace('FLUX', 'WAVE') if 'FLUX' in sci else None
            self.wave_cal_set.append(getattr(self.input, wave) if (wave is not None and hasattr(self.input, wave))
                                     else None)
            neid_ssb = 'SSBJD100'   # neid case

            if neid_ssb in self.input.header['PRIMARY'] and neid_ssb not in self.input.header[sci]:
                self.input.header[sci][neid_ssb] = self.input.header['PRIMARY'][neid_ssb]
            elif neid_ssb not in self.input.header[sci]:     # kpf case
                if obstime_v != None:
                    m_obs = Time(obstime_v).jd - 2400000.5
                    if exptime_v is None:
                        exptime_v = 1.0
                if ('DATE-MID' in self.input.header['PRIMARY']): # kpf case
                    d_obs = 'DATE-MID'
                    exptime = 'EXPTIME' if 'EXPTIME' in self.input.header['PRIMARY'] else None

                    exptime_v = self.input.header['PRIMARY'][exptime] if exptime else 1.0
                    m_obs = Time(self.input.header['PRIMARY'][d_obs]).jd - 2400000.5

                if 'MJD-OBS' not in self.input.header[sci] or self.input.header[sci]['MJD-OBS'] != m_obs:
                    self.input.header[sci]['MJD-OBS'] = m_obs
                    self.input.header[sci]['EXPTIME'] = exptime_v
    
            self.input.header[sci]['MASK'] = self.rv_init['data']['mask_type']
            self.header_set.append(self.input.header[sci] if hasattr(self.input, 'header') and hasattr(self.input, sci)
                                   else None)

        self.total_orderlet = len(self.spectrum_data_set)

        # Order trace algorithm setup
        self.alg = RadialVelocityAlg(self.spectrum_data_set[0], self.header_set[0], self.rv_init,
                                     wave_cal=self.wave_cal_set[0],
                                     segment_limits=self.segment_limits,
                                     order_limits=self.order_limits,
                                     area_limits=self.area_def,
                                     config=self.config, logger=self.logger, ccf_engine=self.ccf_engine,
                                     reweighting_method=self.reweighting_method)

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # input argument must be KPF0
        success = isinstance(self.input, KPF1) and (self.ref_ccf is None or isinstance(self.ref_ccf, np.ndarray))

        return success

    def _post_condition(self) -> bool:
        """
        check for some necessary post condition
        """
        return True

    def _perform(self):
        """
        Primitive action -
        perform radial velocity computation by calling method 'compute_rv_by_cc' from RadialVelocityAlg.

        Returns:
            Level 1 data from the input plus an extension with the cross correlation results from all orders.
            (this part will be updated after level 2 data model is made.)
        """

        # _, nx, ny = self.alg.get_spectrum()

        if self.logger:
            self.logger.info("RadialVelocity: Start crorss correlation to find radial velocity... ")

        output_df = []

        if all( [s is not None and s.size != 0 for s in self.spectrum_data_set]):
            for i in range(self.total_orderlet):
                if i > 0:
                    self.alg.reset_spectrum(self.spectrum_data_set[i], self.header_set[i], self.wave_cal_set[i])
                if self.logger:
                    self.logger.info('RadialVelocity: computing radial velocity on orderlet '+ self.sci_names[i] + '...')

                rv_results = self.alg.compute_rv_by_cc(start_seg=self.start_seg, end_seg=self.end_seg, ref_ccf=self.ref_ccf)
                one_df = rv_results['ccf_df']
                if one_df is None:
                    if self.logger:
                        self.logger.info('RadialVelocity: orderlet ' + self.sci_names[i] + 'error => ' +
                                rv_results['msg'])
                    output_df = []
                    break
                else:
                    assert (not one_df.empty and one_df.values.any())
                    output_df.append(rv_results['ccf_df'])

        self.construct_level2_data(output_df)
        self.output_level2.receipt_add_entry('RadialVelocity', self.__module__, f'config_path={self.config_path}', 'PASS')

        if self.logger:
            self.logger.info("RadialVelocity: Receipt written")

        if self.logger:
            self.logger.info("RadialVelocity: Done!")

        return Arguments(self.output_level2)

    def construct_level2_data(self, output_df):
        if self.output_level2 is None:
            self.output_level2 = KPF2.from_l1(self.input)

        if (len(output_df) == 0):
            return True

        self.make_ccf_table(output_df)

        # make new rv table and append the new one to the existing one if there is
        new_rv_table = self.make_rv_table(output_df)
        self.output_level2.header[self.rv_ext]['star_rv'] = new_rv_table.attrs['star_rv']

        crt_rv_ext = self.output_level2[self.rv_ext] if hasattr(self.output_level2, self.rv_ext) else None
        if crt_rv_ext is None or np.shape(crt_rv_ext)[0] == 0:
            self.output_level2[self.rv_ext] = new_rv_table
            self.output_level2.header[self.rv_ext]['ccd'+str(self.rv_set_idx+1)+'row'] = 0
        else:
            first_row = np.shape(crt_rv_ext)[0]
            new_table_list = {}
            for c_name in self.rv_col_names:
                if c_name in self.rv_col_on_orderlet:
                    for o in range(len(output_df)):
                        c_name_orderlet = c_name+str(o+1)
                        new_list = crt_rv_ext[c_name_orderlet].tolist() + new_rv_table[c_name_orderlet].tolist()
                        new_table_list[c_name_orderlet] = new_list
                else:
                    new_list = crt_rv_ext[c_name].tolist() + new_rv_table[c_name].tolist()
                    new_table_list[c_name] = new_list
            self.output_level2[self.rv_ext] = pd.DataFrame(new_table_list)
            self.output_level2.header[self.rv_ext]['ccd' + str(self.rv_set_idx + 1) + 'row'] = first_row


        for o in range(len(output_df)):
            self.output_level2.header[self.rv_ext]['ccd'+str(self.rv_set_idx+1)+'rv'+str(o+1)] = \
                new_rv_table.attrs['ccd_rv'+str(o+1)]
        self.output_level2.header[self.rv_ext]['ccd'+str(self.rv_set_idx+1)+'rv'] = new_rv_table.attrs['rv']
        self.output_level2.header[self.rv_ext]['ccd'+str(self.rv_set_idx+1)+'jd'] = new_rv_table.attrs['ccd_jd']

        # self.output_level2.header[self.rv_ext]['zb'] = new_rv_table.attrs['zb']    # removed
        return True

    def make_ccf_table(self, output_df):
        total_orderlet = len(output_df)
        total_segment = np.shape(output_df[0].values)[0] - self.alg.ROWS_FOR_ANALYSIS

        all_ccf = np.zeros((self.total_orderlet, total_segment,
                            self.rv_init['data'][RadialVelocityAlgInit.VELOCITY_STEPS]))
        for i in range(total_orderlet):      # make ccf in 3d format, each ccf layer is for one SCI orderlet
            all_ccf[i, :, :] = output_df[i].values[0:total_segment, :]

        self.output_level2[self.ccf_ext] = all_ccf
        self.output_level2.header[self.ccf_ext]['startseg'] = self.start_seg
        self.output_level2.header[self.ccf_ext]['startv'] = \
            (self.rv_init['data'][RadialVelocityAlgInit.VELOCITY_LOOP][0], 'km/sec')
        self.output_level2.header[self.ccf_ext]['stepv'] = \
            (self.rv_init['data']['rv_config'][RadialVelocityAlgInit.STEP], 'km/sec')
        self.output_level2.header[self.ccf_ext]['totalv'] = self.rv_init['data']['velocity_steps']
        self.output_level2.header[self.ccf_ext]['masktype'] = output_df[0].attrs['MASKTYPE']
        for i in range(total_orderlet):
            self.output_level2.header[self.ccf_ext]['ccf'+str(i+1)] = self.sci_names[i]

        return True

    def make_rv_table(self, output_df):
        def f_decimal(num):
            return float("{:.10f}".format(num))

        rv_table = {}

        total_orderlet = len(output_df)
        total_segment = np.shape(output_df[0].values)[0] - self.alg.ROWS_FOR_ANALYSIS
        segment_table = self.alg.get_segment_info()

        velocities = self.rv_init['data'][RadialVelocityAlgInit.VELOCITY_LOOP]
        col_orderlets = np.zeros((total_segment, total_orderlet), dtype=float)
        col_rv = np.zeros(total_segment)
        for s in range(total_segment):
            sum_segment = np.zeros(np.shape(velocities)[0])
            for o in range(total_orderlet):
                ccf_orderlet = output_df[o].values[s, :]   # ccf per orderlet per segment
                sum_segment += ccf_orderlet                # summation per segment of all orderlets
                if self.start_seg <= s <= self.end_seg:
                    _,  orderlet_rv, _, _ = self.alg.fit_ccf(ccf_orderlet, self.alg.get_rv_guess(), velocities,
                                                        self.rv_init['data'][RadialVelocityAlgInit.MASK_TYPE])
                else:
                    orderlet_rv = 0.0
                col_orderlets[s, o] = orderlet_rv
            _, col_rv[s], _, _ = self.alg.fit_ccf(sum_segment, self.alg.get_rv_guess(), velocities,
                                                    self.rv_init['data'][RadialVelocityAlgInit.MASK_TYPE])

        col_sources = np.empty((total_segment, total_orderlet), dtype=object)
        final_sum_ccf = np.zeros(np.shape(velocities)[0])
        for o in range(total_orderlet):
            col_sources[:, o] = self.sci_names[o]
            rv_table[self.RV_COL_ORDERLET+str(o+1)] = col_orderlets[:, o]
            final_sum_ccf += output_df[o].values[-1, :]

        #s_seg = self.start_seg
        #e_seg = self.start_seg + total_segment
        s_seg = 0
        e_seg = total_segment
        rv_table[self.RV_COL_START_W] = segment_table[s_seg:e_seg, RadialVelocityAlg.SEGMENT_W1]
        rv_table[self.RV_COL_END_W] = segment_table[s_seg:e_seg, RadialVelocityAlg.SEGMENT_W2]
        rv_table[self.RV_COL_SEG_NO] = segment_table[s_seg:e_seg, RadialVelocityAlg.SEGMENT_IDX].astype(int)
        rv_table[self.RV_COL_ORD_NO] = segment_table[s_seg:e_seg,  RadialVelocityAlg.SEGMENT_ORD].astype(int)
        rv_table[self.RV_COL_RV] = col_rv
        rv_table[self.RV_COL_RV_ERR] =  np.zeros(total_segment)
        rv_table[self.RV_COL_CCFJD] = np.ones(total_segment) * output_df[0].attrs['CCFJDSUM']
        rv_table[self.RV_COL_BARY] = np.ones(total_segment) * output_df[0].attrs['BARY']

        for o in range(total_orderlet):
            rv_table[self.RV_COL_SOURCE+str(o+1)] = col_sources[:, o]

        results = pd.DataFrame(rv_table)
        for o in range(total_orderlet):
            results.attrs['ccd_rv'+str(o+1)] = output_df[o].attrs['CCF-RVC']
        _, final_rv, _, _ = self.alg.fit_ccf(final_sum_ccf, self.alg.get_rv_guess(), velocities,
                                             self.rv_init['data'][RadialVelocityAlgInit.MASK_TYPE])
        results.attrs['rv'] = (f_decimal(final_rv), 'BaryC RV (km/s)')
        results.attrs['ccd_jd'] = output_df[0].attrs['CCFJDSUM']
        results.attrs['star_rv'] = output_df[0].attrs['STARRV']
        # results.attrs['zb'] = output_df[0].attrs['ZB']    # removed

        return results

    @staticmethod
    def load_csv(filepath, header=None):
        if filepath is not None and os.path.isfile(filepath):
            df = pd.read_csv(filepath, header=header, sep='\s+|\t+|\s+\t+|\t+\s+', engine='python')
            return df.values
        else:
            return None
