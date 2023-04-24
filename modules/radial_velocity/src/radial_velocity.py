# Standard dependencies
"""
    This module defines class `RadialVelocity` which inherits from `KPF1_Primitive` and provides methods to perform
    the event on radial velocity in the recipe.

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
                    - `action.args['bary_corr'] (pd.DataFrame)`: extension name of bary correction table. Defaults to
                      'BARY_CORR'.
                    - `action.args['start_bary_index'] (int)`: starting index in bary correction table. Defaults to 0.
                    - `action.args['rv_correction_by_cal'] (bool)`: if using CAL fiber CCF to correct RV

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
        'exptime': None,
        'bary_corr': 'BARY_CORR',
        'start_bary_index': 0
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
    RV_COL_CAL = 'CAL RV'
    RV_COL_CAL_SOURCE ='source CAL'
    RV_WEIGHTS = 'CCF Weights'
    rv_col_names = [RV_COL_ORDERLET, RV_COL_START_W, RV_COL_END_W, RV_COL_SEG_NO, RV_COL_ORD_NO,
                    RV_COL_RV, RV_COL_RV_ERR, RV_COL_CAL, RV_COL_CCFJD, RV_COL_BARY, RV_COL_SOURCE, RV_COL_CAL_SOURCE, RV_WEIGHTS]
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
                ratio_df = pd.read_csv(action.args['input_ref'], sep='\s+')
                self.ref_ccf = ratio_df

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
        self.is_cal_cor = action.args['rv_correction_by_cal'] if 'rv_correction_by_cal' in args_keys else False
        bary_corr_ext = action.args['bary_corr'] if 'bary_corr' in args_keys else self.default_args_val['bary_corr']
        bc_table = getattr(self.input, bary_corr_ext) if hasattr(self.input, bary_corr_ext) else None
        start_bary_index = action.args['start_bary_index'] \
            if 'start_bary_index' in args_keys else self.default_args_val['start_bary_index']

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
            input_data = getattr(self.input, sci) if self.input is not None and hasattr(self.input, sci) else None
            input_header = self.input.header[sci] if input_data is not None and hasattr(self.input, 'header') else None
            self.spectrum_data_set.append(input_data)
            self.header_set.append(input_header)

            wave = sci.replace('FLUX', 'WAVE') if 'FLUX' in sci else None
            self.wave_cal_set.append(getattr(self.input, wave)
                            if (self.input is not None and wave is not None and hasattr(self.input, wave)) else None)
            neid_ssb = 'SSBJD100'   # neid case
            if input_data is None or input_header is None:
                continue

            if neid_ssb in self.input.header['PRIMARY'] and neid_ssb not in self.input.header[sci]:
                self.input.header[sci][neid_ssb] = self.input.header['PRIMARY'][neid_ssb]   # neid case
            elif neid_ssb not in self.input.header[sci]:     # kpf case
                if obstime_v != None:                        # obs and exptime passed externally
                    m_obs = Time(obstime_v).jd - 2400000.5
                    if exptime_v is None:
                        exptime_v = 1.0
                d_obs = 'DATE-MID'
                if d_obs in self.input.header['PRIMARY']: # get from primary header if the key exists
                    exptime_k = 'EXPTIME' if 'EXPTIME' in self.input.header['PRIMARY'] else None
                    exptime_v = self.input.header['PRIMARY'][exptime_k] if exptime_k else 1.0
                    m_obs = Time(self.input.header['PRIMARY'][d_obs]).jd - 2400000.5

                self.input.header[sci]['MJD-OBS'] = m_obs
                self.input.header[sci]['EXPTIME'] = exptime_v

            for hkey in ['IMTYPE', 'SCI-OBJ', 'SKY-OBJ', 'CAL-OBJ', 'STAR_RV', 'QRV', 'TARGRADV']:
                if hkey in self.input.header['PRIMARY']:
                    self.input.header[sci][hkey] = self.input.header['PRIMARY'][hkey]

            mod, mtype = RadialVelocityAlgInit.MASK_ORDERLET, RadialVelocityAlgInit.MASK_TYPE
            self.input.header[sci]['MASK'] = self.rv_init['data'][mtype] if not self.rv_init['data'][mod]  \
                else self.rv_init['data'][mod][RadialVelocityAlg.get_fiber_object_in_header('kpf', sci)][mtype]


        self.total_orderlet = len(self.spectrum_data_set)
        do_rv_corr = False
        if self.input is not None:
            key_sci = RadialVelocityAlgInit.KEY_SCI_OBJ
            key_cal = RadialVelocityAlgInit.KEY_CAL_OBJ
            sci_obj_v = self.input.header['PRIMARY'][key_sci] if key_sci in self.input.header['PRIMARY'] else None
            if self.rv_init['data'][RadialVelocityAlgInit.MASK_ORDERLET]:
                cal_mask = self.rv_init['data'][RadialVelocityAlgInit.MASK_ORDERLET][key_cal][RadialVelocityAlgInit.MASK_TYPE]
                do_rv_corr = cal_mask in ['lfc', 'thar'] and sci_obj_v.lower() == 'target'
        self.is_cal_cor = self.is_cal_cor and do_rv_corr

        vel_span_pixel = action.args['vel_span_pixel'] if 'vel_span_pixel' in args_keys else None

        try:
            self.alg = RadialVelocityAlg(self.spectrum_data_set[0], self.header_set[0], self.rv_init,
                                    wave_cal=self.wave_cal_set[0],
                                    segment_limits=self.segment_limits,
                                    order_limits=self.order_limits,
                                    area_limits=self.area_def,
                                    config=self.config, logger=self.logger, ccf_engine=self.ccf_engine,
                                    reweighting_method=self.reweighting_method,
                                    bary_corr_table=bc_table, start_bary_index=start_bary_index,
                                    orderlet = self.sci_names[0], vel_span_pixel = vel_span_pixel)
        except Exception as e:
            self.alg = None


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

    @staticmethod
    def is_sci(sci_name):
        return 'sci' in sci_name.lower()

    @staticmethod
    def is_cal(sci_name):
        return 'cal' in sci_name.lower()

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

        if self.alg is None:
            self.logger.info("RadialVelocity: no enough data to start the instance to do cross correlation... ")
            return Arguments(self.output_level2)
        output_df = []

        if all( [s is not None and s.size != 0 for s in self.spectrum_data_set]):
            for i in range(self.total_orderlet):
                if i > 0:
                    self.alg.reset_spectrum(self.spectrum_data_set[i], self.header_set[i], self.wave_cal_set[i],
                                            orderlet=self.sci_names[i])
                if self.logger:
                    self.logger.info('RadialVelocity: computing radial velocity on orderlet '+ self.sci_names[i] + '...')

                rv_results = self.alg.compute_rv_by_cc(start_seg=self.start_seg, end_seg=self.end_seg, ref_ccf=self.ref_ccf)
                one_df = rv_results['ccf_df']
                if one_df is None or one_df.empty or not one_df.values.any():
                    if self.logger:
                        self.logger.info('RadialVelocity: orderlet ' + self.sci_names[i] + ' message => ' +
                                rv_results['msg'])

                output_df.append(one_df)

        # do rv on CAL ccfs

        if all(v is None for v in output_df):
            if self.logger:
                self.logger.info("RadialVelocity: no L2 produced")
            return Arguments(None)

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
            for m in RadialVelocityAlg.vel_range_per_mask.keys():
                self.output_level2.header[self.rv_ext]['vr_' + m] = RadialVelocityAlg.vel_range_per_mask[m]
        else:
            first_row = np.shape(crt_rv_ext)[0]
            new_table_list = {}
            for c_name in self.rv_col_names:
                if c_name in self.rv_col_on_orderlet:
                    for o in range(len(output_df)):
                        if self.is_sci(self.sci_names[o]):
                            c_name_orderlet = c_name+str(o+1)
                            new_list = crt_rv_ext[c_name_orderlet].tolist() + new_rv_table[c_name_orderlet].tolist()
                            new_table_list[c_name_orderlet] = new_list
                else:
                    if c_name in crt_rv_ext and c_name in new_rv_table:
                        new_list = crt_rv_ext[c_name].tolist() + new_rv_table[c_name].tolist()
                        new_table_list[c_name] = new_list
            self.output_level2[self.rv_ext] = pd.DataFrame(new_table_list)
            self.output_level2.header[self.rv_ext]['ccd' + str(self.rv_set_idx + 1) + 'row'] = first_row


        for o in range(len(output_df)):
            if self.is_sci(self.sci_names[o]):
                self.output_level2.header[self.rv_ext]['ccd'+str(self.rv_set_idx+1)+'rv'+str(o+1)] = \
                    new_rv_table.attrs['ccd_rv'+str(o+1)]
                self.output_level2.header[self.rv_ext]['ccd' + str(self.rv_set_idx + 1) + 'erv' + str(o + 1)] = \
                    new_rv_table.attrs['ccd_erv' + str(o+1)]
        self.output_level2.header[self.rv_ext]['ccd'+str(self.rv_set_idx+1)+'rv'] = new_rv_table.attrs['rv']
        self.output_level2.header[self.rv_ext]['ccd'+str(self.rv_set_idx+1)+'erv'] = new_rv_table.attrs['rverr']
        self.output_level2.header[self.rv_ext]['ccd'+str(self.rv_set_idx+1)+'rvc'] = new_rv_table.attrs['rv_cal']
        self.output_level2.header[self.rv_ext]['ccd'+str(self.rv_set_idx+1)+'jd'] = new_rv_table.attrs['ccd_jd']
        self.output_level2.header[self.rv_ext]['rv'+str(self.rv_set_idx+1)+'corr'] = 'T' \
            if new_rv_table.attrs['do_rv_corr'] else 'F'
        self.output_level2.header[self.rv_ext]['rwccfrv']='F'
        # self.output_level2.header[self.rv_ext]['zb'] = new_rv_table.attrs['zb']    # removed
        return True

    def make_ccf_table(self, output_df):
        total_orderlet = len(output_df)
        for o_df in output_df:
            if o_df is not None:
                total_segment = np.shape(o_df.values)[0] - self.alg.ROWS_FOR_ANALYSIS
                break

        all_ccf = np.zeros((self.total_orderlet, total_segment,
                            self.rv_init['data'][RadialVelocityAlgInit.VELOCITY_STEPS]))
        for i in range(total_orderlet):      # make ccf in 3d format, each ccf layer is for one SCI orderlet
            if output_df[i] is not None:
                all_ccf[i, :, :] = output_df[i].values[0:total_segment, :]

        self.output_level2[self.ccf_ext] = all_ccf
        self.output_level2.header[self.ccf_ext]['startseg'] = self.start_seg
        self.output_level2.header[self.ccf_ext]['startv'] = \
            (self.rv_init['data'][RadialVelocityAlgInit.VELOCITY_LOOP][0], 'km/sec')
        self.output_level2.header[self.ccf_ext]['stepv'] = \
            (self.rv_init['data']['rv_config'][RadialVelocityAlgInit.STEP], 'km/sec')
        self.output_level2.header[self.ccf_ext]['totalv'] = self.rv_init['data']['velocity_steps']
        self.output_level2.header[self.ccf_ext]['totalsci'] = sum([1 if self.is_sci(s) else 0 for s in self.sci_names])

        mtype = RadialVelocityAlgInit.MASK_TYPE

        for i in range(total_orderlet):
            ccf_key = None
            if self.is_sci(self.sci_names[i]):
                ccf_key = 'sci_mask'
            elif self.is_cal(self.sci_names[i]):
                ccf_key = 'cal_mask'

            if ccf_key and ccf_key not in self.output_level2.header[self.ccf_ext]:
                if self.rv_init['data'][RadialVelocityAlgInit.MASK_ORDERLET]:
                    fiber_key = self.alg.get_fiber_object_in_header(self.alg.get_instrument().lower(), self.sci_names[i])
                    self.output_level2.header[self.ccf_ext][ccf_key] = \
                        self.rv_init['data'][RadialVelocityAlgInit.MASK_ORDERLET][fiber_key][mtype]
                else:
                    self.output_level2.header[self.ccf_ext][ccf_key] = self.rv_init['data'][mtype]


        for i in range(total_orderlet):
            self.output_level2.header[self.ccf_ext]['ccf'+str(i+1)] = self.sci_names[i]

        return True

    def make_rv_table(self, output_df):
        def f_decimal(num):
            return float("{:.10f}".format(num))

        rv_table = {}

        total_orderlet = len(output_df)
        for o_df in output_df:
            if o_df is not None:
                total_segment = np.shape(o_df.values)[0] - self.alg.ROWS_FOR_ANALYSIS
                break

        segment_table = self.alg.get_segment_info()

        velocities = self.rv_init['data'][RadialVelocityAlgInit.VELOCITY_LOOP]
        col_orderlets = np.zeros((total_segment, total_orderlet), dtype=float)
        col_rv = np.zeros(total_segment)
        col_rv_err = np.zeros(total_segment)
        col_rv_cal = np.zeros(total_segment)
        ins = self.alg.get_instrument().lower()

        sci_mask = None
        cal_mask = None
        for o in range(total_orderlet):
            if self.is_sci(self.sci_names[o]) and output_df[o] is not None:
                sci_mask = RadialVelocityAlg.get_orderlet_masktype(ins, self.sci_names[o], self.rv_init['data'])
            elif self.is_cal(self.sci_names[o]) and output_df[o] is not None:
                cal_mask = RadialVelocityAlg.get_orderlet_masktype(ins, self.sci_names[o], self.rv_init['data'])

        do_corr = self.is_cal_cor

        for s in range(total_segment):
            sum_segment = np.zeros(np.shape(velocities)[0])
            sum_rv = list()

            for o in range(total_orderlet):
                if output_df[o] is None:
                    if self.is_sci(self.sci_names[o]):
                        do_corr = False
                    continue

                ccf_orderlet = output_df[o].values[s, :]   # ccf per orderlet per segment
                if self.is_sci(self.sci_names[o]):
                    sum_segment += ccf_orderlet            # summation per segment of all orderlets


                # mtype = sci_mask if self.is_sci(self.sci_names[o]) else cal_mask
                if self.start_seg <= s <= self.end_seg:
                    orderlet_rv = output_df[o].attrs['RV_SEGMS'][s]
                    # _,  orderlet_rv, _, _, _ = self.alg.fit_ccf(ccf_orderlet, self.alg.get_rv_guess(), velocities,
                    #                                    mtype, rv_guess_on_ccf=(ins == 'kpf'))
                else:
                    orderlet_rv = 0.0

                if self.is_sci(self.sci_names[o]):
                    col_orderlets[s, o] = orderlet_rv
                elif self.is_cal(self.sci_names[o]):
                    col_rv_cal[s] = orderlet_rv

                if self.is_sci(self.sci_names[o]):
                    sum_rv.append(orderlet_rv)            # summation of rv of all orderlets

            if sci_mask is not None:
                _, col_rv[s], _, _, col_rv_err[s] = self.alg.fit_ccf(sum_segment, self.alg.get_rv_guess(), velocities,
                                                    sci_mask,
                                                    rv_guess_on_ccf=(ins == 'kpf'),
                                                    vel_span_pixel=self.alg.get_vel_span_pixel())
                col_rv[s] = sum(sum_rv)/len(sum_rv)


        col_sources = np.empty((total_segment, total_orderlet), dtype=object)
        col_cal_sources = np.empty(total_segment, dtype=object)
        final_sum_ccf = np.zeros(np.shape(velocities)[0])
        cal_rv = 0.0

        # fill in rv to columns orderletx, sourcex, cal rv, cal source
        for o in range(total_orderlet):
            if self.is_cal(self.sci_names[o]):
                col_cal_sources[:] = self.sci_names[o]
                rv_table[self.RV_COL_CAL_SOURCE] = col_cal_sources[:]
                rv_table[self.RV_COL_CAL] = col_rv_cal[:]
                if output_df[o] is not None:
                    cal_rv = output_df[o].attrs['RV_MEAN']   # output_df[o].attrs['CCF-RVC'][0]
            elif self.is_sci(self.sci_names[o]):
                col_sources[:, o] = self.sci_names[o]
                rv_table[self.RV_COL_SOURCE + str(o + 1)] = col_sources[:, o]
                rv_table[self.RV_COL_ORDERLET + str(o + 1)] = (col_orderlets[:, o] - col_rv_cal[:]) \
                    if do_corr else col_orderlets[:, o]
                if output_df[o] is not None:
                    final_sum_ccf += output_df[o].values[-1, :]
            if output_df[o] is not None:
                jd = output_df[o].attrs['CCFJDSEG']
                bary = output_df[o].attrs['BARY']

        s_seg = 0
        e_seg = total_segment
        rv_table[self.RV_COL_START_W] = segment_table[s_seg:e_seg, RadialVelocityAlg.SEGMENT_W1]
        rv_table[self.RV_COL_END_W] = segment_table[s_seg:e_seg, RadialVelocityAlg.SEGMENT_W2]
        rv_table[self.RV_COL_SEG_NO] = segment_table[s_seg:e_seg, RadialVelocityAlg.SEGMENT_IDX].astype(int)
        rv_table[self.RV_COL_ORD_NO] = segment_table[s_seg:e_seg,  RadialVelocityAlg.SEGMENT_ORD].astype(int)
        rv_table[self.RV_COL_RV] = (col_rv - col_rv_cal) if do_corr else col_rv        # col of rv
        rv_table[self.RV_COL_RV_ERR] = col_rv_err                                      # col of rv error
        rv_table[self.RV_COL_CCFJD] = np.ones(total_segment) * jd
        rv_table[self.RV_COL_BARY] = np.ones(total_segment) * bary
        rv_table[self.RV_WEIGHTS] = np.ones(total_segment)

        results = pd.DataFrame(rv_table)

        # ccd1rv[1-3], ccd1erv[1-3],  ccd2rv[1-3], ccd2erv[1-3] (for rv ext header)
        for o in range(total_orderlet):
            if self.is_sci(self.sci_names[o]):
                if output_df[o] is None:
                    results.attrs['ccd_rv' + str(o + 1)] = (0, 'BaryC RV (km/s)')
                    results.attrs['ccd_erv' + str(o + 1)] = 0
                else:
                    # take mean of rv
                    orderlet_rv = output_df[o].attrs['RV_MEAN'] # orderlet_rv = output_df[o].attrs['CCF-RVC'][0]
                    if do_corr:
                        orderlet_rv -= cal_rv
                    results.attrs['ccd_rv' + str(o + 1)] = (f_decimal(orderlet_rv), 'BaryC RV (km/s)')
                    results.attrs['ccd_erv'+ str(o + 1)] = output_df[o].attrs['CCF-ERV']
            if output_df[o] is not None:
                ccfjd =  output_df[o].attrs['CCFJDSUM']
                starrv = output_df[o].attrs['STARRV']

        final_rv = final_rv_err = 0.0
        if sci_mask is not None:
            _, final_rv, _, _, final_rv_err = self.alg.fit_ccf(final_sum_ccf, self.alg.get_rv_guess(), velocities,
                                             sci_mask,
                                             rv_guess_on_ccf=(ins == 'kpf'),
                                             vel_span_pixel=self.alg.get_vel_span_pixel())
            final_rv = RadialVelocityAlg.weighted_rv(rv_table[self.RV_COL_RV], total_segment, None)

        if self.RV_COL_CAL in rv_table:
            cal_rv = RadialVelocityAlg.weighted_rv(rv_table[self.RV_COL_CAL], total_segment, None)

            # final_rv_err /= len(good_idx)**0.5                                         # RV error divided by sqrt of number of measurements (good_idx number of orders)

        # ccd1rv, ccd2rv, ccd1erv ccd2erv, cal rv

        results.attrs['rv'] = (f_decimal(final_rv - cal_rv), 'Bary-corrected RV (km/s)') \
            if do_corr and final_rv != 0.0 else (f_decimal(final_rv), 'Bary-corrected RV (km/s)')
        results.attrs['rverr'] = f_decimal(final_rv_err)
        results.attrs['ccd_jd'] = ccfjd
        results.attrs['star_rv'] = starrv
        results.attrs['rv_cal'] = (f_decimal(cal_rv), 'Cal fiber RV (km/s)')      # ccd1crv
        results.attrs['do_rv_corr'] = do_corr

        return results

    @staticmethod
    def load_csv(filepath, header=None):
        if filepath is not None and os.path.isfile(filepath):
            df = pd.read_csv(filepath, header=header, sep='\s+|\t+|\s+\t+|\t+\s+', engine='python')
            return df.values
        else:
            return None
