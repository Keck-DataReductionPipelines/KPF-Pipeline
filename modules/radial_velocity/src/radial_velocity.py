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
                    - `action.args[3] (str|list)`: Extension name(s) associated with the level 1 science data.
                    - `action.args['area_def'] (list, optional)`: pixel area, [start_y, end_y, start_x, end_x],
                      to be processed. Defaults to None.
                    - `action.args['segment_def'] (str)`: csv file defining segment wavelength information.
                      Each row in the csv includes items of index, starting and ending wavelength, related order index.
                      Defaults to None.
                    - `action.args['order_def'] (str)`: csv file defining order limits information.
                      Each row in the csv includes items of order index, pixels relative to the left and right border.
                      Defaults to None.
                    - `action.args['ccf_ext'] (str)`: Extension name containing the ccf results.
                    - `action.args['rv_ext'] (str)`: Extension name containing rv results.
                    - `action.args['input_ref'] (np.ndarray|str|pd.DataFrame, optional)`: Reference or weighs for
                      reweighting ccf orders. Defaults to None.
                    - `action.args['reweighting_method'] (str, optional)`: reweighting method. Defaults to None.
                    - `action.args['start_seg'] (int)`: Index of first segment to be processed. Defaults to None.
                    - `action.args['end_seg'] (int)`: Index of last segment to be processed. Defaults to None.
                    - `action.args['bary_corr'] (pd.DataFrame)`: extension name of barycentric correction table.
                      Defaults to 'BARY_CORR'.
                    - `action.args['start_bary_index'] (int)`: starting index in barycentric correction table.
                      Defaults to 0.
                    - `action.args['rv_correction_by_cal'] (bool)`: if using CAL fiber CCF to correct RV of SCI fiber.
                      Defaults to False.
                    - `action.args['ccf_engine'] (str)`: using Python ('python') or C ('c') version CCF engine.
                      Defaults to 'c'.
                    - `action.args['rv_set'] (int)`: the index of ccd per ccd list that L1 data is associated with.
                      ex. 0 for 'GREEN_CCD' and 1 for 'RED_CCD' in terms of KPF. Defaults to 0.
                    - `action.args['ins'] (str)`: instrument name. Defaults to None or value of 'INSTRUME'
                      in the primary header.
                    - `action.args['reweighting_masks'] (list)`: masks allowing reweighting. Defaults to None to include
                      all available masks.
                    - `action.args['exptime'] (float)`: exposure time. Defaults to the value of keyword 'EXPTIME'
                      in primary header.
                    - `action.args['obstime'] (float)`: observation time. Defaults to the value of keyword 'DATE-MID' in
                      primary header for KPF.
                    - `action.args['vel_span_pixel'] (float)`: velocity span per pixel (km.sec). Defaults to None.

                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of radial velocity in the master
                  config file associated with the recipe.

            and the following attributes are defined to initialize the object,

                - `input (kpfpipe.models.level1.KPF1)`: Instance of `KPF1`, assigned by `action.args[0]`.
                - `rv_init (dict)`: Result from radial velocity init.
                - `output_level2 (kpfpipe.models.level2.KPF2)`: Instance of `KPF2`, assigned by `action.args[2]`.
                - `od_names (list)`: Name of the orderlet to be processed, assigned by `action.args[2]`.
                - `area_def (list)`: Pixel area to be processed.
                - `segment_limits (np.ndarray)`: table defining segment wavelength information.
                - `order_limits (np.ndarray)`: table defining order pixel boundaries.
                - `start_seg (int)`: Index of first segment to be processed.
                - `end_seg (int)`: Index of last segment to be processed.
                - `reweighting_method (str)`: reweighting method associated with `action.args['reweighting_method']`.
                - `ccf_ext (str)`: ccf extension name.
                - `rv_ext (str)`: rv extension name.
                - `rv_set_idx (int)`: ccd index for the ccd that spectral data is associated with.
                - `is_cal_cor (bool)`: if using CAL fiber CCF to correct RV of SCI fiber.
                - `ins (str)`: instrument name.
                - `reweighting_masks (list)`: the masks allowing reweighting.
                - `spectrum_data_set (list)`: a list containing spectral data to process.
                - `wave_cal_set (list)`: a list containing wavelength data associated with the spectral data in
                  `spectrum_data_set`.
                - `header_set (list)`: a list containing the header associated with the spectral data in
                  `spectrum_data_set`.
                - `total_orderlet (int):` total orderlet in `spectrum_data_set`.
                - `is_solar_data (bool):` if the observation is for target solar.
                - `ref_ccf (numpy.ndarray)`: Reference or ratio of cross correlation values for scaling the computation
                  of cross correlation, associated with `action.args['input_ref']`.
                - `ccf_engine (str)`: ccf engine, 'python' or 'c', associated with `action.args['ccf_engine']`.
                - `reweighted (dict)`: containing key/value as orderlet_name/'T' or 'F' to indicate if the orderlet is
                  CCF reweighted.
                - `config_path (str)`: Path of config file for radial velocity.
                - `config (configparser.ConfigParser)`: Config context.
                - `logger (logging.Logger)`: Instance of logging.Logger.
                - `alg (RadialVelocityAlg)`: Instance of RadialVelocityAlg which has operation codes for the
                  computation of radial velocity.

        * Method `__perform`:

            RadialVelocity returns the L2 wrapped in `Arguments` class object. L2 has the extensions containing CCF and
            RV values.

    Usage:
        For the recipe, the radial velocity event is issued like::


            :
            data_ext_rv = [['GREEN_SCI_FLUX1', 'GREEN_SCI_FLUX2', 'GREEN_SCI_FLUX3', 'GREEN_CAL_FLUX', 'GREEN_SKY_FLUX'],
                           ['RED_SCI_FLUX1', 'RED_SCI_FLUX2', 'RED_SCI_FLUX3', 'RED_CAL_FLUX', 'RED_SKY_FLUX']]
            lev1_data = kpf1_from_fits(input_L1_file, data_type='KPF')
            rv_init = RadialVelocityInit(...)

            rv_data = None
            for idx in [0, 1]:
                ratio_ref = RadialVelocityReweightingRef(...)       # load reweighting ratio table for green or red ccd
                :
                # idx 0 is for green_ccd, 1 is for red_ccd
                rv_data = RadialVelocity(lev1_data, rv_init, rv_data, data_ext_rv[idx],
                        ccf_ext="GREEN_CCD_CCF",        # "RED_CCD_CCF" for idx = 1
                        rv_ext="RV",
                        area_def=[0, 34, 500, -500],    # [0, 31, 500, -500] for idx = 1
                        start_seg=0,
                        end_seg=34,                     # 31 for idx = 1
                        rv_set=0,                       # 1  for idx = 1
                        ccf_engine='c',
                        start_bary_index=0,             # 35 for idx = 1
                        rv_correction_by_cal=False,
                        reweighting_method='ccf_static',
                        input_ref=ratio_ref,
                        reweighting_masks=['espresso'])
            :

        where `rv_data` is KPF2 object wrapped in `Arguments` class object.
"""

import configparser
import numpy as np
import os
import os.path
import pandas as pd
import math
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
    RV_COL_CAL_ERR = 'CAL error'
    RV_COL_SKY_ERR = 'SKY error'
    RV_COL_CCFJD = 'CCFJD'
    RV_COL_BARY = 'Bary_RVC'
    RV_COL_SOURCE = 'source'
    RV_COL_CAL = 'CAL RV'
    RV_COL_CAL_SOURCE ='source CAL'
    RV_COL_SKY = 'SKY RV'
    RV_COL_SKY_SOURCE ='source SKY'
    RV_WEIGHTS = 'CCF Weights'
    rv_col_names = [RV_COL_ORDERLET, RV_COL_START_W, RV_COL_END_W, RV_COL_SEG_NO, RV_COL_ORD_NO,
                    RV_COL_RV, RV_COL_RV_ERR, RV_COL_CAL, RV_COL_CAL_ERR, RV_COL_SKY, RV_COL_SKY_ERR,
                    RV_COL_CCFJD, RV_COL_BARY,
                    RV_COL_SOURCE, RV_COL_CAL_SOURCE, RV_COL_SKY_SOURCE, RV_WEIGHTS]
    rv_col_on_orderlet = [RV_COL_ORDERLET, RV_COL_SOURCE]

    orderlet_key_map = {'kpf':{
        'sci_flux1': {'ccf': {'mask': 'sci_mask', 'name': 'ccf1'}, 'rv': {'rv':'rv1', 'erv':'erv1', 'ccf':'ccf1'}},
        'sci_flux2': {'ccf': {'mask': 'sci_mask', 'name': 'ccf2'}, 'rv': {'rv': 'rv2', 'erv': 'erv2', 'ccf':'ccf2'}},
        'sci_flux3': {'ccf': {'mask': 'sci_mask', 'name': 'ccf3'}, 'rv': {'rv': 'rv3', 'erv': 'erv3', 'ccf':'ccf3'}},
        'sky_flux': {'ccf': {'mask': 'sky_mask', 'name': 'ccf5'}, 'rv': {'rv': 'rvs', 'erv': 'ervs', 'ccf':'ccfs'}},
        'cal_flux': {'ccf': {'mask': 'cal_mask', 'name': 'ccf4'}, 'rv': {'rv': 'rvc', 'erv': 'ervc', 'ccf':'ccfc'}}},
        'neid': {
            'sci': {'ccf': {'mask': 'sci_mask', 'name': 'ccf1'}, 'rv': {'rv': 'rv1', 'erv': 'erv1', 'ccf':'ccf1'}}
        }
    }
    orderlet_rv_col_map = {'kpf':{
        'sci_flux1': {'orderlet':'orderlet1', 'source':'source1', 'erv': ''},
        'sci_flux2': {'orderlet':'orderlet2', 'source':'source2', 'erv': ''},
        'sci_flux3': {'orderlet': 'orderlet3', 'source': 'source3', 'erv': ''},
        'sky_flux': {'orderlet': RV_COL_SKY,  'source': RV_COL_SKY_SOURCE, 'erv': RV_COL_SKY_ERR},
        'cal_flux': {'orderlet': RV_COL_CAL, 'source': RV_COL_CAL_SOURCE, 'erv': RV_COL_CAL_ERR}},
        'neid': {
            'sci': {'orderlet': 'orderlet1', 'source': 'source1', 'erv': ''}
        }
    }


    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        # Initialize parent class
        KPF1_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        self.input = action.args[0]
        self.rv_init = action.args[1]
        self.output_level2 = action.args[2]
        self.od_names = action.args[3] if isinstance(action.args[3], list) else [action.args[3]]
        self.ref_ccf = None

        if 'input_ref' in args_keys:
            if isinstance(action.args['input_ref'], np.ndarray) or isinstance(action.args['input_ref'], pd.DataFrame):
                self.ref_ccf = action.args['input_ref']
            elif isinstance(action.args['input_ref'], str) and os.path.exists(action.args['input_ref']):
                self.ref_ccf = pd.read_csv(action.args['input_ref'], sep='\s+')

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

        p_header = self.input.header['PRIMARY'] if self.input is not None else None

        if 'ins' in args_keys and action.args['ins']:
            self.ins = action.args['ins']
        elif p_header is not None and 'INSTRUME' in p_header:
            self.ins = p_header['INSTRUME'].lower()
        else:
            self.ins = None

        self.reweighting_masks = action.args['reweighting_masks'] if 'reweighting_masks' in args_keys else None
        if self.reweighting_masks is None or not self.reweighting_masks:
            if self.ins.lower() == 'kpf':
                self.reweighting_masks = ['espresso', 'lfc', 'thar', 'etalon']
            else:
                self.reweighting_masks = None

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
        self.sci_names = list()
        for sci in self.od_names:
            input_data = getattr(self.input, sci) if self.input is not None and hasattr(self.input, sci) else None
            input_header = self.input.header[sci] if input_data is not None and hasattr(self.input, 'header') else None
            self.spectrum_data_set.append(input_data)
            self.header_set.append(input_header)
            if self.is_sci(sci):
                self.sci_names.append(sci)
            wave = sci.replace('FLUX', 'WAVE') if 'FLUX' in sci else None
            self.wave_cal_set.append(getattr(self.input, wave)
                            if (self.input is not None and wave is not None and hasattr(self.input, wave)) else None)
            neid_ssb = 'SSBJD100'   # neid case
            if input_data is None or input_header is None:
                continue

            # p_header exists as input_data is not None
            if neid_ssb in p_header and neid_ssb not in self.input.header[sci]:
                self.input.header[sci][neid_ssb] = p_header[neid_ssb]   # neid case
            elif neid_ssb not in self.input.header[sci]:     # kpf case
                if obstime_v != None:                        # obs and exptime passed externally
                    m_obs = Time(obstime_v).jd - 2400000.5
                    if exptime_v is None:
                        exptime_v = 1.0
                d_obs = 'DATE-MID'
                if d_obs in p_header: # get from primary header if the key exists
                    exptime_k = 'EXPTIME' if 'EXPTIME' in p_header else None
                    exptime_v = p_header[exptime_k] if exptime_k else 1.0
                    m_obs = Time(p_header[d_obs]).jd - 2400000.5

                self.input.header[sci]['MJD-OBS'] = m_obs
                self.input.header[sci]['EXPTIME'] = exptime_v

            for hkey in ['IMTYPE', 'SCI-OBJ', 'SKY-OBJ', 'CAL-OBJ', 'STAR_RV', 'QRV', 'TARGRADV', 'OBJECT']:
                if hkey in p_header:
                    self.input.header[sci][hkey] = p_header[hkey]

            mod, mtype = RadialVelocityAlgInit.MASK_ORDERLET, RadialVelocityAlgInit.MASK_TYPE
            self.input.header[sci]['MASK'] = None
            if not self.rv_init['data'][mod]:
                self.input.header[sci]['MASK'] = self.rv_init['data'][mtype]
            elif self.ins is not None:
                self.input.header[sci]['MASK'] = \
                    self.rv_init['data'][mod][RadialVelocityAlg.get_fiber_object_in_header(self.ins, sci)][mtype]

        self.total_orderlet = len(self.spectrum_data_set)

        do_rv_corr = False
        self.is_solar_data = False
        if p_header is not None:
            key_sci = RadialVelocityAlgInit.KEY_SCI_OBJ
            key_cal = RadialVelocityAlgInit.KEY_CAL_OBJ
            key_sky = RadialVelocityAlgInit.KEY_SKY_OBJ

            sci_obj_v = p_header[key_sci] if key_sci in p_header else None
            if self.rv_init['data'][RadialVelocityAlgInit.MASK_ORDERLET]:
                cal_mask = self.rv_init['data'][RadialVelocityAlgInit.MASK_ORDERLET][key_cal][RadialVelocityAlgInit.MASK_TYPE]
                do_rv_corr = cal_mask in ['lfc', 'thar', 'etalon'] and sci_obj_v.lower() == 'target'

            self.is_solar_data = 'OBJECT' in p_header and p_header['OBJECT'].lower() == 'sun' \
                                     and key_sci in p_header and p_header[key_sci].lower() == 'target' \
                                     and key_sky in p_header and p_header[key_sky].lower() == 'target'
        self.is_cal_cor = self.is_cal_cor and do_rv_corr
        self.mask_collection = []

        if self.rv_init['data'][RadialVelocityAlgInit.MASK_ORDERLET]:
            for mt_key in self.rv_init['data'][RadialVelocityAlgInit.MASK_ORDERLET].keys():
                m_type = self.rv_init['data'][RadialVelocityAlgInit.MASK_ORDERLET][mt_key][RadialVelocityAlgInit.MASK_TYPE]
                if m_type in RadialVelocityAlg.vel_range_per_mask and m_type not in self.mask_collection:
                    self.mask_collection.append(m_type)

        vel_span_pixel = action.args['vel_span_pixel'] if 'vel_span_pixel' in args_keys else None
        self.reweighted = {}

        try:
            if self.ins is None:
                self.alg = None
            else:
                self.alg = RadialVelocityAlg(self.spectrum_data_set[0], self.header_set[0], self.rv_init,
                                    wave_cal=self.wave_cal_set[0],
                                    segment_limits=self.segment_limits,
                                    order_limits=self.order_limits,
                                    area_limits=self.area_def,
                                    config=self.config, logger=self.logger, ccf_engine=self.ccf_engine,
                                    reweighting_method=self.reweighting_method,
                                    bary_corr_table=bc_table, start_bary_index=start_bary_index,
                                    orderlet = self.od_names[0], vel_span_pixel = vel_span_pixel)
        except Exception as e:
            self.alg = None

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # input argument must be KPF0
        success = isinstance(self.input, KPF1)
        return success

    def _post_condition(self) -> bool:
        """
        check for some necessary post condition
        """
        return True

    def is_sci(self, sci_name):
        return 'sci' in sci_name.lower()

    def is_cal(self, sci_name):
        return 'cal' in sci_name.lower()

    def is_sky(self, sci_name):
        return 'sky' in sci_name.lower()

    def get_ratio_ccf(self,  od_name):
        ratio_ccf = None

        def is_mask_enable(mask):
            return mask is not None and \
                   (self.reweighting_masks is None or any([m.lower() in mask.lower() for m in self.reweighting_masks]))

        if self.ref_ccf is not None:
            if self.reweighting_method.lower() == 'ccf_static':
                m_type = self.input.header[od_name]['MASK']
                if is_mask_enable(m_type):
                    idx = [c.lower() for c in self.ref_ccf.columns].index(m_type.lower())
                    if idx >= 0:
                        col_idx = np.array([0, idx], dtype=int)
                        ratio_ccf = self.ref_ccf.values[:, col_idx]
            else:
                ratio_ccf = self.ref_ccf.values if isinstance(self.ref_ccf, pd.DataFrame) else self.ref_ccf
        return ratio_ccf

    def get_rv_unit(self, od_name, is_final=False):
        rv_unit = 'Bary-corrected RV (km/s)'
        rvc_unit = 'Cal fiber RV (km/s)'
        rvs_unit = 'Sky fiber RV (km/s)'
        sky_note = ', including SKY RV ' if self.is_solar_data else ''

        if self.is_cal(od_name):
            unit = rvc_unit
        elif self.is_sky(od_name):
            unit = rvs_unit
        else:
            if is_final:
                unit = rv_unit + sky_note
            else:
                unit = rv_unit
        return unit

    def _perform(self):
        """
        Primitive action -
        perform radial velocity computation by calling method 'compute_rv_by_cc' from RadialVelocityAlg.

        Returns:
            Level 2 data including extensions containing CCFs for all CCDs and RVs for all orderlets
        """

        # _, nx, ny = self.alg.get_spectrum()

        if self.logger:
            self.logger.info("RadialVelocity: Start crorss correlation to find radial velocity... ")

        if self.alg is None:
            self.logger.info("RadialVelocity: no enough data to start the instance to do cross correlation... ")
            return Arguments(self.output_level2)
        output_df = {}

        if all( [s is not None and s.size != 0 for s in self.spectrum_data_set]):
            for i in range(self.total_orderlet):
                if i > 0:
                    self.alg.reset_spectrum(self.spectrum_data_set[i], self.header_set[i], self.wave_cal_set[i],
                                            orderlet=self.od_names[i])
                if self.logger:
                    self.logger.info('RadialVelocity: computing radial velocity on orderlet '+ self.od_names[i] + '...')

                ratio_ccf = self.get_ratio_ccf(self.od_names[i])
                self.reweighted[self.od_names[i]] = 'T' if ratio_ccf is not None else 'F'
                rv_results = self.alg.compute_rv_by_cc(start_seg=self.start_seg, end_seg=self.end_seg, ref_ccf=ratio_ccf)
                one_df = rv_results['ccf_df']
                if one_df is None or one_df.empty or not one_df.values.any():
                    if self.logger:
                        self.logger.info('RadialVelocity: orderlet ' + self.od_names[i] + ' message => ' +
                                rv_results['msg'])
                output_df[self.od_names[i]] = one_df

        # do rv on CAL ccfs
        all_none = [output_df[k] is None for k in output_df.keys()]
        if all(all_none):
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

    def get_map_key(self, od_name):
        for k in self.orderlet_key_map[self.ins].keys():
            if k in od_name.lower():
                return k
        return ''

    def add_ext_key(self, ext, key, key_val):
        KEY8 = 8
        key_name = key[0:KEY8] if len(key) > KEY8 else key
        if key_name not in ext:
            ext[key_name] = key_val

    def construct_level2_data(self, output_df):
        if self.output_level2 is None:
            self.output_level2 = KPF2.from_l1(self.input)

        if (len(output_df.keys()) == 0):
            return True

        self.make_ccf_table(output_df)
        # make new rv table and append the new one to the existing one if there is
        new_rv_table = self.make_rv_table(output_df)
        self.output_level2.header[self.rv_ext]['star_rv'] = new_rv_table.attrs['star_rv']

        # ccfxrw: if rw is done on ccfx
        for od_name in self.od_names:
            od_key = self.get_map_key(od_name)
            ccf_key = self.orderlet_key_map[self.ins][od_key]['rv']['ccf']
            if 'rw'+ccf_key not in self.output_level2.header[self.rv_ext]:
                self.output_level2.header[self.rv_ext]['rw'+ccf_key] = self.reweighted[od_name]

        # velocity range for non espresso mask
        for m in RadialVelocityAlg.vel_range_per_mask.keys():
            if m in self.mask_collection:
                self.add_ext_key(self.output_level2.header[self.rv_ext], 'vr_' + m,
                                 RadialVelocityAlg.vel_range_per_mask[m])

        crt_rv_ext = self.output_level2[self.rv_ext] if hasattr(self.output_level2, self.rv_ext) else None
        if crt_rv_ext is None or np.shape(crt_rv_ext)[0] == 0:
            self.output_level2[self.rv_ext] = new_rv_table
            self.output_level2.header[self.rv_ext]['ccd'+str(self.rv_set_idx+1)+'row'] = 0
        else:
            first_row = np.shape(crt_rv_ext)[0]
            new_table_list = {}
            for c_name in self.rv_col_names:
                if c_name in self.rv_col_on_orderlet:
                    for od_name in self.sci_names:
                        c_name_orderlet = self.orderlet_rv_col_map[self.ins][self.get_map_key(od_name)][c_name]
                        new_list = crt_rv_ext[c_name_orderlet].tolist() + new_rv_table[c_name_orderlet].tolist()
                        new_table_list[c_name_orderlet] = new_list
                else:
                    if c_name in crt_rv_ext and c_name in new_rv_table:
                        new_list = crt_rv_ext[c_name].tolist() + new_rv_table[c_name].tolist()
                        new_table_list[c_name] = new_list
            self.output_level2[self.rv_ext] = pd.DataFrame(new_table_list)
            self.output_level2.header[self.rv_ext]['ccd' + str(self.rv_set_idx + 1) + 'row'] = first_row

        ccd_p = 'ccd'+ str(self.rv_set_idx+1)
        ccd_ = 'ccd_'

        for od_name in self.od_names:
            od_key = self.get_map_key(od_name)
            rv_key = self.orderlet_key_map[self.ins][od_key]['rv']['rv']
            erv_key = self.orderlet_key_map[self.ins][od_key]['rv']['erv']
            self.output_level2.header[self.rv_ext][ccd_p+rv_key] = new_rv_table.attrs[ccd_+rv_key]
            self.output_level2.header[self.rv_ext][ccd_p+erv_key] = new_rv_table.attrs[ccd_+erv_key]


        self.output_level2.header[self.rv_ext][ccd_p+'rv'] = new_rv_table.attrs[ccd_+'rv']
        self.output_level2.header[self.rv_ext][ccd_p+'erv'] = new_rv_table.attrs[ccd_+'erv']
        self.output_level2.header[self.rv_ext][ccd_p+'jd'] = new_rv_table.attrs[ccd_+'jd']
        self.output_level2.header[self.rv_ext]['rv'+str(self.rv_set_idx+1)+'corr'] = 'T' \
            if new_rv_table.attrs['do_rv_corr'] else 'F'

        return True

    def make_ccf_table(self, output_df):
        total_orderlet = len(output_df.keys())
        total_segment = 0
        for o_name, o_df in output_df.items():
            if o_df is not None:
                total_segment = np.shape(o_df.values)[0] - self.alg.ROWS_FOR_ANALYSIS
                break

        if total_segment == 0:
            return True
        original_ccf = np.zeros((self.total_orderlet, total_segment,
                            self.rv_init['data'][RadialVelocityAlgInit.VELOCITY_STEPS]))
        reweight_ccf = np.zeros((self.total_orderlet, total_segment,
                            self.rv_init['data'][RadialVelocityAlgInit.VELOCITY_STEPS]))
        for i in range(total_orderlet):      # make ccf in 3d format, each ccf layer is for one SCI orderlet
            if output_df[self.od_names[i]] is not None:
                reweight_ccf[i, :, :] = output_df[self.od_names[i]].values[0:total_segment, :]
                if self.reweighted[self.od_names[i]] == 'T':
                    original_ccf[i, :, :] = output_df[self.od_names[i]].attrs['ORI_CCF'][0:total_segment, :]
                else:
                    original_ccf[i, :, :] = output_df[self.od_names[i]].values[0:total_segment, :]

        # extension xxx_RW to contain reweighted ccf for kpf case
        if self.ins.lower() == 'kpf':
            ccf_exts = [self.ccf_ext, self.ccf_ext+'_RW']
            # if any(x == 'T' for x in self.reweighted.values()) else [self.ccf_ext]
            self.output_level2[ccf_exts[0]] = original_ccf
            if len(ccf_exts) > 1:     # only assign reweighted ccf if reweighting is done on some orderlet
                self.output_level2[ccf_exts[1]] = reweight_ccf
        else:
            ccf_exts = [self.ccf_ext]
            self.output_level2[ccf_exts[0]] = reweight_ccf[i, :, :]
        for ext in ccf_exts:
            self.output_level2.header[ext]['startseg'] = self.start_seg
            self.output_level2.header[ext]['startv'] = \
                (self.rv_init['data'][RadialVelocityAlgInit.VELOCITY_LOOP][0], 'km/sec')
            self.output_level2.header[ext]['stepv'] = \
                (self.rv_init['data']['rv_config'][RadialVelocityAlgInit.STEP], 'km/sec')
            self.output_level2.header[ext]['totalv'] = self.rv_init['data']['velocity_steps']
            self.output_level2.header[ext]['totalsci'] = len(self.sci_names)

        for i in range(total_orderlet):
            map_key = self.get_map_key(self.od_names[i])
            for ext in ccf_exts:
                self.output_level2.header[ext][self.orderlet_key_map[self.ins][map_key]['ccf']['name']] = \
                    self.od_names[i]

        for i in range(total_orderlet):
            map_key = self.get_map_key(self.od_names[i])
            ccf_key = self.orderlet_key_map[self.ins][map_key]['ccf']['mask']

            for ext in ccf_exts:
                if ccf_key not in self.output_level2.header[ext]:
                    self.output_level2.header[ext][ccf_key] = \
                        RadialVelocityAlg.get_orderlet_masktype(self.ins, self.od_names[i], self.rv_init['data'])
        return True

    def make_rv_table(self, output_df):
        def f_decimal(num):
            return float("{:.10f}".format(num))

        rv_table = {}
        for o_name, o_df in output_df.items():
            if o_df is not None:
                total_segment = np.shape(o_df.values)[0] - self.alg.ROWS_FOR_ANALYSIS
                total_vel = np.shape(o_df.values)[1]
                break

        segment_table = self.alg.get_segment_info()

        col_rv = np.zeros(total_segment)
        col_rv_err = np.zeros(total_segment)
        ins = self.alg.get_instrument().lower()

        sci_mask = None
        do_corr = self.is_cal_cor

        # rv, source and error rv columns for each orderlet
        orderlet_cols = {}
        source_cols = {}
        erv_cols = {}

        # mean rv and erv for each orderlet
        orderlet_rvs = {}
        orderlet_ervs = {}


        # cal rv per segment
        cal_rvs = np.zeros(total_segment)
        cal_rv = 0.0

        jd = None
        bary = None
        starrv = None
        for idx, od_name in enumerate(self.od_names):
            map_key = self.get_map_key(od_name)

            # set up rv col, erv col, source sol, per orderlet
            if map_key:
                orderlet_cols[od_name] = self.orderlet_rv_col_map[self.ins][map_key]['orderlet']
                source_cols[od_name] = self.orderlet_rv_col_map[self.ins][map_key]['source']
                erv_cols[od_name] = self.orderlet_rv_col_map[self.ins][map_key]['erv']
                if orderlet_cols[od_name]:
                    rv_table[orderlet_cols[od_name]] = np.zeros(total_segment)
                if erv_cols[od_name]:
                    rv_table[erv_cols[od_name]] = np.zeros(total_segment)
                if source_cols[od_name]:
                    rv_table[source_cols[od_name]] = np.array([od_name] * total_segment, dtype=object)
            else:
                continue   # this should not happen unless the specified orderlet name is wrong

        sci_ccf_ratio = None
        ccf_summ_h_rw = np.zeros((total_segment, total_vel))
        ccf_summ_h_rw_nonnorm = np.zeros((total_segment, total_vel))
        ccf_summ_v_rw = np.zeros(total_vel)        # sum CCF across orderlet per segment
        ccf_summ_v_rw_nonnorm = np.zeros(total_vel)                  # sum CCF summation across orders
        for idx, od_name in enumerate(self.od_names):
            b_sci = od_name in self.sci_names
            if output_df[od_name] is None :             # skip no ccf and no cal correction if some sci ccf is none
                if b_sci:
                    do_corr = False
                orderlet_rvs[od_name] = 0.0
                orderlet_ervs[od_name] = 0.0
                continue

            if jd is None:
                jd = output_df[od_name].attrs['CCFJDSEG']
            if bary is None:
                bary = output_df[od_name].attrs['BARY']
            if starrv is None:
                starrv = output_df[od_name].attrs['STARRV']

            if sci_mask is None and b_sci:
                sci_mask = RadialVelocityAlg.get_orderlet_masktype(ins, od_name, self.rv_init['data'])
                sci_ccf_ratio = self.get_ratio_ccf(od_name)

            # set rv column and rv error column for each orderlet
            if orderlet_cols[od_name]:
                rv_table[orderlet_cols[od_name]][self.start_seg:self.end_seg+1] = \
                    output_df[od_name].attrs['RV_SEGMS'][self.start_seg:self.end_seg+1]
            if erv_cols[od_name]:   # no erv column for sci orderlet
                rv_table[erv_cols[od_name]][self.start_seg:self.end_seg + 1] = \
                    output_df[od_name].attrs['ERV_SEGMS'][self.start_seg:self.end_seg + 1]

            # rv and erv for each orderlet
            # method 1: get rv and rv error per orderlet by CCF summation
            orderlet_rvs[od_name] = output_df[od_name].attrs['CCF-RVC'][0]  # method2: output_df[od_name].attrs['RV_MEAN']
            orderlet_ervs[od_name] = output_df[od_name].attrs['CCF-ERV']    # method2: output_df[od_name].attrs['ERV_MEAN']

            # ccf summation across orderlet
            # for rv, using un-reweighted or reweighted ccf
            # for rv error, using un-reweighted or non-normalized reweighted ccf
            if b_sci or (self.is_sky(od_name) and self.is_solar_data):
                ccf_summ_h_rw[self.start_seg:self.end_seg+1, :] += output_df[od_name].values[self.start_seg:self.end_seg+1, :]
                ccf_summ_h_rw_nonnorm[self.start_seg:self.end_seg+1, :] += \
                    output_df[od_name].attrs['CCF_NONNORM'][self.start_seg:self.end_seg+1, :]
                ccf_summ_v_rw += output_df[od_name].values[-1, :]
                ccf_summ_v_rw_nonnorm += output_df[od_name].attrs['CCF_NONNORM'][-1, :]
            elif self.is_cal(od_name):
                cal_rvs = rv_table[orderlet_cols[od_name]]
                cal_rv =  output_df[od_name].attrs['CCF-RVC'][0]  # record rv per segments and final rv for cal orderlet

        # method 1: for rv and rv error column, based on summation of ccf across orderlet
        for r in range(self.start_seg, self.end_seg+1):
            _, col_rv[r], _, _, col_rv_err[r] = RadialVelocityAlg.fit_ccf(ccf_summ_h_rw[r, :], self.alg.get_rv_guess(),
                                                self.rv_init['data'][RadialVelocityAlgInit.VELOCITY_LOOP],
                                                sci_mask,
                                                # self.input.header[od_name]['MASK'],
                                                rv_guess_on_ccf=(self.ins.lower() == 'kpf'),
                                                vel_span_pixel=self.alg.get_vel_span_pixel() if sci_ccf_ratio is None else None)
            if sci_ccf_ratio is not None:
                _, _, _, _, col_rv_err[r] = RadialVelocityAlg.fit_ccf(ccf_summ_h_rw_nonnorm[r, :], self.alg.get_rv_guess(),
                                                self.rv_init['data'][RadialVelocityAlgInit.VELOCITY_LOOP],
                                                sci_mask,
                                                # self.input.header[od_name]['MASK'],
                                                rv_guess_on_ccf=(self.ins.lower() == 'kpf'),
                                                vel_span_pixel=self.alg.get_vel_span_pixel())
        rv_table[self.RV_COL_RV] = col_rv
        rv_table[self.RV_COL_RV_ERR] = col_rv_err

        # cal correction
        for sci_name in self.sci_names:
            if do_corr:
                rv_table[orderlet_cols[sci_name]] -= cal_rvs

        s_seg = 0
        e_seg = total_segment
        rv_table[self.RV_COL_START_W] = segment_table[s_seg:e_seg, RadialVelocityAlg.SEGMENT_W1]
        rv_table[self.RV_COL_END_W] = segment_table[s_seg:e_seg, RadialVelocityAlg.SEGMENT_W2]
        rv_table[self.RV_COL_SEG_NO] = segment_table[s_seg:e_seg, RadialVelocityAlg.SEGMENT_IDX].astype(int)
        rv_table[self.RV_COL_ORD_NO] = segment_table[s_seg:e_seg,  RadialVelocityAlg.SEGMENT_ORD].astype(int)
        rv_table[self.RV_COL_CCFJD] = np.ones(total_segment) * jd
        rv_table[self.RV_COL_BARY] = np.ones(total_segment) * bary
        rv_table[self.RV_WEIGHTS] = np.ones(total_segment) if sci_ccf_ratio is None else sci_ccf_ratio[:, -1][0:total_segment]

        results = pd.DataFrame(rv_table)

        for od_name in self.od_names:
            map_key = self.get_map_key(od_name)
            if map_key:
                od_rv = orderlet_rvs[od_name] - cal_rv if do_corr and self.is_sci(od_name) else orderlet_rvs[od_name]
                unit = self.get_rv_unit(od_name)
                results.attrs['ccd_'+self.orderlet_key_map[self.ins][map_key]['rv']['rv']] = (f_decimal(od_rv), unit)
                results.attrs['ccd_'+self.orderlet_key_map[self.ins][map_key]['rv']['erv']] = f_decimal(orderlet_ervs[od_name])

        final_rv = final_rv_err = 0.0

        # compute rv & rv error for the chip
        if sci_mask is not None:
            _, final_rv, _, _, final_rv_err = RadialVelocityAlg.fit_ccf(ccf_summ_v_rw, self.alg.get_rv_guess(),
                     self.rv_init['data'][RadialVelocityAlgInit.VELOCITY_LOOP],
                     sci_mask,
                     rv_guess_on_ccf=(self.ins.lower() == 'kpf'),
                     vel_span_pixel=self.alg.get_vel_span_pixel() if sci_ccf_ratio is None else None)

            if sci_ccf_ratio is not None:
                _, _, _, _, final_rv_err =  RadialVelocityAlg.fit_ccf(ccf_summ_v_rw_nonnorm, self.alg.get_rv_guess(),
                      self.rv_init['data'][RadialVelocityAlgInit.VELOCITY_LOOP],
                      sci_mask,
                      rv_guess_on_ccf=(self.ins.lower() == 'kpf'), vel_span_pixel=self.alg.get_vel_span_pixel())

        unit = self.get_rv_unit(self.sci_names[0], is_final=True)
        results.attrs['ccd_rv'] = (f_decimal(final_rv - cal_rv), unit) \
            if do_corr and final_rv != 0.0 else (f_decimal(final_rv), unit)
        results.attrs['ccd_erv'] = f_decimal(final_rv_err)
        results.attrs['ccd_jd'] = jd[0]
        results.attrs['star_rv'] = starrv
        results.attrs['do_rv_corr'] = do_corr

        return results

    @staticmethod
    def load_csv(filepath, header=None):
        if filepath is not None and os.path.isfile(filepath):
            df = pd.read_csv(filepath, header=header, sep='\s+|\t+|\s+\t+|\t+\s+', engine='python')
            return df.values
        else:
            return None
