# Standard dependencies
"""
    This module defines class `RadialVelocityReweighting` which inherits from `KPF1_Primitive` and provides
    methods to perform the event on radial velocity CCF order reweighting in the recipe.

    Description:
        * Method `__init__`:

            RadialVelocityReweighting constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `RadialVelocityReweighting` event issued in the recipe:

                    - `action.args[0] (str|KPF2)`: one file or one KPF2 with ccf for reweighting.
                    - `action.args[1] (str)`: Reweighting method.
                    - `action.args[2] (str|pandas.DataFrame|np.ndarray)`: The ratio table file or ratio table
                      for ccf reweighting.
                    - `action.args[3] (int)`: total segment for the ccf data.
                    - `action.args['ccf_ext'] (str)`: The HDU name  in fits file for the HDU with ccf data.
                      Defaults to 'CCF'.
                    - `action.args['rv_ext'] (str)`: The HDU name in fits file for the HDU with rv data.
                      Defaults to 'RV'.
                    - `action.args['rv_ext_idx'] (int)`: The set index in the extension containing rv data.
                      Defaults to 0. For KPF, 0 is for green ccd, and 1 is for red ccd.
                    - 'action.args['ccf_start_index'] (int)`: The segment index that the first row of ccf_data is
                      associated with.

                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of radial velocity in the master
                  config file associated with the recipe.

            and the following attributes are defined to initialize the object,

                - `ccf_data (dict)`: ccf data per orderlet.
                - `reweighting_method (str)`: Reweighting method.
                - `total_order (int)`: Total order for reweighting.
                - `ccf_ref (np.ndarray)`: Ratio table or Referece ccf for reweighting.
                - `ccf_start_index (int)`: The order index that the first row of ccf_data is associated with.
                - `config_path (str)`: Path of config file for radial velocity.
                - `config (configparser.ConfigParser)`: Config context.
                - `logger (logging.Logger)`: Instance of logging.Logger.
                - `ccf_ext (str)`: name of hdu containing ccf data
                - `rv_ext (str)`: name of hdu containing rv data
                - `rv_ext_idx (int)`: the set index per ccd.
                - `lev2_obj (KPF2)`: L2 instance.
                - `instrument (str)`: instrument.
                - `reweighting_mask (list)`: list of masks allowing reweighting.
                - `is_solar_data (bool)`: if solar is the observation target.
                - `original_ccf_exts (list)`: extensions containing original ccf.
                - `rw_ccf_exts (list)`: extensions containing reweighted ccf.
                - `ccf_dim (int)`: ccf extension dimension.
                - `ccf_ext_header`: header associated with ccf extension.
                - `orderlet_ccf (dict)`: index in the ccf extension data per orderlet.
                - `sci_orderlets (list)`: list of science orderlets.
                - `cal_orderlets (list)`: list of cal orderlets.
                - `sky_orderlets (list)`: list of sky orderlets.
                - `mask_type (dict)`: mask type per orderlet.
                - `totals (int)`: total segment in ccf data.
                - `totalv (int)`: total velocity steps in ccf data.
                - `total_orderlet (int)`: total orderlet in ccf data.
                - `od_names (list)`: names of the orderlets included in the ccf data.

        * Method `__perform`:

            RadialVelocityReweighting returns the result in `Arguments` object containing a level 2 data model
            object with the reweighted ccf orders.

        * Note:
            Action input and return output tentatively uses ccf data in the format of KPF1 object.
            It will be refactored into KPF2 style when level2 data model is implemented.

    Usage:
        For the recipe, the reweighting event is issued like::

            area_def = [0, 34, 500, -500]
            total_segment = area_def[1] - area_def[0] + 1
            ratio_ref = RadialVelocityReweightingRef(None, reweighting_method, ...)

            lev2_rv = kpf2_from_fits(output_lev2_file, data_type=data_type)

            rv_data = RadialVelocityReweighting(lev2_rv,
                                         reweighting_method,
                                         ratio_ref,
                                         total_segment,
                                         ccf_ext="GREEN_CCF",
                                         rv_ext="RV",
                                         rv_ext_idx=0,           # index for green ccd
                                         ccf_start_idx=0,
                                         reweighting_mask=['espresso']
                                    )
            :

        where `rv_data` is KPF2 object wrapped in `Arguments` class object.
"""

import configparser
import numpy as np
import pandas as pd
import os.path

# Pipeline dependencies
from kpfpipe.primitives.level2 import KPF2_Primitive
from kpfpipe.models.level2 import KPF2

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

from modules.radial_velocity.src.alg import RadialVelocityAlg
from modules.radial_velocity.src.alg_rv_init import RadialVelocityAlgInit
from modules.radial_velocity.src.radial_velocity import RadialVelocity as RV

DEFAULT_CFG_PATH = 'modules/optimal_extraction/configs/default.cfg'


class RadialVelocityReweighting(KPF2_Primitive):

    default_agrs_val = {
        'order_name': 'SCI'
    }

    RV_COL_ORDERLET = RV.RV_COL_ORDERLET
    RV_COL_START_W = RV.RV_COL_START_W
    RV_COL_END_W = RV.RV_COL_END_W
    RV_COL_SEG_NO = RV.RV_COL_SEG_NO
    RV_COL_ORD_NO = RV.RV_COL_ORD_NO
    RV_COL_RV = RV.RV_COL_RV
    RV_COL_RV_ERR = RV.RV_COL_RV_ERR
    RV_COL_CAL_ERR = RV.RV_COL_CAL_ERR
    RV_COL_SKY_ERR = RV.RV_COL_SKY_ERR
    RV_COL_CCFJD = RV.RV_COL_CCFJD
    RV_COL_BARY = RV.RV_COL_BARY
    RV_COL_SOURCE = RV.RV_COL_SOURCE
    RV_COL_CAL = RV.RV_COL_CAL
    RV_COL_CAL_SOURCE = RV.RV_COL_CAL_SOURCE
    RV_COL_SKY = RV.RV_COL_SKY
    RV_COL_SKY_SOURCE = RV.RV_COL_SKY_SOURCE
    RV_WEIGHTS = RV.RV_WEIGHTS
    rv_col_names = RV.rv_col_names

    rv_col_on_orderlet = RV.rv_col_on_orderlet

    orderlet_key_map = RV.orderlet_key_map
    orderlet_rv_col_map = RV.orderlet_rv_col_map

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        # Initialize parent class
        KPF2_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        self.ccf_ext = action.args['ccf_ext'] if 'ccf_ext' in args_keys else 'CCF'
        self.ccf_start_index = action.args['ccf_start_index'] if 'ccf_start_index' in args_keys else 0
        self.rv_ext = action.args['rv_ext'] if 'rv_ext' in args_keys else 'RV'
        self.rv_ext_idx = action.args['rv_ext_idx'] if 'rv_ext_idx' in args_keys else 0

        lev2_obj = None

        if isinstance(action.args[0], str):
            if os.path.exists(action.args[0]):
                lev2_obj = KPF2.from_fits(action.args[0])
        elif isinstance(action.args[0], KPF2):
            lev2_obj = action.args[0]

        self.lev2_obj = lev2_obj
        p_header = self.lev2_obj.header['PRIMARY'] if self.lev2_obj else None
        if 'ins' in args_keys and action.args['ins']:
            self.instrument = action.args['ins']
        elif p_header is not None and 'INSTRUME' in p_header:
            self.instrument = p_header['INSTRUME'].lower()
        else:
            self.instrument = None

        # for the use of 'ccf_static'
        self.reweighting_masks = None
        if 'reweighting_masks' in args_keys:
            self.reweighting_masks = action.args['reweighting_masks']
        if self.reweighting_masks is None or not self.reweighting_masks:
            if self.instrument.lower() == 'kpf':
                self.reweighting_masks = ['espresso', 'lfc', 'thar', 'etalon']
            else:
                self.reweighting_masks = None

        self.is_solar_data = p_header is not None and self.instrument == 'kpf' and \
                             'OBJECT' in p_header and p_header['OBJECT'].lower() == 'sun' and \
                             (RadialVelocityAlgInit.KEY_SCI_OBJ in p_header and
                              p_header[RadialVelocityAlgInit.KEY_SCI_OBJ].lower() == 'target') and \
                             (RadialVelocityAlgInit.KEY_SKY_OBJ in p_header and
                              p_header[RadialVelocityAlgInit.KEY_SKY_OBJ].lower() == 'target')

        self.ccf_data = {}              # original ccf in terms of orderlet name
        self.ccf_ext_header = None

        self.ccf_dim = 2
        self.total_orderlet = 1
        self.totals = 0    # total segment in ccf
        self.totalv = 0    # total velocities in ccf
        self.mask_type = {}
        self.sci_orderlets = list()
        self.cal_orderlets = list()
        self.sky_orderlets = list()
        self.od_names = list()
        self.orderlet_ccf = {}
        self.rw_ccf_exts = None         # 2D or 3D containing reweighted ccf
        self.is_solor_data = False

        if self.ccf_ext and hasattr(lev2_obj, self.ccf_ext):
            self.ccf_ext_header = lev2_obj.header[self.ccf_ext]
            ccf_ext_data = lev2_obj[self.ccf_ext]
            self.original_ccf_exts = np.copy(ccf_ext_data)
            self.rw_ccf_exts = np.copy(ccf_ext_data)
            if ccf_ext_data.any():
                self.ccf_dim = self.ccf_ext_header['NAXIS']
                if self.ccf_dim == 2:
                    self.totals, self.totalv = np.shape(ccf_ext_data)
                    self.ccf_data[self.ccf_ext_header['CCF1']] = ccf_ext_data
                    self.orderlet_ccf[self.ccf_ext_header['CCF1']] = 0
                elif self.ccf_dim == 3:
                    self.total_orderlet, self.totals, self.totalv = np.shape(ccf_ext_data)
                    for i in range(self.total_orderlet):
                        od_name = self.ccf_ext_header['CCF'+str(i+1)]
                        self.orderlet_ccf[od_name] = i
                        self.ccf_data[od_name] = ccf_ext_data[i, :, :]
                        self.od_names.append(od_name)
                        if self.is_sci(od_name):
                            self.mask_type[od_name] = \
                                self.ccf_ext_header['SCI_MASK'] if 'SCI_MASK' in self.ccf_ext_header else None
                            self.sci_orderlets.append(od_name)
                        elif self.is_cal(od_name):
                            self.mask_type[od_name] = \
                                self.ccf_ext_header['CAL_MASK'] if 'CAL_MASK' in self.ccf_ext_header else None
                            self.cal_orderlets.append(od_name)
                        elif self.is_sky(od_name):
                            self.mask_type[od_name] = \
                                self.ccf_ext_header['SKY_MASK'] if 'SKY_MASK' in self.ccf_ext_header else None
                            self.sky_orderlets.append(od_name)
                        else:
                            self.mask_type[od_name] = None

        self.reweighting_method = action.args[1]
        self.ccf_ref = None
        if isinstance(action.args[2], str) and os.path.exists(action.args[2]):
            self.ccf_ref = pd.read_csv(action.args[2])
        elif isinstance(action.args[2], pd.DataFrame) or isinstance(action.args[2], np.ndarray):
            self.ccf_ref = action.args[2]

        self.total_segment = action.args[3] if action.args[3] is not None else np.shape(self.ccf_ref)[0]

        # already processed (reweighted) rows
        self.processed_row = action.args['processed_row'] if 'processed_row' in args_keys else None

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['radial_velocity']
        except:
            self.config_path = DEFAULT_CFG_PATH

        self.config.read(self.config_path)
        self.vel_span_pixel =  action.args['vel_span_pixel'] if 'vel_span_pixel' in args_keys \
            else RadialVelocityAlg.comp_velocity_span_pixel(None, self.config, self.instrument)

        # start a logger
        self.logger = None
        # self.logger = start_logger(self.__class__.__name__, self.config_path)
        if not self.logger:
            self.logger = self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        return True

    def _post_condition(self) -> bool:
        """
        check for some necessary post condition
        """
        return True

    def _perform(self):
        """
        Primitive action -
        perform reweghting ccf orders.

        Returns:
            Reweighted ccf orders in pd.DataFrame format
            (this part will be updated after level 2 data model is made.)
        """

        if self.lev2_obj is None or not self.ccf_data or self.ccf_ref is None or \
                self.total_segment is None or self.instrument is None:
            if self.logger:
                self.logger.info("RadialVelocityReweighting: No level2 data, no reweighting data or no data in " + self.ccf_ext)
            return Arguments(self.lev2_obj)

        if self.ccf_dim != 2 and self.ccf_dim != 3:
            if self.logger:
                self.logger.info("RadialVelocityReweighting: incorrect dimension of " + self.ccf_ext)
            return Arguments(self.lev2_obj)

        is_rv_ext = hasattr(self.lev2_obj, self.rv_ext) and not self.lev2_obj[self.rv_ext].empty

        do_corr = False
        if is_rv_ext:
            do_corr_key = ('rv' + str(self.rv_ext_idx + 1) + 'corr')
            do_corr = self.lev2_obj.header[self.rv_ext][do_corr_key] == 'T' \
                if do_corr_key in self.lev2_obj.header[self.rv_ext] else False

        ccf_w = self.ccf_ext_header['TOTALV'] if 'TOTALV' in self.ccf_ext_header else self.totalv
        start_v = self.ccf_ext_header['STARTV']
        v_intv = self.ccf_ext_header['STEPV']
        velocities = np.array([start_v + i * v_intv for i in range(ccf_w)])

        # do cal rv col first
        ccf_ref_list = {}       # ccf ref per orderlet

        rw_ccf_ext = self.ccf_ext + '_RW'

        for o in range(self.total_orderlet-1, -1, -1):
            if self.logger:
                self.logger.info("RadialVelocityReweighting: reweighting on ccd " + str(self.rv_ext_idx+1) +
                                 " orderlet " + str(o+1) + "...")
            od_name = self.od_names[o]
            result_ccf_data = self.ccf_data[od_name]
            ccf_ref = None
            if self.reweighting_method.lower() == 'ccf_static':
                mtype = self.mask_type[od_name]
                is_mask_enable = mtype is not None and \
                                 (self.reweighting_masks is None or
                                  any([m.lower() in mtype.lower() for m in self.reweighting_masks]))
                if is_mask_enable:
                    for idx, r_col in enumerate(self.ccf_ref.columns):
                        if mtype is not None and r_col.lower() == mtype.lower():
                            col_idx = np.array([0, idx], dtype=int)
                            ccf_ref = self.ccf_ref.values[:, col_idx]
                            break
            else:
                ccf_ref = self.ccf_ref.values if isinstance(self.ccf_ref, pd.DataFrame) else self.ccf_ref

            ccf_ref_list[od_name] = ccf_ref
            if ccf_ref is None:
                if self.logger:
                    self.logger.info("RadialVelocityReweighting: No reweighting for " + od_name)
                continue

            rw_ccf, new_total_seg = RadialVelocityAlg.reweight_ccf(result_ccf_data, self.total_segment, ccf_ref,
                                                                   self.reweighting_method, s_seg=self.ccf_start_index,
                                                                   do_analysis=True, velocities=velocities)
            (self.ccf_data[od_name])[0:new_total_seg, :] = rw_ccf[0:new_total_seg, :]
            if self.instrument.lower() == 'kpf':
                self.rw_ccf_exts[self.orderlet_ccf[od_name], 0:new_total_seg, :] = rw_ccf[0:new_total_seg, :]
            else:
                self.lev2_obj[self.ccf_ext][o, 0:new_total_seg, :] = rw_ccf[0:new_total_seg, :]

            # if existing ccf table with summation row
            if np.shape(self.ccf_data[od_name])[1] >= new_total_seg + RadialVelocityAlg.ROWS_FOR_ANALYSIS and not is_rv_ext:
                self.ccf_data[od_name][-1, :] = rw_ccf[-1]

        if self.instrument.lower() == 'kpf':
            self.create_rw_ccf(rw_ccf_ext)

        # update rv table
        if is_rv_ext:
            if self.logger:
                self.logger.info("RadialVelocityReweighting: start updating rv for each segment")
            new_rv_table = self.update_rv_table(velocities, do_corr, ccf_ref_list)
            self.lev2_obj[self.rv_ext] = new_rv_table
            for att in new_rv_table.attrs:
                if att in self.lev2_obj.header[self.rv_ext]:
                    self.lev2_obj.header[self.rv_ext][att] = new_rv_table.attrs[att]


        self.lev2_obj.receipt_add_entry('RadialVelocityReweighting on '+ self.ccf_ext,
                                    self.__module__, f'config_path={self.config_path}', 'PASS')
        if self.logger:
            self.logger.info("RadialVelocityReweighting: Receipt written")
        if self.logger:
            self.logger.info("RadialVelocityReweighting: Done!")

        return Arguments(self.lev2_obj)

    def is_sci(self, sci_name):
        return RV.is_sci(self, sci_name)

    def is_cal(self, sci_name):
        return RV.is_cal(self, sci_name)

    def is_sky(self, sci_name):
        return RV.is_sky(self, sci_name)

    def get_map_key(self, od_name):
        for k in self.orderlet_key_map[self.instrument].keys():
            if k in od_name.lower():
                return k
        return ''

    def get_rv_unit(self, od_name, is_final=False):
        return RV.get_rv_unit(self, od_name, is_final=is_final)

    def create_rw_ccf(self, rw_ext):
        if not self.lev2_obj[rw_ext]:
            self.lev2_obj[rw_ext] = self.rw_ccf_exts
        for key in self.ccf_ext_header.keys():
            if key not in ['XTENSION', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'PCOUNT', 'GCOUNT', 'EXTNAME']:
                self.lev2_obj.header[rw_ext][key] = self.ccf_ext_header[key]

    def update_rv_table(self, velocities, do_corr, ccf_ref_list):
        rv_ext_values = self.lev2_obj[self.rv_ext].values
        rv_ext_columns = self.lev2_obj[self.rv_ext].columns
        rv_ext_header = self.lev2_obj.header[self.rv_ext]
        ccdrow = 'ccd'+str(self.rv_ext_idx+1)+'row'

        # for only one ccd or old L2 file with no such key defined
        rv_start_idx = (ccdrow in rv_ext_header and rv_ext_header[ccdrow]) or \
                       (self.processed_row is not None and self.processed_row) or 0

        def f_decimal(num):
            return float("{:.10f}".format(num))

        def col_idx_rv_table(colname):
            if colname in rv_ext_columns:
                return rv_ext_columns.tolist().index(colname)
            else:
                return -1

        seg_size = self.totals
        cal_idx = col_idx_rv_table(RadialVelocityReweighting.RV_COL_CAL)
        rv_idx = col_idx_rv_table(self.RV_COL_RV)
        rv_err_idx = col_idx_rv_table(self.RV_COL_RV_ERR)

        ods_idx = {}        # column index for each orderlet
        scis_idx = []       # column index list for sci orderlet
        sci_ccf_ref = None  # ccf ref for sci orderlet
        for od_name in self.od_names:
            map_key = self.get_map_key(od_name)
            if map_key:
                ods_idx[od_name] = col_idx_rv_table(self.orderlet_rv_col_map[self.instrument][map_key]['orderlet'])
                if self.is_sci(od_name):
                    scis_idx.append(ods_idx[od_name])
                    if sci_ccf_ref is None:
                        sci_ccf_ref = ccf_ref_list[od_name]

        total_vel = np.shape(self.original_ccf_exts)[2]
        ccf_summ_h_rw = np.zeros((seg_size, total_vel))
        ccf_summ_h_rw_nonnorm = np.zeros((seg_size, total_vel))
        ccf_summ_v_rw = np.zeros(total_vel)
        ccf_summ_v_rw_nonnorm = np.zeros(total_vel)
        new_rv = {}                     # new rv per orderlet
        new_rv_err = {}                 # new rv error per orderlet
        new_ccd_rv = None
        new_ccd_rv_err = None
        new_cal_rv = None
        new_cal_rvs = np.zeros(seg_size)

        def get_new_rv(ccfs, orderlet_name, is_rv_err=False):
            if is_rv_err:
                vel_pix = RadialVelocityAlg.comp_velocity_span_pixel(None, self.config, self.instrument)
                _, _, _, _, new_rv_err = RadialVelocityAlg.fit_ccf(ccfs,
                                        RadialVelocityAlg.get_rv_estimation(rv_ext_header),
                                        velocities, self.mask_type[orderlet_name],
                                        rv_guess_on_ccf=(self.instrument == 'kpf'),
                                        vel_span_pixel=vel_pix)
                return new_rv_err
            else:
                _, new_rv, _, _, _ = RadialVelocityAlg.fit_ccf(ccfs,
                                        RadialVelocityAlg.get_rv_estimation(rv_ext_header),
                                        velocities, self.mask_type[orderlet_name],
                                        rv_guess_on_ccf=(self.instrument == 'kpf'))
                return new_rv

        for od_name in self.od_names:
            if ccf_ref_list[od_name] is None:
                continue

            rw_ccf_nonnorm, _ = RadialVelocityAlg.reweight_ccf(
                                self.original_ccf_exts[self.orderlet_ccf[od_name], 0:seg_size, :],
                                seg_size,
                                ccf_ref_list[od_name],
                                self.reweighting_method,
                                do_analysis=False,
                                velocities=velocities,
                                normalized=False)
            for s in range(seg_size):
                rv_ext_values[s+rv_start_idx, ods_idx[od_name]] = \
                    get_new_rv(self.rw_ccf_exts[self.orderlet_ccf[od_name], s, :], od_name)

                # update rv error for non sci orderlet
                if not self.is_sci(od_name):
                    od_rv_err = get_new_rv(rw_ccf_nonnorm[s, :], od_name, is_rv_err=True)
                    map_key = self.get_map_key(od_name)
                    rv_err_nonsci_idx = col_idx_rv_table(self.orderlet_rv_col_map[self.instrument][map_key]['erv'])
                    rv_ext_values[s+rv_start_idx, rv_err_nonsci_idx] = od_rv_err

                # summ ccf across sci orderlet
                if self.is_sci(od_name) or (self.is_sky(od_name) and self.is_solar_data):
                    ccf_summ_h_rw[s, :] += self.ccf_data[od_name][s, :]
                    ccf_summ_h_rw_nonnorm[s, :] += rw_ccf_nonnorm[s, :]

            # summ ccf across segments => final ccf per orderlet
            ccf_summ = np.sum(self.ccf_data[od_name][0:seg_size, :], axis=0)
            ccf_summ_nonnorm = np.sum(rw_ccf_nonnorm[0:seg_size, :], axis=0)

            # rv and rv error for each orderlet
            new_rv[od_name] = get_new_rv(ccf_summ, od_name)
            new_rv_err[od_name] = get_new_rv(ccf_summ_nonnorm, od_name, is_rv_err=True)
            if self.is_cal(od_name):
                new_cal_rv = new_rv[od_name]
                new_cal_rvs = rv_ext_values[rv_start_idx:rv_start_idx+seg_size, cal_idx]

            # summ final ccf across orderlet
            if self.is_sci(od_name) or (self.is_sky(od_name) and self.is_solar_data):
                ccf_summ_v_rw += ccf_summ
                ccf_summ_v_rw_nonnorm += ccf_summ_nonnorm

        # rv and rv error per ccd
        if np.any(ccf_summ_v_rw):
            new_ccd_rv = get_new_rv(ccf_summ_v_rw, self.sci_orderlets[0])
            new_ccd_rv_err = get_new_rv(ccf_summ_v_rw_nonnorm, self.sci_orderlets[0], is_rv_err=True)

        # update rv and rv error column based on summation across sci orderlets per segment
        for s in range(seg_size):
            rv_ext_values[s+rv_start_idx, rv_idx] = get_new_rv(ccf_summ_h_rw[s, :], self.sci_orderlets[0])
            rv_ext_values[s+rv_start_idx, rv_err_idx] = get_new_rv(ccf_summ_h_rw_nonnorm[s, :],
                                                                       self.sci_orderlets[0],
                                                                       is_rv_err=True)
        # weight column
        ccf_weight = sci_ccf_ref if len(np.shape(sci_ccf_ref)) >= 2 else sci_ccf_ref.reshape(sci_ccf_ref.size, 1)
        ccf_ref_idx = col_idx_rv_table(self.RV_WEIGHTS)
        if ccf_ref_idx >= 0:
            rv_ext_values[rv_start_idx:rv_start_idx+seg_size, ccf_ref_idx] = ccf_weight[0:seg_size, -1]

        # do correction on orderletx columns
        if do_corr:
            for s_idx in scis_idx:
                rv_ext_values[rv_start_idx:rv_start_idx+seg_size, s_idx] -= new_cal_rvs
            rv_ext_values[rv_start_idx:rv_start_idx+seg_size, rv_idx] -= new_cal_rvs

        # create a new rv table to take rv_ext_values with rv update
        new_rv_table = {}
        for c_name in self.rv_col_names:
            if c_name in self.rv_col_on_orderlet:
                for od_name in self.od_names:
                    if not self.is_sci(od_name):
                        continue
                    c_name_orderlet = self.orderlet_rv_col_map[self.instrument][self.get_map_key(od_name)][c_name]
                    idx = col_idx_rv_table(c_name_orderlet)

                    new_rv_table[c_name_orderlet] = rv_ext_values[:, idx].tolist()
            else:
                idx = col_idx_rv_table(c_name)
                if idx >= 0:
                    new_rv_table[c_name] = rv_ext_values[:, idx].tolist()

        results = pd.DataFrame(new_rv_table)      # update rv table
        cal_rv_key = 'ccd'+str(self.rv_ext_idx + 1) + 'rvc'
        if cal_rv_key in rv_ext_header:
            cal_rv = rv_ext_header[cal_rv_key] if new_cal_rv is None else new_cal_rv
        else:
            cal_rv = 0.0

        ccd_p = 'ccd'+str(self.rv_ext_idx+1)

        if new_ccd_rv is not None:
            unit = self.get_rv_unit(self.sci_orderlets[0], is_final=True)
            results.attrs[ccd_p+'rv'] = (f_decimal(new_ccd_rv - cal_rv), unit) \
                if do_corr else (f_decimal(new_ccd_rv), unit)
        if new_ccd_rv_err is not None:
            results.attrs[ccd_p+'erv'] = f_decimal(new_ccd_rv_err)

        for od_name in new_rv.keys():
            map_key = self.get_map_key(od_name)
            if map_key:
                od_rv = new_rv[od_name] - cal_rv if do_corr and self.is_sci(od_name) else new_rv[od_name]
                unit = self.get_rv_unit(od_name)

                results.attrs[ccd_p+self.orderlet_key_map[self.instrument][map_key]['rv']['rv']] = \
                    (f_decimal(od_rv), unit)
                results.attrs[ccd_p+self.orderlet_key_map[self.instrument][map_key]['rv']['erv']] = \
                    f_decimal(new_rv_err[od_name])

                ccf_key = self.orderlet_key_map[self.instrument][map_key]['rv']['ccf']
                rwkey = 'rw'+ccf_key
                rwkey = rwkey[0:8] if len(rwkey) > 8 else rwkey
                results.attrs[rwkey] = 'T' if ccf_ref_list[od_name] is not None else 'F'
        return results

    def reweight_rv(self, rv_colname,  weighting_ratio, is_sigma=False):
        rv_table = self.lev2_obj[self.rv_ext]
        rv_ext_header = self.lev2_obj.header[self.rv_ext]
        seg_size = np.shape(self.lev2_obj[self.ccf_ext])[1]
        ccdrow = 'ccd' + str(self.rv_ext_idx + 1) + 'row'

        if ccdrow in rv_ext_header:
            rv_start_idx = rv_ext_header[ccdrow]
        elif self.processed_row is not None:
            rv_start_idx = self.processed_row
        else:
            rv_start_idx = 0

        rv_arr = (rv_table[rv_colname].to_numpy())[rv_start_idx:rv_start_idx+seg_size]
        if is_sigma:
            w_rv = RadialVelocityAlg.weighted_rv_error(rv_arr, seg_size, weighting_ratio)
        else:
            w_rv = RadialVelocityAlg.weighted_rv(rv_arr, seg_size, weighting_ratio)
        return w_rv

