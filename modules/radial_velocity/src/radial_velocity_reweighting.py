# Standard dependencies
"""
    This module defines class `RadialVelocityReweighting` which inherits from `KPF1_Primitive` and provides
    methods to perform the event on radial velocity CCF orderes reweighting in the recipe.

    Description:
        * Method `__init__`:

            RadialVelocityReweighting constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `RadialVelocityReweighting` event issued in the recipe:

                    - `action.args[0] (str|KPF2)`: one file or one KPF2 with ccf for reweighting.
                    - `action.args[1] (str)`: Reweighting method.
                    - `action.args[2] (pandas.DataFrame|np.ndarray)`: The ratio table or reference ccf for reweighting.
                    - `action.args[3] (int)`: total segment for the ccf data.
                    - `action.args['ccf_ext'] (str)`: The HDU name  in fits file for the HDU with ccf data.
                      Defaults to 'CCF'.
                    - `action.args['rv_ext'] (str)`: The HDU name in fits file for the HDU with rv data.
                      Defaults to 'RV'.
                    - `action.args['rv_ext_idx'] (int)`: The set index in the extension containing rv data.
                      Defaults to 0.
                    - 'action.args['ccf_start_index'] (int)`: The segment index that the first row of ccf_data is
                      associated with.

                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of radial velocity in the master
                  config file associated with the recipe.

            and the following attributes are defined to initialize the object,

                - `ccf_data (np.ndarray)`: ccf data.
                - `reweighting_method (str)`: Reweighting method.
                - `total_order (int)`: Total order for reweighting.
                - `ccf_ref (np.ndarray)`: Ratio table or Referece ccf for reweighting.
                - `ccf_start_index (int)`: The order index that the first row of ccf_data is associated with.
                - `config_path (str)`: Path of config file for radial velocity.
                - `config (configparser.ConfigParser)`: Config context.
                - `logger (logging.Logger)`: Instance of logging.Logger.
                - `ccf_ext`: name of hdu containing ccf data

        * Method `__perform`:

            RadialVelocityReweighting returns the result in `Arguments` object containing a level 1 data model
            object with the reweighted ccf orders.

        * Note:
            Action input and return output tentatively uses ccf data in the format of KPF1 object.
            It will be refactored into KPF2 style when level2 data model is implemented.

    Usage:
        For the recipe, the optimal extraction event is issued like::

            :
            lev1_data = kpf1_from_fits(input_L1_file, data_type='KPF')
            rv_data = RadialVelocityReweighting(lev1_data, <reweighting_method>, <ratio_ref>, <total_order>
                      <rv_init>, ccf_hdu_index=12, ccf_start_idx=0)
            :
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

DEFAULT_CFG_PATH = 'modules/optimal_extraction/configs/default.cfg'


class RadialVelocityReweighting(KPF2_Primitive):

    default_agrs_val = {
        'order_name': 'SCI'
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
    RV_COL_CAL_SOURCE = 'source CAL'
    RV_COL_SKY = 'SKY RV'
    RV_COL_SKY_SOURCE ='source SKY'
    RV_WEIGHTS = 'CCF Weights'
    rv_col_names = [RV_COL_ORDERLET, RV_COL_START_W, RV_COL_END_W, RV_COL_SEG_NO, RV_COL_ORD_NO,
                    RV_COL_RV, RV_COL_RV_ERR, RV_COL_CAL, RV_COL_CAL_ERR, RV_COL_SKY, RV_COL_SKY_ERR,
                    RV_COL_CCFJD, RV_COL_BARY, RV_COL_SOURCE, RV_COL_CAL_SOURCE, RV_COL_SKY_SOURCE, RV_WEIGHTS]
    rv_col_on_orderlet = [RV_COL_ORDERLET, RV_COL_SOURCE]

    orderlet_key_map = {'kpf':{
        'sci_flux1': {'ccf': {'mask': 'sci_mask', 'name': 'ccf1'}, 'rv': {'rv':'rv1', 'erv':'erv1'}},
        'sci_flux2': {'ccf': {'mask': 'sci_mask', 'name': 'ccf2'}, 'rv': {'rv': 'rv2', 'erv': 'erv2'}},
        'sci_flux3': {'ccf': {'mask': 'sci_mask', 'name': 'ccf3'}, 'rv': {'rv': 'rv3', 'erv': 'erv3'}},
        'sky_flux': {'ccf': {'mask': 'sky_mask', 'name': 'ccf5'}, 'rv': {'rv': 'rvs', 'erv': 'ervs'}},
        'cal_flux': {'ccf': {'mask': 'cal_mask', 'name': 'ccf4'}, 'rv': {'rv': 'rvc', 'erv': 'ervc'}}},
        'neid':{
            'sci': {'ccf': {'mask': 'sci_mask', 'name': 'ccf1'}, 'rv': {'rv':'rv1', 'erv':'erv1'}}
        }
    }

    orderlet_rv_col_map = {'kpf':{
        'sci_flux1': {'orderlet':'orderlet1', 'source':'source1', 'erv': ''},
        'sci_flux2': {'orderlet':'orderlet2', 'source':'source2', 'erv': ''},
        'sci_flux3': {'orderlet': 'orderlet3', 'source': 'source3', 'erv': ''},
        'sky_flux': {'orderlet': RV_COL_SKY,  'source': RV_COL_SKY_SOURCE, 'erv': RV_COL_SKY_ERR},
        'cal_flux': {'orderlet': RV_COL_CAL, 'source': RV_COL_CAL_SOURCE, 'erv': RV_COL_CAL_ERR}},
        'neid':{
            'sci': {'orderlet':'orderlet1', 'source':'source1', 'erv': ''}
        }
    }

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

        self.is_solar_data = p_header is not None and self.instrument == 'kpf' and \
                             'OBJECT' in p_header and p_header['OBJECT'].lower() == 'sun' and \
                             (RadialVelocityAlgInit.KEY_SCI_OBJ in p_header and
                              p_header[RadialVelocityAlgInit.KEY_SCI_OBJ].lower() == 'target') and \
                             (RadialVelocityAlgInit.KEY_SKY_OBJ in p_header and
                              p_header[RadialVelocityAlgInit.KEY_SKY_OBJ].lower() == 'target')

        self.ccf_data = {}
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

        if self.ccf_ext and hasattr(lev2_obj, self.ccf_ext):
            self.ccf_ext_header = lev2_obj.header[self.ccf_ext]
            ccf_ext_data = lev2_obj[self.ccf_ext]
            if ccf_ext_data.any():
                self.ccf_dim = self.ccf_ext_header['NAXIS']
                if self.ccf_dim == 2:
                    self.totals, self.totalv = np.shape(ccf_ext_data)
                    self.ccf_data[self.ccf_ext_header['CCF1']] = ccf_ext_data
                elif self.ccf_dim == 3:
                    self.total_orderlet, self.totals, self.totalv = np.shape(ccf_ext_data)
                    for i in range(self.total_orderlet):
                        od_name = self.ccf_ext_header['CCF'+str(i+1)]
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
        rv_ext_header = self.lev2_obj.header[self.rv_ext] if is_rv_ext else None

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
        ccf_ref_list = {}

        for o in range(self.total_orderlet-1, -1, -1):
            if self.logger:
                self.logger.info("RadialVelocityReweighting: reweighting on ccd " + str(self.rv_ext_idx+1) +
                                 " orderlet " + str(o+1) + "...")
            od_name = self.od_names[o]
            result_ccf_data = self.ccf_data[od_name]
            ccf_ref = None
            if self.reweighting_method.lower() == 'ccf_static':
                for idx, r_col in enumerate(self.ccf_ref.columns):
                    if self.mask_type[od_name] is not None and r_col.lower() == self.mask_type[od_name].lower():
                        col_idx = np.array([0, idx], dtype=int)
                        ccf_ref = self.ccf_ref.values[:, col_idx]
                        break
            else:
                ccf_ref = self.ccf_ref.values if isinstance(self.ccf_ref, pd.DataFrame) else self.ccf_ref

            if ccf_ref is None:
                if self.logger:
                    self.logger.info("RadialVelocityReweighting: No reweighting ratio for " + self.ccf_ext + " found")
                return Arguments(self.lev2_obj)

            ccf_ref_list[od_name] = ccf_ref
            rw_ccf, new_total_seg = RadialVelocityAlg.reweight_ccf(result_ccf_data, self.total_segment, ccf_ref,
                                                                   self.reweighting_method, s_seg=self.ccf_start_index,
                                                                   do_analysis=True, velocities=velocities)
            (self.ccf_data[od_name])[0:new_total_seg, :] = rw_ccf[0:new_total_seg, :]     # update ccf value for each orderlet
            self.lev2_obj[self.ccf_ext][o, 0:new_total_seg, :] = rw_ccf[0:new_total_seg, :]

            # if existing ccf table with summary row
            if np.shape(self.ccf_data[od_name])[1] >= new_total_seg + RadialVelocityAlg.ROWS_FOR_ANALYSIS and not is_rv_ext:
                self.ccf_data[od_name][-1, :] = rw_ccf[-1]

        # update rv table
        if is_rv_ext:
            if self.logger:
                self.logger.info("RadialVelocityReweighting: start updating rv for each segment")
            self.update_rv_table(self.total_orderlet, velocities, do_corr, ccf_ref_list)

        # update rv keyword, ccd[1-2]rv[1-3], ccd[1-2]rv, ccd[1-2]rvc
        cal_rv = 0.0
        ods_rv = list()
        for o in range(self.total_orderlet):
            map_key = self.get_map_key(self.od_names[o])
            rv_colname = self.orderlet_rv_col_map[self.instrument][map_key]['orderlet']
            ccd_rv = self.reweight_rv(rv_colname, ccf_ref_list[self.od_names[o]])
            ods_rv.append(ccd_rv)

            if self.od_names[o] in self.cal_orderlets:
                cal_rv = ccd_rv

        if is_rv_ext:
            done_sci = False
            for o in range(self.total_orderlet):
                map_key = self.get_map_key(self.od_names[o])
                rv_key = 'ccd' + str(self.rv_ext_idx + 1) + self.orderlet_key_map[self.instrument][map_key]['rv']['rv']
                if self.is_sci(self.od_names[o]) and do_corr:
                    self.lev2_obj.header[self.rv_ext][rv_key] = ods_rv[o] - cal_rv    # ccd[1-2]rv[1-3]
                else:
                    self.lev2_obj.header[self.rv_ext][rv_key] = ods_rv[o]             # ccd[1-2]rv[1-3], ccd[1-2][rvc, rvs]

                if done_sci and self.is_sci(self.od_names[o]):
                    continue

                if self.is_sci(self.od_names[o]):
                    rv_col = self.RV_COL_RV
                    rv_key = 'ccd'+str(self.rv_ext_idx + 1)+'rv'
                    ccd_rv = self.reweight_rv(rv_col, ccf_ref_list[self.od_names[o]])
                    rv_ext_header[rv_key] = self.f_decimal(ccd_rv - cal_rv) if do_corr else self.f_decimal(ccd_rv)
                    done_sci = True

                    erv_col = self.RV_COL_RV_ERR
                    erv_key = 'ccd' + str(self.rv_ext_idx + 1) + 'erv'
                else:
                    erv_col = self.orderlet_rv_col_map[self.instrument][map_key]['erv']
                    erv_key = 'ccd' + str(self.rv_ext_idx + 1) + self.orderlet_key_map[self.instrument][map_key]['rv']['erv']

                ccd_erv = self.reweight_rv(erv_col, ccf_ref_list[self.od_names[o]], is_sigma=True)
                rv_ext_header[erv_key] = self.f_decimal(ccd_erv)

        rv_ext_header['rwccfrv'] = 'T'

        self.lev2_obj.receipt_add_entry('RadialVelocityReweighting on '+ self.ccf_ext,
                                    self.__module__, f'config_path={self.config_path}', 'PASS')
        if self.logger:
            self.logger.info("RadialVelocityReweighting: Receipt written")
        if self.logger:
            self.logger.info("RadialVelocityReweighting: Done!")

        return Arguments(self.lev2_obj)

    @staticmethod
    def f_decimal(num):
        return float("{:.10f}".format(num))

    @staticmethod
    def is_sci(sci_name):
        return 'sci' in sci_name.lower()

    @staticmethod
    def is_cal(sci_name):
        return 'cal' in sci_name.lower()

    @staticmethod
    def is_sky(sci_name):
        return 'sky' in sci_name.lower()

    def get_map_key(self, od_name):
        for k in self.orderlet_key_map[self.instrument].keys():
            if k in od_name.lower():
                return k
        return ''

    def update_rv_table(self, total_orderlet, velocities, do_corr, ccf_ref_list):
        rv_ext_values = self.lev2_obj[self.rv_ext].values
        rv_ext_columns = self.lev2_obj[self.rv_ext].columns
        rv_ext_header = self.lev2_obj.header[self.rv_ext]
        ccdrow = 'ccd'+str(self.rv_ext_idx+1)+'row'

        # for only one ccd or old L2 file with no such key defined
        rv_start_idx = (ccdrow in rv_ext_header and rv_ext_header[ccdrow]) or \
                       (self.processed_row is not None and self.processed_row) or 0

        def col_idx_rv_table(colname):
            if colname in rv_ext_columns:
                return rv_ext_columns.tolist().index(colname)
            else:
                return -1

        seg_size = self.totals
        cal_idx = col_idx_rv_table(RadialVelocityReweighting.RV_COL_CAL)
        rv_idx = col_idx_rv_table(self.RV_COL_RV)

        ods_idx = {}
        scis_idx = []
        sci_ccf_ref = None
        for od_name in self.od_names:
            map_key = self.get_map_key(od_name)
            if map_key:
                ods_idx[od_name] = col_idx_rv_table(self.orderlet_rv_col_map[self.instrument][map_key]['orderlet'])
                if self.is_sci(od_name):
                    scis_idx.append(ods_idx[od_name])
                    if sci_ccf_ref is None:
                        sci_ccf_ref = ccf_ref_list[od_name]

        for s in range(seg_size):
            sum_rv = list()

            for od_name in self.od_names:
                ccf_orderlet = self.ccf_data[od_name][s, :]
                _, od_rv, _, _, _ = RadialVelocityAlg.fit_ccf(ccf_orderlet,
                                                                RadialVelocityAlg.get_rv_estimation(rv_ext_header),
                                                                velocities, self.mask_type[od_name],
                                                                rv_guess_on_ccf=(self.instrument == 'kpf'))

                rv_ext_values[s + rv_start_idx, ods_idx[od_name]] = od_rv
                if self.is_sci(od_name) or (self.is_sky(od_name) and self.is_solar_data):
                    sum_rv.append(od_rv)
            rv_ext_values[s + rv_start_idx, rv_idx] = sum(sum_rv)/len(sum_rv) if len(sum_rv) > 0 else 0.0

        ccf_weight = sci_ccf_ref if len(np.shape(sci_ccf_ref)) >= 2 else sci_ccf_ref.reshape(sci_ccf_ref.size, 1)
        ccf_ref_idx = col_idx_rv_table(self.RV_WEIGHTS)
        if ccf_ref_idx >= 0:
            rv_ext_values[rv_start_idx:rv_start_idx+seg_size, ccf_ref_idx] = ccf_weight[0:seg_size, -1]
        # error columns
        sci_done = False
        for od_name in self.od_names:
            if self.is_sci(od_name) and not sci_done:
                err_col = self.RV_COL_RV_ERR
                sci_done = True
            elif not self.is_sci(od_name):
                err_col = self.orderlet_rv_col_map[self.instrument][self.get_map_key(od_name)]['erv']
            else:
                continue

            od_ref = ccf_ref_list[od_name]
            od_weight = od_ref if len(np.shape(od_ref)) >= 2 else od_ref.reshape(od_ref.size, 1)
            e_idx = col_idx_rv_table(err_col)
            rv_ext_values[rv_start_idx:rv_start_idx+seg_size, e_idx] *= od_weight[0:seg_size, -1]

        # do correction on orderletx columns
        if do_corr:
            for s_idx in scis_idx:
                rv_ext_values[rv_start_idx:rv_start_idx+seg_size, s_idx] -= \
                    rv_ext_values[rv_start_idx:rv_start_idx+seg_size, cal_idx]
                rv_ext_values[rv_start_idx:rv_start_idx+seg_size, rv_idx] -= \
                    rv_ext_values[rv_start_idx:rv_start_idx+seg_size, cal_idx]

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

        self.lev2_obj[self.rv_ext] = pd.DataFrame(new_rv_table)      # update rv table
        return True

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

