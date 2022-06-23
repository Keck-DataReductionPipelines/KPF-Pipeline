# Standard dependencies
"""
    This module defines class `RadialVelocityReweighting` which inherits from `KPF1_Primitive` and provides
    methods to perform the event on radial velocity CCF orderes reweighting in the recipe.

    Attributes:
        RadialVelocityReweighting

    Description:
        * Method `__init__`:

            RadialVelocityReweighting constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `RadialVelocityReweighting` event issued in the recipe:

                    - `action.args[0] (str|KPF2)`: one file or one KPF2 with ccf for reweighting.
                    - `action.args[1] (str)`: Reweighting method.
                    - `action.args[2] (pandas.DataFrame|np.ndarray)`: The ratio table or reference ccf for reweighting.
                    - `action.args[3] (int)`: total segment for the ccf data.
                    - `action.args[4] (dict)`: Result from the init work made by `RadialVelocityInit` which makes
                      mask lines and velocity steps based on star and other module associated configuration for
                      radial velocity computation.
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
                - `rv_init (dict)`: Result from radial velocity init.
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
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level2 import KPF2_Primitive
from kpfpipe.models.level2 import KPF2


# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

from modules.radial_velocity.src.alg import RadialVelocityAlg

DEFAULT_CFG_PATH = 'modules/optimal_extraction/configs/default.cfg'


class RadialVelocityReweighting(KPF2_Primitive):

    default_agrs_val = {
        'order_name': 'SCI'
    }
    RV_COL_ORDERLET = 'orderlet'
    RV_COL_START_W = 's_wavelength'
    RV_COL_END_W = 'e_wavelength'
    RV_COL_SEG_NO = 'segment no.'
    RV_COL_RV = 'RV'
    RV_COL_RV_ERR = 'RV error'
    RV_COL_CCFJD = 'CCFJD'
    RV_COL_SOURCE = 'source'
    rv_col_names = [RV_COL_ORDERLET, RV_COL_START_W, RV_COL_END_W, RV_COL_SEG_NO, RV_COL_RV, RV_COL_RV_ERR,
                    RV_COL_CCFJD, RV_COL_SOURCE]
    rv_col_on_orderlet = [RV_COL_ORDERLET, RV_COL_SOURCE]

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
        self.ccf_data = None
        lev2_obj = None

        if isinstance(action.args[0], str):
            if os.path.exists(action.args[0]):
                lev2_obj = KPF2.from_fits(action.args[0])
        elif isinstance(action.args[0], KPF2):
            lev2_obj = action.args[0]

        self.lev2_obj = lev2_obj
        self.ccf_data = None
        if self.ccf_ext and hasattr(lev2_obj, self.ccf_ext):
            self.ccf_data = lev2_obj[self.ccf_ext]

        self.reweighting_method = action.args[1]
        self.ccf_ref = action.args[2].values if isinstance(action.args[2], pd.DataFrame) else action.args[2]
        self.total_segment = action.args[3] if action.args[3] is not None else np.shape(self.ccf_ref)[0]
        self.rv_init = action.args[4]
        self.processed_row = action.args['processed_row'] if 'processed_row' in args_keys else None
        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['radial_velocity']
        except:
            self.config_path = DEFAULT_CFG_PATH

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
        # input argument must be KPF0
        success = self.lev2_obj is not None and self.ccf_data is not None and isinstance(self.ccf_data, np.ndarray) \
                  and isinstance(self.ccf_ref, np.ndarray) and self.total_segment is not None

        return success

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

        if self.ccf_data.size == 0:
            if self.logger:
                self.logger.info("RadialVelocityReweighting: No data in " + self.ccf_ext)
            return Arguments(self.lev2_obj)

        header = self.lev2_obj.header[self.ccf_ext]
        ccf_dim = header['NAXIS']

        assert (ccf_dim == 2 or ccf_dim == 3)

        total_orderlet = 1

        if ccf_dim == 2:
            ccf_data = self.ccf_data
            self.ccf_data = np.zeros((1, np.shape(ccf_data)[0], np.shape(ccf_data)[1]))  # convert 2D to 3D
            self.ccf_data[0, :, :] = ccf_data
        elif ccf_dim == 3:
            total_orderlet = np.shape(self.ccf_data)[0]

        velocities = self.rv_init['data']['velocity_loop']
        is_rv_ext = hasattr(self.lev2_obj, self.rv_ext) and not self.lev2_obj[self.rv_ext].empty
        rv_ext_header = self.lev2_obj.header[self.rv_ext] if is_rv_ext else self.lev2_obj.header[self.ccf_ext]

        final_sum_ccf = np.zeros(np.shape(velocities)[0])

        for o in range(total_orderlet):
            if self.logger:
                self.logger.info("RadialVelocityReweighting: reweighting on ccd " + str(self.rv_ext_idx+1) +
                                 " orderlet " + str(o+1) + "...")
            result_ccf_data = self.ccf_data[o, :, :]
            rw_ccf, new_total_seg = RadialVelocityAlg.reweight_ccf(result_ccf_data, self.total_segment, self.ccf_ref,
                                                                   self.reweighting_method, s_seg=self.ccf_start_index,
                                                                   do_analysis=True, velocities=velocities)
            self.ccf_data[o, 0:new_total_seg, :] = rw_ccf[0:new_total_seg, :]     # update ccf value for each orderlet

            # if existing ccf table with summary row
            if np.shape(self.ccf_data)[1] >= new_total_seg + RadialVelocityAlg.ROWS_FOR_ANALYSIS and not is_rv_ext:
                self.ccf_data[o, -1, :] = rw_ccf[-1]
            final_sum_ccf += rw_ccf[-1]
            _, ccd_rv, _, _ = RadialVelocityAlg.fit_ccf(rw_ccf[-1], RadialVelocityAlg.get_rv_estimation(rv_ext_header,
                                                                                    self.rv_init['data']), velocities)

            # update rv on each orderlet in rv extension
            if is_rv_ext:
                self.lev2_obj.header[self.rv_ext]['ccd'+str(self.rv_ext_idx+1)+'rv'+str(o+1)] = ccd_rv

        self.lev2_obj[self.ccf_ext] = self.ccf_data                 # update ccf extension

        # update final rv on all orderlets
        _, ccd, _, _ = RadialVelocityAlg.fit_ccf(final_sum_ccf,
                                    RadialVelocityAlg.get_rv_estimation(rv_ext_header, self.rv_init['data']),
                                    velocities)
        rv_ext_header['ccd'+ str(self.rv_ext_idx+1) + 'rv'] = ccd        # header: ccdnrv

        if is_rv_ext:
            if self.logger:
                self.logger.info("RadialVelocityReweighting: start updating rv for each segment")
            self.update_rv_table(total_orderlet, velocities)

        self.lev2_obj.receipt_add_entry('RadialVelocityReweighting on '+ self.ccf_ext,
                                    self.__module__, f'config_path={self.config_path}', 'PASS')
        if self.logger:
            self.logger.info("RadialVelocityReweighting: Receipt written")
        if self.logger:
            self.logger.info("RadialVelocityReweighting: Done!")

        return Arguments(self.lev2_obj)

    def update_rv_table(self, total_orderlet, velocities):
        rv_ext_values = self.lev2_obj[self.rv_ext].values
        rv_ext_header = self.lev2_obj.header[self.rv_ext]
        rv_name_list = [name.lower() for name in self.rv_col_names]
        # rv_start_idx = self.rv_ext_idx * self.total_segment
        ccdrow = 'ccd'+str(self.rv_ext_idx+1)+'row'
        if ccdrow in rv_ext_header:
            rv_start_idx = rv_ext_header[ccdrow]
        elif self.processed_row is not None:
            rv_start_idx = self.processed_row
        else:
            rv_start_idx = 0   # for only one ccd or old L2 file with no such key defined

        rv_orderlet_colnames = [self.RV_COL_ORDERLET + str(o + 1) for o in range(total_orderlet)]
        def col_idx_rv_table(colname, orderlet_idx=0):
            colname = colname.lower()

            if 'orderlet' in colname:
                return orderlet_idx
            elif 'source' in colname:
                return rv_name_list.index('source') + total_orderlet - 1 + orderlet_idx
            else:
                return rv_name_list.index(colname) + total_orderlet - 1

        seg_size = self.total_segment

        for s in range(seg_size):
            sum_segment = np.zeros(np.shape(velocities)[0])
            for o in range(total_orderlet):
                ccf_orderlet = self.lev2_obj[self.ccf_ext][o, s, :]
                sum_segment += ccf_orderlet
                _, orderlet_rv, _, _ = RadialVelocityAlg.fit_ccf(ccf_orderlet,
                                                                RadialVelocityAlg.get_rv_estimation(rv_ext_header,
                                                                self.rv_init['data']), velocities)
                c_idx = col_idx_rv_table(rv_orderlet_colnames[o], o)
                rv_ext_values[s+rv_start_idx, c_idx] = orderlet_rv

            # update orderletn column at segment s
            _, seg_rv, _, _ = RadialVelocityAlg.fit_ccf(sum_segment,
                                            RadialVelocityAlg.get_rv_estimation(rv_ext_header, self.rv_init['data']),
                                            velocities)
            rv_ext_values[s+rv_start_idx, col_idx_rv_table(self.RV_COL_RV)] = seg_rv     # update rv column\

        new_rv_table = {}
        for c_name in self.rv_col_names:
            if c_name in self.rv_col_on_orderlet:
                for o in range(total_orderlet):
                    c_name_orderlet = c_name+str(o+1)
                    idx = col_idx_rv_table(c_name, o)
                    new_rv_table[c_name_orderlet] = rv_ext_values[:, idx].tolist()
            else:
                idx = col_idx_rv_table(c_name)
                new_rv_table[c_name] = rv_ext_values[:, idx].tolist()

        self.lev2_obj[self.rv_ext] = pd.DataFrame(new_rv_table)      # update rv table
        return True

