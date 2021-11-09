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
                    - `action.args[3] (int)`: total order for the ccf data.
                    - `action.args[4] (dict)`: Result from the init work made by `RadialVelocityInit` which makes
                      mask lines and velocity steps based on star and other module associated configuration for
                      radial velocity computation.
                    - `action.args['ccf_ext'] (str)`: The HDU name  in fits file for the HDU with ccf data.
                      Defaults to 'CCF'.
                    - `action.args['rv_ext'] (str)`: The HDU name in fits file for the HDU with rv data.
                      Defaults to 'RV'.
                    - `action.args['rv_ext_idx'] (int)`: The set index in the extension containing rv data.
                      Defaults to 1.
                    - 'action.args['ccf_start_index'] (int)`: The order index that the first row of ccf_data is
                      associated with.

                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of radial velocity in the master
                  config file associated with the recipe.

            and the following attributes are defined to initialize the object,

                - `ccf_data (np.ndarray)`: ccf data.
                - `lev1_file (str)`: file containing ccf data.
                - `lev1_data (KPF1)`: data containing ccf data.
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
from astropy.io import fits
import os.path
import datetime

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

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        # Initialize parent class
        KPF2_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        self.ccf_ext = action.args['ccf_ext'] if 'ccf_ext' in args_keys else 'CCF'
        self.ccf_start_index = action.args['ccf_start_index'] if 'ccf_start_index' in args_keys else 0
        self.rv_ext = action.args['rv_ext'] if 'rv_ext' in args_keys else 'RV'
        self.rv_ext_idx = action.args['rv_ext_idx'] if 'rv_ext_idx' in args_keys else 1
        self.ccf_data = None
        self.lev1_input = None
        self.lev1_file = None
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
        self.total_order = action.args[3]
        self.rv_init = action.args[4]

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
                  and isinstance(self.ccf_ref, np.ndarray)

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

        header = self.lev2_obj.header[self.ccf_ext]
        at_idx_key = 'CCF_' + str(self.rv_ext_idx) + '_AT'
        if at_idx_key in header:
            at_idx = header[at_idx_key]
            ny = header['TOTALORD']
        else:
            at_idx = 0
            ny, nx = np.shape(self.ccf_data)

        if ny < self.total_order:
            self.total_order = ny

        result_ccf_data = self.ccf_data[at_idx:at_idx+self.total_order, :].copy()

        # assume the first row of ccf data (self.ccf_data) and the ccf from the observation template (or the ratio,
        # i.e. self.ccf_ref) are related to the same order index.

        velocities = self.rv_init['data']['velocity_loop']
        rw_ccf, new_total_order = \
            RadialVelocityAlg.reweight_ccf(result_ccf_data, self.total_order, self.ccf_ref,
                                            self.reweighting_method, s_order=self.ccf_start_index,
                                            do_analysis=True, velocities=velocities)

        self.update_level2_data(rw_ccf, new_total_order)
        self.lev2_obj.receipt_add_entry('RadialVelocityReweighting on '+ self.ccf_ext,
                                    self.__module__, f'config_path={self.config_path}', 'PASS')
        if self.logger:
            self.logger.info("RadialVelocityReweighting: Receipt written")
        if self.logger:
            self.logger.info("RadialVelocityReweighting: Done!")

        return Arguments(self.lev2_obj)

    def update_level2_data(self, rw_ccf, total_order):
        at_idx_key = 'CCF_' + str(self.rv_ext_idx) + '_AT'
        if at_idx_key in self.lev2_obj.header[self.ccf_ext]:
            at_idx = self.lev2_obj.header[self.ccf_ext][at_idx_key]
            self.lev2_obj[self.ccf_ext][at_idx:at_idx+total_order, :] = rw_ccf[0:total_order]
            self.lev2_obj.header[self.ccf_ext]['STARTORD'] = str(self.ccf_start_index)
            self.lev2_obj.header[self.ccf_ext]['ENDORDER'] = str(self.ccf_start_index + total_order - 1)
            self.lev2_obj.header[self.ccf_ext]['TOTALORD'] = str(total_order)
        else:
            self.lev2_obj[self.ccf_ext] = rw_ccf[0:total_order]

        if not self.rv_ext or not self.rv_ext_idx:
            return True

        rv_orders = [0.0] * total_order
        velocities = rw_ccf[total_order + 1]
        rv_guess = RadialVelocityAlg.get_rv_estimation(self.lev2_obj.header[self.ccf_ext], self.rv_init['data'])

        for i in range(total_order):
            if np.any(rw_ccf[i, :] != 0.0):
                _, rv_orders[i], _, _ = RadialVelocityAlg.fit_ccf(rw_ccf[i, :], rv_guess, velocities)

        rv_table = {}
        _, new_rv, _, _ = RadialVelocityAlg.fit_ccf(rw_ccf[total_order + RadialVelocityAlg.ROWS_FOR_ANALYSIS - 1, :],
                                                    rv_guess, velocities)
        new_rv_set = ("{:.10f}".format(new_rv), ' Baryc RV (km/s)')

        at_idx_key = 'RV_' + str(self.rv_ext_idx) + '_AT'
        if hasattr(self.lev2_obj, self.rv_ext) and (at_idx_key in self.lev2_obj.header[self.rv_ext]):
            if isinstance(self.lev2_obj[self.rv_ext], pd.DataFrame):
                crt_rv = self.lev2_obj[self.rv_ext]['RV'].tolist() if 'RV' in self.lev2_obj[self.rv_ext] else []

            start_idx = self.lev2_obj.header[self.rv_ext][at_idx_key]
            crt_rv[start_idx:start_idx+total_order] = rv_orders[0:total_order]

            rv_table['RV'] = crt_rv
            self.lev2_obj.header[self.rv_ext]['RV_'+str(self.rv_ext_idx)] = new_rv_set
        else:
            rv_table['RV'] = rv_orders
            self.lev2_obj.header[self.rv_ext]['CCF-RVC'] = new_rv_set
        self.lev2_obj[self.rv_ext] = pd.DataFrame(rv_table)

        return True

