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

                    - `action.args[0] (str|KPF1)`: one file or one KPF1 with ccf for reweighting.
                    - `action.args[1] (str)`: Reweighting method.
                    - `action.args[2] (pandas.DataFrame|np.ndarray)`: The ratio table or reference ccf for reweighting.
                    - `action.args[3] (int)`: total order for the ccf data.
                    - `action.args[4] (dict)`: Result from the init work made by `RadialVelocityInit` which makes
                      mask lines and velocity steps based on star and other module associated configuration for
                      radial velocity computation.
                    - `action.args['ccf_hdu_index'] (int)`: The HDU index in fits file for the HDU with ccf data.
                      Defaults to None.
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
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.models.level1 import KPF1

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

from modules.radial_velocity.src.alg import RadialVelocityAlg
from modules.radial_velocity.src.alg import RadialVelocityAlgInit

DEFAULT_CFG_PATH = 'modules/optimal_extraction/configs/default.cfg'


class RadialVelocityReweighting(KPF1_Primitive):

    default_agrs_val = {
        'order_name': 'SCI1'
    }

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        # Initialize parent class
        KPF1_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        self.ccf_hdu_index = action.args['ccf_hdu_index'] if 'ccf_hdu_index' in args_keys else None
        self.ccf_start_index = action.args['ccf_start_index'] if 'ccf_start_index' in args_keys else 0
        self.ccf_data = None
        self.lev1_input = None
        self.lev1_file = None
        self.jd = None

        if isinstance(action.args[0], str):
            self.lev1_file = action.args[0]

            if os.path.exists(action.args[0]):
                hdulist = fits.open(action.args[0])
                ccf_hdu = None
                if hdulist is not None:
                    if self.ccf_hdu_index is not None:
                        ccf_hdu = hdulist[self.ccf_hdu_index]
                    else:
                        for hdu in hdulist:
                            if 'EXTNAME' in hdu.header and hdu.header['EXTNAME'].lower().startswith('ccf'):
                                ccf_hdu = hdu
                                break

                if ccf_hdu is not None:
                    self.jd = ccf_hdu.header['CCFJDSUM'] if 'CCFJDSUM' in ccf_hdu.header else 0.0
                    self.ccf_data = ccf_hdu.data if not isinstance(ccf_hdu.data, fits.fitsrec.FITS_rec) \
                            else pd.DataFrame(ccf_hdu.data).values
        elif isinstance(action.args[0], KPF1):
            self.lev1_input = action.args[0]
            self.ccf_data = action.args[0].extension['CCF'].values
            self.jd = action.args[0].header['CCF']['CCFJDSUM']

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
        success = self.ccf_data is not None and isinstance(self.ccf_data, np.ndarray) \
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

        ny, nx = np.shape(self.ccf_data)
        if ny < self.total_order:
            self.total_order = ny

        result_ccf_data = self.ccf_data[0:self.total_order, :].copy()

        # assume the first row of ccf data (self.ccf_data) and the ccf from the observation template (or the ratio,
        # i.e. self.ccf_ref) are related to the same order index.

        velocities = self.rv_init['data']['velocity_loop']
        rw_ccf = RadialVelocityAlg.reweight_ccf(result_ccf_data, self.total_order, self.ccf_ref,
                                                self.reweighting_method, s_order=self.ccf_start_index,
                                                do_analysis=True, velocities=velocities)

        rv_est = self.rv_init['data']['rv_config'][RadialVelocityAlgInit.START_RV]
        if self.lev1_input is None:       # TBD: data type containing ccf, change to an instance of level 2 data later
            self.lev1_input = KPF1()
        else:
            self.lev1_input.del_extension('CCF')

        self.lev1_input.create_extension('CCF')

        # form DataFrame for extension 'CCF'
        ccf_table = {}
        for i in range(len(velocities)):
            ccf_table['vel-' + str(i)] = rw_ccf[:, i]
        ccf_df = pd.DataFrame(ccf_table)

        ccf_fit, ccf_mean, g_x, g_y = RadialVelocityAlg.fit_ccf(
            rw_ccf[self.total_order + RadialVelocityAlg.ROWS_FOR_ANALYSIS - 1, :],
            rv_est, velocities)

        ccf_df.attrs['CCFJDSUM'] = self.jd
        ccf_df.attrs['CCF-RVC'] = "{:.10f}".format(ccf_mean) + ' Baryc RV (km/s)'
        ccf_df.attrs['CCFSTART'] = str(rv_est)
        ccf_df.attrs['CCFSTEP']  = str(self.rv_init['data']['rv_config'][RadialVelocityAlgInit.STEP])
        ccf_df.attrs['STARTORD'] = str(self.ccf_start_index)
        ccf_df.attrs['ENDORDER'] = str(self.ccf_start_index+self.total_order-1)

        self.lev1_input.extension['CCF'] = ccf_df
        self.lev1_input.receipt_add_entry('RadialVelocityReweighting',
                                    self.__module__, f'config_path={self.config_path}', 'PASS')
        if self.logger:
            self.logger.info("RadialVelocityReweighting: Receipt written")
        if self.logger:
            self.logger.info("RadialVelocityReweighting: Done!")

        return Arguments(self.lev1_input)



