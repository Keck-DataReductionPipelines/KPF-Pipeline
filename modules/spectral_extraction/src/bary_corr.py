# Standard dependencies
"""
    This module defines class BaryCorrTable which inherits from `KPF0_Primitive` and provides methods to build
    BARY_CORR table for L1 data in the recipe.

    Description:
        * Method `__init__`:

            BaryCorrTable constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `BaryCrrTable` event issued in the recipe:

                    - `action.args[0] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing primary header and
                      expmeter_sci data to produce data for bary correction table.
                    - `action.args[1] (kpfpipe.models.level0.KPF1)`: Instance of `KPF1` containing extension of bary
                      correction table if existing.
                    - `action.args[2] (int)`: total rows for bary correction table.
                    - `action.args[3] (int)`: total orders of the ccd to be processed.
                    - `action.args['ext_bary_table']: (str, optional)`: name of extension with bary correction table.
                      Defaults to `BARY_CORR`.
                    - `action.args['ext_expmeter_sci_table']: (str, optional)`: name of extension with expmeter science
                      data. Defaults to `EXPMETER_SCI`.
                    - `action.args['wls_ext'] (str, optional)`: Name of extension containing wavelength solution data.
                      Defaults to None.
                    - `action.args['start_bary_index'] (int, optional)`: start index for the ccd to be processed in
                      bary correction table. Defaults to 0.
                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of spectral extraction in the master
                  config file associated with the recipe.

            and the following attributes are defined to initialize the object,

                - `lev0_obj (kpfpipe.models.level0.KPF0)`: Instance of `KPF0`, assigned by `actions.args[0]`.
                - `lev1_obj (kpfpipe.models.level0.KPF1)`:  Instance of `KPF1`, assigned by `actions.args[1]`.
                - `total_orders (int)`: total rows of bary correction table.
                - `ccd_orders (int)`: total orders of the ccd to be processed.
                - `ext_bary (str)`: name of the extension containing bary correction table.
                - `start_index (int)`: start index for the ccd to be processed in bary correction table.
                - `df_bc (pd.DataFrame)`: bary correction table.
                - `wls_data (np.ndarray)`: wavelength solution data.
                - `config_path (str)`: Path of config file for spectral extraction.
                - `config (configparser.ConfigParser)`: Config context per the file defined by `config_path`.
                - `logger (logging.Logger)`: Instance of logging.Logger.
                - `alg_table (modules.order_trace.src.alg.BaryCorrTableAlg)`: Instance of `BaryCorrTableAlg` which
                  has operation codes to produce bary correction table.


        * Method `__perform`:

            BaryCorrTable returns the result in `Arguments` object which contains a level 1 data object (`KPF1`)
            with the bary correction table result.
    Usage:
        For the recipe, the bary correction event is issued like (red ccd for KPF)::

            :
            lev0_data = kpf0_from_fits(input_lev0_file, data_type=data_type)
            output_data = kpf1_from_fits(input_lev1_file, data_type=data_type)

            # for red CCD
            output_data = BaryCorrTable(lev0_data, output_data, 67, 32,
                                        start_bary_index=35,
                                        wls_ext='RED_SCI_WAVE1',
                                        start_bary_index=35)
            :

        where `output_data` is KPF1 object wrapped in `Arguments` class object.
"""


import configparser

# Pipeline dependencies
# from kpfpipe.logger import start_logger
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.models.level0 import KPF0
from kpfpipe.models.level1 import KPF1

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.spectral_extraction.src.alg_bary_corr import BaryCorrTableAlg
import numpy as np

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/spectral_extraction/configs/default.cfg'


class BaryCorrTable(KPF0_Primitive):
    default_args_val = {
                    'ext_bary_table': 'BARY_CORR',
                    'ext_expmeter_sci_table': 'EXPMETER_SCI',
                    'start_bary_index': 0,
                    'wls_ext': None,
                    'overwrite': True
                }

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:

        # Initialize parent class
        KPF0_Primitive.__init__(self, action, context)

        args_keys = [item for item in action.args.iter_kw() if item != "name"]
        # input argument
        self.lev0_obj = action.args[0]      # no process if lev0_obj is None (need primary header)
        self.lev1_obj = action.args[1]      # no process if lev1_obj is None
        self.total_orders = action.args[2]
        self.ccd_orders = action.args[3]
        self.ext_bary = self.get_args_value('ext_bary_table', action.args, args_keys)
        ext_expmeter_sci = self.get_args_value('ext_expmeter_sci_table', action.args, args_keys)
        self.start_index = self.get_args_value('start_bary_index', action.args, args_keys)
        self.is_overwrite = self.get_args_value('overwrite', action.args, args_keys)

        df_em = self.lev0_obj[ext_expmeter_sci] \
            if self.lev0_obj is not None and hasattr(self.lev0_obj, ext_expmeter_sci) else None  # expmeter_sci from lev0
        self.df_bc = self.lev1_obj[self.ext_bary] \
            if self.lev1_obj is not None and hasattr(self.lev1_obj, self.ext_bary) else None   # bary_corr from lev 1

        wls_ext = self.get_args_value('wls_ext', action.args, args_keys)
        if wls_ext is not None and self.lev1_obj and hasattr(self.lev1_obj, wls_ext):
            self.wls_data = self.lev1_obj[wls_ext]
            if self.ccd_orders is None:
                self.ccd_orders = np.shape(self.wls_data)[0]
        else:
            self.wls_data = None

        p_header = self.lev0_obj.header['PRIMARY'] if self.lev0_obj is not None else None  # primary header

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            config_path = context.config_path['spectral_extraction']

        except Exception as e:
            config_path = DEFAULT_CFG_PATH

        self.config.read(config_path)
        # start a logger
        self.logger = None
        if not self.logger:
            self.logger = self.context.logger
        self.logger.info('Loading config from: {}'.format(config_path))

        try:
            if self.lev1_obj is None:
                self.alg_table = None
            else:
                self.alg_table = BaryCorrTableAlg(df_em, self.df_bc, p_header, self.wls_data,
                                          self.total_orders, self.ccd_orders,
                                          start_bary_index=self.start_index,
                                          config=self.config,
                                          logger=self.logger)
        except Exception as e:
            self.alg_table = None

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # input argument must be KPF0
        success = isinstance(self.lev0_obj, KPF0) and isinstance(self.lev1_obj, KPF1)

        return success

    def _post_condition(self) -> bool:
        """
        Check for some necessary post conditions
        """
        return True

    def _perform(self):
        """
        Primitive action -
        perform bary correction table computation by calling method `build_bary_corr_table` from BaryCorrTableAlg and
        produce DataFrame containing bary correction table and update the lev1 data instance.

        Returns:
            pandas.DataFrame containing bary correction table.

        """

        if self.logger:
            self.logger.info("BaryCorrTable: starting bary correction table computation...")

        if self.alg_table is None:
            if self.logger:
                self.logger.info("BaryCorrTable: table is not built due to insufficient data about table size and from primary header")
            return Arguments(self.lev1_obj)

        bc_table = self.alg_table.build_bary_corr_table()

        if bc_table is not None:
            self.lev1_obj[self.ext_bary] = bc_table
            if 'BCV_UNIT' in bc_table.attrs:
                self.lev1_obj.header[self.ext_bary]['BCV_UNIT'] = bc_table.attrs['BCV_UNIT']
        # print(bc_table)
        s_idx, e_idx = self.alg_table.get_index_range()
        self.lev1_obj.receipt_add_entry('BaryCorrTable', self.__module__,
                                             f'bary correction table from {s_idx} to {e_idx} is computed', 'PASS')

        if self.logger:
            self.logger.info('BaryCorrTable: done with bary correction table')

        return Arguments(self.lev1_obj)

    def get_args_value(self, key: str, args: Arguments, args_keys: list):
        if key in args_keys:
            v = args[key]
        else:
            v = self.default_args_val[key]

        return v
