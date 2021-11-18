# Standard dependencies
"""
    This module defines class OptimalExtraction which inherits from `KPF0_Primitive` and provides methods to perform
    the event on order rectification in the recipe.

    Attributes:
        OrderRectification

    Description:
        * Method `__init__`:

            OrderRectification constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `OptimalExtraction` event issued in the recipe:

                    - `action.args[0] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing spectrum data for
                      optimal extraction.
                    - `action.args[1] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing flat data and order
                      trace result.
                    - `action.args['order_name'] (str|list, optional)`: Name or list of names of the order to be
                      processed. Defaults to 'SCI'.
                    - `action.args['start_order'] (int, optional)`: Index of the first order to be processed.
                      Defaults to 0.
                    - `action.args['max_result_order'] (int, optional)`: Total orders to be processed, Defaults to -1.
                    - `action.args['rectification_method'] (str, optional)`: Rectification method, '`norect`',
                      '`vertial`', or '`normal`', to rectify the curved order trace. Defaults to '`norect`',
                      meaning no rectification.
                    - `action.args['clip_file'] (str, optional)`:  Prefix of clip file path. Defaults to None.
                      Clip file is used to store the polygon clip data for the rectification method
                      which is not NoRECT.

                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of optimal extraction in the master
                  config file associated with the recipe.

            and the following attributes are defined to initialize the object,

                - `input_spectrum (kpfpipe.models.level0.KPF0)`: Instance of `KPF0`, assigned by `actions.args[0]`.
                - `input_flat (kpfpipe.models.level0.KPF0)`:  Instance of `KPF0`, assigned by `actions.args[1]`.
                - `order_name (str)`: Name of the order to be processed.
                - `start_order (int)`: Index of the first order to be processed.
                - `max_result_order (int)`: Total orders to be processed.
                - `rectification_method (int)`: Rectification method code as defined in `OptimalExtractionAlg`.
                - `extraction_method (str)`: Extraction method code as defined in `OptimalExtractionAlg`.
                - `config_path (str)`: Path of config file for optimal extraction.
                - `config (configparser.ConfigParser)`: Config context.
                - `logger (logging.Logger)`: Instance of logging.Logger.
                - `clip_file (str)`: Prefix of clip file path.
                - `alg (modules.order_trace.src.alg.OptimalExtractionAlg)`: Instance of `OptimalExtractionAlg` which
                  has operation codes for the computation of optimal extraction.

        * Method `__perform`:

            OrderRectification returns the result in `Arguments` object which contains a level 0 data object (`KPF0`)
            with the rectification results replacing the raw image.

    Usage:
        For the recipe, the optimal extraction event is issued like::

            :
            lev0_data = kpf0_from_fits(input_lev0_file, data_type=data_type)
            op_data = OrderRectification(lev0_data, lev0_flat_data,
                                        order_name=order_name,
                                        rectification_method=rect_method,
                                        clip_file="/clip/file/folder/fileprefix")
            :
"""


import configparser
import pandas as pd
import numpy as np

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
from modules.optimal_extraction.src.alg import OptimalExtractionAlg

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/optimal_extraction/configs/default.cfg'

class OrderRectification(KPF0_Primitive):
    default_args_val = {
                    'order_name': 'SCI',
                    'max_result_order': -1,
                    'start_order': 0,
                    'rectification_method': 'norect',  # 'norect', 'normal', 'vertical',
                    'clip_file': None,
                    'data_extension': 'DATA',
                    'trace_extension': None,
                    'trace_file': None,
                    'poly_degree': 3,
                    'origin': [0, 0]
            }

    NORMAL = 0
    VERTICAL = 1
    NoRECT = 2

    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:

        # Initialize parent class
        KPF0_Primitive.__init__(self, action, context)

        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        # input argument
        # action.args[0] is for level 0 fits
        # action.args[1] is for level 0 flat with order trace result extension
        self.input_spectrum = action.args[0]  # kpf0 instance
        self.input_flat = action.args[1]      # kpf0 instance with flat data
        self.orderlet_names = self.get_args_value('orderlet_names', action.args, args_keys)
        self.max_result_order = self.get_args_value("max_result_order", action.args, args_keys)
        self.start_order = self.get_args_value("start_order", action.args, args_keys)  # for the result of order trace
        self.rectification_method = self.get_args_value("rectification_method", action.args, args_keys)
        self.extraction_method = OptimalExtractionAlg.NOEXTRACT
        self.clip_file = self.get_args_value("clip_file", action.args, args_keys)

        self.data_ext =  self.get_args_value('data_extension', action.args, args_keys)
        order_trace_ext = self.get_args_value('trace_extension', action.args, args_keys)
        order_trace_file = self.get_args_value('trace_file', action.args, args_keys)

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['optimal_extraction']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        # start a logger
        self.logger = None
        if not self.logger:
            self.logger = self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        # Order trace algorithm setup
        spec_data = self.input_spectrum[self.data_ext] \
            if self.input_spectrum is not None and hasattr(self.input_spectrum, self.data_ext) else None
        spec_header = self.input_spectrum.header[self.data_ext] \
            if (self.input_spectrum is not None and hasattr(self.input_spectrum, self.data_ext)) else None

        self.order_trace_data = None
        if order_trace_file:
            self.order_trace_data = pd.read_csv(order_trace_file, header=0, index_col=0)
            poly_degree = self.get_args_value('poly_degree', action.args, args_keys)
            origin = self.get_args_value('origin', action.args, args_keys)
            order_trace_header = {'STARTCOL': origin[0], 'STARTROW': origin[1], 'POLY_DEG': poly_degree}
        elif order_trace_ext and hasattr(self.input_flat, order_trace_ext):
            self.order_trace_data = self.input_flat[order_trace_ext]
            order_trace_header = self.input_flat.header[order_trace_ext]

        self.alg = OptimalExtractionAlg(self.input_flat[self.data_ext] if hasattr(self.input_flat, self.data_ext) else None,
                                        self.input_flat.header[self.data_ext] if hasattr(self.input_flat, self.data_ext) else None,
                                        spec_data,
                                        spec_header,
                                        self.order_trace_data,
                                        order_trace_header,
                                        config=self.config, logger=self.logger,
                                        rectification_method=self.rectification_method,
                                        extraction_method=self.extraction_method,
                                        clip_file=self.clip_file)

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # input argument must be KPF0

        if self.input_flat is None:          # no flat, rectify spectrum
            success = False
        else:
            success = (self.input_spectrum is None or isinstance(self.input_spectrum, KPF0)) and \
                      isinstance(self.input_flat, KPF0) and self.order_trace_data is not None

        return success

    def _post_condition(self) -> bool:
        """
        Check for some necessary post conditions
        """
        return True

    def _perform(self):
        """
        Primitive action -
        perform optimal extraction by calling method `extract_spectrum` from OptimalExtractionAlg and create an instance
        of level 1 data (KPF1) to contain the analysis result.

        Returns:
            Level 1 data containing optimal extraction result.

        """
        # rectification_method: OptimalExtractAlg.NoRECT(fastest) OptimalExtractAlg.VERTICAL, OptimalExtractAlg.NORMAL
        # extraction_method: 'optimal' (default), 'sum', or 'rectonly'

        # either the input spectrum or flat is already rectified

        if self.input_spectrum is not None:
            if OptimalExtractionAlg.RECTIFYKEY in self.input_spectrum.header[self.data_ext]:
                self.logger.info("OrderRectification: the order of the spectrum is rectified already")
                return Arguments(self.input_spectrum)
        else:
            if OptimalExtractionAlg.RECTIFYKEY in self.input_flat.header[self.data_ext]:
                self.logger.info("OrderRectification: the order of the flat is rectified already")
                return Arguments(self.input_flat)

        if self.logger:
            self.logger.info("OrderRectification: rectifying order...")

        all_order_names = self.orderlet_names if type(self.orderlet_names) is list else [self.orderlet_names]
        all_orders = []
        for order_name in all_order_names:
            o_set = self.alg.get_order_set(order_name)
            if o_set.size > 0 :
                s_order = self.start_order if self.start_order is not None else 0
                e_order = min((s_order + self.max_result_order), len(o_set)) \
                        if (self.max_result_order is not None and self.max_result_order > 0) else len(o_set)

                o_set = o_set[s_order:e_order]
            all_orders.extend(o_set)

        all_orders = np.sort(all_orders)

        opt_ext_result = self.alg.extract_spectrum(order_set=all_orders)

        assert('optimal_extraction_result' in opt_ext_result and
               isinstance(opt_ext_result['optimal_extraction_result'], pd.DataFrame))

        data_df = opt_ext_result['optimal_extraction_result']
        rect_on = opt_ext_result['rectification_on']

        level0_obj = self.input_spectrum if rect_on == 'spectrum' else self.input_flat

        self.update_level0_data(data_df, level0_obj)
        level0_obj.receipt_add_entry('OrderRectification', self.__module__, f'order trace is rectified', 'PASS')

        if self.logger:
            self.logger.info("OrderRectification: Receipt written")

        if self.logger:
            self.logger.info("OrderRectification: Done for rectifying the flux!")

        return Arguments(level0_obj)

    def update_level0_data(self, data_result, lev0_obj):
        # img_d = np.where(np.isnan(data_result.values), 0.0, data_result.values)
        lev0_obj[self.data_ext] = data_result.values
        for att in data_result.attrs:
            lev0_obj.header[self.data_ext][att] = data_result.attrs[att]

    def get_args_value(self, key: str, args: Arguments, args_keys: list):
        if key in args_keys:
            v = args[key]
        else:
            v = self.default_args_val[key]

        if key == 'rectification_method':
            if v is not None and isinstance(v, str):
                if v.lower() == 'normal':
                    method = OptimalExtractionAlg.NORMAL
                elif v.lower() == 'vertical':
                    method = OptimalExtractionAlg.VERTICAL
                else:
                    method = OptimalExtractionAlg.NoRECT
            else:
                method = OptimalExtractionAlg.NoRECT
            return method
        else:
            if key == 'data_extension' or key == 'trace_extension':
                if v is None:
                    v = self.default_args_val[key]
            return v
