# Standard dependencies 
"""
    This module defines class `OrderTrace` which inherits from `KPF0_Primitive` and provides methods to perform the
    event on order trace calculation in the recipe.

    Attributes:
        OrderTrace

    Description:
        * Method `__init__`:

            OrderTrace constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `OrderTrace` event issued in the recipe:

                    - `action.args[0] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing image data for order
                      trace extraction.
                    - `action.args['data_row_range'] (list, optional)`: Row range of the level 0 data to be processed.
                      If the number is less than 0, it stands for the position relative to the last row of the image.
                    - `action.args['data_col_range'] (list, optional)`: column range of the level 0 data to be
                      processed. If the number is less than 0, it stands for the position relative to the last column
                      of the image.

                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of order trace  in the master
                  config file associated with the recipe.

            and the following attributes are defined to initialize the object,

                - `input (kpfpipe.models.level0.KPF0)`: Instance of `KPF0`,  assigned by `actions.args[0]`.
                - `flat_data (numpy.array)`:  2D spectral data associated with `actions.args[0]`.
                - `data_row_range (list)`: Row range of the data to be processed. The row is counted from
                  the first row in case the number is greater than or equal to 0, or from the last row in case
                  the number is less than 0.
                - `data_col_range (list)`: Column range of the data to be processed. The column is counted from
                  first column in case the number is greater than or equal to 0, or from the last column
                  in case the number is less than 0.
                - `config_path (str)`: Path of config file for the computation of order trace.
                - `config (configparser.ConfigParser)`: Config context.
                - `logger (logging.Logger)`: Instance of logging.Logger.
                - `alg (modules.order_trace.src.alg.OrderTraceAlg)`: Instance of `OrderTraceAlg` which has operation
                  codes for the computation of order trace.


        * Method `__perform`:

            OrderTrace returns the result in `Arguments` object which contains the original input
            level 0 data object (`KPF0`) plus an extension with the order trace result.

    Usage:
        For the recipe, the order trace event is issued like::

            :
            flat_data = kpf0_from_fits(input_flat_file, data_type=data_type)
            ot_data = OrderTrace(flat_data, data_row_range=[0, 2000])
            :

        where `flat_data` is level 0 data (`KPF0`) object stored in `Arguments` object.

"""


import configparser
import pandas as pd

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.models.level0 import KPF0

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.order_trace.src.alg import OrderTraceAlg
import ast
import numpy as np

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/order_trace/configs/default.cfg'


class OrderTrace(KPF0_Primitive):
    def __init__(self, 
                 action: Action,
                 context: ProcessingContext) -> None:

        # Initialize parent class
        KPF0_Primitive.__init__(self, action, context)

        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        # input argument
        self.input = action.args[0]
        row, col = np.shape(self.input.data)
        self.row_range = [0, row-1]
        self.col_range = [0, col-1]
        self.cols_to_reset = None
        self.rows_to_reset = None

        if 'data_row_range' in args_keys and action.args['data_row_range'] is not None:
            self.row_range = self.find_range(action.args['data_row_range'], row)

        if 'data_col_range' in args_keys and action.args['data_col_range'] is not None:
            self.col_range = self.find_range(action.args['data_col_range'], col)

        if 'cols_to_reset' in args_keys and action.args['cols_to_reset'] is not None:
            self.cols_to_reset =action.args['cols_to_reset']

        if 'rows_to_reset' in args_keys and action.args['rows_to_reset'] is not None:
            self.rows_to_reset =action.args['rows_to_reset']

        self.flat_data = self.input.data
        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['order_trace']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        # start a logger
        self.logger = None
        # self.logger = start_logger(self.__class__.__name__, self.config_path)
        if not self.logger:
            self.logger = self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        # Order trace algorithm setup
        self.alg = OrderTraceAlg(self.flat_data, config=self.config, logger=self.logger)


    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # input argument must be KPF0
        success = isinstance(self.input, KPF0) and \
            isinstance(self.input.data, np.ndarray)

        return success

    def _post_condition(self) -> bool:
        """
        check for some necessary post condition
        """
        return True

    def _perform(self):
        """
        Primitive action -
        perform radial velocity computation by calling methods from OrderTraceAlg.

        Returns:
            Level 0 data from the input plus an extension containing the order trace result.
        """

        self.alg.set_data_range([self.col_range[0], self.col_range[1],
                                self.row_range[0], self.row_range[1]])

        # 1) Locate cluster
        if self.logger:
            self.logger.info("OrderTrace: locating cluster...")
        cluster_xy = self.alg.locate_clusters(self.rows_to_reset, self.cols_to_reset)

        # 2) assign cluster id and do basic cleaning
        if self.logger:
            self.logger.info("OrderTrace: assigning cluster id and cleaning...")
        x, y, index = self.alg.form_clusters(cluster_xy['x'], cluster_xy['y'])

        power = self.alg.get_poly_degree()
        # 3) advanced cleaning and border cleaning
        if self.logger:
            self.logger.info("OrderTrace: advanced cleaning...")
        new_x, new_y, new_index, all_status = self.alg.advanced_cluster_cleaning_handler(index, x, y)
        new_x, new_y, new_index = self.alg.clean_clusters_on_borders(new_x, new_y, new_index)

        # 5) Merge cluster
        if self.logger:
            self.logger.info("OrderTrace: merging cluster...")
        c_x, c_y, c_index = self.alg.merge_clusters_and_clean(new_index, new_x, new_y)

        # 6) Find width
        if self.logger:
            self.logger.info("OrderTrace: finding width...")
        all_widths, cluster_coeffs = self.alg.find_all_cluster_widths(c_index, c_x, c_y, power_for_width_estimation=3)

        if self.logger:
            self.logger.info("OrderTrace: writing cluster into dataframe...")

        df = self.alg.write_cluster_info_to_dataframe(all_widths, cluster_coeffs)

        assert(isinstance(df, pd.DataFrame))
        
        self.input.create_extension('ORDER_TRACE_RESULT')
        self.input.extension['ORDER_TRACE_RESULT'] = df

        for att in df.attrs:
            self.input.header['ORDER_TRACE_RESULT'][att] = df.attrs[att]

        self.input.receipt_add_entry('OrderTrace', self.__module__, f'config_path={self.config_path}', 'PASS')
        if self.logger:
            self.logger.info("OrderTrace: Receipt written")

        if self.logger:
            self.logger.info("OrderTrace: Done!")

        return Arguments(self.input)

    @staticmethod
    def find_range(range_des, limit):
        tmp_range = ast.literal_eval(range_des) if isinstance(range_des, str) else range_des

        if isinstance(tmp_range, list) and len(tmp_range) == 2:
            tmp_range = [int(t) for t in tmp_range]
            return tmp_range
        tmp_range = [0, limit-1]
        return tmp_range