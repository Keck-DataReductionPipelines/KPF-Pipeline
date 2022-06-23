# Standard dependencies 
"""
    This module defines class `OrderTraceCombine` which combines multiple order trace tabular results
    into one table.

    Attributes:
        OrderTraceCombine

    Description:
        * Method `__init__`:

            OrderTraceCombine constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `OrderTraceCombine` event issued in the recipe:

                    - `action.args[0] (int)`: Total order trace tables to be combined.
                    - `action.args[0] (list)`: list of kpfpipe.models.level0.KPF0 instance(s) containing image data
                      for order trace extraction.
                    - `action.args[1] (list)`: list of string(s) containing the path of order trace results associated
                      with the images of previous argument.
                    - `action.args['data_extension'] (string, optional)`: name of the extension with the image. `data`
                      is the default.
                    - `action.args['for_cal'] (list, optional)`: list of object names that each order trace result
                      are associated to. ex: ['ldls' , 'ldls']
                    - `action.args['for_fiber'] (list, optional)`: list of object names that each order trace result
                      are associated to. ex: [['sci', 'cal'], ['sky', 'cal']]
                    - `action.args['trace_range`] (list, optional)`: list of the interested trace range from
                      each order result files.
                    - `action.args['common_fiber'] (string)`: common trace among the input of all order trace results.
                    - `action.args['output_path'] (list)`: the rows with the object to be extracted and combined.
                    - `action.args['fitting_poly_degree'] (int, optional)`: Order of polynomial used to fit the trace.
                      Defaults to None. The value overrides the number defined in the configuration file for the module
                      if it is defined.

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
                - `result_path (str)`: File or extension containing the output.
                - `alg (modules.order_trace.src.alg.OrderTraceAlg)`: Instance of `OrderTraceAlg` which has operation
                  codes for the computation of order trace.
                - `poly_degree (int)`: Order of polynimial for order trace fitting.


        * Method `__perform`:

            OrderTraceCombine returns the result in `Arguments` object which contains the original input
            level 0 data object (`KPF0`) plus an extension with the order trace result.

"""

import configparser

# Pipeline dependencies
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.models.level0 import KPF0

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
import numpy as np
import pandas as pd

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/order_trace/configs/default.cfg'


class OrderTraceCombine(KPF0_Primitive):
    def __init__(self, 
                 action: Action,
                 context: ProcessingContext) -> None:

        # Initialize parent class
        KPF0_Primitive.__init__(self, action, context)
        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        # input argument
        self.total_order_trace = action.args[0]   # total order trace
        input_flat = action.args[1]
        self.input_flat = []
        self.flat_data = []

        for inputs in input_flat:
            if isinstance(inputs, str):                 # from a file path
                self.input_flat.append(KPF0.from_fits(inputs))
            elif isinstance(inputs, KPF0):              # from a KPF0 object
                self.input_flat.append(inputs)
            elif isinstance(inputs, list):              # from list of KPF0 or file path
                f_data_set = []
                for f_data in inputs:
                    if isinstance(f_data, KPF0):
                        f_data_set.append(f_data)
                    elif isinstance(f_data, str):
                        f_data_set.append(KPF0.from_fits(f_data))
                    else:
                        f_data_set.append(None)
                self.input_flat.append(f_data_set)
            else:
                self.input_flat.append(None)

            # input kpf0 files
        order_trace_table = action.args[2]   # order trace results or files

        self.order_trace_table = \
            [ot.values if isinstance(ot, pd.DataFrame)
             else pd.read_csv(ot, header=0, index_col=0).values for ot in order_trace_table]

        # data extension in kpf0 files
        data_ext = action.args['data_extension']

        self.flat_data = \
            [[i_flat[data_ext] for i_flat in self.input_flat[i]] if isinstance(self.input_flat[i], list)
             else self.input_flat[i][data_ext] for i in range(self.total_order_trace)]

        if 'fitting_poly_degree' in args_keys and action.args['fitting_poly_degree'] is not None:
            self.poly_degree = action.args['fitting_poly_degree']
        else:
            self.poly_degree = 3

        self.for_cal = action.args['for_cal'] if 'for_cal' in args_keys else None
        self.for_fiber = action.args['for_fiber'] if 'for_fiber' in args_keys else None
        self.common_fiber = action.args['common_fiber'] if 'common_fiber' in args_keys else None
        self.output_path = action.args['output_path'] if 'output_path' in args_keys else None
        self.trace_range = action.args['trace_range'] if 'trace_range' in args_keys else None
        self.mixed_trace_range = action.args['mixed_trace_range'] if 'mixed_trace_range' in args_keys else None

        # import pdb;pdb.set_trace()
        if self.trace_range is None:
            self.trace_range = [[0, np.shape(ot_data)[0]] for ot_data in self.order_trace_table]
        else:
            for idx in range(self.total_order_trace):
                total_trace = np.shape(self.order_trace_table[idx])[0]
                for i in [0, 1]:
                    self.trace_range[idx][i] = int(self.trace_range[idx][i])
                    if self.trace_range[idx][i] < 0:
                        self.trace_range[idx][i] += total_trace

        if self.mixed_trace_range is not None:
            for idx in range(self.total_order_trace):
                if self.mixed_trace_range[idx] is None:
                    continue

                total_trace = np.shape(self.order_trace_table[idx])[0]
                for i in [0, 1]:
                    self.mixed_trace_range[idx][i] = int(self.mixed_trace_range[idx][i])
                    if self.mixed_trace_range[idx][i] < 0:
                        self.mixed_trace_range[idx][i] += total_trace
        # import pdb;pdb.set_trace()
        if isinstance(self.flat_data[0], list):
            self.row, self.col = np.shape(self.flat_data[0][0])
        else:
            self.row, self.col = np.shape(self.flat_data[0])

        # import pdb;pdb.set_trace()
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


    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # input argument must be KPF0

        success = (self.total_order_trace == len(self.input_flat) ) and \
                  (self.total_order_trace == len(self.order_trace_table)) and \
                  (self.total_order_trace == len(self.trace_range))

        return success

    def _post_condition(self) -> bool:
        """
        check for some necessary post condition
        """
        return True

    def get_overlap_trace(self, y_pos_list, y_pos):
        y_c2, y_lower2, y_upper2 = y_pos

        dist = self.row
        found_pos_idx = -1
        for p_idx in range(np.shape(y_pos_list)[0]):
            y_c, y_lower, y_upper = y_pos_list[p_idx]
            if y_upper2 < y_lower:        # tested curve above current curve
                break
            elif y_lower2 > y_upper:      # tested curve under current curve
                continue
            elif abs(y_c-y_c2) < dist:
                dist = abs(y_c-y_c2)
                found_pos_idx = p_idx

        return found_pos_idx

    def get_order_y_center(self, order_trace, center_x):
        trace_coeffs = np.flip(order_trace[0:self.poly_degree + 1])
        widths = order_trace[self.poly_degree + 1:self.poly_degree + 3]
        y_c = np.polyval(trace_coeffs, center_x)
        y_l, y_u = y_c-widths[0], y_c+widths[1]

        return center_x, [y_c, y_l, y_u]

    def get_trace_range(self, ot_idx, combine_type):
        if combine_type == 'add':
            order_trace_range = self.trace_range[ot_idx]
        else:
            order_trace_range = self.mixed_trace_range[ot_idx]

        total_order = np.shape(self.order_trace_table[ot_idx])[0]
        if isinstance(order_trace_range, list):
            order_trace_range = [int(order_trace_range[0]), min(int(order_trace_range[1]), total_order-1)]
        else:
            if isinstance(order_trace_range, int):
                order_trace_range = [int(order_trace_range), total_order - 1]
            else:
                order_trace_range = [0, total_order-1]

        return order_trace_range

    def init_first_trace(self, first_idx, combine_type='add'):
        # start and end trace index
        first_order_trace = self.get_trace_range(first_idx, combine_type)
        t_range = np.arange(first_order_trace[0], first_order_trace[1]+1, 1, dtype=int)

        trace_table = self.order_trace_table[first_idx]

        center_x = self.col//2
        # import pdb;pdb.set_trace()
        result_table = trace_table[t_range, :]
        t_center_y_pos = np.zeros((np.shape(result_table)[0], 3))

        for idx, od in enumerate(t_range):
            x_c, y_pos = self.get_order_y_center(trace_table[od], center_x)
            t_center_y_pos[idx, :] = y_pos
        # import pdb;pdb.set_trace()
        new_order = t_center_y_pos[:, 0].argsort()
        t_center_y_pos = t_center_y_pos[new_order]
        result_table = result_table[new_order]

        return t_center_y_pos, result_table

    def add_next_trace(self, crt_table, ot_idx, y_pos):
        trace_range = np.arange(self.trace_range[ot_idx][0], self.trace_range[ot_idx][1]+1, dtype=int)
        trace_table = self.order_trace_table[ot_idx]
        center_x = self.col // 2

        extract_order_idx = []
        extract_y_center = []
        # import pdb;pdb.set_trace()
        for od in trace_range:
            found_overlap = -1
            x_c,  y_next_pos = self.get_order_y_center(trace_table[od], center_x)

            if self.common_fiber is not None and self.common_fiber in self.for_fiber[ot_idx]:
                found_overlap = self.get_overlap_trace(y_pos, y_next_pos)

            if found_overlap != -1:
                continue

            extract_order_idx.append(od)        # order to extract
            extract_y_center.append(y_next_pos)
        # import pdb;pdb.set_trace()
        new_table = np.concatenate((crt_table, trace_table[extract_order_idx]), axis=0)
        new_y_center = np.concatenate((y_pos, np.array(extract_y_center)), axis=0)

        new_order = np.argsort(new_y_center[:, 0])
        new_table = new_table[new_order]
        new_y_center = new_y_center[new_order]
        return new_table, new_y_center

    def mix_next_trace(self, crt_table, ot_idx, y_pos):
        if self.mixed_trace_range is None or self.mixed_trace_range[ot_idx] is None:
            return crt_table, y_pos

        mixed_trace_range_ot = self.get_trace_range(ot_idx, 'mix')
        mixed_trace_orders = np.arange(mixed_trace_range_ot[0], mixed_trace_range_ot[1]+1, 1, dtype=int)
        trace_table = self.order_trace_table[ot_idx]

        for od in mixed_trace_orders:
            c_x, crt_y_pos = self.get_order_y_center(trace_table[od], self.col//2)
            if crt_y_pos[1] > y_pos[-1][2]:       # out of the range
                break
            pos_idx = self.get_overlap_trace(y_pos, crt_y_pos)
            # print('pos_idx: '+str(pos_idx))

            if pos_idx == -1:
                continue

            new_t_row = self.mix_trace_location(crt_table, pos_idx, trace_table, od)
            crt_table[pos_idx, :] = new_t_row
            y_c = np.polyval(np.flip(new_t_row[0:self.poly_degree+1]), self.col//2)
            y_lower, y_upper = y_c - new_t_row[self.poly_degree+1], y_c + new_t_row[self.poly_degree+2]
            y_pos[pos_idx, :] = [y_c, y_lower, y_upper]
        # import pdb;pdb.set_trace()
        return crt_table, y_pos

    def mix_trace_location(self, crt_table, crt_idx, new_trace_table, new_idx):
        coeff1 = np.flip(crt_table[crt_idx][0:self.poly_degree+1])
        coeff2 = np.flip(new_trace_table[new_idx][0:self.poly_degree+1])
        crt_x1, crt_x2 = crt_table[crt_idx][self.poly_degree+3:self.poly_degree+5]
        new_x1, new_x2 = new_trace_table[new_idx][self.poly_degree+3:self.poly_degree+5]
        widths_1 = crt_table[crt_idx][self.poly_degree+1:self.poly_degree+3]
        widths_2 = new_trace_table[new_idx][self.poly_degree+1:self.poly_degree+3]

        crt_x_vals = np.arange(int(crt_x1), int(crt_x2), 1, dtype=int)
        new_x_vals = np.arange(int(new_x1), int(new_x2), 1, dtype=int)
        crt_y_vals = np.polyval(coeff1, crt_x_vals)
        new_y_vals = np.polyval(coeff2, new_x_vals)

        x_set = np.concatenate((crt_x_vals, new_x_vals), axis=0)
        y_set = np.concatenate((crt_y_vals, new_y_vals), axis=0)
        sort_idx = np.argsort(x_set)
        x_set = x_set[sort_idx]
        y_set = y_set[sort_idx]

        new_coeffs = np.polyfit(x_set, y_set, self.poly_degree)
        new_widths = [(widths_1[0]+widths_2[0])/2, (widths_1[1]+widths_2[1])/2]
        new_x_range = [min(crt_x1, new_x1), max(crt_x2, new_x2)]
        mix_trace = np.flip(new_coeffs).tolist() + new_widths + new_x_range
        return np.array(mix_trace)

    def _perform(self):
        """
        Primitive action - perform combine order trace_result

        Returns:
            DataFrame instance containing the combine order trace result.
        """

        # get the first set of order trace table to combine
        first_idx = -1
        for i in range(self.total_order_trace):
            if self.order_trace_table[i] is not None:
                first_idx = i
                break

        # import pdb;pdb.set_trace()
        new_table = None
        new_y_center = None

        if first_idx != -1:
            init_center_y, init_table = self.init_first_trace(first_idx, 'add')

            new_table = init_table.copy()
            new_y_center = init_center_y.copy()
            # import pdb;pdb.set_trace()
            for f in range(first_idx+1, self.total_order_trace):
                if self.order_trace_table[f] is not None:
                    new_table, new_y_center = self.add_next_trace(new_table, f, new_y_center)
                    # import pdb;pdb.set_trace()

        mix_idx = -1
        if self.mixed_trace_range is not None and \
                isinstance(self.mixed_trace_range, list) and \
                len(self.mixed_trace_range) == self.total_order_trace:
            for i in range(self.total_order_trace):
                if self.mixed_trace_range[i] is not None:
                    mix_idx = i
                    break
        # import pdb;pdb.set_trace()
        # handle mixed order trace
        if mix_idx != -1:
            mix_init_y, mix_init_table = self.init_first_trace(mix_idx, 'mix')
            # import pdb;pdb.set_trace()
            mix_center_y = mix_init_y.copy()
            mix_table = mix_init_table.copy()

            for f in range(mix_idx+1, self.total_order_trace):
                if self.order_trace_table[f] is not None:
                    mix_table, mix_center_y = self.mix_next_trace(mix_table, f, mix_center_y)
                # import pdb;pdb.set_trace()
            if mix_center_y is not None and mix_table is not None:
                new_table = np.concatenate((new_table, mix_table), axis=0) if new_table is not None else mix_table
                new_y_center = np.concatenate((new_y_center, mix_center_y), axis=0) \
                    if new_y_center is not None else mix_center_y

                new_order = np.argsort(new_y_center[:, 0])
                new_table = new_table[new_order]

        trace_table = {}

        column_names = ['Coeff'+str(i) for i in range(self.poly_degree+1)]
        for i in range(self.poly_degree+1):
            trace_table[column_names[i]] = new_table[:, i]

        trace_table['BottomEdge'] = new_table[:, self.poly_degree+1]
        trace_table['TopEdge'] = new_table[:, self.poly_degree+2]
        trace_table['X1'] = new_table[:, self.poly_degree+3]
        trace_table['X2'] = new_table[:, self.poly_degree+4]
        df = pd.DataFrame(trace_table)
        assert(isinstance(df, pd.DataFrame))

        if self.output_path:
            df.to_csv(self.output_path)

        # import pdb;pdb.set_trace()
        if self.logger:
            self.logger.info("OrderTraceCombine: order trace combine")

        if self.logger:
            self.logger.info("OrderTraceCombine: Done!")

        return Arguments(df)

