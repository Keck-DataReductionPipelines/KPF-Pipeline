# Standard dependencies
"""
    This module defines class OrderMask which inherits from `KPF0_Primitive` and provides methods to perform
    the event on order Mask in the recipe.

    Attributes:
        OrderNask

    Description:
        * Method `__init__`:

            OrderMask constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `OrderRectification` event issued in the recipe:

                    - `action.args[0] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing level 0 data for
                      making order mask based on the order trace result.
                    - `action.args[1] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` to contain order mask result.
                    - `action.args['data_extension']: (str, optional)`: the name of the extension containing data.
                    - `action.args['trace_file']: (str, optional)`: the name file containing order trace results.
                    - `action.args['orderlet_names'] (str|list, optional)`: Name or list of names of all orderlets
                      included in the order trace result. Defaults to 'SCI1'.
                    - `action.args['orderlets_on_image'] (str|list, optional)`: Name or list of names of the orderlets
                      appear on the L0 image. Defaults to None.
                    - `action.args['start_order'] (int, optional)`: Index of the first orderlet of the first order
                      assuming the first trace in the order trace file starts with index 0. Defaults to 0.
                    - `action.args['poly_degree']: (str, optional)`: Polynomial degree for order trace curve fitting.
                      Defaults to 3.
                    - `action.args['origin']: (list, optional)`: Origin of the image where the order trace is related
                      to. Defaults to [0, 0]
                    - `action.args['orderlet_widths']: (dict, optional)`: Orderlet widths to replace the edge widths
                      from the order trace file. Defaults to {} or None.

                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of order trace in the master
                  config file associated with the recipe.

            and the following attributes are defined to initialize the object,

                - `img_data (numpy.ndarray)`: raw flux data for order trace mask.
                - `output_level0 (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` to contain order mask result.
                - `orderlet_names (list)`: list of orderlet names per order.
                - `data_ext (str)`: extension of flux data to be processed.
                - `orderlets_on_image (list)`: list of names of the orderlets appearing on the L0 image.
                - `order_trace_data (pandas.DataFrame): order trace data including polynomial coefficients,
                  top/bottom edges and horizontal coverage for each order trace.
                - `config_path (str)`: Path of config file for spectral extraction.
                - `config (configparser.ConfigParser)`: Config context per the file defined by `config_path`.
                - `logger (logging.Logger)`: Instance of logging.Logger.
                - `alg (modules.order_trace.src.alg.OrderMaskAlg)`: Instance of `OrderMaskAlg` which has operation
                  codes for the computation of order mask.

        * Method `__perform`:

            OrderRectification returns the result in `Arguments` object which contains a level 0 data object (`KPF0`)
            with the rectification results replacing the raw image.

    Usage:
        For the recipe, the spectral extraction event is issued like::

            :
            lev0_data = kpf0_from_fits(input_lev0_flat, data_type=data_type)
            op_data = OrderMask(lev0_data, NULL, orderlet_names=order_names,
                                        rectification_method=rect_method,
                                        clip_file="/clip/file/folder/fileprefix")
            :
"""


import configparser
import pandas as pd
import numpy as np
import os
import numbers

# Pipeline dependencies
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.models.level0 import KPF0

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.order_trace.src.alg_order_mask import OrderMaskAlg

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/spectral_extraction/configs/default.cfg'

class OrderMask(KPF0_Primitive):
    default_args_val = {
                    'start_order': 0,
                    'data_extension': 'DATA',
                    'trace_file': None,
                    'poly_degree': 3,
                    'origin': [0, 0],
                    'orderlets_on_image': None,
                    'orderlet_widths': None
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
        input_flux = action.args[0]         # kpf0 instance for L0 data
        self.output_level0 = action.args[1] # kpf0 output containing order mask
        o_names = self.get_args_value('orderlet_names', action.args, args_keys)
        if o_names is not None and o_names:
            self.orderlet_names = o_names if type(o_names) is list else [o_names]
        else:
            self.orderlet_names = None
        start_order = self.get_args_value("start_order", action.args, args_keys)  # the index of first orderlet

        self.data_ext =  self.get_args_value('data_extension', action.args, args_keys)
        order_trace_file = self.get_args_value('trace_file', action.args, args_keys)
        self.orderlets_on_image = action.args["orderlets_on_image"] if "orderlets_on_image" in args_keys \
            else self.orderlet_names

        od_widths = action.args['orderlet_widths'] if 'orderlet_widths' in args_keys else None
        if od_widths is not None and isinstance(od_widths, list) and od_widths:
            self.orderlet_widths = od_widths
        else:
            self.orderlet_widths = None

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['order_trace']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        # start a logger
        self.logger = None
        if not self.logger:
            self.logger = self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        # Order trace algorithm setup
        self.img_data = input_flux[self.data_ext] if input_flux is not None and hasattr(input_flux, self.data_ext) \
            else None

        self.order_trace_data = None
        order_trace_header = None
        if order_trace_file and os.path.isfile(order_trace_file):
            self.order_trace_data = pd.read_csv(order_trace_file, header=0, index_col=0)
            poly_degree = self.get_args_value('poly_degree', action.args, args_keys)
            origin = self.get_args_value('origin', action.args, args_keys)
            order_trace_header = {'STARTCOL': origin[0], 'STARTROW': origin[1], 'POLY_DEG': poly_degree}

        try:
            if self.orderlets_on_image is None:
                self.alg = None
            else:
                self.alg = OrderMaskAlg(self.img_data,
                                self.order_trace_data,
                                order_trace_header,
                                orderlet_names=self.orderlet_names,
                                start_order=start_order,
                                config=self.config, logger=self.logger)
        except Exception as e:
            self.alg = None

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        return True

    def _post_condition(self) -> bool:
        """
        Check for some necessary post conditions
        """
        return True

    def _perform(self):
        """
        Primitive action -
        perform order mask by calling method ``create_order_mask from OrderMaskAlg and create an array of
        of data in which the pixels covering the traces are masked.

        Returns:
            2D array containing order mask data

        """

        if self.logger:
            self.logger.info("OrderMask: creating order mask...")
        if self.alg is None:
            if self.logger:
                self.logger.info("OrderMask: no flux data or order trace data or no orderlet on image for "+self.data_ext)
            return Arguments(None)

        order_mask_result = None
        total_orderlet = len(self.orderlet_names)

        def is_eligible_width_number(w_num):
            return (isinstance(w_num, numbers.Number) or w_num.isnumeric()) and float(w_num) >= 0.0

        if self.orderlet_widths is not None and len(self.orderlet_names) != len(self.orderlet_widths):
            if self.logger:
                self.logger.info("OrderMask: size of orderlet width not match the size of orderlet names")
            self.orderlet_widths = None

        for idx in range(total_orderlet):
            order_name = self.orderlet_names[idx]
            o_width = self.orderlet_widths[idx] if self.orderlet_widths is not None else None
            if o_width is not None:
                if not isinstance(o_width, list):           # list
                    o_width = [o_width, o_width]

                if len(o_width) < 1:                # empty list
                    o_width = None
                elif len(o_width) == 1:
                    o_width = [o_width[0], o_width[0]]
                else:
                    o_width = [o_width[0], o_width[1]]

                if o_width is not None:
                   o_width = [float(o_width[0]), float(o_width[1])] \
                       if all([is_eligible_width_number(o_width[i]) for i in range(2)]) else None

            o_name = order_name if order_name in self.orderlets_on_image else None
            order_mask_result = self.alg.get_order_mask(o_name, s_width=o_width)

            assert ('order_mask_result' in order_mask_result and
                    isinstance(order_mask_result['order_mask_result'], np.ndarray))

        if self.output_level0 is None:
            self.output_level0 = KPF0()

        self.output_level0[self.data_ext] = order_mask_result['order_mask_result']
        self.output_level0.header['PRIMARY']['IMTYPE'] = 'OrderMask'
        self.output_level0.receipt_add_entry('OrderMask', self.__module__, f'order mask is created for {self.data_ext}',
                                             'PASS')

        if self.logger:
            self.logger.info("OrderMask: Receipt written")

        return Arguments(self.output_level0)

    def get_args_value(self, key: str, args: Arguments, args_keys: list):
        if key in args_keys:
            v = args[key]
        else:
            v = self.default_args_val[key]

        return v
