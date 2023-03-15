import numpy as np
import pandas as pd
from astropy.io import fits
from modules.Utils.config_parser import ConfigHandler
from modules.Utils.alg_base import ModuleAlgBase

class OrderMaskAlg(ModuleAlgBase):
    """
    This module defines class 'OrderMaskAlg' and methods to perform order trace mask.
    A 2D array of the same size as that of the input spectrum data is returned and the pixels that are covered by the
    order traces are set to be True, otherwise the pixels are set to be False. The coverage is determined by the order
    trace location, the vertical edges and the horizontal range defined in the order trace data plus the origin defined
    in the order trace header regarding where the trace position is related to.

    Args:
        spectrum_data (numpy.ndarray): 2D spectrum raw data.
        order_trace_data (Union[numpy.ndarray, pandas.DataFrame]): order trace data including polynomial coefficients,
            top/bottom edges and horizontal coverage of the order trace.
        order_trace_header (dict): dict instance containing order trace info including the origin and polynomial degree
            to fit the trace.
        start_order (int): the index of the first orderlet of the first order. It is based on the assumption that
            the index of the first trace in the order_trace_data is counted as 0.
            A negative number like -n means the first n traces are not included in the order_trace_data.
        order_mask_data (numpy.ndarray): 2D array to contain the order mask data. Defaults to None.
        config (configparser.ConfigParser, optional): config context. Defaults to None.
        logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.
        logger_name (str, optional): Name of the logger defined for the OrderMaskAlg instance.

    Attributes:
        spectrum_flux (numpy.ndarray): Numpy array storing 2d spectrum data for spectral extraction. None is allowed.
        poly_order (int): Polynomial order for the approximation made on the order trace.
        origin (list): The origin of the image from the original raw image.
        instrument (str): Instrument of the observation.
        config_ins (ConfigHandler): Instance of ConfigHandler related to section for the instrument or 'PARAM' section.
        order_trace (numpy.ndarrary): Order trace results from order trace module including polynomial coefficients,
            top/bottom edges and  area coverage of the order trace.
        total_order (int): Total order in order trace object.
        order_coeffs (numpy.ndarray): Polynomial coefficients for order trace from higher to lower order.
        dim_w, dim_h (int): image dimension, width and height.
        order_edges (numpy.ndarray): Bottom and top edges along order trace.
        order_xrange (numpy.ndarray): Left and right boundary of order trace.
        order_mask_data (numpy.ndarray): order mask per spectrum flux and order trace result.
        orderlet_names (list): orderlet names for each order in flat file.
        start_order (int): the index of the first orderlet of the first order.
        total_trace_per_order (int): total orderlets (traces) per order in flat file.
        order_mask_data (numpy.ndarray): 2D array to contain the order mask data.

    Raises:
        TypeError: If there is type error for `spectrum_data`, or `order_trace_data`.
        TypeError: If there is type error for `order_trace_header`.
        TypeError: If there is no flux data for order mask.
    """

    X = 0
    Y = 1

    # @profile
    def __init__(self, spectrum_data, order_trace_data, order_trace_header,
                 orderlet_names=None,
                 start_order=0,
                 order_mask_data=None,
                 config=None, logger=None,
                 logger_name=None):

        if spectrum_data is None:
            raise TypeError("no flux data for order_mask, cannot construct object from OrderMaskAlg")
        elif not isinstance(spectrum_data, np.ndarray):
            raise TypeError('flux data type error, cannot construct object from OrderMaskAlg')
        elif not spectrum_data.any():
            raise TypeError('empty flux data, cannot construct object from OrderMaskAlg')

        if not isinstance(order_trace_data, np.ndarray) and not isinstance(order_trace_data, pd.DataFrame):
            raise TypeError('order trace data error, cannot construct object from OrderMaskAlg')
        if not isinstance(order_trace_header, dict) and not isinstance(order_trace_header, fits.header.Header):
            raise TypeError('type: ' + str(type(order_trace_header)) +
                            ' flux header type error, cannot construct object from OrderMaskAlg')

        ModuleAlgBase.__init__(self, logger_name, config, logger)

        self.spectrum_flux = spectrum_data
        self.poly_order = order_trace_header['POLY_DEG'] if 'POLY_DEG' in order_trace_header else 3

        # origin of the image
        self.origin = [order_trace_header['STARTCOL'] if 'STARTCOL' in order_trace_header else 0,
                       order_trace_header['STARTROW'] if 'STARTROW' in order_trace_header else 0]

        ins = self.config_param.get_config_value('instrument', '') if self.config_param is not None else ''
        self.instrument = ins.upper()
        # section of instrument or 'PARAM'
        self.config_ins = ConfigHandler(config, ins, self.config_param)

        self.total_order = np.shape(order_trace_data)[0]
        # convert order_trace_data into np.ndarray type
        if isinstance(order_trace_data, pd.DataFrame):
            self.order_trace = order_trace_data.values
        else:
            self.order_trace = np.array(order_trace_data)

        # coefficients from 0 to high exponent
        self.order_coeffs = np.flip(self.order_trace[:, 0:self.poly_order+1], axis=1)
        self.dim_w = np.shape(self.spectrum_flux)[1]
        self.dim_h = np.shape(self.spectrum_flux)[0]

        self.order_edges = None
        self.order_xrange = None

        # orderlet names in flat image, the index in order trace list for the first orderlet of first order
        self.orderlet_names = orderlet_names
        self.start_order = start_order
        self.total_trace_per_order = len(orderlet_names) \
            if ((isinstance(orderlet_names, list)) and (len(orderlet_names) > 0)) else None

        self.order_mask_data = np.zeros_like(self.spectrum_flux).astype(bool) if order_mask_data is None \
            else order_mask_data

    def get_config_value(self, prop, default=''):
        """ Get defined value from the config file.

        Search the value of the specified property from instrument section. The default value is returned if not found.

        Args:
            prop (str): Name of the parameter to be searched.
            default (Union[int, float, str], optional): Default value for the searched parameter.

        Returns:
            Union[int, float, str]: Value for the searched parameter.

        """
        return self.config_ins.get_config_value(prop, default)

    def get_order_edges(self, idx=0):
        """ Get the top and bottom edges of the specified order.

        Args:
            idx (int, optional): Index of the order in the order trace array. Defaults to zero.

        Returns:
            numpy.ndarray: Bottom and top edges of the order, `idx`. The first in the array is the bottom edge,
            and the second in the array is the top edge.

        """
        if self.order_edges is None:
            trace_col = np.shape(self.order_trace)[1]
            if trace_col >= self.poly_order+3:
                self.order_edges = self.order_trace[:, self.poly_order+1: self.poly_order+3]
            else:
                self.order_edges = np.repeat(np.ones((1, 2))*self.get_config_value('width_default', 6),
                                             self.total_order, axis=0)

        return self.order_edges[idx, :] if self.total_order > idx >= 0 else self.order_edges[0, :]

    def get_order_xrange(self, idx=0):
        """ Get the left and right x boundaries of the specified order.

        Args:
            idx (int, optional): Index of the order in the order trace array. Defaults to zero.

        Returns:
            numpy.ndarray: Left and right boundaries of order, `idx`. The first in the array is the left end,
            and the second in the array is the right end.

        """
        if self.order_xrange is None:
            trace_col = np.shape(self.order_trace)[1]
            if trace_col >= self.poly_order + 5:
                self.order_xrange = self.order_trace[:, self.poly_order + 3: self.poly_order + 5].astype(int)
            else:
                self.order_xrange = np.repeat(np.array([0, self.dim_w - 1], dtype=int).reshape((1, 2)),
                                              self.total_order, axis=0)
        if idx >= self.total_order or idx < 0:
            idx = 0
        return self.order_xrange[idx, :]

    def set_order_mask(self, order_idx):
        if order_idx < 0 or order_idx >= self.total_order:
            raise Exception("wrong order index number")

        # horizontal range
        xrange = self.get_order_xrange(order_idx)
        x_o = self.origin[self.X]
        x_step = np.arange(xrange[0], xrange[1]+1).astype(int)
        coeffs = self.order_coeffs[order_idx]
        y_mid = np.polyval(coeffs, x_step) + self.origin[self.Y]

        # top and bottom widths
        y_widths = self.get_order_edges(order_idx)

        y_lower = np.ceil(y_mid - y_widths[0]).astype(int)
        y_lower = np.where(y_lower <= 0, 0, y_lower)

        y_upper = np.floor(y_mid + y_widths[1]).astype(int)
        y_upper = np.where(y_upper >= self.dim_h, self.dim_h-1, y_upper)

        x_img_step = x_step + x_o
        for idx, x in enumerate(x_img_step):
            self.order_mask_data[y_lower[idx]:y_upper[idx], x] = True

        return self.order_mask_data

    def get_orderlet_names(self):
        """ Get Orderlette names defined in config.

        Returns:
            list: list of orderlet names
        """
        if self.orderlet_names is None:
            o_list = self.get_config_value('orderlet_names', "['SCI']")
            if isinstance(o_list, list):
                self.orderlet_names = o_list
            elif isinstance(o_list, str):
                self.orderlet_names = o_list.split(',')

        return self.orderlet_names

    def get_total_orderlets_from_image(self):
        """ Get total orderlets from level 0 image, defined in config

        Returns:
            int: total orderdelettes.
        """
        if self.total_trace_per_order is None:
            self.total_trace_per_order = self.get_config_value("total_image_orderlets", 1)

        return self.total_trace_per_order

    def get_orderlet_index(self, order_name):
        """ Find the index of the order name in the orderlet name list.

        Args:
            order_name (str): Fiber name

        Returns:
            int: index of order name in the orderlet name list. If not existing, return is 0.

        """
        all_names = self.get_orderlet_names()
        order_name_idx = all_names.index(order_name) if order_name in all_names else 0
        order_name_idx = order_name_idx % self.total_trace_per_order

        return order_name_idx

    def get_order_set(self, order_name):
        """ Get the list of the trace index eligible for order mask process.

        Args:
            order_name (str): Fiber name.

        Returns:
            list: list of the trace index.

        """
        orderlet_index = self.get_orderlet_index(order_name)
        traces_per_order = self.total_trace_per_order

        if orderlet_index < traces_per_order:
            o_set_ary = np.arange(orderlet_index, self.total_order, traces_per_order, dtype=int) + self.start_order
            o_set = o_set_ary[np.where((o_set_ary < self.total_order) & (o_set_ary >= 0))]
        else:
            o_set = np.array([])

        return o_set

    def get_order_mask(self, order_name, show_time=False, print_debug=None):
        """ From 2D flux to 2D with order masked.

        Args:
            order_name (str): orderlet name
            show_time (bool, optional):  Show running time of the steps. Defaults to False.
            print_debug (str, optional): Print debug information to stdout if it is provided as empty string,
                a file with path `print_debug` if it is non empty string, or no print if it is None.
                Defaults to None.

        Returns:
            dict: order mask result, like::

                    {
                        'order_mask_result':  np.ndarray
                    }

        """

        if order_name is None:
            return {'order_mask_result': self.order_mask_data}

        self.add_file_logger(print_debug)
        self.enable_time_profile(show_time)
        order_set = self.get_order_set(order_name)

        self.d_print('OrderMaskAlg: do order mask on ', len(order_set), ' orders')

        t_start = self.start_time()

        for idx_out in range(len(order_set)):
            c_order = order_set[idx_out]
            self.d_print('OrderMaskAlg: ', c_order, ' edges: ', self.get_order_edges(c_order),
                         ' xrange: ', self.get_order_xrange(c_order))

            self.set_order_mask(c_order)
            t_start = self.time_check(t_start, '**** time [' + str(c_order) + ']: ')

        return {'order_mask_result': self.order_mask_data}
