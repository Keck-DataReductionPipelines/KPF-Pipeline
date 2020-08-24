import configparser
import numpy as np
import math
import time
import pandas as pd
from astropy.io import fits
import re

# Pipeline dependencies
# from kpfpipe.logger import start_logger
# from kpfpipe.primitives.level0 import KPF0_Primitive
# from kpfpipe.models.level0 import KPF0


class OptimalExtractionAlg:
    """
    This module defines class 'OptimalExtractionAlg' and methods to perform the optimal or summation
    extraction which reduces 2D spectrum to 1D spectrum for each order. The process includes 2 steps.
    In the first step, the flux of each order from both spectral data and flat data is
    processed and output to a new data set by using either rectification method or not.
    The second step performs either the optimal or summation extraction to reduce the output data of
    the first step into 1D data for each order.

    For the first step, the pixels along the normal or vertical direction of the order trace are collected and
    processed column by column in 3 types of methods,

        - no rectification method: collecting the pixels within the edge size along the north-up direction
          of the order and taking the pixel flux in full to result the output pixel.
        - vertical rectification method: collecting the pixels within the edge size along the north-up direction
          of the order and performing fractional summation over the collected pixels to result the output pixel.
          The output pixel coverage is based on the vector along the vertical direction and
          the weighting for the fractional summation is based on the overlapping between the collected pixels
          and the output pixel.
        - normal rectification method: collecting the pixels within the edge size along the normal direction
          of the order and performing fractional summation over the collected pixels to result the output pixel.
          The output pixel coverage is based on the vector along the normal direction and the weighting for the
          fractional summation is based on the overlapping between the collected pixels and the output pixel.

    For the second step, either optimal or summation extraction is performed to reduce the 2D data along each order
    into 1D data. By using optimal extraction, the output pixel of the first step from the spectrum data are weighted
    and summed up column by column and the weighting is based on the associated output pixel of the first
    step from the flat data. By using summation extraction, the pixels are summed up directly.


    Args:
        flat_data (numpy.ndarray): 2D flat data.
        spectrum_data (numpy.ndarray): 2D spectrum data.
        spectrum_header (fits.header.Header): fits header of spectrum data.
        order_trace_data (Union[numpy.ndarray, pandas.DataFrame]): order trace data including polynomial coeffients,
            top/bottom edges and area coverage of the order trace.
        order_trace_header (dict): fits header of order trace extension.
        config (configparser.ConfigParser, optional): config context. Defautls to None.
        logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.

    Attributes:
        logger (logging.Logger): Instance of logging.Logger.
        instrument (str): Imaging instrument.
        flat_flux (numpy.ndarray): Numpy array storing 2d flat data.
        spectrum_flux (numpy.ndarray): Numpy array storing 2d spectrum data for optimal extraction.
        spectrum_header (fits.header.Header): Header of the fits for spectrum data.
        dim_width (int): Width of spectrum data/flat data.
        dim_height (int): Height of spectrum data/flat data.
        poly_order (int): Polynomial order for the approximation made on the order trace.
        config_param (configparser.SectionProxy): Related to 'PARAM' section or the section associated with
            the instrument if it is defined in the config file.
        order_trace (numpy.ndarrary): Order trace results from order trace module including polynomial coefficients,
            top/bottom edges and  area coverage of the order trace.
        total_order (int): Total order in order trace object.
        order_coeffs (numpy.ndarray): Polynomial coefficients for order trace from higher to lower order.
        order_edges (numpy.ndarray): Bottom and top edges along order trace.
        order_xrange (numpy.ndarray): Left and right boundary of order trace.
        debug_output (str): File path for the file that the debug information is printed to. The printing goes to
            standard output if it is an empty string or no printing is made if it is None.
        is_time_profile (bool): Print out the time status while running.
        is_debug (bool): Print out the debug information while running.

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
        TypeError: If there is type error for `flat_data`, `spectrum_data`, or `order_trace_data`.
        TypeError: If there is type error for `spectrum_header` or `order_trace_header`.
        Exception: If the order size between spectrum data and order trace data is not the same.

    """

    X = 0
    Y = 1
    NORMAL = 0
    VERTICAL = 1
    NoRECT = 2
    OPTIMAL = 'optimal'
    SUM = 'sum'
    rectifying_method = ['normal rectification', 'vertical rectification', 'no rectification' ]

    def __init__(self, flat_data, spectrum_data, spectrum_header,  order_trace_data, order_trace_header,
                 config=None, logger=None):

        if not isinstance(flat_data, np.ndarray):
            raise TypeError('flat data type error, cannot construct object from OptionalExtractionAlg')
        if not isinstance(spectrum_data, np.ndarray):
            raise TypeError('flux data type error, cannot construct object from OptionalExtractionAlg')
        if not isinstance(order_trace_data, np.ndarray) and not isinstance(order_trace_data, pd.DataFrame):
            raise TypeError('flux data type error, cannot construct object from OptionalExtractionAlg')
        if not isinstance(spectrum_header, fits.header.Header):
            raise TypeError('flux header type error, cannot construct object from OptionalExtractionAlg')
        if not isinstance(order_trace_header, dict) and not isinstance(order_trace_header, fits.header.Header):
            raise TypeError('type: ' + type(order_trace_header) +
                            ' flux header type error, cannot construct object from OptionalExtractionAlg')

        self.logger = logger
        self.flat_flux = flat_data
        self.spectrum_flux = spectrum_data
        self.spectrum_header = spectrum_header
        rows, cols = np.shape(self.flat_flux)
        self.dim_width = cols
        self.dim_height = rows
        self.poly_order = order_trace_header['POLY_DEG'] if 'POLY_DEG' in order_trace_header else 3
        p_config = config['PARAM'] if config is not None and config.has_section('PARAM') else None
        self.instrument = p_config.get('instrument', '') if p_config is not None else ''
        ins = self.instrument.upper()
        self.config_param = config[ins] if ins and config.has_section(ins) else p_config
        self.total_order = np.shape(order_trace_data)[0]
        if isinstance(order_trace_data, pd.DataFrame):
            self.order_trace = order_trace_data.values
        else:
            self.order_trace = np.array(order_trace_data)
        self.order_coeffs = np.flip(self.order_trace[:, 0:self.poly_order+1], axis=1)
        self.order_edges = None
        self.order_xrange = None

        self.is_debug = True if self.logger else False
        self.debug_output = None
        self.is_time_profile = False

    def get_config_value(self, prop, default=''):
        """ Get defined value from the config file.

        Search the value of the specified property fom config section. The default value is returned if not found.

        Args:
            prop (str): Name of the parameter to be searched.
            default (Union[int, float, str], optional): Default value for the searched parameter.

        Returns:
            Union[int, float, str]: Value for the searched parameter.

        """
        if self.config_param is not None:
            if isinstance(default, int):
                return self.config_param.getint(prop, default)
            elif isinstance(default, float):
                return self.config_param.getfloat(prop, default)
            else:
                return self.config_param.get(prop, default)
        else:
            return default

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
        if idx >= self.total_order or idx < 0:
            idx = 0
        return self.order_edges[idx, :]

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
                self.order_xrange = np.repeat(np.array([0, self.dim_width-1], dtype=int).reshape((1, 2)),
                                              self.total_order, axis=0)
        if idx >= self.total_order or idx < 0:
            idx = 0
        return self.order_xrange[idx, :]

    def get_spectrum_size(self):
        """ Get the dimension of the spectrum data.

        Returns:
            tuple: Dimension of the data,

                * (*int*): Width of the data.
                * (*int*): Height of the data.

        """
        return self.dim_width, self.dim_height

    def get_spectrum_order(self):
        """ Get the total order of the order trace or the spectrum data.

        Returns:
            int: The total order.

        """
        return self.total_order

    def get_instrument(self):
        """ Get imaging instrument.

        Returns:
            str: Instrument name.
        """
        return self.instrument

    def enable_debug_print(self, to_print=True):
        """ Enable or disable debug printing.

        Args:
            to_print (bool, optional): Print out the debug information of the execution. Defaults to False.

        Returns:
            None.

        """
        self.is_debug = to_print or bool(self.logger)

    def enable_time_profile(self, is_time=False):
        """ Enable or disable time profiling printing.

        Args:
            is_time (bool, optional): Print out the time of the execution. Defaults to False.

        Returns:
            None.

        """
        self.is_time_profile = is_time

    def d_print(self, *args, end='\n', info=False):
        """ Print out running status to logger or debug information to a file.

        Args:
            *args: Variable length argument list to print.
            end (str, optional): Specify what to print at the end.
            info (bool): Print out for information level, not for debug level.

        Returns:
            This function handles the print-out to the logger defined in the config file or other file as specified in
            :func:`~alg.OptimalExtractionAlg.add_file_logger()`.

        """
        if self.is_debug:
            out_str = ' '.join([str(item) for item in args])
            if self.logger:
                if info:
                    self.logger.info(out_str)
                else:
                    self.logger.debug(out_str)
            if self.debug_output is not None and not info:
                if self.debug_output:
                    with open(self.debug_output, 'a') as f:
                        f.write(' '.join([str(item) for item in args]) + end)
                        f.close()
                else:
                    print(out_str, end=end)

    def t_print(self, *args):
        if self.is_time_profile and self.logger:
            out_str = ' '.join([str(item) for item in args])
            self.logger.info(out_str)

    def update_spectrum_flux(self, bleeding_cure_file=None):
        """ Update the spectrum flux per specified bleeding cure file or 'nan_pixels' set in config file.

        Args:
            bleeding_cure_file (str, optional): Filename of bleeding cure file if there is. Defaults to None.

        Returns:
            None.

        """
        if bleeding_cure_file is not None:
            correct_data, correct_header = fits.getdata(bleeding_cure_file, header=True)
            if np.shape(correct_data) == np.shape(self.spectrum_flux):
                correct_method = self.get_config_value('correct_method')
                if correct_method == 'sub':
                    self.spectrum_flux = self.spectrum_flux - correct_data

        nan_pixels = self.get_config_value('nan_pixels').replace(' ', '')
        if nan_pixels:
            pixel_groups = re.findall("^\\((.*)\\)$", nan_pixels)

            if len(pixel_groups) > 0:            # group of nan pixels is set
                res = [i.start()+1 for i in re.finditer('\\],\\[', pixel_groups[0])]
                res.insert(0, -1)
                res.append(len(pixel_groups[0]))
                idx_groups = [pixel_groups[0][res[i]+1:res[i+1]] for i in range(len(res)-1)]
                for group in idx_groups:
                    idx_set = re.findall("^\\[([0-9]*):([0-9]*),([0-9]*):([0-9]*)\\]$", group)
                    if len(idx_set) > 0 and len(idx_set[0]) == 4:
                        y_idx, x_idx = idx_set[0][0:2], idx_set[0][2:4]
                        r_idx = [int(y_idx[i]) if y_idx[i] else (0 if i == 0 else self.dim_height) for i in range(2)]
                        c_idx = [int(x_idx[i]) if x_idx[i] else (0 if i == 0 else self.dim_width) for i in range(2)]
                        self.spectrum_flux[r_idx[0]:r_idx[1], c_idx[0]:c_idx[1]] = np.nan

    def collect_data_from_order(self, coeffs, widths, xrange, data_group, s_rate=1, sum_extraction=True):
        """ Collect the spectral data along the order per polynomial fit data and no rectification.

        Args:
            coeffs (numpy.ndarray): Polynomial coefficients starting from higher order.
            widths (numpy.ndarray): Bottom and top edges of the order, i.e. `widths[0]` & `widths[1]`.
            xrange (numpy.ndarray):  Horizontal coverage of the order in terms of two ends along x axis.
            data_group (list): Set of input data from various sources such as spectral data and flat data.
            s_rate (Union[list, float], optional): Sampling rate from input domain to output domain for 2D data.
                Defaults to 1.
            sum_extraction(bool, optioanl): Flag to indicate if performing summation on collected data
                column by column. Defaults to False.

        Returns:
            dict: Information of non rectified data from the order including the dimension, like::

                {
                    'y_center': int
                        # the vertical position where to locate the order in output domain
                    'width': list
                        # adjusted bottom and top edges, i.e. [<bottom edge>, <top edge>]
                    'dim': list
                        # dimension of data in 'out_data', [<x_dimension>, <y_dimension>]
                    'out_data': list
                        # collected flux based the data set from input parameter data_group
                }

        """

        input_y_dim, input_x_dim = np.shape(data_group[0])
        sampling_rate = []
        if type(s_rate).__name__ == 'list':
            sampling_rate.extend(s_rate)
        else:
            sampling_rate.extend([s_rate, s_rate])

        output_x_dim = input_x_dim * sampling_rate[self.X]
        output_y_dim = input_y_dim * sampling_rate[self.Y]

        # self.d_print('output_x_dim: ', output_x_dim, 'sampling_rate: ', sampling_rate)

        # construct corners map between input and output
        x_output_step = np.arange(0, output_x_dim, dtype=int)      # x step in output domain, including 0 & output_x_dim
        x_step = self.get_input_pos(x_output_step, sampling_rate[self.X])    # x step in input domain

        # x step compliant to xrange
        x_step = x_step[np.where(np.logical_and(x_step >= xrange[0], x_step <= xrange[1]))[0]]
        x_output_step = self.get_output_pos(x_step, sampling_rate[self.X]).astype(int)

        y_mid = np.polyval(coeffs, x_step)                                    # y position of spectral trace
        v_border = np.array([np.amax(y_mid), np.amin(y_mid)])
        # the vertical position to locate the order in output domain
        y_output_mid = math.floor(np.mean(v_border)*sampling_rate[self.Y])    # a number, output y center
        # self.d_print('y_output_mid: ', y_output_mid)

        output_widths = self.get_output_pos(widths, sampling_rate[self.Y]).astype(int)  # width of output
        upper_width = min(output_widths[1], output_y_dim - 1 - y_output_mid)
        lower_width = min(output_widths[0], y_output_mid)
        # self.d_print('no rectify: width at output: ', upper_width, lower_width)

        y_size = 1 if sum_extraction is True else (upper_width+lower_width)
        total_data_group = len(data_group)
        out_data = [np.zeros((y_size, output_x_dim)) for _ in range(0, total_data_group)]

        # x_output_step in sync with x_step,
        input_widths = np.array([self.get_input_pos(y_o, sampling_rate[self.Y])
                                                            for y_o in range(-lower_width, upper_width)])
        input_x = np.floor(x_step).astype(int)

        for s_x, o_x in enumerate(x_output_step):               # ox: 0...x_dim-1, out_data: 0...x_dim-1, corners: 0...
            # if o_x % 1000 == 0:
            #    self.d_print(o_x, end=" ")

            x_i = input_x[s_x]
            y_input = np.floor(input_widths + y_mid[s_x]).astype(int)
            y_input_idx = np.where((y_input <= (input_y_dim - 1)) & (y_input >= 0))[0]
            y_input = y_input[y_input_idx]
            for i in range(0, total_data_group):
                if sum_extraction is True:
                    out_data[i][0, o_x] = np.sum(data_group[i][y_input, x_i])
                else:
                    out_data[i][y_input_idx , o_x] = data_group[i][y_input, x_i]

        result_data = {'y_center': y_output_mid,
                       'width': [upper_width, lower_width],
                       'dim': [output_y_dim, output_x_dim],
                       'out_data': [out_data[i] for i in range(0, total_data_group)]}

        return result_data

    def rectify_spectrum_curve(self, coeffs, widths, xrange, data_group, s_rate=1, sum_extraction=True,
                               direction=NORMAL):
        """ Rectify the order trace based on the pixel collection method.

        Parameters:
            coeffs (numpy.ndarray): Polynomial coefficients starting from higher order.
            widths (numpy.ndarray): Bottom and top edges of the order.
            xrange (numpy.ndarray):  Horizontal coverage of the order in terms of two ends along x axis.
            data_group (list): Set of input data from various sources such as spectral data and flat data.
            s_rate (list or number, optional): Sampling rate from input domain to output domain for 2D data.
                Defaults to 1.
            sum_extraction(bool, optional): Flag to indicate if performing summation on collected data
                column by column. Defaults to True.
            direction (int, optional): Types of data collection methods for rectification.
                Defualts to NORMAL.

                - NORMAL: collect data along the normal direction of the order.
                - VERTICAL: collect data along the vertical direction of the order.

        Returns:
            dict:  Information of rectified data from the order including the dimension, like::

                {
                    'y_center': int
                        # the vertical position where to locate the order in output domain.
                    'width': list
                         # adjusted bottom and top edges, i.e. [<bottom edge>, <top edge>].
                    'dim': list
                        # dimension of data in 'out_data', [<x_dimension>, <y_dimension>].
                    'out_data': list
                        # collected data based the data set from parameter 'data_group'.
                }

        """

        input_y_dim, input_x_dim = np.shape(data_group[0])
        sampling_rate = []
        if type(s_rate).__name__ == 'list':
            sampling_rate.extend(s_rate)
        else:
            sampling_rate.extend([s_rate, s_rate])

        output_x_dim = input_x_dim * sampling_rate[self.X]
        output_y_dim = input_y_dim * sampling_rate[self.Y]

        # self.d_print('output_x_dim: ', output_x_dim, 'sampling_rate: ', sampling_rate)

        # construct corners map between input and output
        x_output_step = np.arange(0, output_x_dim+1, dtype=int)    # x step in output domain, including 0 & output_x_dim
        x_step = self.get_input_pos(x_output_step, sampling_rate[self.X])  # x step in input domain

        # x step compliant to xrange
        x_step = x_step[np.where(np.logical_and(x_step >= xrange[0], x_step <= (xrange[1]+1)))[0]]
        x_output_step = self.get_output_pos(x_step, sampling_rate[self.X]).astype(int)
        y_mid = np.polyval(coeffs, x_step)          # spectral trace value at mid point

        if direction == self.NORMAL:
            y_norm_step = self.poly_normal(x_step, coeffs, sampling_rate[self.Y])   # curve norm along x in input domain
        else:
            y_norm_step = self.vertical_normal(x_step, sampling_rate[self.Y])   # vertical norm along x in input domain

        v_border = np.array([np.amax(y_mid), np.amin(y_mid)])
        # self.d_print('v_border: ', v_border)
        # the vertical position to locate the order in output domain
        y_output_mid = math.floor(np.mean(v_border)*sampling_rate[self.Y])
        # self.d_print('y_output_mid: ', y_output_mid)

        output_widths = self.get_output_pos(widths, sampling_rate[self.Y]).astype(int)  # width of output
        upper_width = min(output_widths[1], output_y_dim - 1 - y_output_mid)
        lower_width = min(output_widths[0], y_output_mid)
        self.d_print('rectify: width at output: ', upper_width, lower_width)

        corners_at_mid = np.vstack((x_step, y_mid)).T
        # self.d_print('corners_at_mid: ', corners_at_mid)

        y_size = 1 if sum_extraction is True else (upper_width+lower_width)
        total_data_group = len(data_group)
        out_data = [np.zeros((y_size, output_x_dim)) for _ in range(0, total_data_group)]

        input_upper_corners = np.zeros((upper_width+1, x_step.size, 2))
        input_lower_corners = np.zeros((lower_width+1, x_step.size, 2))
        input_upper_corners[0] = corners_at_mid.copy()
        input_lower_corners[0] = corners_at_mid.copy()

        for o_y in range(1, upper_width+1):
            next_upper_corners = self.go_vertical(input_upper_corners[o_y-1], y_norm_step, 1)
            input_upper_corners[o_y] = next_upper_corners

        for o_y in range(1, lower_width+1):
            next_lower_corners = self.go_vertical(input_lower_corners[o_y-1], y_norm_step, -1)
            input_lower_corners[o_y] = next_lower_corners

        s_x = 0
        for o_x in x_output_step[0:-1]:               # o_x: 0...x_dim-1, out_data: 0...x_dim-1, corners: 0...
            for o_y in range(0, upper_width):
                input_corners = input_upper_corners[o_y:o_y+2, s_x:s_x+2].reshape((4, 2))[[0, 2, 3, 1]]
                for i in range(0, total_data_group):
                    flux = self.compute_output_flux(input_corners, data_group[i], input_x_dim, input_y_dim,
                                                    direction)
                    if sum_extraction is True:
                        out_data[i][0, o_x] += flux
                    else:
                        out_data[i][lower_width+o_y, o_x] = flux

            for o_y in range(0, lower_width):
                input_corners = input_lower_corners[o_y:o_y+2, s_x:s_x+2].reshape((4, 2))[[2, 0, 1, 3]]
                for i in range(0, total_data_group):
                    flux = self.compute_output_flux(input_corners, data_group[i], input_x_dim, input_y_dim,
                                                    direction)
                    if sum_extraction is True:
                        out_data[i][0, o_x] += flux
                    else:
                        out_data[i][lower_width-o_y-1, o_x] = flux
            s_x += 1

        # self.d_print(' ')

        result_data = {'y_center': y_output_mid,
                       'width': [upper_width, lower_width],
                       'dim': [output_y_dim, output_x_dim],
                       'out_data': [out_data[i] for i in range(0, total_data_group)]}

        return result_data

    def get_flux_from_order(self, coeffs, widths, x_range, in_data, flat_flux, s_rate=1, norm_direction=None):
        """  Collect the data along the order by either rectifying the pixels or not.

        The data collection is based on the following 2 types of methods,

            - rectification method: the pixels along the order are selected depending on the edge size
              (i.e. `widths`) and the direction (i.e. `norm_direction`). With that, all pixels appearing at
              either vertical or normal direction of the order are collected, weighted and summed up.
              The weighting for each pixel is based on the area of that pixel contributing to the pixel
              after rectification.
            - no rectification method: the pixels along the vertical direction of the order are collected
              in full depending on the edge size.

        Parameters:
            coeffs (numpy.ndarray): Polynomial coefficients starting from higher order.
            widths (numpy.ndarray): Bottom and top edges of the orders.
            x_range (numpy.ndarray): Horizontal coverage of the order in terms of two ends along x axis.
            in_data (numpy.ndarray): 2D spectral data.
            flat_flux (numpy.ndarray): 2D flat data.
            s_rate (Union[list, float], optional): sampling rate from input domain to output domain for 2D data.
                Defaults to 1.
            norm_direction(int, optional): Rectification method. Defaults to None.

                - None: no rectification.
                - VERTICAL: pixels at the north and south direction along the order are collected to
                  be rectified.
                - NORMAL: pixels at the normal direction of the order are collected to be rectified.

        Returns:
            dict: Information related to the order data, like::

                {
                    'order_data': numpy.ndarray
                            # extracted spectrum data from the order using rectification or not.
                    'order_flat': numpy.ndarray
                            # extracted flat data from the order using rectification or not.
                    'data_height': int        # height of 'order_data'.
                    'data_width': int         # width of 'order_data'.
                    'out_y_center': int       # y center position where 'order_data' is located.
                }

        Raises:
            AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
            Exception: If there is unmatched data size between spectrum data and flat data.
            Exception: If the size of polynomial coefficients is not enough for the order it represents for.
            Exception: If bottom or top edge is missing.
            Exception: If the left or right border is missing.
            Exception: If the rectification method is invalid.

        """
        if np.shape(in_data) != np.shape(flat_flux):
            raise Exception("unmatched data size between spectrum data and flat data")
        if np.size(coeffs) != (self.poly_order+1):
            raise Exception("polynomial coefficient error")

        if np.size(widths) < 2:
            raise Exception("bottom or top edge is missing")

        if np.size(x_range) < 2:
            raise Exception("left or right border is missing")

        if norm_direction > self.NoRECT or norm_direction < self.NORMAL:
            raise Exception("invalid rectification method")

        if norm_direction is None or norm_direction == self.NoRECT:
            flux_results = self.collect_data_from_order(coeffs, widths, x_range, [in_data, flat_flux], s_rate,
                                                        sum_extraction=False)
        else:
            flux_results = self.rectify_spectrum_curve(coeffs, widths, x_range, [in_data, flat_flux], s_rate,
                                                       direction=norm_direction, sum_extraction=False)

        height = sum(flux_results.get('width'))
        width = flux_results.get('dim')[1]
        s_data = flux_results.get('out_data')[0]
        f_data = flux_results.get('out_data')[1]

        return {'order_data': s_data, 'order_flat': f_data, 'data_height': height, 'data_width': width,
                'out_y_center': flux_results.get('y_center')}

    @staticmethod
    def optimal_extraction(s_data, f_data, data_height, data_width):
        """ Do optimal extraction on collected pixels along the order.

        This optimal extraction method does the calculation based on the variance of the spectrum data and the
        weighting based on the flat data.

        Args:
            s_data (numpy.ndarray): 2D spectral data collected for one order.
            f_data (numpy.ndarray): 2D flat data collected for one order.
            data_height (int): Height of the 2D data for optimal extraction.
            data_width (int): Width of the 2D data for optimal extraction.

        Returns:
            dict: Information of optimal extraction result, like::

                {
                    'extraction': numpy.ndarray   # optimal extraction result.
                }

        Raises:
            AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
            Exception: If there is unmatched size between collected order data and associated flat data.
            Exception: If the data size doesn't match to the given dimension.

        """

        if np.shape(s_data) != np.shape(f_data):
            raise Exception("unmatched size between collected order data and associated flat data")

        if np.shape(s_data)[0] != data_height or np.shape(s_data)[1] != data_width:
            raise Exception("unmatched data size with the given dimension")

        w_data = np.zeros((1, data_width))

        # taking weighted summation on spectral data of each column,
        # the weight is based on flat data.
        # formula: sum((f/sum(f)) * s/variance)/sum((f/sum(f))^2)/variance) ref. Horne 1986

        d_var = np.full((data_height, 1), 1.0)  # set the variance to be 1.0
        w_sum = np.sum(f_data, axis=0)
        nz_idx = np.where(w_sum != 0.0)[0]
        p_data = f_data[:, nz_idx]/w_sum[nz_idx]
        num = p_data * s_data[:, nz_idx]/d_var
        dem = np.power(p_data, 2)/d_var
        w_data[0, nz_idx] = np.sum(num, axis=0)/np.sum(dem, axis=0)

        # for x in range(0, data_width):
        #    w_sum = sum(f_data[:, x])
        #    if w_sum != 0.0:                        # pixel is not out of range
        #        p_data = f_data[:, x] / w_sum
        #        num = p_data * s_data[:, x] / d_var
        #        dem = np.power(p_data, 2) / d_var
        #        w_data[0, x] = np.sum(num) / np.sum(dem)

        return {'extraction': w_data}


    @staticmethod
    def summation_extraction(s_data):
        """ Spectrum extraction by summation on collected pixels (rectified or non-rectified)

        Args:
            s_data (numpy.ndarray): Collected data for spectrum extraction.

        Returns:
            dict: Information related to the order data, like::

                {
                    'extraction': numpy.ndarray   # optimal extraction result.
                }

        """
        out_data = np.sum(s_data, axis=0)

        return {'extraction': out_data}

    @staticmethod
    def fill_2d_with_data(from_data, to_data, to_pos, from_pos=0):
        """ Fill a band of 2D data into another 2D container starting from and to specified vertical positions.

        Args:
            from_data (numpy.ndarray): Band of data to be copied from.
            to_data (numpy.ndarray): 2D area to copy the data to.
            to_pos (int): The vertical position of `to_data` where `from_data` is copied to.
            from_pos (int): the vertical position of `from_data` where the data is copied from. The default is 0.

        Returns:
            numpy.ndarray: 2D data with `from_data` filled in.

        """

        to_y_dim = np.shape(to_data)[0]
        from_y_dim = np.shape(from_data)[0]

        if to_pos < 0 or to_pos >= to_y_dim or from_pos < 0 or from_pos >= from_y_dim:  # out of range, do nothing
            return to_data

        h = min((to_y_dim - to_pos), from_y_dim-from_pos)
        to_data[to_pos:, :] = from_data[from_pos:from_pos+h, :]
        return to_data

    @staticmethod
    def vertical_normal(pos_x, sampling_rate):
        """ Calculate the vertical vector at the specified x position per vertical sampling rate.

        Args:
            pos_x (numpy.ndarray): x position.
            sampling_rate (float): Vertical sampling rate.

        Returns:
            numpy.ndarray: Vertical vector at position `pos_x`.

        """

        v_norms = np.zeros((pos_x.size, 2))
        d = 1.0/sampling_rate
        v_norms[:, 1] = d

        return v_norms

    @staticmethod
    def poly_normal(pos_x, coeffs, sampling_rate=1):
        """ Calculate the normal vector at the specified x position per vertical sampling rate.

        Args:
            pos_x (numpy.ndarray): x position.
            coeffs (numpy.ndarray): Coefficients of the polynomial fit to the order from higher order to lower.
            sampling_rate (int, optional): Vertical sampling rate. Defaults to 1.

        Returns:
            numpy.ndarray: Normal vectors at positions in `pos_x`.

        """

        d_coeff = np.polyder(coeffs)
        tan = np.polyval(d_coeff, pos_x)
        v_norms = np.zeros((pos_x.size, 2))
        norm_d = np.sqrt(tan*tan+1)*sampling_rate
        v_norms[:, 0] = -tan/norm_d
        v_norms[:, 1] = 1.0/norm_d

        return v_norms

    @staticmethod
    def get_input_pos(output_pos, s_rate):
        """ Get associated position of input domain per output position and sampling rate.

        Args:
            output_pos (numpy.ndarray): Position of output domain.
            s_rate (flat): Sampling ratio between input domain and output domain, i.e. *input*s_rate = output*.

        Returns:
            numpy.ndarray: Position of input domain.

        """
        return output_pos/s_rate

    @staticmethod
    def get_output_pos(input_pos: np.ndarray, s_rate: float):
        """ Get associated output position per input position and sampling rate.

        Args:
            input_pos (numpy.ndarray): Position of input domain.
            s_rate (float): Sampling rate.

        Returns:
            numpy.ndarray: Position of output domain.

        """
        if isinstance(input_pos, np.ndarray):
            return np.floor(input_pos*s_rate)     # position at output cell domain is integer based

    @staticmethod
    def go_vertical(crt_pos, norm, direction=1):
        """ Get new positions by starting from a set of positions and traversing along the specified direction.

        Args:
            crt_pos (numpy.ndarray): Current positions.
            norm (numpy.ndarray): Vector to traverse.
            direction (int, optional): Traverse direction. Defaults to 1.

        Returns:
            numpy.ndarray: New position at given traversing direction.

        """

        new_pos = crt_pos + direction * norm

        return new_pos

    def compute_output_flux(self, input_corners, input_data, input_x_dim, input_y_dim, vertical_normal):
        """ Compute weighted flux covered by a polygon area.

        Compute the flux within a polygon using polygon clipping algorithm if the polygon corners are collected in
        normal direction or checking the area coverage of each pixel inside the polygon if the polygon corners are
        collected in vertical direction.

        Args:
            input_corners(numpy.ndarray): Polygon corners at input domain in counterclockwise order.
            input_data(numpy.ndarray): Input data.
            input_x_dim(int): Width of input data
            input_y_dim(int): Height of input data
            vertical_normal(int): the method regarding how the corners are collected, NORMAL or VERTICAL.

        Returns:
            float : Flux value for the polygon.

        """

        x_list = input_corners[:, self.X]
        y_list = input_corners[:, self.Y]

        x_1 = max(0, math.floor(np.amin(x_list)))
        x_2 = min(input_x_dim, math.ceil(np.amax(x_list)))
        y_1 = max(0, math.floor(np.amin(y_list)))
        y_2 = min(input_y_dim, math.ceil(np.amax(y_list)))

        if vertical_normal == self.VERTICAL:
            flux_vertical, total_area_vertical = self.compute_flux_from_vertical_clipping(input_corners,
                                                                                          [x_1, x_2, y_1, y_2],
                                                                                          input_data)
            return flux_vertical

        flux_polygon, total_area_polygon = self.compute_flux_from_polygon_clipping(input_corners,
                                                                                   [x_1, x_2, y_1, y_2],
                                                                                   input_data)
        return flux_polygon

    def compute_flux_from_polygon_clipping(self, poly_corners, border_points, input_data):
        """ Compute flux on pixels covered by one polygon formed in the normal direction of the order.

        Collect pixels covered by the specified polygon and compute weighted summation on the pixels.

        Args:
            poly_corners (numpy.ndarray): Corners of the polygon.
            border_points (list): Area covered by `poly_corners`, i.e. *[<left_x>, <right_x>, <bottom_y>, <top_y>]*.
            input_data (numpy.ndarray): Imaging data - spectrum data or flat data.

        Returns:
            tuple: Weighted summation of flux over the polygon,

                * **flux** (*float*): Weighted summation of the flux over the polygon.
                * **total_area** (*float*): Total overlapping area between the collected pixels and the polygon.

        """
        x_1, x_2, y_1, y_2 = border_points
        total_area = 0.0
        flux = 0.0
        for x in range(x_1, x_2):
            for y in range(y_1, y_2):
                if input_data[y, x] != 0.0:
                    new_corners = self.polygon_clipping(poly_corners, [[x, y], [x, y+1], [x+1, y+1], [x+1, y]], 4)
                    area = self.polygon_area(new_corners)
                    total_area += area
                    flux += area * input_data[y, x]

        new_flux = flux/total_area
        return new_flux, total_area

    def compute_flux_from_vertical_clipping(self, poly_corners, border_points, input_data):
        """ Compute flux on pixels covered by specified polygon formed in the vertical direction of the order.

        Collect pixels covered by the specified polygon and compute weighted summation on the pixels.
        The computation is made to be more efficient than that of
        :func:`~alg.OptimalExtractionAlg.compute_flux_from_polygon_clipping()`
        due to that two sides of the polygon are formed in the vertical direction.

        Args:
            poly_corners (numpy.ndarray): Corners of the polygon.
            border_points (list): Area covered by `poly_corners`, i.e. *[<left_x>, <right_x>, <bottom_y>, <top_y>]*.
            input_data (numpy.ndarray): Imaging data - spectrum data or flat data.

        Returns:
            tuple: Weighted summation of flux over the polygon,

                * **flux** (*float*): Flux of weighted summation of the flux over the polygon.
                * **total_area** (*float*): Total overlapping area between the collected pixels and the polygon.

        """
        # make mark on vertical grid line
        x1, x2, y1, y2 = border_points  # grid boundary of poly_corners
        y_grid = np.arange(y1, y2+1, dtype=float)

        border_x1 = np.amin(poly_corners[:, self.X])        # x range of poly_corners,
        border_x2 = np.amax(poly_corners[:, self.X])
        border_x = np.arange(math.floor(border_x1), math.ceil(border_x2)+1, dtype=float)  # horizontal coverage along x
        border_x[0] = border_x1 if border_x1 != border_x[0] else border_x[0]
        border_x[-1] = border_x2 if border_x2 != border_x[-1] else border_x[-1]

        # y top and bottom ends of poly_corners at each point in border_x
        d_x = border_x2 - border_x1
        bottom_y = ((border_x - border_x1)*poly_corners[3, self.Y] + (border_x2 - border_x)*poly_corners[0, self.Y])/d_x
        top_y = ((border_x-border_x1)*poly_corners[2, self.Y] + (border_x2 - border_x)*poly_corners[1, self.Y])/d_x

        mark_y = []
        for i in range(len(border_x)):
            # vertical coverage at each point in border_x
            border_y = np.arange(math.floor(bottom_y[i]), math.ceil(top_y[i])+1, dtype=float)
            border_y[0] = bottom_y[i]
            border_y[-1] = top_y[i]
            mark_y.append(border_y)

        rows, cols = (y2-y1, x2-x1)
        cell_corners = [[list() for _ in range(cols)] for _ in range(rows)]  # corners in each cell starting from x1, y1

        for x_ni in range(np.size(border_x)-1):
            # collect corners & points_at_borders: [<point at border 1>, <point at border 2>]

            y_line1 = mark_y[x_ni]
            y_line2 = mark_y[x_ni+1]
            sy1 = np.where((y_line1 - y_grid[0]) >= 0)[0][0]            # first index in y_line1 covered by y_grid
            y_line1_sy = np.where((y_grid - y_line1[sy1]) <= 0)[0][-1]  # first index from y_grid
            sy2 = np.where((y_line2 - y_grid[0]) >= 0)[0][0]            # first index in y_line2 covered by y_grid
            y_line2_sy = np.where((y_grid - y_line2[sy2]) <= 0)[0][-1]  # first index from y_grid
            ey1 = np.where((y_line1 - y_grid[-1]) < 0)[0][-1]           # last index in y_line1 covered by y_grid
            y_line1_ey = np.where((y_line1[ey1] - y_grid) >= 0)[0][-1]  # last index from y_grid
            ey2 = np.where((y_line2 - y_grid[-1]) < 0)[0][-1]           # last index in y_line2 covered by y_grid
            y_line2_ey = np.where((y_line2[ey2] - y_grid) >= 0)[0][-1]  # last index from y_grid

            min_sy_idx = min(y_line1_sy, y_line2_sy)   # y index on y_grid
            max_ey_idx = max(y_line1_ey, y_line2_ey)

            # collect the intersect points at y position in y_grid & border points in the cell starting from the same y
            v_cell_info = [{'inter_points': list(), 'border_points': [list(), list()]} for _ in y_grid]
            for y_idx in range(min_sy_idx, max_ey_idx+1):
                if min(y_line1[sy1], y_line2[sy2]) < y_grid[y_idx] < max(y_line1[sy1], y_line2[sy2]):
                    x_inter = (abs(y_grid[y_idx] - y_line1[sy1]) * border_x[x_ni+1] +
                               abs(y_line2[sy2] - y_grid[y_idx]) * border_x[x_ni])/abs(y_line1[sy1]-y_line2[sy2])
                    if y_line1[sy1] < y_line2[sy2]:   # line1 is lower
                        v_cell_info[y_idx]['inter_points'] = [border_x[x_ni], x_inter]
                    else:
                        v_cell_info[y_idx]['inter_points'] = [x_inter, border_x[x_ni+1]]
                if min(y_line1[ey1], y_line2[ey2]) < y_grid[y_idx] < max(y_line1[ey1], y_line2[ey2]):
                    x_inter = (abs(y_grid[y_idx] - y_line1[ey1]) * border_x[x_ni+1] +
                               abs(y_line2[ey2] - y_grid[y_idx]) * border_x[x_ni])/abs(y_line1[ey1]-y_line2[ey2])
                    if y_line1[ey1] < y_line2[ey2]:
                        v_cell_info[y_idx]['inter_points'] = [x_inter, border_x[x_ni+1]]
                    else:
                        v_cell_info[y_idx]['inter_points'] = [border_x[x_ni], x_inter]
            if y_line1[sy1] > y_grid[y_line1_sy]:
                v_cell_info[y_line1_sy]['border_points'][0] = [border_x[x_ni], y_line1[sy1]]
            if y_line2[sy2] > y_grid[y_line2_sy]:
                v_cell_info[y_line2_sy]['border_points'][1] = [border_x[x_ni+1], y_line2[sy2]]

            if y_line1[ey1] > y_grid[y_line1_ey]:
                v_cell_info[y_line1_ey]['border_points'][0] = [border_x[x_ni], y_line1[ey1]]
            if y_line2[ey2] > y_grid[y_line2_ey]:
                v_cell_info[y_line2_ey]['border_points'][1] = [border_x[x_ni+1], y_line2[ey2]]

            bottom_p = [[border_x[x_ni], y_line1[sy1]], [border_x[x_ni+1], y_line2[sy2]]]
            if y_line1_sy < y_line2_sy:
                bottom_p[1] = bottom_p[0]
            elif y_line1_sy > y_line2_sy:
                bottom_p[0] = bottom_p[1]

            top_p = [[border_x[x_ni], y_line1[ey1]], [border_x[x_ni+1], y_line2[ey2]]]
            if y_line1_ey > y_line2_ey:
                top_p[1] = top_p[0]
            elif y_line2_ey > y_line1_ey:
                top_p[0] = top_p[1]

            for y_idx in range(min_sy_idx, max_ey_idx):
                corners = list()
                corners.append(bottom_p[0])
                if (len(v_cell_info[y_idx]['border_points'][0]) > 0) and \
                   (v_cell_info[y_idx]['border_points'][0] != bottom_p[0]):
                    corners.append(v_cell_info[y_idx]['border_points'][0])

                y_c = y_grid[y_idx+1]
                if len(v_cell_info[y_idx+1]['inter_points']) > 0:
                    corners.append([v_cell_info[y_idx+1]['inter_points'][0], y_c])
                    corners.append([v_cell_info[y_idx+1]['inter_points'][1], y_c])
                else:
                    corners.extend([[border_x[x_ni], y_c], [border_x[x_ni+1], y_c]])

                next_bottom = [corners[-2], corners[-1]]  # the last two corners just added

                if (len(v_cell_info[y_idx]['border_points'][1]) > 0) and \
                   (v_cell_info[y_idx]['border_points'][1] != corners[-1]):
                    corners.append(v_cell_info[y_idx]['border_points'][1])

                if bottom_p[1] != corners[-1] and bottom_p[1] != corners[0]:
                    corners.append(bottom_p[1])
                bottom_p = next_bottom

                cell_corners[y_idx][x_ni] = corners

            cell_corners[max_ey_idx][x_ni] = [bottom_p[0], top_p[0], top_p[1], bottom_p[1]]

        total_area = 0.0
        flux = 0.0
        for y in range(rows):
            for x in range(cols):
                if len(cell_corners[y][x]) == 0 or input_data[y1+y, x1+x] == 0:
                    continue

                # corners = np.array(cell_corners[y][x])-np.array([x1, y1])
                corners = np.array(cell_corners[y][x])
                area = self.polygon_area(corners)
                total_area += area
                flux += area * input_data[y1+y, x1+x]

        return flux, total_area

    def polygon_clipping(self, poly_points, clipper_points, clipper_size):
        """ Clip a polygon by an area enclosed by a set of straight lines of 2D domain.

        Args:
            poly_points (numpy.ndarray): Corners of polygon in counterclockwise order.
            clipper_points (list): Corners of clipping area in counterclockwise order.
            clipper_size (int): Total sides of the clipping area.

        Returns:
            numpy.ndarray: Corners of the polygon after clipping in counterclockwise order.

        """

        new_poly_points = [[poly_points[i, 0], poly_points[i, 1]] for i in range(clipper_size)]
        for i in range(clipper_size):
            k = (i+1) % clipper_size
            new_poly_points = self.clip(new_poly_points, clipper_points[i][0], clipper_points[i][1],
                                        clipper_points[k][0], clipper_points[k][1])

        new_corners = self.remove_duplicate_point(new_poly_points)
        return np.array(new_corners)

    @staticmethod
    def remove_duplicate_point(corners):
        """ Remove the duplicate points from a list of corner points of a polygon.

        Args:
            corners (list): Corner points of a polygon.

        Returns:
            list: Corner points of the polygon.

        """
        new_corners = []
        for c in corners:
            if c not in new_corners:
                new_corners.append(c)

        return new_corners

    @staticmethod
    def polygon_area(corners):
        """ Calculate the polygon area per polygon corners.

        Args:
            corners (numpy.ndarray): Corners of a polygon.

        Returns:
            float: Area of the polygon.

        """
        polygon_size = np.shape(corners)[0]
        area = 0.0
        for i in range(polygon_size):
            k = (i+1) % polygon_size
            area += corners[i, 0]*corners[k, 1] - corners[k, 0]*corners[i, 1]

        return abs(area)/2

    def clip(self, poly_points, x1, y1, x2, y2):
        """ Clipping a polygon by a vector on 2D domain.

        Some corners of the polygons, `poly_points`, are replaced by the intersection points between the vector and
        the polygon after clipping.

        Args:
            poly_points (list): List of corners of the polygon.
                Each corner is a list containing values for x and y coordinates.
            x1 (int): x of end point 1 of the vector.
            y1 (int): y of end point 1 of the vector.
            x2 (int): x of end point 2 of the vector.
            y2 (int): y of end point 2 of the vector.

        Returns:
            list: Updated corners of the polygon after being clipped by the vector.

        """
        new_points = []
        poly_size = len(poly_points)

        for i in range(poly_size):
            k = (i+1) % poly_size

            ix = poly_points[i][0]
            iy = poly_points[i][1]
            kx = poly_points[k][0]
            ky = poly_points[k][1]

            # position of first point w.r.t. clipper line
            i_pos = (x2 - x1) * (iy - y1) - (y2 - y1) * (ix - x1)
            # position of second point w.r.t. clipper line
            k_pos = (x2 - x1) * (ky - y1) - (y2 - y1) * (kx - x1)

            if i_pos < 0 and k_pos < 0:             # both are inside, take the second
                new_points.append([kx, ky])
            elif i_pos >= 0 and k_pos < 0:          # only the second is inside, take the intersect and the second one
                if i_pos == 0:
                    new_points.append([ix, iy])
                else:
                    intersect_p = self.line_intersect(x1, y1, x2, y2, ix, iy, kx, ky)
                    new_points.append([intersect_p[0], intersect_p[1]])
                new_points.append([kx, ky])
            elif i_pos < 0 and k_pos >= 0:          # onlyt the first inside,  take the intersect
                if k_pos == 0:
                    new_points.append([kx, ky])
                else:
                    intersect_p = self.line_intersect(x1, y1, x2, y2, ix, iy, kx, ky)
                    new_points.append([intersect_p[0], intersect_p[1]])

        return new_points

    @staticmethod
    def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
        """ Find the intersection of two lines on 2D by assuming there is the intersetion between these 2 lines.

        Args:
            x1 (int): x of end point 1 of line 1.
            y1 (int): y of end point 1 of line 1.
            x2 (int): x of end point 2 of line 1.
            y2 (int): y of end point 2 of line 1.
            x3 (int): x of end point 1 of line 2.
            y3 (int): y of end point 1 of line 2.
            x4 (int): x of end point 2 of line 2.
            y4 (int): y of end point 2 of line 2.

        Returns:
            list: Intersect point containing values of x and y coordinates.

        """
        den = (x1-x2)*(y3-y4) - (x3-x4)*(y1-y2)
        num_x = (x1*y2 - x2*y1) * (x3 - x4) - (x1 - x2) * (x3*y4 - x4*y3)
        num_y = (x1*y2 - x2*y1) * (y3 - y4) - (y1 - y2) * (x3*y4 - x4*y3)

        return [num_x/den, num_y/den]

    def write_data_to_dataframe(self, result_data):
        """ Write optimal extraction result to an instance of Pandas DataFrame.

        Args:
            result_data (numpy.ndarray): Optimal extraction result.  Each row of the array corresponds to the reduced
                1D data of one order.

        Returns:
            Pandas.DataFrame: Instance of DataFrame containing the extraction result plus the following attributes:

                - *MJD-OBS*: modified Julian date of the observation.
                - *EXPTIME*: exposure time of the observation.
                - *TOTALORD*: total order in the result data.
                - *DIMWIDTH*: Width of the order in the result data.

        """
        header_keys = list(self.spectrum_header.keys())

        mjd = 58834.102377407
        if 'SSBJD100' in header_keys:
            mjd = self.spectrum_header['SSBJD100'] - 2400000.5
        elif 'OBSJD' in header_keys:
            mjd = self.spectrum_header['OBSJD'] - 2400000.5
        elif 'OBS MJD' in header_keys:
            mjd = self.spectrum_header['OBS MJD']
        exptime = self.spectrum_header['EXPTIME'] if 'EXPTIME' in header_keys else 600.0

        total_order, dim_width = np.shape(result_data)
        # result_table = {'order_'+str(i): result_data[i, :] for i in range(total_order) }
        # df_result = pd.DataFrame(result_table)
        df_result = pd.DataFrame(result_data)
        df_result.attrs['MJD-OBS'] = mjd
        df_result.attrs['OBSJD'] = mjd + 2400000.5
        df_result.attrs['EXPTIME'] = exptime
        df_result.attrs['TOTALORD'] = total_order
        df_result.attrs['DIMWIDTH'] = dim_width

        return df_result

    def time_check(self, t_start, step_msg):
        """Count and display the execution time.

        Args:
            t_start (float): Start time to count.
            step_msg (str): Message to print.

        Returns:
            float: End of time.

        """
        t_end = time.time()
        self.t_print(step_msg, (t_end - t_start), 'sec.')
        return t_end

    def add_file_logger(self, filename=None):
        """ Add file to log debug information.

        Args:
            filename (str, optional): Filename of the log file. Defaults to None.

        Returns:
            None.

        """
        self.enable_debug_print(filename is not None)
        self.debug_output = filename

    def extract_spectrum(self, rectification_method=NoRECT, extraction_method=OPTIMAL,
                         order_set=None,
                         show_time=False,
                         print_debug=None,
                         bleeding_file=None):
        """ Optimal extraction from 2D flux to 1D. Rectification step is optional.

        Args:
            rectification_method (int): There are three methods used to collect pixels from orders of spectrum data
                and flat dta for optimal extraction,

                - OptimalExtractionAlg.NoRECT: Pixels at the north-up direction along the order are collected.
                  No rectification. (the fastest computation).
                - OptimalExtractionAlg.VERTICAL: Pixels at the north-up direction along the order are collected
                  to be rectified.
                - OptimalExtractionAlg.NORMAL: Pixels at the normal direction of the order are collected to
                  be rectified.
            extraction_method (str, optional): There are two extraction methods performing extraction on collected
                flux along the order. Defaults to OPTIMAL.

                - OptimalExtractionAlg.OPTIMAL (i.e. 'optimal'): for optimal extraction.
                - OptimalExtractionAlg.SUM (i.e. 'sum'): for summation extraction.

            order_set (numpy.ndarray, optional): Set of orders to extract. Defaults to None for all orders.
            show_time (bool, optional):  Show running time of the steps. Defaults to False.
            print_debug (str, optional): Print debug information to stdout if it is provided as empty string,
                a file with path `print_debug` if it is non empty string, or no print if it is None.
                Defaults to None.
            bleeding_file (str, optioanl): Bleeding cure file, such as that for PARAS data. Defaults to None.

        Returns:
            dict: Optimal extraction result from 2D spectrum data, like::

                    {
                        'optimal_extraction_result':  Padas.DataFrame
                                    # table storing optimal extraction result.
                                    # each row of the table containing the optimal extraction
                                    # result for one order.
                    }

        """

        self.add_file_logger(print_debug)
        self.enable_time_profile(show_time)
        self.update_spectrum_flux(bleeding_file)

        dim_width, dim_height = self.get_spectrum_size()
        total_order = self.get_spectrum_order()
        if order_set is None:
            if self.instrument and self.get_instrument().upper() == 'NEID':
                order_set = np.arange(0, total_order, 2, dtype=int)
            else:
                order_set = np.arange(0, total_order, dtype=int)

        out_data = np.zeros((order_set.size, dim_width))

        self.d_print("do ", self.rectifying_method[rectification_method], extraction_method,
                     ' on ', order_set.size, ' orders', info=True)

        t_start = time.time()
        for idx_out in range(order_set.size):
            c_order = order_set[idx_out]
            self.d_print(c_order, ' edges: ', self.get_order_edges(c_order),
                         ' xrange: ', self.get_order_xrange(c_order), end=" ", info=True)
            order_flux = self.get_flux_from_order(self.order_coeffs[c_order], self.get_order_edges(c_order),
                                                  self.get_order_xrange(c_order), self.spectrum_flux, self.flat_flux,
                                                  norm_direction=rectification_method)
            result = dict()
            if 'optimal' in extraction_method:
                result = self.optimal_extraction(order_flux.get('order_data'), order_flux.get('order_flat'),
                                                 order_flux.get('data_height'), order_flux.get('data_width'))
            elif 'sum' in extraction_method:
                result = self.summation_extraction(order_flux.get('order_data'))
            if 'extraction' in result:
                self.fill_2d_with_data(result.get('extraction'), out_data, idx_out)

            t_start = self.time_check(t_start, '**** time ['+str(c_order)+']: ')
        self.d_print(" ")
        data_df = self.write_data_to_dataframe(out_data)
        return {'optimal_extraction_result': data_df}

    @staticmethod
    def result_test(target_data, data_result):
        t_y, t_x = np.shape(target_data)
        r_y, r_x = np.shape(data_result)

        if t_y != r_y or t_x != r_x:
            return {'result': 'error', 'msg': 'dimension is not the same'}

        not_nan_data_idx = np.argwhere(~np.isnan(data_result))
        not_nan_target_idx = np.argwhere(~np.isnan(target_data))

        if np.size(not_nan_data_idx) != np.size(not_nan_target_idx):
            return {'result': 'error', 'msg': 'NaN data different'}
        elif np.size(not_nan_data_idx) != 0:
            if not (np.array_equal(not_nan_data_idx, not_nan_target_idx)):
                return {'result': 'error', 'msg': 'NaN data different'}
            else:
                not_nan_target = target_data[~np.isnan(target_data)]
                not_nan_data = data_result[~np.isnan(data_result)]
                diff_idx = np.where(not_nan_target - not_nan_data)[0]

                if diff_idx.size > 0:
                    return {'result': 'error', 'msg': 'data is not the same at ' + str(diff_idx.size) + ' points'}

        return {'result': 'ok'}

    @staticmethod
    def update_wavecal_from_existing_L1(fiber, wave_key, L1_wave_data, header, data_obj, header_obj, wave_start_order = 0):
        wave_end_order = min(np.shape(data_obj[fiber][1, :, :])[0] + wave_start_order,
                             np.shape(L1_wave_data[fiber][1, :, :])[0])
        data_obj[fiber][1, :, :] = L1_wave_data[fiber][1, wave_start_order:wave_end_order, :]
        header_obj[wave_key] = header[wave_key]
        header_obj[fiber+'_FLUX']['SSBZ100'] = header['PRIMARY']['SSBZ100']
        header_obj[fiber+'_FLUX']['SSBJD100'] = header['PRIMARY']['SSBJD100']

