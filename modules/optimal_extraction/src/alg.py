import configparser
import numpy as np
import math
import time
import pandas as pd
from astropy.io import fits
import re
from modules.Utils.config_parser import ConfigHandler

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
        config_param (ConfigHandler): Related to 'PARAM' section or the section associated with
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
    HORIZONTAL = 0
    NORMAL = 0
    VERTICAL = 1
    NoRECT = 2
    OPTIMAL = 'optimal'
    SUM = 'sum'
    V_UP = 0
    H_RIGHT = 1
    V_DOWN = 2
    H_LEFT = 3
    V1 = 'v_1'
    V2 = 'v_2'

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
            raise TypeError('type: ' + str(type(order_trace_header)) +
                            ' flux header type error, cannot construct object from OptionalExtractionAlg')

        self.logger = logger
        self.flat_flux = flat_data
        self.spectrum_flux = spectrum_data
        self.spectrum_header = spectrum_header
        rows, cols = np.shape(self.flat_flux)
        self.dim_width = cols
        self.dim_height = rows

        self.poly_order = order_trace_header['POLY_DEG'] if 'POLY_DEG' in order_trace_header else 3

        # origin of the image
        self.origin = [ order_trace_header['STARTCOL'] if 'STARTCOL' in order_trace_header else 0,
                        order_trace_header['STARTROW'] if 'STARTROW' in order_trace_header else 0]

        config_h = ConfigHandler(config, 'PARAM')
        self.instrument = config_h.get_config_value('instrument', '')
        ins = self.instrument.upper()
        # section of instrument or 'PARAM'
        self.config_param = ConfigHandler(config, ins, config_h)
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
        self.total_image_orderlettes = None
        self.orderlette_names = None

    def get_config_value(self, prop, default=''):
        """ Get defined value from the config file.

        Search the value of the specified property fom config section. The default value is returned if not found.

        Args:
            prop (str): Name of the parameter to be searched.
            default (Union[int, float, str], optional): Default value for the searched parameter.

        Returns:
            Union[int, float, str]: Value for the searched parameter.

        """
        return self.config_param.get_config_value(prop, default)

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
        """ Print out message with time stamp.

        Args:
            *args: Variable length argument list to print including the time stamp.

        Returns:
            This function handles the time stamp related print-out to the logger defined in the config file.
        """
        """
        if self.is_time_profile:
            out_str = ' '.join([str(item) for item in args])
            print(out_str)
        """
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
        x_o = self.origin[self.X]
        x_step = x_step[np.where(np.logical_and(x_step >= (xrange[0]+x_o), x_step <= (xrange[1]+x_o)))[0]]
        x_output_step = self.get_output_pos(x_step, sampling_rate[self.X]).astype(int)

        y_mid = np.polyval(coeffs, x_step-x_o) + self.origin[self.Y]  # y position of spectral trace
        v_border = np.array([np.amax(y_mid), np.amin(y_mid)])
        # the vertical position to locate the order in output domain
        y_output_mid = math.floor(np.mean(v_border)*sampling_rate[self.Y])    # a number, output y center
        # self.d_print('y_output_mid: ', y_output_mid)

        output_widths = self.get_output_pos(widths, sampling_rate[self.Y]).astype(int)  # width of output
        # output_widths = np.around(self.get_output_pos(widths, sampling_rate[self.Y])).astype(int)
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
                    out_data[i][y_input_idx, o_x] = data_group[i][y_input, x_i]

        # out data starting from origin [0, 0] contains the reduced flux associated with the data range
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
        x_o = self.origin[self.X]
        x_step = x_step[np.where(np.logical_and(x_step >= (xrange[0]+x_o), x_step <= (xrange[1]+x_o+1)))[0]]
        x_output_step = self.get_output_pos(x_step, sampling_rate[self.X]).astype(int)

        y_mid = np.polyval(coeffs, x_step-x_o) + self.origin[self.Y]      # spectral trace value at mid point

        # y_output_step = np.arange(0, output_y_dim+1, dtype=int)
        # y_step = self.get_input_pos(y_output_step, sampling_rate[self.Y])  # y step in input domain

        if direction == self.NORMAL:
            # curve norm along x in input domain
            y_norm_step = self.poly_normal(x_step-x_o, coeffs, sampling_rate[self.Y])
        else:
            # vertical norm along x in input domain
            y_norm_step = self.vertical_normal(x_step-x_o, sampling_rate[self.Y])

        v_border = np.array([np.amax(y_mid), np.amin(y_mid)])
        # self.d_print('v_border: ', v_border)
        # the vertical position to locate the order in output domain
        y_output_mid = math.floor(np.mean(v_border)*sampling_rate[self.Y])
        # self.d_print('y_output_mid: ', y_output_mid)

        output_widths = self.get_output_pos(widths, sampling_rate[self.Y]).astype(int)  # width of output
        upper_width = min(output_widths[1], output_y_dim - 1 - y_output_mid)
        lower_width = min(output_widths[0], y_output_mid)
        self.d_print('rectify: width at output: ', upper_width, lower_width)

        corners_at_mid = np.vstack((x_step, y_mid)).T     # for x and y in data range, relative to 2D origin [0, 0]
        # self.d_print('corners_at_mid: ', corners_at_mid)

        y_size = 1 if sum_extraction is True else (upper_width+lower_width)
        total_data_group = len(data_group)
        out_data = [np.zeros((y_size, output_x_dim)) for _ in range(0, total_data_group)]

        all_input_corners = np.zeros((upper_width + lower_width + 1, x_step.size, 2))    # for x & y
        all_input_corners[lower_width] = corners_at_mid.copy()

        for o_y in range(1, upper_width+1):
            next_upper_corners = self.go_vertical(all_input_corners[lower_width+o_y-1], y_norm_step, 1)
            all_input_corners[lower_width+o_y] = next_upper_corners

        for o_y in range(1, lower_width+1):
            next_lower_corners = self.go_vertical(all_input_corners[lower_width-o_y+1], y_norm_step, -1)
            all_input_corners[lower_width-o_y] = next_lower_corners

        v2_borders = None

        upper_pixels = list(range(lower_width, upper_width + lower_width))
        lower_pixels = list(range(lower_width - 1, -1, -1))
        for i, o_x in enumerate(x_output_step[0:-1]):    # for x output associated with the data range
            # if i % 100 == 0:
            #    print(i, end=" ")
             if direction == self.NORMAL or direction == self.VERTICAL:
                if i == 0:
                    v1_borders = self.collect_v_borders(all_input_corners, i)
                else:
                    v1_borders = v2_borders

                v2_borders = self.collect_v_borders(all_input_corners, i+1)
                h_borders = self.collect_h_borders(v1_borders, v2_borders)

                # each border: vertex_1, vertex_2, direction, intersect_with_borders={direction, pos, loc}
                for pixel_list in [upper_pixels, lower_pixels]:
                    for o_y in pixel_list:
                        borders = [
                                v1_borders[o_y].copy(), h_borders[o_y+1].copy(),
                                v2_borders[o_y].copy(), h_borders[o_y].copy()
                            ]

                        # adjust v1 and v2 in clockwise direction
                        for n in [self.V_DOWN, self.H_LEFT]:
                            borders[n][self.V1],  borders[n][self.V2] = borders[n][self.V2], borders[n][self.V1]
                        flux = self.compute_flux_for_output_pixel(borders, data_group, total_data_group)
                        for n in range(0, total_data_group):
                            if sum_extraction is True:
                                out_data[n][0, o_x] += flux[n]
                            else:
                                out_data[n][o_y, o_x] = flux[n]

        result_data = {'y_center': y_output_mid,
                       'width': [upper_width, lower_width],
                       'dim': [output_y_dim, output_x_dim],
                       'out_data': [out_data[i] for i in range(0, total_data_group)]}
        return result_data

    def intersect_cell_borders(self, vertex_1, vertex_2):
        """Find out the intersection of a line segment with the horizontal and vertical grid lines.

        Args:
            vertex_1 (list): End point 1 of the line segment.
            vertex_2 (list): Eng point 2 of the line segment.

        Returns:
            dict: A dict instance containing the intersect information, like::

                {
                    'min_cell_x': int
                        # min x of the cell coverage of the line segment.
                    'max_cell_x': int
                        # max x of the cell coverage of the line segment.
                    'min_cell_y': int
                        # min y of the cell coverage of the line segment.
                    'max_cell_y': int
                        # max y of the cell coverage of the line segment.
                    'v_borders': dict
                        # collection of intersection with vertical grid lines, like:
                        {
                            <grid_line_x_position>:
                            {
                                'direction': VERTICAL,
                                'pos': float,
                                      # intersected position along y axis.
                                'axis_idx': self.X
                                      # coordinate index in 'loc' of the grid line.
                                'loc': list,
                                      # cell location of the intersected position.
                            }
                            :
                        }

                    'h_borders': dict
                        # collection of intersection with horizontal grid lines, like:
                         {
                            <grid_line_y_position>:
                            {
                                'direction': HORIZONTAL,
                                'pos': float,
                                        # intersected position along x axis.
                                'axis_idx': self.Y
                                        # coordinate index in 'loc' of the grid line.
                                'loc': list,
                                        # cell location of the intersected position.
                            }
                            :
                        }
                }

        """
        x_ends = [vertex_1[self.X], vertex_2[self.X]]
        y_ends = [vertex_1[self.Y], vertex_2[self.Y]]

        x_min = min(x_ends[0], x_ends[1])
        x_max = max(x_ends[0], x_ends[1])
        y_min = min(y_ends[0], y_ends[1])
        y_max = max(y_ends[0], y_ends[1])
        y_dim, x_dim = np.shape(self.spectrum_flux)
        min_cell_x = max(0, math.floor(x_min))
        max_cell_x = min(x_dim, math.ceil(x_max))
        min_cell_y = max(0, math.floor(y_min))
        max_cell_y = min(y_dim, math.ceil(y_max))

        h_borders = dict()
        v_borders = dict()

        # def make_border(direct, pos, pos_cells):
        # pos could be fractional, loc falls on the position increment by one
        def make_border(direction, loc, pos):
            if direction == self.VERTICAL:
                loc_x = loc
                loc_y = math.floor(pos)
                axis_idx = self.X
            else:
                loc_y = loc
                loc_x = math.floor(pos)
                axis_idx = self.Y

            new_border = {'direction': direction, 'pos': pos, 'axis_idx': axis_idx,  'loc': [loc_x, loc_y]}
            return new_border

        def intersect_xline(x1, x3, y3, x4, y4):
            if y3 == y4:        # vertial x line interact with horozontal line
                return y3
            if x1 == x3:        # one end meets the vertical line
                return y3
            if x1 == x4:
                return y4

            c_xy = x3 * y4 - x4 * y3
            den = (x3 - x4)
            num_y = x1 * (y3 - y4) + c_xy      # x1*(x3-x4) + x3*y4-x4*y3
            return num_y/den

        def intersect_yline(y1, x3, y3, x4, y4):
            if x3 == x4:       # horizontal y line interacts with vertical line
                return x3
            if y1 == y3:
                return x3
            if y1 == y4:
                return x4

            c_xy = x3 * y4 - x4 * y3
            den = y3 - y4
            num_x = y1 * (x3 - x4) - c_xy       # y1*(x4-x3)+x3*y4-x4*y3
            return num_x/den

        # end_xy = x_ends[0]*y_ends[1]-x_ends[1]*y_ends[0]   # x3*y4-x4*y3
        # find intersection with vertical cell border (vertical cell line)
        for c_x in range(min_cell_x, max_cell_x+1):
            if x_min <= c_x <= x_max:
                if x_ends[0] != x_ends[1]:          # not a vertical line overlapping cell border
                    y_p = intersect_xline(c_x, x_ends[0], y_ends[0], x_ends[1], y_ends[1])

                    i_border = make_border(self.VERTICAL, c_x, y_p)
                    v_borders[c_x] = i_border

        # find intersection with horizontal cell border (horizontal cell line)
        for c_y in range(min_cell_y, max_cell_y+1):
            if y_min <= c_y <= y_max:
                if y_ends[0] != y_ends[1]:          # not a horizontal line overlapping cell border
                    x_p = intersect_yline(c_y, x_ends[0], y_ends[0], x_ends[1], y_ends[1])

                    i_border = make_border(self.HORIZONTAL, c_y, x_p)
                    h_borders[c_y] = i_border

        intersect_borders = {
                    'orig_v1': vertex_1,
                    'orig_v2': vertex_2,
                    'min_cell_x': min_cell_x,
                    'max_cell_x': max_cell_x,
                    'min_cell_y': min_cell_y,
                    'max_cell_y': max_cell_y,
                    'v_borders': v_borders,
                    'h_borders': h_borders
            }
        return intersect_borders

    def collect_v_borders(self, corners: np.ndarray, start_idx: int):
        """ Collect vertical borders of output pixels.

        Find vertical borders of the rectified cells along either normal or vertical direction
        from the order trace.

        Args:
            corners (list): A set of corners of rectified cells along the normal or vertical
                direction of the order.
            start_idx (int): Starting index of corners of the polygon representing the rectified
                output cell.

        Returns:
            list: Collection of all vertical borders along either normal or vertical direction
            from an order trace, like::

                [
                    {
                        'v1': list   # end point 1 of a border
                        'v2': list   # end point 2 of a border
                        'direction': VERTICAL
                        'intersect_with_borders': dict
                                     # grid line intersection of the border,
                                     # see 'intersect_cell_borders' for the detail
                    }
                    :
                ]

        """

        all_borders = list()
        total_borders = np.shape(corners)[0] - 1

        for idx in range(total_borders):
            vertex_1 = [corners[idx, start_idx, self.X], corners[idx, start_idx, self.Y]]
            vertex_2 = [corners[idx+1, start_idx, self.X], corners[idx+1, start_idx, self.Y]]

            v1 = self.V1
            v2 = self.V2

            border = {
                    v1: vertex_1,
                    v2: vertex_2,
                    'direction': self.VERTICAL,
                    'intersect_with_borders': self.intersect_cell_borders(vertex_1, vertex_2)
            }

            all_borders.append(border)

        return all_borders

    def collect_h_borders(self, v1_borders, v2_borders):
        """Collect horizontal borders of the output pixels.

        Find horizontal borders of the rectified cells along either normal or vertical direction from the order trace.

        Args:
            v1_borders: Left side of vertical borders of the rectified cells collected by
                        :func:`~alg.OptimalExtractionAlg.collect_v_borders()`
            v2_borders: Right side of vertical borders of the rectified cells collected by
                        :func:`~alg.OptimalExtractionAlg.collect_v_borders()`

        Returns:
            list: Collection of all horizontal borders along either normal or vertical direction
            from an order trace, like::

                [
                    {
                        'v1': list,   # end point 1 of a border
                        'v2': list,   # end point 2 of a border
                        'direction': HORIZONTAL,
                        'intersect_with_borders': dict,
                                      # grid line intersection of the border,
                                      # see 'intersect_cell_borders' for the detail
                    }
                    :
                ]

        """
        all_borders = list()
        total_borders = len(v1_borders) + 1
        v1 = self.V1
        v2 = self.V2

        for idx in range(total_borders):
            if idx < (total_borders - 1):
                vertex_1 = v1_borders[idx][v1]
                vertex_2 = v2_borders[idx][v1]
            else:
                vertex_1 = v1_borders[idx-1][v2]
                vertex_2 = v2_borders[idx-1][v2]

            border = {
                    v1: vertex_1,
                    v2: vertex_2,
                    'direction': self.HORIZONTAL,
                    'intersect_with_borders': self.intersect_cell_borders(vertex_1, vertex_2)
            }
            all_borders.append(border)
        return all_borders

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
            x_range (numpy.ndarray): Horizontal coverage of the order in terms of
                two ends along x axis.
            in_data (numpy.ndarray): 2D spectral data.
            flat_flux (numpy.ndarray): 2D flat data.
            s_rate (Union[list, float], optional): sampling rate from input domain to output domain for 2D data.
                Defaults to 1.
            norm_direction(int, optional): Rectification method. Defaults to None.

                - None: no rectification.
                - VERTICAL: pixels at the north and south direction along the order are
                  collected to be rectified.
                - NORMAL: pixels at the normal direction of the order are collected to be rectified.

        Returns:
            dict: Information related to the order data, like::

                {
                    'order_data': numpy.ndarray
                        # extracted spectrum data from the order using rectification # or not.
                    'order_flat': numpy.ndarray
                        # extracted flat data from the order using rectification or not.
                    'data_height': int        # height of 'order_data'.
                    'data_width': int         # width of 'order_data'.
                    'out_y_center': int
                        # y center position where 'order_data' is located.
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
        out_data = np.sum(s_data, axis=0).reshape(1, -1)

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

    def compute_flux_for_output_pixel(self, borders, input_data, total_data_group):
        """ Compute weighted flux covered by one output pixel.

            Compute the weighted flux per spectrum data and flat data covered for the area of one output pixel.
            The area is represented by polygon borders and the polygon is collected in either normal direction or
            vertical direction to the order. The weight depends on the area overlap between the output pixel and the
            input pixel.

            Args:
                borders(list): Borders of the coverage of one output pxiel.
                input_data(list): Input data group. each element is numpy.ndarray.
                total_data_group(int): total data group

            Returns:
                list : Flux value on observation data and flat data for the area covered by one output pixel.

        """
        x_1 = min([border['intersect_with_borders']['min_cell_x'] for border in borders])
        x_2 = max([border['intersect_with_borders']['max_cell_x'] for border in borders])
        y_1 = min([border['intersect_with_borders']['min_cell_y'] for border in borders])
        y_2 = max([border['intersect_with_borders']['max_cell_y'] for border in borders])

        flux_polygon, total_area_polygon = self.compute_flux_from_polygon_clipping2(borders,
                                                                [x_1, x_2, y_1, y_2], input_data, total_data_group)
        return flux_polygon

    def compute_flux_from_polygon_clipping2(self, borders, clipper_borders, input_data, total_data_group):
        """ Compute flux on pixels covered by one polygon formed in the normal or vertical direction of the order.

        Collect input pixels covered by the output pixel and compute weighted summation on the input pixels.

        Args:
            borders(list): Borders of the coverage of one output pxiel.
            clipper_borders (list): Rectangular area covered by the clipper boundaries.
            input_data (list): Imaging data - list of numpy.ndarray containing spectrum data and flat data.
            total_data_group (int): total data group.

        Returns:
            tuple: Weighted summation of flux for spectrum data and flat data over the area covered by
            the output pixel.

            * **flux** (*list*): Weighted summation of the flux for spectrum and flat data.
            * **total_area** (*float*): Total overlapping area between the collected pixels and the output pixel.
        """
        x_1, x_2, y_1, y_2 = clipper_borders
        total_area = 0.0
        flux = np.zeros(total_data_group)

        # print('poly borders: ', borders)
        # print('clipper: ', x_1, x_2, y_1, y_2)
        for x in range(x_1, x_2):
            for y in range(y_1, y_2):
                # stime = time.time()
                # if len([d_group[y, x] for d_group in input_data if d_group[y, x] != 0.0]) > 0:
                new_corners = self.polygon_clipping2(borders, [[x, y], [x, y+1], [x+1, y+1], [x+1, y]], 4)
                area = self.polygon_area(new_corners)
                total_area += area
                for n in range(total_data_group):
                    if input_data[n][y, x] != 0.0:
                        flux[n] += area * input_data[n][y, x]
                    # print('x, y = ', x, y, ' area = ', area)
                # self.time_check(stime, 'time for one clipper pixel ')
        # print('total_area = ', total_area, ' flux=', flux)
        new_flux = flux/total_area
        return new_flux, total_area

    def polygon_clipping2(self, borders, clipper_points, clipper_size):
        """ Clip a polygon by an area enclosed by a set of straight lines of 2D domain.

        Args:
            borders(list): Borders of the coverage of one output pixel in counterclockwise order.
            clipper_points (list): Corners of clipping area in counterclockwise order.
            clipper_size (int): Total sides of the clipping area.

        Returns:
            numpy.ndarray: Corners of the polygon after clipping in counterclockwise order.

        """
        new_poly_borders = [b.copy() for b in borders]

        # print('\nclipper_points: ', clipper_points)
        # do clipping on each side of the clipper
        for i in range(clipper_size):
            k = (i+1) % clipper_size
            # print('\ni = ', i )
            new_poly_borders = self.clip2(new_poly_borders, clipper_points[i][0], clipper_points[i][1],
                                          clipper_points[k][0], clipper_points[k][1], i)
            # for b in new_poly_borders:
            #    print(b)

        new_corners = self.remove_duplicate_point2(new_poly_borders)
        # print('\n new corners: ', new_corners)
        return np.array(new_corners)

    @staticmethod
    def remove_duplicate_point2(borders):
        """ Remove the duplicate points from a list of corner points of a polygon.

        Args:
            borders (list): Borders of a polygon.

        Returns:
            list: Corner points of the polygon.

        """
        new_corners = []
        for b in borders:
            v1 = b[OptimalExtractionAlg.V1]
            v2 = b[OptimalExtractionAlg.V2]
            if v1 not in new_corners:
                new_corners.append(v1)
            if v2 not in new_corners:
                new_corners.append(v2)

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

    def clip2(self, poly_borders, x1, y1, x2, y2, clipper_direction):
        """ Clipping a polygon by a vector on 2D domain.

        The end points in the borders of the polygons, `poly_borders`, may be replaced by the intersected points
        between the clipping vector and the polygon borders.

        Args:
            poly_borders (list): List of borders of the polygon.
                Each border is a dict instance as described in :func:`~alg.OptimalExtractionAlg.collect_v_borders()`
                or :func:`~alg.OptimalExtractionAlg.collect_h_borders()`.
            x1 (int): x of end point 1 of the vector.
            y1 (int): y of end point 1 of the vector.
            x2 (int): x of end point 2 of the vector.
            y2 (int): y of end point 2 of the vector.
            clipper_direction (int): Clipping vector direction.

        Returns:
            list: Updated borders of the polygon after being clipped by the vector.

        """
        poly_size = len(poly_borders)
        new_poly_borders = list()

        def set_pos(border_pos, point_pos, is_bigger_inside=True):
            p_pos = 0
            if point_pos > border_pos:
                p_pos = -1 if is_bigger_inside else 1
            elif point_pos < border_pos:
                p_pos = 1 if is_bigger_inside else -1
            return p_pos

        # pick the reserved one if two intersection points are found
        def get_intersect_point(border_set, clipper_at, clipper_axis):
            p_axis = self.X if clipper_axis == self.Y else self.Y
            if clipper_at in border_set:
                res_point = [0.0, 0.0]
                res_point[p_axis] = border_set[clipper_at]['pos']
                res_point[clipper_axis] = clipper_at
                return res_point
            else:
                return None

        v1 = self.V1
        v2 = self.V2

        for i in range(poly_size):
            p_border = poly_borders[i]

            i_idx = v1
            k_idx = v2
            ix, iy = p_border[i_idx][0], p_border[i_idx][1]
            kx, ky = p_border[k_idx][0], p_border[k_idx][1]

            intersect_borders = p_border['intersect_with_borders']
            v_borders = intersect_borders['v_borders']
            h_borders = intersect_borders['h_borders']

            if clipper_direction == self.V_UP or clipper_direction == self.V_DOWN:
                i_pos = set_pos(x1, ix, clipper_direction == self.V_UP)
                k_pos = set_pos(x1, kx, clipper_direction == self.V_UP)
            else:
                i_pos = set_pos(y1, iy, clipper_direction == self.H_LEFT)
                k_pos = set_pos(y1, ky, clipper_direction == self.H_LEFT)

            if i_pos <= 0 and k_pos <= 0:                 # from inside|border to inside|border, keep the border
                new_poly_borders.append(p_border)
            # from outside to inside or inside to outside, replace outside with clipped pt.
            elif (i_pos > 0 and k_pos < 0) or (i_pos < 0 and k_pos > 0):
                if x1 == x2:
                    i_point = get_intersect_point(v_borders, x1, self.X)
                else:           # y1 == y2
                    i_point = get_intersect_point(h_borders, y1, self.Y)

                if i_point is not None:
                    if i_pos > 0:
                        p_border[i_idx] = i_point      # clipping point
                    else:
                        p_border[k_idx] = i_point
                    new_poly_borders.append(p_border)
                else:
                    print("no intersect found ", x1, x2, y1, y2)

        new_poly_size = len(new_poly_borders)
        ret_poly_borders = list()

        for i in range(new_poly_size):
            k = (i+1) % new_poly_size
            p1 = new_poly_borders[i][v2]
            p2 = new_poly_borders[k][v1]

            ret_poly_borders.append(new_poly_borders[i])

            if p1 != p2:
                intersect_borders = self.intersect_cell_borders(p1, p2)
                a_border = {
                        v1: p1,
                        v2: p2,
                        'direction': -1,
                        'intersect_with_borders': intersect_borders
                }
                ret_poly_borders.append(a_border)

        return ret_poly_borders

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
        df_result.attrs['ORDEROFF'] = self.start_row_index()
        df_result.attrs['DIMWIDTH'] = dim_width
        df_result.attrs['FROMIMGX'] = self.origin[self.X]
        df_result.attrs['FROMIMGY'] = self.origin[self.Y]

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

    def get_total_orderlettes_from_image(self):
        """ Get total orderlettes from level 0 image, defined in config

        Returns:
            int: total orderdelettes.
        """
        if self.total_image_orderlettes is None:
            self.total_image_orderlettes = self.get_config_value("total_image_orderlettes", 1)

        return self.total_image_orderlettes

    def get_orderlette_names(self):
        """ Get Orderlette names defined in config.

        Returns:
            list: list of orderlette names
        """
        if self.orderlette_names is None:
            o_names_str = self.get_config_value('orderlette_names')
            order_names = list()
            if o_names_str is not None:
                o_names = o_names_str.strip('][ ').split(',')
                for o_nm in o_names:
                    order_names.append(o_nm.strip("' "))

                if len(order_names) == 0:
                    order_names.append('SCI1')
                self.orderlette_names = order_names

        return self.orderlette_names

    def get_orderlette_index(self, order_name:str):
        """ Find the index of the order name in the orderlette name list.

        Args:
            order_name (str): Fiber name

        Returns:
            int: index of order name in the orderlette name list. If not existing, return is 0.

        """
        all_names = self.get_orderlette_names()
        order_name_idx = all_names.index(order_name) if order_name in all_names else 0

        return order_name_idx

    def start_row_index(self):
        """ The row index for the flux of the first oder in the output.

        Returns:
            int: the row index.
        """
        return self.get_config_value('start_order', 0)

    def get_order_set(self, order_name=''):
        """ Get the list of the trace index eligible for optimal extraction process.

        Args:
            order_name (str): Fiber name.

        Returns:
            list: list of the trace index.

        """
        orderlette_index = self.get_orderlette_index(order_name)
        traces_per_order = self.get_total_orderlettes_from_image()

        if orderlette_index < traces_per_order:
            o_set = np.arange(orderlette_index, self.total_order, traces_per_order, dtype=int)
        else:
            o_set = np.array([])

        return o_set

    def extract_spectrum(self, rectification_method=NoRECT, extraction_method=OPTIMAL,
                         order_set=None,
                         order_name=None,
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

        if order_set is None:
            order_set = self.get_order_set(order_name)

        start_row_at = self.start_row_index()
        order_data_size = order_set.size + start_row_at if order_set.size > 0 else 0
        out_data = np.zeros((order_data_size, dim_width))

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
                self.fill_2d_with_data(result.get('extraction'), out_data, idx_out+start_row_at)
            t_start = self.time_check(t_start, '**** time ['+str(c_order)+']: ')
        self.d_print(" ")
        data_df = self.write_data_to_dataframe(out_data)
        return {'optimal_extraction_result': data_df}

