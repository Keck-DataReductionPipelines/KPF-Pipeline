import configparser
import numpy as np
import json
from scipy import linalg, ndimage
import math
from astropy.modeling import models, fitting
import csv
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
    This module defines class 'OptimalExtractionAlg' and methods to perform the optimal extraction from 2D spectrum to
    1D spectrum. Prior to the step of optimal extraction, the order trace is either rectified or not. It the order is
    rectified, the pixels in either vertical direction or normal direction along the trace are collected and rectified
    column by column.
    """

    X = 0
    Y = 1
    NORMAL = 0
    VERTICAL = 1
    NoRECT = 2
    def __init__(self, flat_data, spectrum_data, config=None):
        self.flat_flux = flat_data.data
        self.spectrum_flux = spectrum_data.data
        self.original_spectrum_flux = self.spectrum_flux.copy()
        self.spectrum_header = spectrum_data.header['PRIMARY']

        rows, cols = np.shape(self.flat_flux)
        self.dim_width = cols
        self.dim_height = rows

        self.poly_order = flat_data.header['ORDERTRACE']['POLY_ORD']
        self.config_param = config['PARAM'] if (config is not None and config.has_section('PARAM')) else None
        self.config_debug = config['DEBUG'] if (config is not None and config.has_section('DEBUG')) else None
        self.order_trace = flat_data.order_trace_result

        order_trace = flat_data.order_trace_result
        self.total_order = np.shape(order_trace)[0]
        self.order_coeffs = np.flip(self.order_trace[:, 0:self.poly_order+1], axis=1)
        self.order_edges = None
        self.order_xrange = None

        self.is_debug = False if self.config_debug is None else self.config_debug.getboolean('debug', False)
        self.debug_output = '' if self.config_debug is None else self.config_debug.get('debug_path', '')

    def get_config_value(self, property: str, default=''):
        """
        get defined value from the config file
        """
        if self.config_param is not None:
            if isinstance(default, int):
                return self.config_param.getint(property, default)
            elif isinstance(default, float):
                return self.config_param.getfloat(property, default)
            else:
                return self.config_param.get(property, default)
        else:
            return default

    def get_order_edges(self, idx: int = 0):
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

    def get_order_xrange(self, idx: int = 0):
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
        return self.dim_width, self.dim_height

    def get_spectrum_order(self):
        return self.total_order

    def enable_debug_print(self, to_print=True):
        """
        enable or disable debug printing
        """
        self.is_debug = to_print

    def redirect_debug_output(self, direct_file=''):
        if direct_file == 'no':
            self.enable_debug_print(False)
        else:
            self.enable_debug_print(True)
            self.debug_output = direct_file

    def d_print(self, *args, end='\n'):
        if self.is_debug:
            if self.debug_output:
                with open(self.debug_output, 'a') as f:
                    f.write(' '.join([str(item) for item in args])+end)
                    f.close()
            else:
                print(' '.join([str(item) for item in args]), end=end)

    def update_spectrum_flux(self, bleeding_cure_file: str=None):
        """
        update the spectrum flux per specified bleeding cure file or  'nan_pixels' set in config file
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
                idx_groups = [ pixel_groups[0][res[i]+1:res[i+1]] for i in range(len(res)-1)]
                for group in idx_groups:
                    idx_set = re.findall("^\\[([0-9]*):([0-9]*),([0-9]*):([0-9]*)\\]$", group)
                    if len(idx_set) > 0 and len(idx_set[0]) == 4:
                        y_idx, x_idx = idx_set[0][0:2], idx_set[0][2:4]
                        r_idx = [int(y_idx[i]) if y_idx[i] else (0 if i == 0 else self.dim_height) for i in range(2)]
                        c_idx = [int(x_idx[i]) if x_idx[i] else (0 if i == 0 else self.dim_width) for i in range(2)]
                        self.spectrum_flux[r_idx[0]:r_idx[1], c_idx[0]:c_idx[1]] = np.nan

    def collect_data_from_order(self, coeffs: np.ndarray, widths: np.ndarray, xrange: np.ndarray,
                                data_group: list, s_rate=[1, 1], sum_extraction=True):
        """
        collect the spectral data per order by polynomial fit
        Parameters:
            coeffs (array): polynomial coefficients starting from zero order
            widths (array): lower and upper edges of the order
            xrange (array):  x coverage of the order in terms of two ends at x axis
            data_group (list): set of input data from various sources such as spectral data and flat data
            s_rate (list or number): sampling rate from input domain to output domain for 2D data
            sum_extraction(bool): flag to indicate if performing summation on collected data column by column

        Returns:
            spectral_info (dict): information of the order including dimension and the data

        """

        input_y_dim, input_x_dim = np.shape(data_group[0])
        sampling_rate = []
        if type(s_rate).__name__ == 'list':
            sampling_rate.extend(s_rate)
        else:
            sampling_rate.append([s_rate, s_rate])

        output_x_dim = input_x_dim * sampling_rate[self.X]
        output_y_dim = input_y_dim * sampling_rate[self.Y]

        # self.d_print('output_x_dim: ', output_x_dim, 'sampling_rate: ', sampling_rate)

        # construct corners map between input and output
        x_output_step = np.arange(0, output_x_dim, dtype=int)      # x step in output domain, including 0 & output_x_dim
        x_step = self.get_input_pos(x_output_step, sampling_rate[self.X])    # x step in input domain

        # x step compliant to xrange
        x_step = x_step[np.where(np.logical_and(x_step >= xrange[0], x_step <= xrange[1]))[0]]
        x_output_step = self.get_output_pos(x_step, sampling_rate[self.X]).astype(int)

        y_mid = np.polyval(coeffs, x_step)                            # y position of spectral trace
        v_border = np.array([np.amax(y_mid), np.amin(y_mid)])
        # the vertical position to locate the order in output domain
        y_output_mid = self.get_output_pos(np.mean(v_border), sampling_rate[self.Y])
        # self.d_print('y_output_mid: ', y_output_mid)

        output_widths = self.get_output_pos(widths, sampling_rate[self.Y]).astype(int)  # width of output
        upper_width = min(output_widths[1], output_y_dim - 1 - y_output_mid)
        lower_width = min(output_widths[0], y_output_mid)
        self.d_print('edges of order at output: ', upper_width, lower_width)

        y_size = 1 if sum_extraction is True else (upper_width+lower_width)
        total_data_group = len(data_group)
        out_data = [np.zeros((y_size, output_x_dim)) for _ in range(0, total_data_group)]

        # x_output_step in sync with x_step,
        s_x = 0
        for o_x in x_output_step:               # ox: 0...x_dim-1, out_data: 0...x_dim-1, corners: 0...
            # if o_x % 1000 == 0:
            #    self.d_print(o_x, end=" ")
            for o_y in range(0, upper_width):
                y_i = int(math.floor(y_mid[s_x]+o_y))
                x_i = int(math.floor(x_step[s_x]))
                for i in range(0, total_data_group):
                    if sum_extraction is True:
                        out_data[i][0, o_x] += data_group[i][y_i, x_i]
                    else:
                        out_data[i][lower_width+o_y, o_x] = data_group[i][y_i, x_i]

            for o_y in range(0, lower_width):
                y_i = int(math.floor(y_mid[s_x]-o_y-1))
                x_i = int(math.floor(x_step[s_x]))
                for i in range(0, total_data_group):
                    if sum_extraction is True:
                        out_data[i][0, o_x] += data_group[i][y_i, x_i]
                    else:
                        out_data[i][lower_width-o_y-1, o_x] = data_group[i][y_i, x_i]

            s_x += 1

        result_data = {'y_center': y_output_mid,
                       'width': [upper_width, lower_width],
                       'dim': [output_y_dim, output_x_dim],
                       'out_data': [out_data[i] for i in range(0, total_data_group)]}

        return result_data

    def rectify_spectrum_curve(self, coeffs: np.ndarray, widths: np.ndarray, xrange: np.ndarray,
                               data_group: list, s_rate=[1, 1], sum_extraction: bool = True, direction: int = NORMAL):
        """
        Rectify the order trace by collecting the pixels in vertical or normal direction of the order

        Parameters:
            coeffs (array): polynomial coefficients starting from zero order
            widths (array): lower and upper edges of the order
            xrange (array):  x coverage of the order in terms of two ends at x axis
            data_group (list): set of input data from various sources such as spectral data and flat data
            s_rate (list or number): sampling rate from input domain to output domain for 2D data
            sum_extraction(bool): flag to indicate if performing summation on collected data column by column
            direction (int): data collection method for rectification.
                             NORMAL: collect data along the normal direction of the order
                             VERTICAL: collect data along the vertical direction

        Returns:
            spectral_info (dict): information of straightened order including dimension and the data

        """

        input_y_dim, input_x_dim = np.shape(data_group[0])
        sampling_rate = []
        if type(s_rate).__name__ == 'list':
            sampling_rate.extend(s_rate)
        else:
            sampling_rate.append([s_rate, s_rate])

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
        y_output_mid = self.get_output_pos(np.mean(v_border), sampling_rate[self.Y])
        # self.d_print('y_output_mid: ', y_output_mid)

        output_widths = self.get_output_pos(widths, sampling_rate[self.Y]).astype(int)  # width of output
        upper_width = min(output_widths[1], output_y_dim - 1 - y_output_mid)
        lower_width = min(output_widths[0], y_output_mid)
        self.d_print('width at output: ', upper_width, lower_width)

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
            # if o_x % 100 == 0:
            #    print(o_x, end=" ")
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

    def get_flux_from_order(self, coeffs: np.ndarray, widths: np.ndarray, x_range: np.ndarray, in_data: np.ndarray,
                            flat_flux: np.ndarray, s_rate: list = [1, 1], norm_direction: int = None):
        """
        Collect the data around the order with either rectifying the pixels or not.
        If the rectification method is specified, the pixels along the order are selected depending on the edge size
        (i.e. widths) and the edge direction (i.e. norm_direction). With that, the pixels appearing at
        either vertical or normal direction of the order are weighted and summed up. The weighting for each pixel
        is determined based on the area of that pixel contributing to the pixel after rectification.
        If no rectification method is specified, the pixels along the vertical direction the order are collected
        in full depending on the edge size only.

        Parameters:
            coeffs (array): polynomial coefficients starting from zero order
            widths (array): lower and upper edges of the orders
            x_range (array): x coverage of the order in terms of two ends at x axis
            in_data (array): 2D spectral data
            flat_flux (array): 2D flat data
            s_rate (list or number): sampling rate from input domain to output domain
            norm_direction(int): rectification method. optional.
                                 None: no rectification.
                                 VERTICAL: pixels at  the north and south direction along the order are collected to
                                           be rectified.
                                 NORMAL: pixels at the normal direction of the order are collected to be rectified.

        Returns:
            out (dict): information related to the order data, like
                        {'order_data': <2D data>, 'order_flat': <2D data>,
                         'data_height': <height of the order data>,
                         'data_width': <width of the order data>,
                         'out_y_center': <y center position where the 2D result data is located>
                        }
        """

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
    def optimal_extraction(s_data: np.ndarray, f_data: np.ndarray, data_height: int,
                           data_width: int, y_center: int):
        """
        Do optimal extraction from 2D data to 1D data on collected pixels along the order

        Parameters:
            s_data (array): 2D spectral data collected for one order
            f_data (array): 2D flat data collected for one order
            data_height (int); height of the 2D data for optimal extraction
            data_width (int): width of the 2D data for optimal extraction
            y_center (int): y position where the 1D result is located in the output domain

         Returns:
            out (array): 1D data result of optimal extraction
        """
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

        return {'order_data': s_data, 'order_flat': f_data, 'extraction': w_data,
                'y_center': y_center}

    @staticmethod
    def optimal_extraction_weight_only(s_data: np.ndarray, f_data: np.ndarray, data_width: int, y_center: int):
        """
        Do optimal extraction from 2D data to 1D data on collected pixels along the order column. The formula
        that comprises the extraction algorithm mainly calculates the weighted summation over the collected pixels and
        the weighting is determined by the ratio between flat data of each pixel  over the summation of those at the
        same column. Currently, this function is not in use for optimal extraction.

        Parameters:
            s_data (array): 2D spectral data collected for one order
            f_data (array): 2D flat data collected for one order
            data_width (int): width of the 2D data for optimal extraction
            y_center (int): y position where the 1D result is located in the output domain

         Returns:
            out (array): 1D data result of optimal extraction
        """
        w_data = np.zeros((1, data_width))

        # taking weighted summation on spectral data for each column,
        # the weight is based on flat data.

        w_sum = np.sum(f_data, axis=0)
        non_zero_idx = np.where(w_sum != 0.0)[0]
        w_data[0, non_zero_idx] = np.sum(s_data[:, non_zero_idx]*f_data[:, non_zero_idx], axis=0)/w_sum[non_zero_idx]

        return {'order_data': s_data, 'order_flat': f_data, 'extraction': w_data,
                'y_center': y_center}

    @staticmethod
    def summation_extraction(s_data: np.ndarray, y_center: int):
        """
        Spectrum extraction by summation on collected pixels (rectified or non-rectified)
        """
        out_data = np.sum(s_data, axis=0)

        return {'order_data': s_data, 'extraction': out_data, 'y_center': y_center}

    @staticmethod
    def fill_2d_with_data(from_data: np.ndarray, to_data: np.ndarray, to_pos: int, from_pos: int = 0):
        """
        Fill a band of 2D data into another band of 2D to the vertical position in 'to_data'

        Parameters:
            from_data(array): band of data to be copied from
            to_data(array): 2D area to copy the data to
            to_pos(number): the vertical position in 'to_data' where 'from_data' is copied to
            from_pos(number): the vertical position in 'from_data' where the data is copied from. The default is 0.
        Returns:
            out (array): 2D data with 'from_data' filled in.
        """

        to_y_dim = np.shape(to_data)[0]
        from_y_dim = np.shape(from_data)[0]

        if to_pos < 0 or to_pos >= to_y_dim or from_pos < 0 or from_pos >= from_y_dim:  # out of range, do nothing
            return to_data

        h = min((to_y_dim - to_pos), from_y_dim-from_pos)
        to_data[to_pos:, :] = from_data[from_pos:from_pos+h, :]
        return to_data

    @staticmethod
    def vertical_normal(pos_x: np.ndarray, sampling_rate: float):
        """
        Calculate the vertical vector at the specified x position per vertical sampling rate
        """

        v_norms = np.zeros((pos_x.size, 2))
        d = 1.0/sampling_rate
        v_norms[:, 1] = d

        return v_norms

    @staticmethod
    def poly_normal(pos_x: np.ndarray, coeffs: np.ndarray, sampling_rate: int = 1):
        """
        Calculate the normal vector at the specified x position per vertical sampling rate
        """

        d_coeff = np.polyder(coeffs)
        tan = np.polyval(d_coeff, pos_x)
        v_norms = np.zeros((pos_x.size, 2))
        norm_d = np.sqrt(tan*tan+1)*sampling_rate
        v_norms[:, 0] = -tan/norm_d
        v_norms[:, 1] = 1.0/norm_d

        return v_norms

    @staticmethod
    def get_input_pos(output_pos: np.ndarray, s_rate: float):
        """
        Get associated position at input domain per output position and sampling rate

        Parameters:
            output_pos (array): position on output domain
            s_rate (number): sampling ratio between input domain and output domain, input*s_rate = output

        Returns:
            out (array): position on input domain
        """

        return output_pos/s_rate

    @staticmethod
    def get_output_pos(input_pos: np.ndarray, s_rate: float):
        """
        get the associated output position per input position and sampling rate

        Parameters:
            input_pos (array): position on input domain
            s_rate (float): sampling rate

        Returns:
            out (array): position on output domain
        """

        if isinstance(input_pos, np.ndarray):
            return np.floor(input_pos*s_rate)     # position at output cell domain is integer based
        else:
            return math.floor(input_pos*s_rate)

    @staticmethod
    def go_vertical(crt_pos: np.ndarray, norm: np.ndarray, direction: int = 1):
        """
        Get 2D position from current position by traversing in given direction (or reverse direction)

        Parameters:
            crt_pos (array): current position
            norm (array): vector of unit length
            direction (number): traverse in given or reverse direction

        Returns:
            out (array): new position at given (or reverse) direction
        """

        new_pos = crt_pos + direction * norm

        return new_pos

    def compute_output_flux(self, input_corners: np.ndarray, input_data: np.ndarray, input_x_dim: int, input_y_dim: int,
                            vertical_normal: int):
        """
        compute the flux within a polygon using polygon clipping algorithm if the polygon corners are collected in
        normal direction or checking the area coverage of each pixel inside the polygon if the polygon corners are
        collected in vertical direction.

        Parameters:
            input_corners(array): polygon corners at input domain in counterclockwise order
            input_data(array): input data
            input_x_dim(number): width of input data
            input_y_dim(number): height of input data
            vertical_normal(int): the method how the corners are collected, NORMAL or VERTICAL
        Returns:
            flux(number): flux value
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

    def compute_flux_from_polygon_clipping(self, poly_corners: np.ndarray, border_points: list, input_data: np.ndarray):
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

        return flux, total_area

    def compute_flux_from_vertical_clipping(self, poly_corners: np.ndarray, border_points: list,
                                            input_data: np.ndarray):
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
        cell_corners = [[None for _ in range(cols)] for _ in range(rows)]  # corners in each cell starting from x1, y1

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
            v_cell_info = [{'inter_points': None, 'border_points': [None, None]} for _ in y_grid]
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
                if (v_cell_info[y_idx]['border_points'][0] is not None) and \
                   (v_cell_info[y_idx]['border_points'][0] != bottom_p[0]):
                    corners.append(v_cell_info[y_idx]['border_points'][0])

                y_c = y_grid[y_idx+1]
                if v_cell_info[y_idx+1]['inter_points'] is not None:
                    corners.append([v_cell_info[y_idx+1]['inter_points'][0], y_c])
                    corners.append([v_cell_info[y_idx+1]['inter_points'][1], y_c])
                else:
                    corners.extend([[border_x[x_ni], y_c], [border_x[x_ni+1], y_c]])

                next_bottom = [corners[-2], corners[-1]]  # the last two corners just added

                if (v_cell_info[y_idx]['border_points'][1] is not None) and \
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
                if cell_corners[y][x] is None or input_data[y1+y, x1+x] == 0:
                    continue

                # corners = np.array(cell_corners[y][x])-np.array([x1, y1])
                corners = np.array(cell_corners[y][x])
                area = self.polygon_area(corners)
                total_area += area
                flux += area * input_data[y1+y, x1+x]

        return flux, total_area

    def polygon_clipping(self, poly_points: np.ndarray, clipper_points: list, clipper_size: int):
        """
        New polygon points after performing the clipping based on the specified clipping area
        """

        new_poly_points = [[poly_points[i, 0], poly_points[i, 1]] for i in range(clipper_size)]
        for i in range(clipper_size):
            k = (i+1) % clipper_size
            new_poly_points = self.clip(new_poly_points, clipper_points[i][0], clipper_points[i][1],
                                        clipper_points[k][0], clipper_points[k][1])

        new_corners = self.remove_duplicate_point(new_poly_points)
        return np.array(new_corners)

    @staticmethod
    def remove_duplicate_point(corners: np.ndarray):
        """
        Remove the duplicate points from a list of corner points
        """

        new_corners = []
        for c in corners:
            if c not in new_corners:
                new_corners.append(c)

        return new_corners

    @staticmethod
    def polygon_area(corners: np.ndarray):
        """
        Calculate the polygon area per polygon corners
        """

        polygon_size = np.shape(corners)[0]
        area = 0.0
        for i in range(polygon_size):
            k = (i+1) % polygon_size
            area += corners[i, 0]*corners[k, 1] - corners[k, 0]*corners[i, 1]

        return abs(area)/2

    def clip(self, poly_points: np.ndarray, x1: int, y1: int, x2: int, y2: int):
        """
        Polygon clipping
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
    def line_intersect(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float):
        den = (x1-x2)*(y3-y4) - (x3-x4)*(y1-y2)
        num_x = (x1*y2 - x2*y1) * (x3 - x4) - (x1 - x2) * (x3*y4 - x4*y3)
        num_y = (x1*y2 - x2*y1) * (y3 - y4) - (y1 - y2) * (x3*y4 - x4*y3)

        return [num_x/den, num_y/den]

    def write_data_to_dataframe(self, result_data: np.ndarray):
        """
        Write optimal extraction result to Pandas DataFrame Object
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
        df_result = pd.DataFrame(result_data)
        df_result.attrs['MJD-OBS'] = mjd
        df_result.attrs['EXPTIME'] = exptime
        df_result.attrs['TOTALORD'] = total_order
        df_result.attrs['DIMWIDTH'] = dim_width

        return df_result

    def extract_spectrum(self, rectification_method: int = NoRECT, extraction_method: str = 'optimal',
                         order_set: np.ndarray = None,
                         print_progress: str = None,
                         bleeding_file: str = None):
        """
        Optimal extraction from 2D flux to 1D. Rectification step is optional.

        Parameters:
            rectification_method (int): NoRECT: no rectification.
                    VERTICAL: pixels at  the north-up direction along the order are collected to be rectified.
                    NORMAL: pixels at the normal direction of the order are collected to be rectified.
            extraction_method (str): extraction method. 'optimal' for optimal extraction or
                                    'sum' for summation on extraction.
            order_set (array): set of order to extract.
            print_progress (str): output debug information to stdout,  a file or no print per the following values,
                                     'no': no display,
                                     empty string or a string: print out to stdout or a file per string value.
                                     None: print out to the debug channel as the setting in DEBUG section of .cfg file.
            bleeding_file (str): bleeding cure file

        Returns:
            out (array): 1D spectrum data
        """
        if print_progress is not None:
            self.redirect_debug_output(print_progress)

        self.update_spectrum_flux(bleeding_file)

        dim_width, dim_height = self.get_spectrum_size()
        total_order = self.get_spectrum_order()
        if order_set is None:
            order_set = np.arange(0, total_order, dtype=int)

        out_data = np.zeros((order_set.size, dim_width))

        for idx_out in range(order_set.size):
            c_order = order_set[idx_out]
            self.d_print(c_order, ' edges: ', self.get_order_edges(c_order),
                         ' xrange: ', self.get_order_xrange(c_order), end=" ")
            order_flux = self.get_flux_from_order(self.order_coeffs[c_order], self.get_order_edges(c_order),
                                                  self.get_order_xrange(c_order), self.spectrum_flux, self.flat_flux,
                                                  norm_direction=rectification_method)

            # check element with nan np.argwhere(np.isnan(order_flux.get('order_data'))), paras data has nan in spectrum
            result = dict()
            if 'optimal' in extraction_method:
                result = self.optimal_extraction(order_flux.get('order_data'), order_flux.get('order_flat'),
                                                 order_flux.get('data_height'), order_flux.get('data_width'),
                                                 order_flux.get('out_y_center'))
            elif 'sum' in extraction_method:
                result = self.summation_extraction(order_flux.get('order_data'), order_flux.get('out_y_center'))
            if 'extraction' in result:
                self.fill_2d_with_data(result.get('extraction'), out_data, idx_out)
        self.d_print(" ")
        data_df = self.write_data_to_dataframe(out_data)

        return {'optimal_extraction_result': data_df}

    @staticmethod
    def result_test(target_file: str, data_result: np.ndarray):
        target_data = fits.getdata(target_file)
        t_y, t_x = np.shape(target_data)
        r_y, r_x = np.shape(data_result)

        if t_y != r_y or t_x != r_x:
            return {'result': 'error', 'msg': 'dimension is not the same'}

        diff_data = target_data - data_result
        diff_idx = np.where((diff_data != 0.0) & (~np.isnan(diff_data)))[0]

        if diff_idx.size > 0:
            return {'result': 'error', 'msg': 'data is not the same at '+diff_idx.tostring()}

        return {'result': 'ok'}