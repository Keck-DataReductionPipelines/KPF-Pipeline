"""This module does polygon clipping on simulated spectral order trace described in fits style"""

from __future__ import print_function
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import math
import numpy as np
import copy
import csv
from numpy.polynomial.polynomial import polyval, polyder
from numpy import sqrt, square
import time

X = 0
Y = 1
C0 = 0
C1 = 1
C2 = 2
C3 = 3
NORMAL = 0
VERTICAL = 1

class PolygonClipping2:
    """Class for doing polygon clipping on simulated spectral trace.

    Parameters:
        fits_file (str): path for fits_file which contains the fitting polynomial coefficient for the spectral trace.

    """


    def __init__(self, fits_file=None, order_width=5):
        self.spectral_curve_file = fits_file
        self.order_width = order_width

    def load_paras_spectral(self, filename=None):
        """Load paras spectral trace from the fits file"""
        header_key_map = {
            'NAXIS1': 'xdim',
            'NAXIS2': 'ydim'
        }
        fits_name = self.spectral_curve_file if filename is None else filename
        if fits_name is None:
            return None

        fits_header = fits.open(fits_name)
        if len(fits_header) < 1:
            return None

        spectral_info = {'data': fits_header[0].data}
        header = fits_header[0].header
        header_keys = list(header.keys())
        for k in header_keys:
            if k in header_key_map.keys():
                spectral_info[header_key_map.get(k)] = int(header[k])

        return spectral_info

    def load_correct_data(self, mask_file=None):
        """Load mask file"""
        if mask_file is None:
            return None
        mask_header = fits.open(mask_file)
        mask_data = np.array(mask_header[0].data)
        return mask_data

    def correct_data_by_sub(self, cure_data, in_data):
        cure_size = np.shape(cure_data)
        data_size = np.shape(in_data)
        if cure_size != data_size:
           return in_data;

        new_in_data = in_data - cure_data;
        return new_in_data

    def load_csv_file(self, filename, nx, power=None, delimit=','):
        rows = np.empty((0, power+1))
        widths = np.empty((0, 2))
        xrange = np.empty((0, 2), dtype=int)
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=delimit)
            for row in csv_reader:
                row_num = np.array(row[0:power+1][::-1]).astype(float)
                if len(row) >= (power+2):
                    if len(row) >= (power+3):
                        width_num = np.array(row[power+1:power+3]).astype(float)
                    else:
                        width_num = np.array([row[power+1], row[power+1]]).astype(float)

                    #min_w = np.amin(width_num)

                    #if (min_w > self.order_width):
                    #    width_num = width_num - (min_w - self.order_width)


                    if len(row) >= (power+5):
                        xrange_num = np.array(row[power+3:power+5]).astype(int)
                    else:
                        xrange_num = np.array([0, nx-1]).astype(int)
                else:
                    xrange_num = np.array([0, nx-1]).astype(int)
                    width_num = np.array([self.order_width, self.order_width])

                rows = np.vstack((rows, row_num))
                widths = np.vstack((widths, width_num))
                xrange = np.vstack((xrange, xrange_num))

        return rows, widths, xrange


    def rectify_spectral_curve(self, coeffs, widths, xrange, data_group, s_rate=[1, 1], sum_extraction=True, verbose=False, vertical_normal=NORMAL):
        """Straighten the spectral trace

        Parameters:
            coeffs (array): polynomial coefficient list starting from that of zero order
            widths (array): lower and upper order trace widths
            data_group (array): input data from different source such as spectral data and flat data
            s_rate (list): sampling rate from input domain to output domain
            sum_extraction(bool): flag to indicate if performing summation on rectified curve or not
            verbose(bool): flag for debug


        Returns:
            spectral_info (dict): straightened spectral information including dimension and the data

        """

        input_ydim, input_xdim = np.shape(data_group[0])
        sampling_rate = []
        if type(s_rate).__name__ == 'list':
            sampling_rate.extend(s_rate)
        else:
            sampling_rate.append([s_rate, s_rate])

        output_xdim = input_xdim * sampling_rate[X]
        output_ydim = input_ydim * sampling_rate[Y]

        if verbose is True:
            print('output_xdim: ', output_xdim, 'sampling_rate: ', sampling_rate)

        # construct corners map between input and output

        x_output_step = np.arange(0, output_xdim+1, dtype=int)           # x step in output domain, both ends are included

        #if verbose is True:
        #   print('x_output_step=', x_output_step)

        x_step = self.get_input_pos(x_output_step, sampling_rate[X])  # x step in input domain
        x_step = x_step[np.where(np.logical_and(x_step >= xrange[0], x_step <= (xrange[1]+1)))[0]]
        x_output_step = self.get_output_pos(x_step, sampling_rate[X]).astype(int)
        y_mid = np.polyval(coeffs, x_step)          # spectral trace value at mid point

        if vertical_normal == NORMAL:
            y_norm_step = self.poly_normal(x_step, coeffs, sampling_rate[Y])   # curve norm along x in input domain
        else:
            y_norm_step = self.vertical_normal(x_step, sampling_rate[Y])   # vertical norm along x in input domain

        v_border = np.array([np.amax(y_mid), np.amin(y_mid)])
        if verbose is True:
            print('v_border: ', v_border)

        # the vertical position to locate the output spectral
        v_mid = self.get_output_pos(np.mean(v_border), sampling_rate[Y])
        if verbose is True:
            print('v_mid: ', v_mid)

        output_widths = self.get_output_pos(widths, sampling_rate[Y]).astype(int)  # width of output
        upper_width = min(output_widths[1], output_ydim - 1 - v_mid)
        lower_width = min(output_widths[0], v_mid)
        if verbose is True:
            print('width at output: ', upper_width, lower_width)

        #print("before mid corners")

        corners_at_mid = np.vstack((x_step, y_mid)).T
        if verbose is True:
            print('corners_at_mid: ', corners_at_mid)

        y_size = 1 if sum_extraction is True else (upper_width+lower_width)
        total_data_group = len(data_group)
        out_data = [np.zeros((y_size, output_xdim)) for i in range(0, total_data_group)]

        input_upper_corners = np.zeros((upper_width+1, x_step.size, 2))
        input_lower_corners = np.zeros((lower_width+1, x_step.size, 2))
        input_upper_corners[0] = corners_at_mid.copy()
        input_lower_corners[0] = corners_at_mid.copy()

        for o_y in range(1, upper_width+1):
            next_upper_corners = self.go_vertical(input_upper_corners[o_y-1], y_norm_step, 1)
            input_upper_corners[o_y] = next_upper_corners

        #print("before lower corners")
        #import pdb;pdb.set_trace()
        for o_y in range(1, lower_width+1):
            next_lower_corners = self.go_vertical(input_lower_corners[o_y-1], y_norm_step, -1)
            input_lower_corners[o_y] = next_lower_corners

        #print("before get corners")
        #import pdb;pdb.set_trace()

        s_x = 0
        for o_x in x_output_step[0:-1]:               # ox: 0...xdim-1, out_data: 0...xdim-1, corners: 0...
            if o_x%100 == 0:
                print(o_x, end=" ")
            for o_y in range(0, upper_width):
                input_corners = input_upper_corners[o_y:o_y+2, s_x:s_x+2].reshape((4,2))[[0,2,3,1]]
                for i in range(0, total_data_group):
                    flux = self.compute_output_flux(input_corners, data_group[i], input_xdim, input_ydim,
                                                    vertical_normal, False)
                    if sum_extraction is True:
                        out_data[i][0, o_x] += flux
                    else:
                        out_data[i][lower_width+o_y, o_x] = flux

            for o_y in range(0, lower_width):
                input_corners = input_lower_corners[o_y:o_y+2, s_x:s_x+2].reshape((4,2))[[2,0,1,3]]
                for i in range(0, total_data_group):
                    flux = self.compute_output_flux(input_corners, data_group[i], input_xdim, input_ydim,
                                                    vertical_normal, False)

                    if sum_extraction is True:
                        out_data[i][0, o_x] += flux
                    else:
                        out_data[i][lower_width-o_y-1, o_x] = flux

            s_x += 1
            if verbose is True:
                print('[%d %.2f]' % (o_x, out_data[0][0, o_x]), end=' ')

        print(' ')
        if verbose is True:
            print(' ')
        #print("optimal done")
        #import pdb;pdb.set_trace()

        result_data = {'y_center': v_mid,
                       'width': [upper_width, lower_width],
                       'dim': [output_ydim, output_xdim],
                       'out_data': [out_data[i] for i in range(0, total_data_group)]}

        return result_data

    def collect_data_from_order(self, coeffs, widths, xrange, data_group, s_rate=[1, 1], sum_extraction=True, verbose=False):
        """collect the spectral data per order by polynomial fit
        Parameters:
            coeffs (array): polynomial coefficient list starting from that of zero order
            widths (array): lower and upper order trace widths
            data_group (array): input data from different source such as spectral data and flat data
            s_rate (list): sampling rate from input domain to output domain
            sum_extraction(bool): flag to indicate if performing summation on rectified curve or not
            verbose(bool): flag for debug

        Returns:
            spectral_info (dict): straightened spectral information including dimension and the data

        """

        input_ydim, input_xdim = np.shape(data_group[0])
        sampling_rate = []
        if type(s_rate).__name__ == 'list':
            sampling_rate.extend(s_rate)
        else:
            sampling_rate.append([s_rate, s_rate])

        output_xdim = input_xdim * sampling_rate[X]
        output_ydim = input_ydim * sampling_rate[Y]

        if verbose is True:
            print('output_xdim: ', output_xdim, 'sampling_rate: ', sampling_rate)

        # construct corners map between input and output

        x_output_step = np.arange(0, output_xdim, dtype=int)           # x step in output domain, both ends are included

        #if verbose is True:
        #   print('x_output_step=', x_output_step)

        x_step = self.get_input_pos(x_output_step, sampling_rate[X])  # x step in input domain
        x_step = x_step[np.where(np.logical_and(x_step >= xrange[0], x_step <= xrange[1]))[0]]
        x_output_step = self.get_output_pos(x_step, sampling_rate[X]).astype(int)
        y_mid = np.polyval(coeffs, x_step)                            # spectral trace value at mid point

        v_border = np.array([np.amax(y_mid), np.amin(y_mid)])
        if verbose is True:
            print('v_border: ', v_border)

        # the vertical position to locate the output spectral
        v_mid = self.get_output_pos(np.mean(v_border), sampling_rate[Y])
        if verbose is True:
            print('v_mid: ', v_mid)

        output_widths = self.get_output_pos(widths, sampling_rate[Y]).astype(int)  # width of output
        upper_width = min(output_widths[1], output_ydim - 1 - v_mid)
        lower_width = min(output_widths[0], v_mid)
        if verbose is True:
            print('width at output: ', upper_width, lower_width)

        y_size = 1 if sum_extraction is True else (upper_width+lower_width)
        total_data_group = len(data_group)
        out_data = [np.zeros((y_size, output_xdim)) for i in range(0, total_data_group)]

        # x_output_step in sync with x_step,
        s_x = 0
        for o_x in x_output_step:               # ox: 0...xdim-1, out_data: 0...xdim-1, corners: 0...
            if o_x%1000 == 0:
                print(o_x, end=" ")
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
            if verbose is True:
                print('[%d %.2f]' % (o_x, out_data[0][0, o_x]), end=' ')

        print(' ')
        if verbose is True:
            print(' ')

        result_data = {'y_center': v_mid,
                       'width': [upper_width, lower_width],
                       'dim': [output_ydim, output_xdim],
                       'out_data': [out_data[i] for i in range(0, total_data_group)]}

        return result_data

    def rectify_spectral_curve_by_optimal(self, coeffs, widths, x_range, in_data, flat_data, s_rate=[1, 1], verbose=False, norm_direction=NORMAL):
        if norm_direction is None:
            results = self.collect_data_from_order(coeffs, widths, x_range, [in_data, flat_data], s_rate, sum_extraction=False)
        else:
            results = self.rectify_spectral_curve(coeffs, widths, x_range, [in_data, flat_data], s_rate, sum_extraction=False, vertical_normal=norm_direction)

        height = sum(results.get('width'))
        width = results.get('dim')[1]
        w_data = np.zeros((1, width))

        s_data = results.get('out_data')[0]
        f_data = results.get('out_data')[1]

        for x in range(0, width):
            w_sum = sum(f_data[:, x])

            if w_sum != 0.0:
                w_data[0, x] = np.sum(s_data[:, x] * f_data[:, x])/w_sum

        result_data = {'y_center': results.get('y_center'),
                       'dim': results.get('dim'),
                       'out_data': w_data,
                       'rectified_trace': s_data,
                       'rectified_flat': f_data
                      }

        return result_data

    def rectify_spectral_curve_by_optimal2(self, coeffs, widths, x_range, in_data, flat_data, s_rate=[1, 1], verbose=False, norm_direction=NORMAL):
        """
        Do rectification (optional) and optimal extraction. No rectification is done if norm_direction is NOne.
        """

        if norm_direction is None:
            results = self.collect_data_from_order(coeffs, widths, x_range, [in_data, flat_data], s_rate, sum_extraction=False)
        else:
            results = self.rectify_spectral_curve(coeffs, widths, x_range, [in_data, flat_data], s_rate, sum_extraction=False, vertical_normal=norm_direction)

        height = sum(results.get('width'))
        width = results.get('dim')[1]
        s_data = results.get('out_data')[0]
        f_data = results.get('out_data')[1]

        w_data = self.optimal_extraction(s_data, f_data, width, height)

        result_data = {'y_center': results.get('y_center'),
                       'dim': results.get('dim'),
                       'out_data': w_data,
                       'rectified_trace': s_data,
                       'rectified_flat': f_data
                       }

        return result_data

    def optimal_extraction(self, s_data, f_data, data_width, data_height):
        w_data = np.zeros((1, data_width))

        for x in range(0, data_width):
            w_sum = sum(f_data[:, x])
            d_var = np.full((1, data_height), 1.0)    # set the variance to be 1.0
            if w_sum != 0.0:                          # pixel is not out of range
                p_data = f_data[:, x]/w_sum
                num = p_data * s_data[:, x]/d_var     # sum((f/sum(f)) * s/variance)/sum(((f/sum(f))^2)/variance)) ref. Horne 1986
                dem = np.power(p_data, 2)/d_var
                w_data[0, x] = np.sum(num)/np.sum(dem)

        return w_data


    def rectify_spectral_curve_by_sum(self, coeffs, widths, x_range, in_data, s_rate=[1,1], verbose=False, norm_direction=NORMAL):
        """Straighten the spectral trace and perform the summation on the rectified trace

        Parameters:
            coeffs (array): polynomial coefficient list starting from that of zero order
            widths (array): upper and lower width along the trace
            in_data (array): input data
            s_rate (list): sampling rate from input domain to output domain
            verbose(bool): flag for debug


        Returns:
            spectral_info (dict): straightened spectral information including dimension and the data

        """
        if norm_direction is None:
            results = self.collect_data_from_order(coeffs, widths, x_range, [in_data], s_rate, sum_extraction=False)
        else:
            results = self.rectify_spectral_curve(coeffs, widths, x_range, [in_data], s_rate, sum_extraction=False, vertical_normal=norm_direction)

        if verbose is True:
            print('rectify curve: ', result_data)

        s_data = results.get('out_data')[0]
        return {'y_center': results.get('y_center'),
                'dim': results.get('dim'),
                'out_data': self.sum_curve(s_data, verbose),
                'rectified_trace': s_data
                }

    def sum_curve(self, flat_data, verbose=False):
        """Sum a band of spectral trace

        Parameters:
            flat_data(array): flatten trace
            verbose(bool): flag for debug

        Returns:
            out_data(array): sum extraction of one rectified flat

        """

        sum_data = np.sum(flat_data, axis=0)

        return sum_data

    def fill_2D_to_2D(self, from_data, to_data, from_center, to_center):
        """Fill a band of 2D data into another band of 2D from one vertical position to another one

        Parameters:
            from_data(array): data to be copied from
            to_data(array): 2D area to copy the data to
            from_center(number): locate the center of the from_data
            to_center(number): to the center of the to_data

        """

        s = np.shape(from_data)
        from_xdim = s[1]
        from_ydim = s[0]

        center_dist = to_center - from_center;
        for x in range(0, from_xdim):
            for y in range(0, from_ydim):
                to_data[center_dist+y, x] = from_data[y, x]

    def vertical_normal(self, pos_x, sampling_rate):
        """ Calculate the normal at the specified x position per vertical sampling rate"""

        is_array = True
        if isinstance(pos_x, np.ndarray) is not True:
            is_array = False
            pos_x = np.array([pos_x])

        v_norms = np.empty((0, 2))
        d = 1.0/sampling_rate

        for n in range(0, pos_x.size):
            v_norms = np.vstack((v_norms, [0, d]))

        if is_array:
            return v_norms
        else:
            return v_norms[0]


    def poly_normal(self, pos_x, coeffs, sampling_rate = 1):
        """ Calculate the normal at the specified x position per vertical sampling rate"""

        is_array = True
        if isinstance(pos_x, np.ndarray) is not True:
            is_array = False
            pos_x = np.array([pos_x])

        d_coeff = np.polyder(coeffs)
        tan = np.polyval(d_coeff, pos_x)
        v_norms = np.empty((0, 2))

        norm_d = np.sqrt(tan*tan+1)*sampling_rate

        for n in range(0, pos_x.size):
            v_norms = np.vstack((v_norms, [-tan[n], 1]/norm_d[n]))

        if is_array:
            return v_norms
        else:
            return v_norms[0]

    def get_input_pos(self, output_pos, s_rate):
        """ Get associated position at input domain per output position and sampling rate

        Parameters:
            output_pos (array): position on output domain
            s_rate (number): sampling ratio between input domain and output domain, input*s_rate = output

        Returns:
            out (array): position on input domain
        """

        return output_pos/s_rate

    def get_output_pos(self, input_pos, s_rate):
        """ get the associated output position per input position and sampling rate

        Parameters:
            input_pos (array): position on input domain

        Returns:
            out (array): position on output domain
        """

        if isinstance(input_pos, np.ndarray):
            return np.floor(input_pos*s_rate)     # position at output cell domain is integer based
        else:
            return math.floor(input_pos*s_rate)

    def go_vertical(self, pos, norm, direction = 1):
        """ Get positions of next row along the normal direction at each position

        Parameters:
            pos (array): position array
            norm (array): normal vector of unit length
            direction (number): upward or downward

        Returns:
            out (array): position along norm direction
        """

        new_pos = pos + direction * norm

        return new_pos

    def is_zero_coverage(self, corners, input_data):
        """ Check if all corners are located at zero valued cells

        Parameters:
            corners (array): 4 corners
            input_data (array): data where corners are located at

        Returns:
            ret (bool): True or False

        """

        s = np.shape(input_data) # 0: y dim, 1: x dim
        s_x = s[1] - 1
        s_y = s[0] - 1
        corners_on_cell = [[max(0, min(math.floor(c[X]), s_x)),
                            max(0, min(math.floor(c[Y]), s_y))] for c in corners]
        non_zero_corners = [ c for c in corners_on_cell if input_data[c[Y], c[X]] != 0.0]
        return len(non_zero_corners) == 0

    def is_out_range(self, corners, input_data):
        """ Check if all corners are located out of a domain"""

        size = np.shape(input_data)
        corners_on_cell = [[math.floor(c[X]), math.floor(c[Y])] for c in corners]
        return all([(c[Y] < 0 or c[Y] > size[0] or c[X] < 0 or c[X] > size[1]) for c in corners_on_cell])


    def compute_output_flux(self, input_corners, input_data, input_xdim, input_ydim, vertical_normal, verbose=False):
        """ compute the flux per corners and a matrix of data per polygon clipping algorithm

        Parameters:
            input_corners(array): corners at input domain in counterclockwise order
            input_data(array): input data
            input_xdim(number): width of input data
            input_ydim(number): height of input data
            verbose(bool): flag for debug

        Returns:
            flux(number): flux value

        """

        if verbose is True:
            print('input_corners: ', input_corners)

        x_list = input_corners[:, X]
        y_list = input_corners[:, Y]

        x_1 = np.amin(x_list)
        x_2 = np.amax(x_list)
        y_1 = np.amin(y_list)
        y_2 = np.amax(y_list)

        y_1 = max(0, math.floor(y_1))
        y_2 = min(input_ydim, math.ceil(y_2))
        x_1 = max(0, math.floor(x_1))
        x_2 = min(input_xdim, math.ceil(x_2))

        if verbose is True:
            print('x_1:', x_1, ' x2:', x_2, ' y1:', y_1, ' y2:', y_2)


        flux = 0.0
        if vertical_normal == VERTICAL:
            flux_vertical, total_area_vertical = self.compute_flux_from_vertical_clipping(input_corners, [x_1, x_2, y_1, y_2], input_data)
            return flux_vertical

        flux_polygon, total_area_polygon = self.compute_flux_from_polygon_clipping(input_corners, [x_1, x_2, y_1, y_2], input_data)
        return flux_polygon


    def compute_flux_from_polygon_clipping(self, poly_corners, border_points, input_data, verbose=False):
        x_1, x_2, y_1, y_2 = border_points
        total_area = 0.0
        flux = 0.0
        for x in range(x_1, x_2):
            for y in range(y_1, y_2):
                if verbose is True:
                    print('input_data[', y, x,']: ', input_data[y, x])

            if input_data[y, x] != 0.0:
                new_corners = self.polygon_clipping(poly_corners,[[x, y], [x, y+1], [x+1, y+1], [x+1, y]], 4)
                area = self.polygon_area(new_corners)
                if verbose is True:
                    print('area: ', area)
                total_area += area
                flux += area * input_data[y, x]

        return flux, total_area

    def compute_flux_from_vertical_clipping(self, poly_corners, border_points, input_data):
        # make mark on vertical grid line

        x1, x2, y1, y2 = border_points  # grid boundary of poly_corners
        y_grid = np.arange(y1, y2+1, dtype = float)

        border_x1 = np.amin(poly_corners[:, X])        # x range of poly_corners,
        border_x2 = np.amax(poly_corners[:, X])
        border_x = np.arange(math.floor(border_x1), math.ceil(border_x2)+1, dtype=float)  # horizontal coverage along x
        border_x[0] = border_x1  if border_x1 != border_x[0] else border_x[0]
        border_x[-1] = border_x2 if border_x2 != border_x[-1] else border_x[-1]

        # y top and bottom ends of poly_corners at each point in border_x
        bottom_y = ((border_x - border_x1)*poly_corners[3, Y]+ (border_x2 - border_x)*poly_corners[0, Y])/(border_x2-border_x1)
        top_y = ((border_x-border_x1)*poly_corners[2, Y] + (border_x2 - border_x)*poly_corners[1, Y])/(border_x2-border_x1)

        mark_y = []
        for i in range(len(border_x)):
            # vertical coverage at each point in border_x
            border_y = np.arange(math.floor(bottom_y[i]), math.ceil(top_y[i])+1, dtype=float)
            border_y[0] = bottom_y[i]
            border_y[-1] = top_y[i]
            mark_y.append(border_y)

        x_ni = 0
        rows, cols = (y2-y1, x2-x1)
        cell_corners = [ [None for _ in range(cols)] for _ in range(rows)] # corners set in each cell starting from x1, y1

        for x_ni in range(np.size(border_x)-1):
            # corners & points_at_borders: [<point at border 1>, <point at border 2>]

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
            max_sy_idx = max(y_line1_sy, y_line2_sy)
            min_ey_idx = min(y_line1_ey, y_line2_ey)
            max_ey_idx = max(y_line1_ey, y_line2_ey)

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

                next_bottom = [corners[-2], corners[-1]]

                if (v_cell_info[y_idx]['border_points'][1] is not None) and \
                   (v_cell_info[y_idx]['border_points'][1] != corners[-1]):
                   corners.append(v_cell_info[y_idx]['border_points'][1])

                if bottom_p[1] != corners[-1] and bottom_p[1] != corners[0]:
                    corners.append(bottom_p[1])
                bottom_p = next_bottom

                cell_corners[y_idx][x_ni] = corners

            cell_corners[max_ey_idx][x_ni] = [bottom_p[0], top_p[0], top_p[1], bottom_p[1]]
            x_ni += 1

        total_area = 0.0
        flux = 0.0
        for y in range(rows):
            for x in range(cols):
                if cell_corners[y][x] is None or input_data[y1+y, x1+x] == 0:
                    continue

                #corners = np.array(cell_corners[y][x])-np.array([x1, y1])
                corners = np.array(cell_corners[y][x])
                area = self.polygon_area(corners)
                total_area += area
                flux += area * input_data[y1+y, x1+x]

        return flux, total_area


    def polygon_clipping(self, poly_points, clipper_points, clipper_size):
        """ New polygon points after performing the clipping based on the specified clipping area"""

        #new_poly_points = np.copy(poly_points)
        new_poly_points = []
        for i in range(clipper_size):
            new_poly_points.append([poly_points[i, 0], poly_points[i, 1]])

        for i in range(clipper_size):
            k = (i+1)%clipper_size
            new_poly_points = self.clip(new_poly_points, clipper_points[i][0], clipper_points[i][1],
                                   clipper_points[k][0], clipper_points[k][1])

        new_corners = self.remove_duplicate_point(new_poly_points)
        return np.array(new_corners)

    def remove_duplicate_point(self, corners):
        """ Remove the duplicate points from a list of corner points"""

        new_corners = []
        for c in corners:
            if c not in new_corners:
                new_corners.append(c)

        return new_corners

    def polygon_area(self, corners):
        """ Calculate the polygon area per polygon corners"""

        polygon_size = np.shape(corners)[0]
        area = 0.0
        for i in range(polygon_size):
            k = (i+1)%polygon_size
            area += corners[i, 0]*corners[k, 1] - corners[k, 0]*corners[i, 1]

        return abs(area)/2


    def clip(self, poly_points, x1, y1, x2, y2):
        """ Polygon clipping"""

        new_points = []
        #poly_size = np.shape(poly_points)[0]
        poly_size = len(poly_points)

        for i in range(poly_size):
            k = (i+1)%poly_size

            #ix = poly_points[i, 0]
            #iy = poly_points[i, 1]
            #kx = poly_points[k, 0]
            #ky = poly_points[k, 1]
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
        #return np.array(new_points)
        return new_points


    def line_intersect(self, x1, y1, x2, y2, x3, y3, x4, y4):
        den = (x1-x2)*(y3-y4) - (x3-x4)*(y1-y2)
        num_x = (x1*y2 - x2*y1) * (x3 - x4) - (x1 - x2) * (x3*y4 - x4*y3)
        num_y = (x1*y2 - x2*y1) * (y3 - y4) - (y1 - y2) * (x3*y4 - x4*y3)
        #return np.array([num_x/den, num_y/den])
        return [num_x/den, num_y/den]

    def get_vertical_width(self, data_array, x, y):
        """Computer vertical width up and down between [x, y[0]] and [x+1, y[1]]"""

        array_size = np.shape(data_array)
        y_top = array_size[0]
        y_start = math.floor(min(y))

        # vertical width stop at windth_top and width_bottom
        s_to_top = y_top - 1 - y_start
        s_to_bottom = y_start

        for i in range(y_start, y_top, 1):
            if data_array[i, x] == 0.0:
                s_to_top = i - y_start
                break
        for i in range(y_start - 1, 0, -1):
            if data_array[i, x] == 0.0:
                s_to_bottom = y_start - i
                break

        return [s_to_top, s_to_bottom]

    def get_spectral_vertical_range(self, xdim, ydim, data_array):
        """Find the spectral vertical coverage.

        Parameters:
            xdim (int): dimension along x axis
            ydim (int): dimension along y axis
            data: spectral data in 2D array

        Returns:
            vertical_range (list): vertical coverage range

        """

        y_range = np.nonzero(data_array)[0]
        r_len = len(y_range)

        return [y_range[0], y_range[r_len-1]] if r_len > 0 else None


