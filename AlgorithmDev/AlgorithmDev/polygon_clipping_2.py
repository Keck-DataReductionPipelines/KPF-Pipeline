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
                    flux = self.compute_output_flux(input_corners, data_group[i], input_xdim, input_ydim, False)
                    if sum_extraction is True:
                        out_data[i][0, o_x] += flux
                    else:
                        out_data[i][lower_width+o_y, o_x] = flux

            for o_y in range(0, lower_width):
                input_corners = input_lower_corners[o_y:o_y+2, s_x:s_x+2].reshape((4,2))[[2,0,1,3]]
                for i in range(0, total_data_group):
                    flux = self.compute_output_flux(input_corners, data_group[i], input_xdim, input_ydim, False)
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


    def rectify_spectral_curve_by_optimal(self, coeffs, widths, x_range, in_data, flat_data, s_rate=[1, 1], verbose=False, norm_direction=NORMAL):
        results = self.rectify_spectral_curve(coeffs, widths, x_range, [in_data, flat_data], s_rate, sum_extraction=False, vertical_normal=norm_direction)

        #f_result = self.rectify_spectral_curve(coeffs, widths, x_range, flat_data, s_rate, sum_extraction=False)

        #print('in optimal')

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
        results = self.rectify_spectral_curve(coeffs, widths, x_range, [in_data, flat_data], s_rate, sum_extraction=False, vertical_normal=norm_direction)
        #results = self.rectify_spectral_curve(coeffs, widths, x_range, [in_data, flat_data], s_rate, sum_extraction=False)

        height = sum(results.get('width'))
        width = results.get('dim')[1]
        w_data = np.zeros((1, width))

        s_data = results.get('out_data')[0]
        f_data = results.get('out_data')[1]

        for x in range(0, width):
            w_sum = sum(f_data[:, x])
            d_var = np.full((1, height), 1.0)
            if w_sum != 0.0:
                p_data = f_data[:, x]/w_sum
                num = p_data * s_data[:, x]/d_var
                dem = np.power(p_data, 2)/d_var
                w_data[0, x] = np.sum(num)/np.sum(dem)


        #print(w_data[0, (x_range[0]-2):(x_range[0]+10)])
        result_data = {'y_center': results.get('y_center'),
                       'dim': results.get('dim'),
                       'out_data': w_data,
                       'rectified_trace': s_data,
                       'rectified_flat': f_data
                       }

        return result_data


    def rectify_spectral_curve_by_fractional_sum(self, coeffs, widths, x_range, in_data, flat_data, s_rate=[1, 1], verbose=False, norm_direction=NORMAL):
        results = self.rectify_spectral_curve(coeffs, widths, x_range, [in_data, flat_data], s_rate, sum_extraction=False, vertical_normal=1)

        height = sum(results.get('width'))
        width = results.get('dim')[1]
        w_data = np.zeros((1, width))

        s_data = results.get('out_data')[0]
        f_data = results.get('out_data')[1]

        for x in range(0, width):
            w_sum = sum(f_data[:, x])
            d_var = np.full((1, height), 1.0)
            if w_sum != 0.0:
                p_data = f_data[:, x]/w_sum
                num = p_data * s_data[:, x]/d_var
                dem = np.power(p_data, 2)/d_var
                w_data[0, x] = np.sum(num)/np.sum(dem)

        #print(x_range, w_data[0, (x_range[0]-2):(x_range[0]+10)])
        result_data = {'y_center': results.get('y_center'),
                       'dim': results.get('dim'),
                       'out_data': w_data,
                       'rectified_trace': s_data,
                       'rectified_flat': f_data
                       }

        return result_data


    def rectify_spectral_curve_by_sum(self, coeffs, widths, x_range, in_data, s_rate=[1,1], verbose=False, norm_direction=NORMAL):
        """Straighten the spectral trace and perform the summation on the rectify trace

        Parameters:
            coeffs (array): polynomial coefficient list starting from that of zero order
            widths (array): upper and lower width along the trace
            in_data (array): input data
            s_rate (list): sampling rate from input domain to output domain
            verbose(bool): flag for debug


        Returns:
            spectral_info (dict): straightened spectral information including dimension and the data

        """
        results = self.rectify_spectral_curve(coeffs, widths, x_range, [in_data], s_rate, sum_extraction=False, vertical_normal=norm_direction)
        #result_data = self.rectify_spectral_curve(coeffs, widths, x_range, [in_data], s_rate, False, verbose)
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


    def compute_output_flux(self, input_corners, input_data, input_xdim, input_ydim, verbose=False):
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

        #start=time.time()
        for x in range(x_1, x_2):
            for y in range(y_1, y_2):
                if verbose is True:
                    print('input_data[', y, x,']: ', input_data[y, x])

                if input_data[y, x] != 0.0:
                    #start1 = time.time()
                    new_corners = self.polygon_clipping(input_corners,[[x, y], [x, y+1], [x+1, y+1], [x+1, y]], 4)
                    #start2 = time.time()
                    #print('poly_clipping: ', (start2-start1))
                    area = self.polygon_area(new_corners)
                    #start3 = time.time()
                    #print('area: ', (start3-start2))
                    if verbose is True:
                        print('area: ', area)
                    flux += area * input_data[y, x]

        #end = time.time()
        #print('flux: ', (end-start))
        return flux

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


