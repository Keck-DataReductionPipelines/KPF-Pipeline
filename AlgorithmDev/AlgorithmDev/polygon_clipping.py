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

class PolygonClipping:
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
            'NAXIS2': 'ydim',
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

        new_in_data = np.zeros((cure_size[0], cure_size[1]))
        for y in range(0, cure_size[0]):
            for x in range(0, cure_size[1]):
                new_in_data[y, x] = in_data[y, x] - cure_data[y, x]

        return new_in_data

    def correct_data_by_mask(self, mask_data, in_data):
        mask_size = np.shape(mask_data)
        data_size = np.shape(in_data)
        if mask_size != data_size:
            return;

        new_in_data = np.copy(in_data)

        for r in range(0, mask_size[0]):
            for c in range(0, mask_size[1]):
                if mask_data[r, c] != 0:
                    continue

                num = 0
                total = 0.0
                c1 = max(c-1, 0)
                c2 = min(c+1, mask_size[1]-1)
                r1 = max(r-1, 0)
                r2 = min(r+1, mask_size[0]-1)

                for y in range(r1, r2+1):
                    for x in range(c1, c2+1):
                        if mask_data[y, x] == 1:
                            num = num+1
                            total += in_data[y, x]
                            #print("y = ", y, " x = ", x, " data = ", in_data[y, x], " num = ", num, " total =", total)

                if num != 0:
                    new_in_data[r, c] = total/num
                    print("new data at r, c", r, c, " is ", new_in_data[r, c])
                else:
                    print("num = ", num)

        return new_in_data



    def load_csv_file(self, filename, power = None, delimit=','):
        rows = []
        widths = []

        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=delimit)

            for row in csv_reader:
                if power is not None:
                    row_num = [float(n) for n in row[0:power+1]]
                    if len(row) > (power+1):
                        width_num = [float(n) for n in row[power+1:]]
                        if len(width_num) == 1:
                            width_num.append(width_num[0])
                    else:
                        width_num = [self.order_width, self.order_width]
                else:
                    row_num = [float(n) for n in row]
                    width_num = [self.order_width, self.order_width]

                rows.append(row_num)
                widths.append(width_num[0:2])


        return rows, widths

    def load_simple_spectral_info(self):
        """Load simple spectral trace from the fits file"""
        header_key_map = {
            'NAXIS1': 'xdim',
            'NAXIS2': 'ydim',
        }
        COEFF = 'COEFF'

        if self.spectral_curve_file is None:
            return None

        fits_header = fits.open(self.spectral_curve_file)
        if len(fits_header) < 1:
            return None


        spectral_info = {'data': fits_header[0].data}
        header = fits_header[0].header
        header_keys = list(header.keys())

        coeffs = []
        n_idx = len(COEFF)

        for k in header_keys:
            if k in header_key_map.keys():
                spectral_info[header_key_map.get(k)] = int(header[k])
            elif k.startswith(COEFF):
                coeffs.insert(int(k[n_idx:]), header[k])

        spectral_info.update({'coeffs': coeffs})

        return spectral_info


    def get_data_around_trace(self, coeffs, in_data, s_rate=[1, 1], verbose=False):
        """Get the data around the trace"""
        s = np.shape(in_data)
        input_xdim = s[1]
        input_ydim = s[0]
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
        y_mid = []                                              # spectral trace value at mid point
        x_step = []                                             # x step in input domain
        x_output_step = list(range(0, output_xdim+1))           # x step in output domain, both ends are included

        for o_x in x_output_step:
            crt_x = self.get_input_pos(o_x, sampling_rate[X])
            x_step.insert(o_x, crt_x)
            y_mid.insert(o_x, polyval(crt_x, coeffs))

        import pdb;pdb.set_trace()
        v_border = [max(y_mid), min(y_mid)]
        if verbose is True:
            print('v_border: ', v_border)

        # the vertical position to locate the output spectral
        v_mid = self.get_output_pos(math.floor((v_border[0]+v_border[1])/2), sampling_rate[Y])
        if verbose is True:
            print('v_mid: ', v_mid)

        # import pdb; pdb.set_trace()
        y_upper = min(math.ceil(max(y_mid) + self.order_width), input_ydim)
        y_lower = max(math.floor(min(y_mid) - self.order_width), 0)
        y_width = y_upper - y_lower;

        out_data = np.zeros((y_width, input_xdim))

        for o_y in range(y_lower, y_upper):
            loc_y = o_y - y_lower
            for o_x in range(0, input_xdim):
                out_data[loc_y, o_x] = in_data[o_y, o_x]

        return out_data

    def rectify_spectral_curve(self, coeffs, widths, in_data, s_rate=[1, 1], sum_extraction=True, verbose=False):
        """Straighten the spectral trace

        Parameters:
            coeffs (list): polynomial coefficient list starting from that of zero order
            in_data (array): input data
            s_rate (list): sampling rate from input domain to output domain
            sum_extraction(bool): flag to indicate if performing summation on rectified curve or not
            verbose(bool): flag for debug


        Returns:
            spectral_info (dict): straightened spectral information including dimension and the data

        """

        s = np.shape(in_data)
        input_xdim = s[1]
        input_ydim = s[0]
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
        y_mid = []                                              # spectral trace value at mid point
        x_step = []                                             # x step in input domain
        y_norm_step = []                                        # curve norm along x in input domain
        x_output_step = list(range(0, output_xdim+1))           # x step in output domain, both ends are included

        #if verbose is True:
        #   print('x_output_step=', x_output_step)

        for o_x in x_output_step:
            crt_x = self.get_input_pos(o_x, sampling_rate[X])
            x_step.insert(o_x, crt_x)
            y_mid.insert(o_x, polyval(crt_x, coeffs))
            y_norm_step.insert(o_x, self.poly_normal(crt_x, coeffs, sampling_rate[Y]))

        #print('y_mid: ', y_mid)
        v_border = [max(y_mid), min(y_mid)]
        if verbose is True:
            print('v_border: ', v_border)

        # the vertical position to locate the output spectral
        v_mid = self.get_output_pos(math.floor((v_border[0]+v_border[1])/2), sampling_rate[Y])
        if verbose is True:
            print('v_mid: ', v_mid)

        input_upper_corners = []   # corners above the curve, row based list
        input_lower_corners = []   # corners below the curve, row based list

        for n in range(2):
            output_width = int(self.get_output_pos(widths[n], sampling_rate[Y]))  # width of output
            if n == 1:
                upper_width = min(output_width, output_ydim - 1 - v_mid)
            else:
                lower_width = min(output_width, v_mid)
        if verbose is True:
            print('width at output: ', upper_width, lower_width)

        corners_at_mid = [[x, y] for x, y in zip(x_step, y_mid)]
        if verbose is True:
            print('corners_at_mid: ', corners_at_mid)

        y_size = 1 if sum_extraction is True else (upper_width+lower_width)
        out_data = np.zeros((y_size, output_xdim))

        input_upper_corners.insert(0, corners_at_mid.copy())
        input_lower_corners.insert(0, corners_at_mid.copy())

        for o_y in range(1, upper_width+1):
            next_upper_corners = [self.go_vertical(input_upper_corners[o_y-1][o_x], y_norm_step[o_x], 1) for
                                        o_x in x_output_step]
            input_upper_corners.insert(o_y, next_upper_corners)

        for o_y in range(1, lower_width+1):
            next_lower_corners = [self.go_vertical(input_lower_corners[o_y-1][o_x], y_norm_step[o_x], -1) for
                                        o_x in x_output_step]
            input_lower_corners.insert(o_y, next_lower_corners)

        for o_x in x_output_step[0:-1]:
            if o_x%100 == 0:
                print(o_x, end=" ")
            for o_y in range(0, upper_width):
                input_corners = [input_upper_corners[o_y+c[0]][o_x+c[1]] for c in [[0, 0], [1, 0], [1, 1], [0, 1]]]
                flux = self.compute_output_flux(input_corners, in_data, input_xdim, input_ydim, False)
                if sum_extraction is True:
                    out_data[0, o_x] += flux
                    #if verbose is True:
                    #    print(o_y, o_x, flux, out_data[0, o_x])
                else:
                    out_data[lower_width+o_y, o_x] = flux

            for o_y in range(0, lower_width):
                input_corners = [input_lower_corners[o_y+c[0]][o_x+c[1]] for c in [[1, 0], [0, 0], [0, 1], [1, 1]]]
                flux = self.compute_output_flux(input_corners, in_data, input_xdim, input_ydim, False)
                if sum_extraction is True:
                    out_data[0, o_x] += flux
                else:
                    out_data[lower_width-o_y-1, o_x] = flux

            if verbose is True:
                print('[%d %.2f]' % (o_x, out_data[0, o_x]), end=' ')

        print(' ')
        if verbose is True:
            print(' ')

        result_data = {'y_center': v_mid,
                       'width': [upper_width, lower_width],
                       'dim': [output_ydim, output_xdim],
                       'out_data': out_data}

        return result_data

    def rectify_spectral_curve_by_optimal(self, coeffs, widths, in_data, flat_data, s_rate=[1, 1], verbose=False):
        print('in optimal')
        import pdb;pdb.set_trace()
        s_result = self.rectify_spectral_curve(coeffs, widths, in_data, s_rate, sum_extraction=False)
        f_result = self.rectify_spectral_curve(coeffs, widths, flat_data, s_rate, sum_extraction=False)

        height = sum(s_result.get('width'))
        width = s_result.get('dim')[1]
        w_data = np.zeros((1, width))

        s_data = s_result.get('out_data')
        f_data = f_result.get('out_data')

        for x in range(0, width):
            w_sum = sum(f_data[:, x])
            #w_data_tmp = [ s_data[y, x] * f_data[y, x]/w_sum for y in range(0, height)]
            w_data[0, x] = sum([ s_data[y, x] * f_data[y, x]/w_sum if w_sum != 0.0 else 0.0 for y in range(0, height)])

        print(w_data[0, 1000:1500])
        import pdb;pdb.set_trace()
        result_data = {'y_center': s_result.get('y_center'),
                       'dim': s_result.get('dim'),
                       'out_data': w_data
                       }

        return result_data

    def rectify_spectral_curve_by_optimal2(self, coeffs, widths, in_data, flat_data, s_rate=[1, 1], verbose=False):
            s_result = self.rectify_spectral_curve(coeffs, widths, in_data, s_rate, sum_extraction=False)
            f_result = self.rectify_spectral_curve(coeffs, widths, flat_data, s_rate, sum_extraction=False)

            height = sum(s_result.get('width'))
            width = s_result.get('dim')[1]
            w_data = np.zeros((1, width))

            s_data = s_result.get('out_data')
            f_data = f_result.get('out_data')

            max_data = np.amax(s_data)
            min_data = np.amin(s_data)
            print('max_data: ', max_data, "min_data: ", min_data)

            for x in range(0, width):
                #v = [ math.sqrt(s_data[y, x]) for y in range(0, height)]
                w_sum = sum(f_data[:, x])
                #d_var = [ math.sqrt(s_data[y, x] - min_data) for y in range(0, height)]
                d_var = [1.0 for y in range(0, height)]
                p_data = [ f_data[y, x]/w_sum for y in range(0, height)]
                num = [ p_data[y] * (s_data[y, x]) /d_var[y] for y in range(0, height)]
                dem = [ math.pow(p_data[y], 2)/ d_var[y] for y in range(0, height)]
                w_data[0, x] = sum(num)/sum(dem)

            print(w_data[0, 1000:1500])
            import pdb;pdb.set_trace()

            result_data = {'y_center': s_result.get('y_center'),
                           'dim': s_result.get('dim'),
                           'out_data': w_data,
                           'rectified_trace': s_data,
                           'rectified_flat': f_data
                           }

            return result_data


    def rectify_spectral_curve_by_sum(self, coeffs, in_data, s_rate=[1,1], verbose=False):
        """Straighten the spectral trace and perform the summation on the rectify trace

        Parameters:
            coeffs (list): polynomial coefficient list starting from that of zero order
            in_data (array): input data
            s_rate (list): sampling rate from input domain to output domain
            verbose(bool): flag for debug


        Returns:
            spectral_info (dict): straightened spectral information including dimension and the data

        """

        result_data = self.rectify_spectral_curve(coeffs, in_data, s_rate, True, verbose)
        if verbose is True:
            print('rectify curve: ', result_data)

        return self.sum_curve(result_data.get('out_data'), verbose)


    def sum_curve(self, flat_data, verbose=False):
        """Sum a band of spectral trace

        Parameters:
            flat_data(array): flatten trace
            verbose(bool): flag for debug

        Returns:
            out_data(array): sum extraction of one rectified flat

        """

        sum_data = np.sum(flat_data, axis=0)
        if verbose is True:
            print('sum curve: ', sum_data)

        s = np.shape(sum_data)
        out_data = np.zeros((1, s[0]))
        for x in range(0, s[0]):
            out_data[0, x] = sum_data[x]
        return out_data

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

    def poly_normal(self, pos_x, coeffs, sampling_rate = 1):
        """ Calculate the normal at the specified x position per vertical sampling rate"""

        d_coeff = polyder(coeffs)
        tan = polyval(pos_x, d_coeff)
        v_norm = [-tan, 1]/(sampling_rate * sqrt(tan*tan+1))
        return v_norm

    def get_input_pos(self, output_pos, s_rate):
        """ Get associated position at input domain per output position and sampling rate"""

        return output_pos/s_rate

    def get_output_pos(self, input_pos, s_rate):
        """ get the associated output position per input position and sampling rate"""

        return math.floor(input_pos*s_rate)     # position at output cell domain is integer based

    def go_vertical(self, pos, norm, direction = 1):
        """ Get positions of next row along the normal direction at each position"""

        new_pos = [ p + direction*n for p, n in zip(pos, norm)]
        return new_pos

    def is_zero_coverage(self, corners, input_data):
        """ Check if all corners are located at zero valued cells"""

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
            input_corners(list): corners at input domain
            input_data(array): input data
            input_xdim(number): width of input data
            input_ydim(number): height of input data
            verbose(bool): flag for debug

        Returns:
            flux(number): flux value

        """

        if verbose is True:
            print('input_corners: ', input_corners)

        x_list = [input_corners[i][X] for i in range(0, 4)]
        y_list = [input_corners[i][Y] for i in range(0, 4)]
        x_1 = min(x_list)
        x_2 = max(x_list)
        y_1 = min(y_list)
        y_2 = max(y_list)

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
                    new_corners = self.polygon_clipping(input_corners, [[x, y], [x, y+1], [x+1, y+1], [x+1, y]], 4)
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

        new_poly_points = copy.deepcopy(poly_points)
        for i in range(clipper_size):
            k = (i+1)%clipper_size

            new_poly_points = self.clip(new_poly_points, clipper_points[i][0], clipper_points[i][1],
                                   clipper_points[k][0], clipper_points[k][1])

        return self.remove_duplicate_point(new_poly_points)

    def remove_duplicate_point(self, corners):
        """ Remove the duplicate points from a list of corner points"""

        new_corners = []
        for c in corners:
            if c not in new_corners:
                new_corners.append(c)

        return new_corners

    def polygon_area(self, corners):
        """ Calculate the polygon area per polygon corners"""

        polygon_size = len(corners)
        area = 0.0
        for i in range(polygon_size):
            k = (i+1)%polygon_size
            area += corners[i][0]*corners[k][1] - corners[k][0]*corners[i][1]

        return abs(area)/2


    def clip(self, poly_points, x1, y1, x2, y2):
        """ Polygon clipping"""

        new_points = []
        poly_size = len(poly_points)

        for i in range(poly_size):
            k = (i+1)%poly_size
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


    def line_intersect(self, x1, y1, x2, y2, x3, y3, x4, y4):
        den = (x1-x2)*(y3-y4) - (x3-x4)*(y1-y2)
        num_x = (x1*y2 - x2*y1) * (x3 - x4) - (x1 - x2) * (x3*y4 - x4*y3)
        num_y = (x1*y2 - x2*y1) * (y3 - y4) - (y1 - y2) * (x3*y4 - x4*y3)
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

              
