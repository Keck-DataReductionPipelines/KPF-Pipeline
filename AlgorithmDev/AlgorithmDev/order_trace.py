"""This module computes echelle order curvature and location empirically from fiber flat exposure """

from __future__ import print_function
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import math
import numpy as np
from numpy.polynomial.polynomial import polyval, polyder
from numpy import sqrt, square
from scipy import linalg, ndimage
from astropy.modeling import models, fitting
import csv

GRATING = 0
PRISM = 1
FLAG_DEBUG = False
TOP = 1
BOTTOM = 0
FIT_G = fitting.LevMarLSQFitter()
FIT_ERROR_TH = 2.5
WIDTH_TH = 7
WIDTH_DEFAULT = 6

class OrderTrace:
    """Class for doing order trace computation

    Parameters:
    -----------
        fits_file (str): path for fits_file related to the fiber flat exposure

    """

    # cross-disperser type


    def __init__(self, fits_file=None):
        self.spectral_file = fits_file
        self.spectral_info = None
        self.clusters_all_y = None


    def set_debug(self, on_off_debug = True):
        if on_off_debug:
            import pdb;pdb.set_trace()

    def load_spectral(self, filename=None):
        """Load spectral trace from the fits file"""

        fits_name = self.spectral_file if filename is None else filename
        if fits_name is None:
            return None

        fits_header = fits.open(fits_name)
        if len(fits_header) < 1:
            return None

        im = fits_header[0].data
        size = np.shape(im)

        spectral_info = {'data': im, 'nx': size[1], 'ny': size[0], 'filepath': filename}
        self.spectral_info = spectral_info
        return spectral_info


    def get_spectral_data(self, filename=None):
        """ get spectral data information

        Parameters:
            filename (str): spectral file name

        Returns:
            data (array), nx (number), ny (number): spectral data and dimension or None

        """

        if filename is not None:
            self.load_spectral(filename)

        if self.spectral_info is None:
            return None
        else:
            return self.spectral_info['data'], self.spectral_info['nx'], self.spectral_info['ny']


    def opt_filter(self, y_data, par, weight=None):
        """ Optimal filtering

        Paramters:
            y_data (array): 1D input data
            par (number): filter width for 1D input data or 2D input data along x axis
            weight (array): if set, same size as y_data and contains values between 0 and 1

        Returns:
            f (array): solution of A^t * f = y_data*weight, A is banded matrix formed based on par & weight

        """

        n = y_data.size

        # 1D array
        if y_data.ndim != 1:
            print('opt_filter handles one dimensional y data only')
            return y_data

        if par < 0:
            return y_data

        wgt = np.reshape(weight, (1, -1)) if weight is not None else np.ones((1, n), dtype=np.float64)[0]

        r = y_data*wgt

        # resolve banded matrix by combining a, b, c, abc*f = r
        a = np.ones((1, n), dtype=np.float64)[0] * (-abs(par))
        b = np.hstack([[wgt[0]+abs(par)], wgt[1:n-1]+2.0*abs(par), [wgt[n-1]+abs(par)]])
        c = a.copy()
        a[0] = c[-1] = 0
        f = linalg.solve_banded((1, 1), np.vstack([a, b, c]), r)

        return f

    def remove_vertical_line(self, x_loc, y_loc, imm, len_th = None):
        im_data, ny, nx = self.get_spectral_data()
        th = ny//100 if len_th is None else len_th

        print('start remove vertical long line, th:', th)

        total_y = list()

        #for cx in range(0, nx):
        #    idx_cx = np.where(x_loc == cx)[0]
        #    total_y.append(np.size(idx_cx))

        v_idx = np.array([439, 440, 446, 447])
        for cx in v_idx:
            imm[:, cx] = 0

        pos = np.where(imm>0)

        return pos[1], pos[0], imm


    def locate_clusters(self, filename=None,  filter=20, mask=1, noise=0.0, remove_vertical=False):
        """ Find cluster pixels from 2D data array

        Parameters:
            filename (str): spectral file name or use the one already loaded
            filter (number): the width of the filter for detection of pixels that belong to local maxima

        Returns:
            cluster_info (dict): result of cluster, like
                                     {'x': array([2, 2, 2, 2],
                                      'y': array([4, 5, 6, 7])}
        """

        im_data, total_col, total_row = self.get_spectral_data(filename)

        if im_data is None:
            return None

        imm = np.zeros((total_row, total_col), dtype=np.uint8)

        for col in range(total_col):
            mm = im_data[:, col] + noise - self.opt_filter(im_data[:, col], filter)
            mm_pos = np.where(mm>0, mm, 0)
            h = 0.5*np.sort(mm_pos)[mm_pos.size//2]
            imm[:, col][mm>(h+1)] = mask

        pos = np.where(imm>0)  # ex: (array([4, 5, 6, 7]), array([2, 2, 2, 2]))

        x = pos[1]
        y = pos[0]

        # image correction for stacked_2fiber_flat.fits (temporary fixing)

        if remove_vertical == True:
            x, y, imm = self.remove_vertical_line(x, y, imm)

        return {'x': pos[1], 'y':pos[0], 'im_map': imm}

    def form_cluster(self, x, y, thres=400):
        """ Take x and y coordinates of pixels from (nx, ny) 2D array and identify individual clusters from pixels.

        Parameters:
            x (array): x coordinates for selected pixels
            y (array): y coordinates for selected pixels
            thres (number): threshold used for determining the cluster

        Returns:
            out (array): if no error, array of the same size as  x and y that contains the cluster number for each pixel.
                         Non-cluster members are marked with 0.
                         if there is an error, array of size 1 with value
                         0: "x and y have different size"
                         -1: "x, y cannot exceed the dimension of the spectral data"


        """

        if x.size != y.size:
            return np.array([0])

        _, nx, ny = self.get_spectral_data()

        if np.max(x) >= nx or np.max(x) >= ny:
            return np.array([-1])

        n = x.size
        index = np.zeros(n, dtype=int)

        # segments_at_cy contains segments of one row and segments_all_y contains segments of all rows
        # segment means consecutive pixels along x axis, the begin and ending index from x or y are set

        segments_all_y = list()

        clusters = list()
        c_no = 0   # starting cluster no.
        cluster_at_cy = list()
        cluster_at_py = list()

        for cy in range(ny):
            if cy%10 == 0:
                print(cy, ' ', end='')
            idx_at_cy = np.where(y == cy)[0]   # idx for y at cy
            if idx_at_cy.size == 0:
                continue

            # find horizontal segments at current y position
            segments_at_cy = list()
            x_at_cy = x[idx_at_cy]

            pidx = idx_at_cy[0]
            px = x_at_cy[0]
            segments_at_cy.append([pidx, pidx])

            for i in range(1, idx_at_cy.size):
                cidx = idx_at_cy[i]
                cx = x_at_cy[i]

                if cx == (px+1):
                    segments_at_cy[-1][1] = cidx
                else:
                    segments_at_cy.append([cidx, cidx])
                pidx = cidx
                px = cx

            joined_clusters = cluster_at_cy + cluster_at_py


            for segment in segments_at_cy:
                seg_x1 = x[segment[0]]
                seg_x2 = x[segment[1]]

                connections = list()

                if len(joined_clusters) != 0:
                    # check if connected with any cluster
                    filter_joined_clusters = list(filter(lambda c: (seg_x2 >= (clusters[c]['x1']-1)) and (seg_x1 <= (clusters[c]['x2']+1)), joined_clusters))

                    for c in filter_joined_clusters:
                        # find the segments with covered y
                        for seg in clusters[c]['segments'][::-1]:
                            if cy > (y[seg[0]]+1):
                                break;
                            if seg_x1 <= (x[seg[1]]+1) and seg_x2 >= (x[seg[0]]-1):
                                connections.append(c)                    # collect cluster connected to crt segment
                                break

                if len(connections) == 0:
                    new_cluster = {'segments': [segment],
                                    'x1': seg_x1,
                                    'x2': seg_x2,
                                    'y1': cy,
                                    'y2': cy}
                    clusters.append(new_cluster)
                    cluster_at_cy.append(len(clusters)-1)
                else:     # merge cluster
                    len_c = len(connections)
                    x1_set = np.zeros(len_c+1, dtype=int)
                    x2_set = np.zeros(len_c+1, dtype=int)
                    first_c = connections[0]
                    x1_set[0] = clusters[first_c]['x1']
                    x2_set[0] = clusters[first_c]['x2']
                    if first_c in cluster_at_py:
                        cluster_at_py.remove(first_c)
                        cluster_at_cy.append(first_c)

                    if len(connections) > 1:
                        for i in range(1, len_c):
                            next_c = connections[i]
                            clusters[first_c]['segments'].extend(clusters[next_c]['segments'])
                            x1_set[i] = clusters[next_c]['x1']
                            x2_set[i] = clusters[next_c]['x2']
                            clusters[next_c]['segments'] = list()
                            if next_c in cluster_at_cy:
                                cluster_at_cy.remove(next_c)
                            else:
                                cluster_at_py.remove(next_c)

                    clusters[first_c]['segments'].append(segment)
                    self.sort_cluster_segments(clusters[first_c]['segments'])
                    x1_set[len_c] = seg_x1
                    x2_set[len_c] = seg_x2
                    clusters[first_c]['x1'] = np.amin(x1_set)
                    clusters[first_c]['x2'] = np.amax(x2_set)
                    clusters[first_c]['y2'] = cy

                joined_clusters = cluster_at_py + cluster_at_cy
            cluster_at_py = cluster_at_cy.copy()
            cluster_at_cy = list()

        cluster_no = 1
        for one_cluster in clusters:
            no_segs = len(one_cluster['segments'])
            if no_segs == 0:
                continue

            # test if one single pixel
            if no_segs == 1 and one_cluster['segments'][0][0] == one_cluster['segments'][0][1]:
                index[one_cluster['segments'][0][0]] = 0
            else:
                total_pixel = 0
                for cluster_seg in one_cluster['segments']:
                    total_pixel += (cluster_seg[1]-cluster_seg[0]+1)
                    if total_pixel > thres:    # get this cluster
                        for c_seg in one_cluster['segments']:
                            index[c_seg[0]:(c_seg[1]+1)] = cluster_no
                        cluster_no += 1
                        break
                if total_pixel <= thres:
                    cluster_no += 0

        nregions = max(index)+1 if min(index) == 0  else  max(index)

        return {'index': index, 'nregiopns': nregions}


    def sort_cluster_segments(self, segments):
        """ sort the segment based on the first location number """

        segments.sort(key = lambda s: s[0])
        return segments

    def sort_cluster_on_loc(self, clusters, loc):
        """ sort the clusters base on the location number """

        clusters.sort(key = lambda c: c[loc])
        return clusters

    def remove_unassigned_cluster(self, x, y, index):
        """ remove the pixel which has no cluster number assigned """

        idx_clustered = np.where(index > 0)[0]   # the pixel which is assigned cluster number
        x_r = x[idx_clustered]                     # x, y coordinate of pixel which is assigned cluster number
        y_r = y[idx_clustered]
        index_r = index[idx_clustered]
        return x_r, y_r, index_r

    def clean_clusters_on_border(self, x, y, index, border_y):

        # find all clusters crossing the bottom boundary
        border_cross = np.where(y == border_y)[0]  # index at bottom boundary

        if border_cross.size > 0:
            border_cross = index[border_cross] # cluster number boundary pixels
            border_cross = np.unique(border_cross) # sorted unique bottom boundary cluster number

            # cluster number at bottom (or top) boundry
            for i in range(border_cross.size):
                if border_cross[i] == 0:     # no cluster number assigned
                    continue
                idx_of_c_num = np.where(index == border_cross[i])[0]          # all pixels at this cluster number
                bind = idx_of_c_num[np.where(y[idx_of_c_num] == border_y)[0]] # all pixels at the boundary of this cluster number
                for ii in range(bind.size):
                    idx_to_remove = idx_of_c_num[np.where(x[idx_of_c_num] == x[bind[ii]])[0]]
                    index[idx_to_remove] = 0
        return index

    def write_data_to_fits(self, out_file, index, x, y, nx, ny):
        imm = self.make_2D_data(index, x, y, nx, ny)
        self.make_fits(imm, out_file)

    def make_2D_data(self, index, x, y, selected_clusters=None):
        """ make 2D data based on cluster number and location

        Parameters:
            x (array): x coordinates for selected pixels
            y (array): y coordinates for selected pixels
            index (array): cluster number on selected pixels
            selected_clusters (array) : make 2D data based on selected clusters only

        Returns:
            out (array): 2D data with pixel set as 1 on the selected clusters
        """

        _,nx, ny = self.get_spectral_data()

        imm = np.zeros((ny, nx), dtype=np.uint8)
        if selected_clusters is None:
            ymin = 0
            ymax = ny-1
        else:
            sel = np.where(np.isin(index, selected_clusters))[0]
            ymin = np.amin(y[sel])
            ymax = np.amax(y[sel])

        for cy in range(ny):
            if cy < ymin:
                continue
            elif cy > ymax:
                break;
            y_cond = np.where(y==cy)[0]
            if selected_clusters is None:
                nz_idx_at_cy = y_cond[np.where(index[y_cond] != 0)[0]]
            else:
                nz_idx_at_cy = y_cond[np.where(np.isin(index[y_cond], selected_clusters))[0]]
            imm[cy, x[nz_idx_at_cy]] = 1
        return imm

    def make_fits(self, data, out_file):
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(out_file, overwrite=True)


    def curve_fitting_on_one_cluster(self, cluster_no, index, x, y, power, poly_info=None):
        """ finding polynomial to fit the cluster points.

        Parameters:
            cluster_no (number): number of cluster
            index (array): cluster number for selected pixels specified by x, y
            x (array): x coordinates of selected pixels
            y (array): y coordinates of selected pixels

        Returns:
            poly_info (array): contains coeffs of fitting polynomial from high order and
                               regular border enclosing selected pixels, minumum x, maximum x, minimum y and maximum y.

        """

        if poly_info is None:
            poly_info = np.zeros(power+5)

        idx_at_order = np.where(index == cluster_no)[0]
        x_set = x[idx_at_order]
        y_set = y[idx_at_order]
        sort_idx = np.argsort(x_set)
        x_sorted = x_set[sort_idx]
        y_sorted = y_set[sort_idx]

        poly_info[0:(power+1)] = np.polyfit(x_sorted, y_sorted, power)
        poly_info[power+1] = np.amin(x_set)
        poly_info[power+2] = np.amax(x_set)
        poly_info[power+3] = np.amin(y_set)
        poly_info[power+4] = np.amax(y_set)

        error = math.sqrt(np.square(np.polyval(poly_info[0:power+1], x_sorted) - y_sorted).mean())
        area = [np.amin(x_set), np.amax(x_set), np.amin(y_set), np.amax(y_set)]

        return poly_info, error, area

    def curve_fitting_on_clusters(self, cluster_nos, index, x, y, power=3, poly_info=None, check_errors=True):
        if poly_info is None:
            poly_info = np.zeros(power+5)

        idx_at_order = np.where(np.isin(index, cluster_nos))[0]
        x_set = x[idx_at_order]
        y_set = y[idx_at_order]
        sort_idx = np.argsort(x_set)
        x_sorted = x_set[sort_idx]
        y_sorted = y_set[sort_idx]
        poly_info[0:(power+1)] = np.polyfit(x_sorted, y_sorted, power)
        poly_info[power+1] = np.amin(x_set)
        poly_info[power+2] = np.amax(x_set)
        poly_info[power+3] = np.amin(y_set)
        poly_info[power+4] = np.amax(y_set)

        errors = math.sqrt(np.square(np.polyval(poly_info[0:power+1], x_set) - y_set).mean()) if check_errors else None

        return poly_info, errors

    def curve_fitting_on_all_clusters(self, index, x, y, power):

        max_index = np.amax(index)
        poly_all = np.zeros((max_index+1, power+5))
        errors = np.zeros(max_index+1)

        for c in range(1, max_index+1):
            poly, error, area = self.curve_fitting_on_one_cluster(c, index, x, y, power)
            poly_all[c, ] = poly
            errors[c] = error

        return poly_all, errors


    def merge_two_clusters(self, cluster_nos,  x, y, index, power):
        """ calculate the error and distance in case 2 clusters are merged, the first is the leftmost one """
        poly_info, errors = self.curve_fitting_on_clusters(cluster_nos, index, x, y, power)
        return poly_info, errors

    def distance_between_clusters(self, cluster_nos, x, y, index):
        """ find the distance between the clusters, the first cluster has smaller min_x """

        end_x = np.zeros(2)
        end_y = np.zeros(2)

        for i in range(2):
            idx_c = np.where(index == cluster_nos[i])[0]
            all_x_c = x[idx_c]
            all_y_c = y[idx_c]
            end_idx = np.argsort(all_x_c)[-1] if i == 0 else np.argsort(all_x_c)[0]
            end_x[i] = all_x_c[end_idx]
            end_y[i] = all_y_c[end_idx]

        dist_x = end_x[1] - end_x[0]  if end_x[1] > end_x[0] else 0
        dist_y = abs(end_y[0]-end_y[1])
        return dist_x, dist_y

    def cross_other_cluster(self, polys, cluster_nos_for_polys, cluster_nos, x, y, index, power, sort_map, merged_coeffs, print_result=False):
        #merge_coeffs contains the coeffs and range in case the two cluster is merged
        min_x = int(merged_coeffs[power+1])
        max_x = int(merged_coeffs[power+2])
        min_y = int(merged_coeffs[power+3])
        max_y = int(merged_coeffs[power+4])

        cluster_idx = np.where(np.logical_or(index == cluster_nos[0], index == cluster_nos[1]))[0]
        cluster_x = x[cluster_idx]
        cluster_y = y[cluster_idx]

        two_curve_x1 = polys[cluster_nos_for_polys[0], power+2]
        two_curve_x2 = polys[cluster_nos_for_polys[1], power+1]

        _, nx, ny = self.get_spectral_data()

        all_x = list()
        all_y = list()

        # find x belonging to curves tested to be merged
        for s_x in range(min_x, max_x+1):
            x_idx = np.where(cluster_x == s_x)[0]
            if x_idx.size > 0:
                all_x.append(s_x)
                all_y.append((np.amin(cluster_y[x_idx])+np.amax(cluster_y[x_idx]))/2)

        # test if two curves connected to each other, no vertical cut gap between
        if two_curve_x2 <= two_curve_x1:
            x_idx = np.where(cluster_x == two_curve_x1)[0]
            y_min_1 = np.amin(cluster_y[x_idx])
            y_max_1 = np.amax(cluster_y[x_idx])
            x_idx = np.where(cluster_x == two_curve_x2)[0]
            y_min_2 = np.amin(cluster_y[x_idx])
            y_max_2 = np.amax(cluster_y[x_idx])

            # vertical position overlapped, or very close
            if y_max_1 >= y_min_2 and y_min_1 <= y_max_2:
                #print(cluster_nos, ' connected to each other 1 ',two_curve_x1, two_curve_x2)
                return False
            elif abs(y_max_1-y_min_2) < WIDTH_TH or abs(y_min_1-y_max_2) < WIDTH_TH:
                #print(cluster_nos, ' connected to each other 2', two_curve_x1, two_curve_x2)
                return False


        total_c = np.shape(polys)[0]

        if print_result is True:
            print('in cross_other_cluster ' + str(len(all_x)))

        # find the horizontal gap
        all_x_idx = np.where((all_x - np.roll(all_x, 1)) >= 2)[0]
        gap_x_idx = list()
        for idx in all_x_idx:
            gap_x_idx.append(idx-1)
            gap_x_idx.append(idx)


        gap_x_idx = np.array(gap_x_idx, dtype=int)

        offset = 0

        for c_idx in range(1, total_c):
            if c_idx == cluster_nos_for_polys[0] or c_idx == cluster_nos_for_polys[1] or polys[c_idx, power+1] == -1:
                continue
            if polys[c_idx, power+3] > max_y:
                break
            elif polys[c_idx, power+4] < min_y:
                continue

            # big gap and the test curve fall between

            #if (two_curve_x2 - two_curve_x1) >= offset and  polys[c_idx, power+2] > two_curve_x1 and polys[c_idx, power+1] < two_curve_x2:
            #    print(' curve ', c_idx, ' from ', sort_map[c_idx], ' in big curve gap ', two_curve_x1, two_curve_x2)
            #    return True

            cross_point = dict()


            # find if tested cluster horizontally ovelaps out of gap ends of two merged clusters
            if polys[c_idx, power+1] <= max_x and polys[c_idx, power+2] >= min_x:
                com_min_x = int(max(polys[c_idx, power+1], min_x))
                com_max_x = int(min(polys[c_idx, power+2], max_x))
                com_list = [com_min_x, com_max_x] if com_min_x != com_max_x else [com_min_x]

                for curve_end in com_list:
                    if curve_end in np.array(all_x)[gap_x_idx]:   # the end point of overlap meet the gap ends
                        continue
                    one_y_val = np.polyval(polys[c_idx, 0:power+1], curve_end)
                    merged_y = np.polyval(merged_coeffs[0:power+1], curve_end)

                    # compare the y location of the tested curve and the merged curves
                    if abs(one_y_val - merged_y) < 1:
                        cross_point[int(curve_end)] = 0
                    else:
                        cross_point[int(curve_end)] = (one_y_val - merged_y)/abs(one_y_val-merged_y)

            vals = np.array([ v for v in cross_point.values()])

            # check if tested curve has short horizontal overlap with merged curves and vertically meets with the merged curves at all gap ends
            if np.size(vals) != 0:
                same_y_count = np.size(np.where(vals == 0)[0])
                com_dist = abs(com_max_x - com_min_x)

                # overlap at one pixel and same y or overlap at short range and same y at two ends of overlap
                if (( same_y_count == 1 and com_dist == 0) or ( same_y_count == 2 and com_dist < 10)):
                    #print(' curve ', c_idx, ' from ', sort_map[c_idx], ' connect to ', cluster_nos)
                    continue

            # check the y location at every gap overlapping with the test curve, cross_point records the y position at selected x positions
            for n_idx in range(0, len(gap_x_idx), 2):
                gap1 = gap_x_idx[n_idx]
                gap2 = gap_x_idx[n_idx+1]

                if (polys[c_idx, power+1] < (all_x[gap2] + offset)) and (polys[c_idx, power+2] > (all_x[gap1] - offset)):
                    two_y_val = np.zeros(2)
                    if all_x[gap2] < polys[c_idx, power+1]:
                        two_y_val = np.polyval(polys[c_idx, 0:power+1], np.array([polys[c_idx, power+1], polys[c_idx, power+1]]))
                    elif all_x[gap1] > polys[c_idx, power+2]:
                        two_y_val = np.polyval(polys[c_idx, 0:power+1], np.array([polys[c_idx, power+2], polys[c_idx, power+2]]))
                    else:
                        end1 = max(all_x[gap1], polys[c_idx, power+1])
                        end2 = min(all_x[gap2], polys[c_idx, power+2])
                        two_y_val = np.polyval(polys[c_idx, 0:power+1], np.array([end1, end2]))
                    for i in [0, 1]:
                        #print(' test curve y: ', two_y_val[i], '  merged_y: ', all_y[gap_x_idx[n_idx+i]], ' gap: ', (all_x[gap2]-all_x[gap1]))
                        if abs(two_y_val[i] - all_y[gap_x_idx[n_idx+i]]) < 1:
                            cross_point[all_x[gap_x_idx[n_idx+i]]] = 0
                        else:
                            cross_point[all_x[gap_x_idx[n_idx+i]]] = (two_y_val[i] - all_y[gap_x_idx[n_idx+i]])/abs(two_y_val[i] - all_y[gap_x_idx[n_idx+i]])


            vals = np.array([ v for v in cross_point.values()])

            zero_total = np.size(np.where(vals == 0)[0])
            positive_zero_total = np.size(np.where(np.logical_or(vals == 1, vals == 0))[0])
            negative_zero_total = np.size(np.where(np.logical_or(vals == -1, vals == 0))[0])

            #if positive_zero_total >=1 positive_zero_total == negative_zero_total and  and zero_total == positive_zero_total:
            #    print('test ', c_idx, ' from ', sort_map[c_idx], ' merged original index: ',  cluster_nos, ' vals: ', vals, ' at points: ', cross_point.keys(), ' x: ', \
            #            polys[c_idx, power+1], polys[c_idx, power+2])
            #    import pdb;pdb.set_trace()

            if print_result is True:
                print('test ', c_idx, ' from ', sort_map[c_idx], ' merged original index: ',  cluster_nos, ' vals: ', vals, ' at points: ', cross_point.keys(), ' x: ', \
                           polys[c_idx, power+1], polys[c_idx, power+2])

            if positive_zero_total >= 1 and negative_zero_total >= 1:
                if print_result is True:
                    print('  ', cluster_nos, ' cross ', c_idx, ' from ', sort_map[c_idx], ' x range: ', polys[c_idx, power+1:power+3])
                return True

        return False



    def merge_fitting_curve(self, poly_curves, power, index, x, y, fixed_curves = None, threshold = FIT_ERROR_TH, print_result=True):
        """ merge the curve which is close to each other by max_change merge """
        x_min_c = power+1
        x_max_c = power+2
        y_min_c = power+3
        y_max_c = power+4

        _, nx, ny = self.get_spectral_data()

        max_order = np.amax(index);

        sort_idx_on_miny = np.argsort(poly_curves[:, y_min_c])
        new_polys = poly_curves[sort_idx_on_miny]

        stop_at = 1
        cluster_changed = 0
        non_exist = -1
        short_curve = nx/2
        reserve_curve = nx//20

        m_height = np.median(poly_curves[:, y_max_c] - poly_curves[:, y_min_c])
        log = ''

        kept_curves = []

        c1 = 1
        while(True):
            if c1 > max_order:
                break
            if cluster_changed >= 1:       # stop at when the number of cluster changed is made
                break
            if fixed_curves is not None:
                if sort_idx_on_miny[c1] in fixed_curves:
                    c1 += 1
                    continue

            if print_result is True:
                print("current test curve: c1: "+ str(c1) + " o_c1: "+ str(sort_idx_on_miny[c1]))
            if new_polys[c1, x_min_c] == non_exist or (new_polys[c1, x_max_c] - new_polys[c1, x_min_c] > short_curve):
                kept_curves.append(sort_idx_on_miny[c1])
                c1 += 1
                continue

            v_neighbors = list()

            y_offset = m_height * 1       # may make smaller
            y_lower = max(new_polys[c1, y_min_c] - y_offset, 0)
            y_upper = min(new_polys[c1, y_max_c] + y_offset, ny-1)
            for c2 in range(1, max_order+1):
                if c1 == c2 or new_polys[c2, x_min_c] == non_exist:
                    continue

                if new_polys[c2, y_min_c] > y_upper:
                    break
                if new_polys[c2, y_max_c] < y_lower:
                    continue

                # skip the curve which is horontally overlapped

                if new_polys[c1, x_min_c] < new_polys[c2, x_min_c]:
                    h_overlap = new_polys[c1, x_max_c] - new_polys[c2, x_min_c]
                else:
                    h_overlap = new_polys[c2, x_max_c] - new_polys[c1, x_min_c]

                if h_overlap < nx/20:        # not overlap too much
                    v_neighbors.append(c2)
                    #print('add ', c2, sort_idx_on_miny[c2], new_polys[c2, x_min_c:y_max_c+1])

            o_c1 = sort_idx_on_miny[c1]
            v_neighbors = np.asarray(v_neighbors)
            errors = np.full(v_neighbors.size, ny*ny, dtype=float)
            merged_poly_info = dict()

            # no vertical neighbor, set the cluster to be 0
            if v_neighbors.size > 0:
                v_neighbors = np.sort(v_neighbors)
                for i in range(v_neighbors.size):
                    o_c2 = sort_idx_on_miny[v_neighbors[i]]
                    merged_poly_info[o_c2], errors[i] = self.merge_two_clusters(np.array([o_c1, o_c2]), x, y, index, power)

            if print_result is True:
                print('neighbors: ', v_neighbors, 'neighbors errors: ', errors)

            # no neighbors or no neighbors qualified to merge
            if (v_neighbors.size == 0  or (v_neighbors.size > 0 and np.amin(errors) > threshold)):
                curve_width = new_polys[c1, x_max_c] - new_polys[c1, x_min_c]
                if curve_width > reserve_curve:
                    pass_center = 1 if (new_polys[c1, x_min_c] < short_curve and new_polys[c1, x_max_c] > short_curve) else 0
                    #print('no neighbor, width: ', curve_width,  ' pass center: ', pass_center)
                    if pass_center == 1:
                        kept_curves.append(sort_idx_on_miny[c1])
                        c1 += 1
                        continue

                index = np.where(index==o_c1, 0, index)
                new_polys[c1, x_min_c] = non_exist
                if print_result is True:
                    print("remove: ", c1, ' from: ', o_c1)
                log += 'remove '+str(o_c1)
                cluster_changed += 1
                c1 += 1
                continue

            c_neighbors = v_neighbors[np.where(errors < threshold)[0]]
            c_neighbors_distance = np.zeros(c_neighbors.size)
            x_dists = np.zeros(c_neighbors.size)
            y_dists = np.zeros(c_neighbors.size)

            cross_neighbor = np.zeros(c_neighbors.size)

            for i in range(c_neighbors.size):
                c2 = c_neighbors[i]
                o_c2 = sort_idx_on_miny[c2]
                cluster_nos = np.array([o_c1, o_c2]) if new_polys[c1, x_min_c] < new_polys[c2, x_min_c] \
                                                     else np.array([o_c2, o_c1])
                dist_x, dist_y = self.distance_between_clusters(cluster_nos, x, y, index)
                c_neighbors_distance[i] = dist_x + dist_y
                x_dists[i] = dist_x
                y_dists[i] = dist_y

                if self.cross_other_cluster(new_polys, np.array([c1, c2]), np.array([o_c1, o_c2]), x, y, index, power, \
                                            sort_idx_on_miny, merged_poly_info[o_c2], print_result=print_result):
                    cross_neighbor[i] = 1
                if print_result is True:
                    print('c2: ', c2,  'from',  o_c2, 'c1: ', o_c1, ' dist: ', dist_x, dist_y, (dist_x+dist_y), cross_neighbor[i])
                    #import pdb;pdb.set_trace()

            neighbor_idx = np.where(np.logical_and(x_dists < nx/2, cross_neighbor == 0))[0]

            if neighbor_idx.size == 0:
                curve_width = new_polys[c1, x_max_c] - new_polys[c1, x_min_c]
                if curve_width > reserve_curve:
                    pass_center = 1 if (new_polys[c1, x_min_c] < short_curve and new_polys[c1, x_max_c] > short_curve) else 0
                    #print('no neighbor, width: ', curve_width,  ' pass center: ', pass_center)
                    if pass_center == 1:
                        kept_curves.append(sort_idx_on_miny[c1])
                        c1 += 1
                        continue
                index = np.where(index==o_c1, 0, index)
                new_polys[c1, x_min_c] = non_exist
                if print_result is True:
                    print("remove: ", c1, ' from: ', o_c1)
                log += 'remove '+str(o_c1)
                cluster_changed += 1
                c1 += 1
                continue


            c_neighbors_distance = c_neighbors_distance[neighbor_idx]
            best_neighbors = c_neighbors[neighbor_idx]
            best_neighbor = best_neighbors[np.argsort(c_neighbors_distance)][0]
            o_c2 = sort_idx_on_miny[best_neighbor]
            index = np.where(index==o_c2, o_c1, index)
            if print_result is True:
                print('merge: ', c1, best_neighbor, ' from: ', o_c1, o_c2)
            log += 'merge '+str(o_c1) + ' and ' + str(o_c2)

            new_polys[c1, x_min_c] = min(new_polys[c1, x_min_c], new_polys[best_neighbor, x_min_c])
            new_polys[c1, x_max_c] = max(new_polys[c1, x_max_c], new_polys[best_neighbor, x_max_c])
            new_polys[c1, y_min_c] = min(new_polys[c1, y_min_c], new_polys[best_neighbor, y_min_c])
            new_polys[c1, y_max_c] = max(new_polys[c1, y_max_c], new_polys[best_neighbor, y_max_c])
            new_polys[best_neighbor, x_min_c] = non_exist
            poly_curves[o_c1, x_min_c:y_max_c+1] = new_polys[c1, x_min_c:y_max_c+1]
            poly_curves[o_c1, 0:power+1] = merged_poly_info[o_c2][0:power+1]
            cluster_changed += 1


        return {'status':  'changed' if cluster_changed >= 1 else 'nochange', 'index': index, 'cluster_changed': cluster_changed, 'log': log,
                'kept_curves': kept_curves, 'log': log}

    def reorganize_index_old(self, index, x, y):
        """ remove pixels with unsigned cluster no and reorder the cluster number """

        new_x, new_y, new_index = self.remove_unassigned_cluster(x, y, index)
        max_index = np.amax(new_index)

        idx = 1
        rnt_index = np.zeros(new_index.size, dtype=int)

        for i in range(1, max_index+1):
            idx_at_no = np.where(new_index == i)[0]
            if idx_at_no.size > 0:
                rnt_index[idx_at_no] = idx
                idx += 1

        return new_x, new_y, rnt_index

    def reorganize_index(self, index, x, y, return_map = False):
        """ remove pixels with unsigned cluster no and reorder the cluster number """

        #import pdb;pdb.set_trace()
        new_x, new_y, new_index = self.remove_unassigned_cluster(x, y, index)
        max_index = np.amax(new_index)
        unique_index = np.sort(np.unique(new_index[1:]))
        full_index = np.arange(1, max_index+2)

        not_exist_idx = full_index[np.where(np.logical_not(np.isin(full_index, unique_index)))[0]]
        if (len(not_exist_idx) == 0):
            if return_map is False:
                return new_index, new_x, new_y
            else:
                return new_index, new_x, new_y, dict(zip(unique_index, unique_index))

        result_index = np.zeros(np.size(unique_index))
        rnt_index = np.copy(new_index)

        offset = 0
        bound1 = -1

        #print("not exist idx: ", not_exist_idx)
        for b in not_exist_idx:
            bound2 = b
            inbound_idx = np.where(np.logical_and(new_index > bound1, new_index < bound2))[0]
            #print('bound: ', bound1, bound2, inbound_idx)
            if (np.size(inbound_idx) != 0):
                rnt_index[inbound_idx] = new_index[inbound_idx] - offset
            offset += 1
            bound1 = bound2

        #print("end index reorganize")
        #import pdb;pdb.set_trace()

        unique_result = np.sort(np.unique(rnt_index[1:]))

        if return_map is False:
            return new_x, new_y, rnt_index
        else:
            return new_x, new_y, rnt_index, dict(zip(unique_index,  unique_result))

    def remove_broken_cluster(self, index, x, y, polys):
        """ remove the cluster which has big opening around the middle """

        data_x_center = self.spectral_info['nx']//2
        max_idx = np.amax(index)
        changed = 0

        for c in range(1, max_idx+1):
            border_idx = np.where(index == c)[0]
            x_border_set = x[border_idx]
            x_before_center = x_border_set[np.where(x_border_set < data_x_center)[0]]
            x_after_center = x_border_set[np.where(x_border_set > data_x_center)[0]]
            x_before = np.amax(x_before_center) if x_before_center.size > 0 else 0
            x_after = np.amin(x_after_center) if x_after_center.size > 0 else (self.spectral_info['nx'] -1)

            if (x_after - x_before) > 20:
                index[border_idx] = 0
                changed = 1

        new_x = x.copy()
        new_y = y.copy()
        new_index = index.copy()

        if changed == 1:
            new_x, new_y, new_index = self.reorganize_index(new_index, new_x, new_y)

        return new_x, new_y, new_index

    def get_sorted_index(self, poly_coeffs, cluster_no, power, x_loc):
        max_idx = np.shape(poly_coeffs)[0]-1

        centers = np.zeros(max_idx+1)
        for c in range(1, max_idx+1):
            centers[c] = np.polyval(poly_coeffs[c, 0:power+1], x_loc)

        center_index = np.argsort(centers)
        idx = np.where(center_index==cluster_no)[0]
        return {'idx': idx[0], 'index_v_pos': center_index}

    def get_spectrum_around_cluster(self, cluster_no, index, x, y, poly_coeffs, power):
        max_idx = np.amax(index)
        if cluster_no < 1 or cluster_no > max_idx:
            return None

        size = np.shape(poly_coeffs)
        cluster_idx = np.where(index==cluster_no)[0]
        if cluster_idx.size == 0:
            return None

        cluster_y = y[cluster_idx]
        cluster_x = x[cluster_idx]
        s_data, nx, ny = self.get_spectral_data()
        sort_clusters = self.get_sorted_index(poly_coeffs, cluster_no, power, (nx//2))
        idx = sort_clusters['idx']
        center_index = sort_clusters['index_v_pos']

        y_min = int(poly_coeffs[center_index[idx-1], power+3]) if idx > 1 else 0
        y_max = int(poly_coeffs[center_index[idx+1], power+4]) if idx < max_idx else (ny-1)

        data = s_data[y_min:y_max+1, :]

        return {'data': data, 'min_y': y_min, 'max_y': y_max, 'cluster_x': cluster_x, 'cluster_y': cluster_y}

    def get_cluster_points(self, polys_coeffs, power):
        """ get cluster points on fitting curve only within min_x and max_x of the cluster """

        s_coeffs = np.shape(polys_coeffs)
        nx = self.spectral_info['nx']
        cluster_points = np.zeros((s_coeffs[0], nx), dtype=int)
        for c in range(1, s_coeffs[0]):
            s_x = int(max(0, polys_coeffs[c, power+1]))
            e_x = int(min(nx, polys_coeffs[c, power+2]+1))
            x_val = np.arange(s_x, e_x, dtype=int)
            cluster_points[c, s_x:e_x] = np.round(np.polyval(polys_coeffs[c, 0:power+1], x_val))

        return cluster_points

    def get_cluster_distance_at_x(self, cluster_pixels, coeffs, power, x_loc=None ):
        x_center = x_loc if x_loc is not None else spectral_info['nx']//2
        y_at_center = cluster_pixels[:, x_center]

        s = np.shape(coeffs)
        x_exist = np.zeros(s[0], dtype=bool)
        x_exist[np.where(np.logical_and(x_center >= coeffs[:, power+1], x_center <= coeffs[:, power+2]))[0]] = 1

        y_sort_idx = np.argsort(y_at_center)
        y_at_center_sorted = y_at_center[y_sort_idx]
        x_exist_sorted = x_exist[y_sort_idx]
        peak_width = np.ones(s[0], dtype=float)*10.0

        # distance between (1st, 2nd), (2nd, 3rd) ... (last to the 2nd, last) clusters
        for c in range(1, s[0]-1):
            if x_exist_sorted[c] and x_exist_sorted[c+1]:
                peak_width[y_sort_idx[c]] = abs(y_at_center_sorted[c+1] - y_at_center_sorted[c])//2

        peak_width[0] = abs(y_at_center_sorted[1])
        peak_width[-1] = peak_width[-2]

        return peak_width


    def get_cluster_distance_all_x(self, cluster_pixels, coeffs, power):
        nx = self.spectral_info['nx']
        s = np.shape(coeffs)

        peak_width_all_x = np.zeros((s[0], nx), dtype=float)
        for x in range(nx):
            peak_width_all_x[:, x] = self.get_cluster_distance_at_x(cluster_pixels, coeffs, power, x)

        return peak_width_all_x

    def get_cluster_peak_pixels(self, cluster_pixels, power, poly_coeffs):
        """ get the peak data location for each cluster along x direction """

        spectral_data, nx, ny = self.get_spectral_data()

        # get distance between two fitting curves (from cluster_pixels)
        size = np.shape(cluster_pixels)
        v_dists = self.get_cluster_distance_at_x(cluster_pixels, poly_coeffs, power, nx//2)
        all_peak_pixels = np.zeros((size[0], size[1]), dtype=int)

        # get peaks for each cluster within min_x and max_x range
        for c in range(1, size[0]):
            s_x = int(poly_coeffs[c, power+1])
            e_x = int(poly_coeffs[c, power+2]+1)
            for x in range(s_x, e_x):
                s_y = int(max((cluster_pixels[c, x] - v_dists[c]), 0))
                e_y = int(min((cluster_pixels[c, x] + v_dists[c]), (ny-1)))+1
                s_data = spectral_data[s_y:e_y, x]
                idx = np.argmax(s_data)
                all_peak_pixels[c, x] = idx+s_y

        return all_peak_pixels

    def curve_fitting_on_peaks(self, crt_coeffs, power):
        """ re-fitting the polynomial on the peak data location around the cluster """

        all_cluster_points = self.get_cluster_points(crt_coeffs, power)
        cluster_pixels_at_peaks = self.get_cluster_peak_pixels(all_cluster_points, power, crt_coeffs)
        s = np.shape(crt_coeffs)

        peak_coeffs = crt_coeffs.copy()
        errors = np.zeros(s[0])

        for c in range(1, s[0]):
            s_x = int(peak_coeffs[c, power+1])
            e_x = int(peak_coeffs[c, power+2]+1)
            x_set = np.arange(s_x, e_x, dtype=int)
            y_set = cluster_pixels_at_peaks[c, s_x:e_x]
            peak_coeffs[c, 0:power+1] = np.polyfit(x_set, y_set, power)
            errors[c] = math.sqrt(np.square(np.polyval(peak_coeffs[c, 0:power+1], x_set) - y_set).mean())

        return {'coeffs': peak_coeffs, 'peak_pixels': cluster_pixels_at_peaks, 'cluster_piexls': all_cluster_points,
                'errors': errors}


    def rms_of_polys(self, poly_coeff1, poly_coeff2, power):
        """ root mean square between two polynomial fitting """

        total_cluster = np.shape(poly_coeff1)[0]-1
        rms = np.zeros(total_cluster+1)
        for  c in range(1, total_cluster+1):
            x_set = np.arange(int(poly_coeff1[c, power+1]), int(poly_coeff1[c, power+2])+1)
            y1_clusters = np.polyval(poly_coeff1[c, 0:power+1], x_set)
            y2_clusters = np.polyval(poly_coeff2[c, 0:power+1], x_set)
            rms[c] = np.sqrt(np.mean((y1_clusters - y2_clusters)**2))
        return rms


    def vertical_slope_at_x(self, loc_x, min_y, max_y):
        s_data, _, ny = self.get_spectral_data()
        data = s_data[:, loc_x]
        slope_2points = np.zeros(max_y - min_y + 1)
        slope_3points = np.zeros(max_y - min_y + 1)

        for y in range(min_y, max_y+1):
            if (y < ny -2):
                slope_2points[y - min_y] = data[y+1] - data[y]
                slope_3points[y - min_y] = (data[y+2] - data[y])/2
            elif (y < ny-1) :
                slope_2points[y - min_y] = data[y+1] - data[y]
                slope_3points[y - min_y] = data[y+1] - data[y]

        slope_2nd = np.zeros(max_y - min_y + 1)
        for y in range(min_y, max_y):
            slope_2nd[y-min_y] = slope_2points[y+1-min_y] - slope_2points[y-min_y] if y < ny-2 else 0

        return {'2point_slope': slope_2points, '3point_slope': slope_3points, 'slope_2nd': slope_2nd}



    def find_background_around(self, cluster_no, poly_coeffs, power, cluster_points):
        """ find the background data before and after the cluster.

        Parameters:
            cluster_no (number): cluster number
            poly_coeffs (array): polynomial fitting data for each cluster
            power (numver): power of polynomial curve
            cluster_points (array): cluster points based on the polynomial fitting

        Returns:
            out (array): background data before and after the cluster along x direction
        """


        data, nx, ny = self.get_spectral_data()
        total_cluster = np.shape(poly_coeffs)[0]-1

        index_pos = self.get_sorted_index(poly_coeffs, cluster_no, power, nx//2)
        crt_idx = index_pos['idx']
        sorted_index = index_pos['index_v_pos']

        # background before peak and after peak
        backgrounds = np.zeros((2, nx))
        prev_idx = crt_idx - 1 if crt_idx > 1 else crt_idx
        next_idx = crt_idx + 1 if crt_idx < total_cluster else crt_idx
        three_clusters = np.array([cluster_no, sorted_index[prev_idx], sorted_index[next_idx]])
        min_x = int(np.amax(poly_coeffs[three_clusters, power+1]))
        max_x = int(np.amin(poly_coeffs[three_clusters, power+2]))

        crt_peak_y = cluster_points[cluster_no]
        prev_peak_y = cluster_points[sorted_index[crt_idx-1]] if crt_idx > 1 else np.zeros(nx, dtype=int)
        next_peak_y = cluster_points[sorted_index[crt_idx+1]] if crt_idx < total_cluster else np.ones(nx, dtype=int)*(ny-1)

        prev_mid = ((crt_peak_y+prev_peak_y)//2).astype(int)
        total_prev_data = (crt_peak_y - prev_peak_y) + 1
        collect_prev_no = (total_prev_data * 0.2).astype(int)//2
        collect_prev_no = np.where(collect_prev_no >= 1, collect_prev_no, 1)

        next_mid = ((crt_peak_y+next_peak_y)//2).astype(int)
        total_next_data = (next_peak_y - crt_peak_y) + 1
        collect_next_no = (total_next_data * 0.2).astype(int)//2
        collect_next_no = np.where(collect_next_no >= 1, collect_next_no, 1)

        for x in range(min_x, max_x+1):
            data_collected = data[prev_mid[x]-collect_prev_no[x]:prev_mid[x]+collect_prev_no[x]+1, x]

            hist, bin_edge = np.histogram(data_collected, bins=4)
            max_hist_idx = np.argmax(hist)
            data_idx = np.where(np.logical_and(data_collected >= bin_edge[max_hist_idx],
                                               data_collected <= bin_edge[max_hist_idx+1]))[0]

            backgrounds[0, x] = np.mean(data_collected[data_idx])

            data_collected = data[next_mid[x]-collect_next_no[x]:next_mid[x]+collect_next_no[x]+1, x]
            hist, bin_edge = np.histogram(data_collected, bins=4)
            max_hist_idx = np.argmax(hist)
            data_idx = np.where(np.logical_and(data_collected >= bin_edge[max_hist_idx],
                                               data_collected <= bin_edge[max_hist_idx+1]))[0]
            backgrounds[1, x] = np.mean(data_collected[data_idx])

        for i in range(0, 2):
            if min_x > 0:
                backgrounds[i, 0:min_x]= backgrounds[i, min_x]
            if max_x < nx-1:
                backgrounds[i, max_x+1:] = backgrounds[i, max_x]
        return backgrounds


    def float_to_string(self, afloat):
        new_str = f"{afloat:.4f}"
        return new_str

    def width_of_cluster(self, cluster_no, poly_coeffs, cluster_points, power):
        """ find the width of the cluster

        Parameters:
            cluster_no (number): cluster number
            poly_coeffs (array): polynomial fitting information of each cluster
            power (number): power of polynomial curve

        Returns:
            out: cluster width information including
                    cluster number, width before and after the cluster along x direction, and one width number before
                    and after the cluster among all width numbers
        """


        pow_width = 4
        spec_data, nx, ny = self.get_spectral_data()
        max_cluster_no = np.shape(poly_coeffs)[0]-1
        center_x = nx//2
        index_pos = self.get_sorted_index(poly_coeffs, cluster_no, power, center_x)

        # index of cluster_no in index_pos list
        idx = index_pos['idx']
        idx_v_post = index_pos['index_v_pos']
        prev_idx = idx - 1 if idx > 1 else idx
        next_idx = idx + 1 if idx < max_cluster_no else idx

        three_clusters = np.array([cluster_no, idx_v_post[prev_idx], idx_v_post[next_idx]])
        min_x = int(np.amax(poly_coeffs[three_clusters, power+1]))
        max_x = int(np.amin(poly_coeffs[three_clusters, power+2]))
        x_range = np.array([min_x, max_x])

        # get background data along the curve of cluster_no at upper and lower sides
        background_data = self.find_background_around(cluster_no, poly_coeffs, power, cluster_points)

        # compute the width along x direction every step
        step = 100
        x_loc1 = np.arange(center_x, int(x_range[1])+1, step)
        x_loc2 = np.arange(center_x-step, int(x_range[0])-1, -step)
        x_loc = np.concatenate((np.flip(x_loc2), x_loc1))
        cluster_width_info = list()
        prev_widths = list()
        next_widths = list()

        for xs in x_loc:
            cluster_y = cluster_points[cluster_no, xs]
            cluster_y_next = cluster_points[idx_v_post[idx+1], xs] if idx < max_cluster_no else ny-1
            cluster_y_prev = cluster_points[idx_v_post[idx-1], xs] if idx > 1 else 0

            if idx == 1 and idx < max_cluster_no:
                cluster_y_prev = max(cluster_y - abs(cluster_y_next - cluster_y), 0)
            if idx == max_cluster_no and idx > 1:
                cluster_y_next = min(cluster_y + abs(cluster_y - cluster_y_prev), ny-1)


            next_mid = min(ny-1, ((cluster_y+cluster_y_next)//2+1))
            prev_mid = max(0, ((cluster_y+cluster_y_prev)//2-1))

            next_dist = min((next_mid - cluster_y)//4, 10)
            prev_dist = min((cluster_y - prev_mid)//4, 10)
            next_mid_data = background_data[TOP, xs]
            prev_mid_data = background_data[BOTTOM, xs]

            slope_coeffs_bound = list()
            # finding width at both sides
            x_set=np.arange(prev_mid, cluster_y+1)
            y_set=spec_data[prev_mid:(cluster_y+1), xs]
            slope_coeff = np.polyfit(x_set, y_set, pow_width)
            slope_der = np.polyder(slope_coeff)
            all_roots = np.roots(slope_der)
            roots=all_roots[np.where(all_roots<cluster_y)[0]].real
            #prev_width = cluster_y - np.amax(roots)
            prev_width = self.find_trace_width(cluster_y, roots, xs)
            prev_widths.append(prev_width)
            slope_coeffs_bound.append({'coeffs':slope_coeff, 'bound':[prev_mid, cluster_y],
                                      'x_set': x_set, 'y_set': y_set})


            x_set = np.arange(cluster_y, next_mid+1)
            y_set = spec_data[cluster_y:(next_mid+1), xs]
            slope_coeff = np.polyfit(x_set, y_set, pow_width)
            slope_der = np.polyder(slope_coeff)
            all_roots = np.roots(slope_der)
            roots=all_roots[np.where(all_roots>cluster_y)[0]].real
            #next_width = np.amin(roots)-cluster_y
            next_width = self.find_trace_width(cluster_y, roots, xs, 1)
            next_widths.append(next_width)
            slope_coeffs_bound.append({'coeffs':slope_coeff, 'bound':[cluster_y, next_mid],
                                       'x_set': x_set, 'y_set': y_set})


            info_at_x = {'x': str(xs), 'y': str(cluster_y),
                    'n_mid': str(next_mid), 'p_mid': str(prev_mid),
                    'backgd0': self.float_to_string(prev_mid_data),
                    'backgd1':self.float_to_string(next_mid_data),
                    'data': self.float_to_string(spec_data[cluster_y, xs]),
                    'width0': self.float_to_string(prev_width), 'width1': self.float_to_string(next_width)}

            next_slope = list()
            for y in range(cluster_y+1, next_mid-next_dist):
                x_set_1 = np.arange(cluster_y, y+1)
                y_set_1 = spec_data[cluster_y:(y+1), xs]
                x_set_2 = np.arange(y, next_mid+1)
                y_set_2 = spec_data[y:(next_mid+1), xs]
                slope_coeff1 = np.polyfit(x_set_1, y_set_1, 1)
                slope_coeff2 = np.polyfit(x_set_2, y_set_2, 1)
                next_slope.append([y, slope_coeff1[0], slope_coeff2[0], spec_data[y, xs]])

            prev_slope = list()
            for y in range(cluster_y-1, prev_mid+prev_dist, -1):
                x_set_1 = np.arange(y, cluster_y+1)
                y_set_1 = spec_data[y:(cluster_y+1), xs]
                x_set_2 = np.arange(prev_mid, y+1)
                y_set_2 = spec_data[prev_mid:(y+1), xs]
                slope_coeff1 = np.polyfit(x_set_1, y_set_1, 1)
                slope_coeff2 = np.polyfit(x_set_2, y_set_2, 1)
                prev_slope.append([y, slope_coeff1[0], slope_coeff2[0], spec_data[y, xs]])

            cluster_width_info.append({'x': xs, 'width_info': info_at_x, 'slopes_next': next_slope,
                                       'slopes_prev': prev_slope, 'slope_coeffs': slope_coeffs_bound })

        avg_pwidth = self.find_val_from_histogram(np.array(prev_widths))
        avg_nwidth = self.find_val_from_histogram(np.array(next_widths))

        self.values_at_width(avg_pwidth, avg_nwidth, cluster_points[cluster_no, center_x], center_x)

        return {'cluster_no': cluster_no,
                'width_info_all_x': cluster_width_info,
                'avg_pwidth': avg_pwidth,
                'avg_nwidth': avg_nwidth,
                'prev_widths': prev_widths,
                'next_widths': next_widths}

    def width_of_cluster_by_gaussian(self, cluster_no, poly_coeffs, cluster_points, power):
        """ find the width of the cluster

        Parameters:
            cluster_no (number): cluster number
            poly_coeffs (array): polynomial fitting information of each cluster
            power (number): power of polynomial curve

        Returns:
            out: cluster width information including
                    cluster number, width before and after the cluster along x direction, and one width number before
                    and after the cluster among all width numbers
        """


        spec_data, nx, ny = self.get_spectral_data()
        max_cluster_no = np.shape(poly_coeffs)[0]-1
        center_x = nx//2
        index_pos = self.get_sorted_index(poly_coeffs, cluster_no, power, center_x)

        # index of cluster_no in index_pos list
        idx = index_pos['idx']
        idx_v_post = index_pos['index_v_pos']
        prev_idx = idx - 1 if idx > 1 else idx
        next_idx = idx + 1 if idx < max_cluster_no else idx

        three_clusters = np.array([cluster_no, idx_v_post[prev_idx], idx_v_post[next_idx]])
        min_x = int(np.amax(poly_coeffs[three_clusters, power+1]))
        max_x = int(np.amin(poly_coeffs[three_clusters, power+2]))

        x_range = np.array([min_x, max_x])

        # get background data along the curve of cluster_no at upper and lower sides
        background_data = self.find_background_around(cluster_no, poly_coeffs, power, cluster_points)

        # compute the width along x direction every step
        step = 100
        x_loc1 = np.arange(center_x, int(x_range[1])+1, step)
        x_loc2 = np.arange(center_x-step, int(x_range[0])-1, -step)
        x_loc = np.concatenate((np.flip(x_loc2), x_loc1))
        cluster_width_info = list()
        prev_widths = list()
        next_widths = list()
        prev_centers = list()
        next_centers = list()

        for xs in x_loc:
            cluster_y = cluster_points[cluster_no, xs]
            cluster_y_next = cluster_points[idx_v_post[idx+1], xs] if idx < max_cluster_no else ny-1
            cluster_y_prev = cluster_points[idx_v_post[idx-1], xs] if idx > 1 else 0

            if idx == 1 and idx < max_cluster_no:
                cluster_y_prev = max(cluster_y - abs(cluster_y_next - cluster_y), 0)
            if idx == max_cluster_no and idx > 1:
                cluster_y_next = min(cluster_y + abs(cluster_y - cluster_y_prev), ny-1)


            next_mid = min(ny-1, ((cluster_y+cluster_y_next)//2+1))
            prev_mid = max(0, ((cluster_y+cluster_y_prev)//2-1))

            next_dist = min((next_mid - cluster_y)//4, 10)
            prev_dist = min((cluster_y - prev_mid)//4, 10)
            next_mid_data = background_data[TOP, xs]
            prev_mid_data = background_data[BOTTOM, xs]

            slope_coeffs_bound = list()
            # finding width at both sides
            x_set=np.arange(prev_mid, cluster_y+1)
            y_set=spec_data[prev_mid:(cluster_y+1), xs]
            new_x_set, new_y_set = self.mirror_data(x_set, y_set, 1)
            gaussian_fit_prev, prev_width, prev_center = self.fit_width_by_gaussian(new_x_set, new_y_set, cluster_y, xs)
            prev_widths.append(prev_width)
            prev_centers.append(prev_center)
            slope_coeffs_bound.append({'gaussian':gaussian_fit_prev, 'bound':[prev_mid, cluster_y],
                                                  'x_set': x_set, 'y_set': y_set})


            x_set = np.arange(cluster_y, next_mid+1)
            y_set = spec_data[cluster_y:(next_mid+1), xs]
            new_x_set, new_y_set = self.mirror_data(x_set, y_set, 0)
            gaussian_fit_next, next_width, next_center = self.fit_width_by_gaussian(new_x_set, new_y_set, cluster_y, xs)
            next_widths.append(next_width)
            next_centers.append(next_center)
            slope_coeffs_bound.append({'gaussian':gaussian_fit_next, 'bound':[cluster_y, next_mid],
                                                   'x_set': x_set, 'y_set': y_set})

            info_at_x = {'x': str(xs), 'y': str(cluster_y),
                    'x_set': x_set, 'y_set': y_set,
                    'n_mid': str(next_mid), 'p_mid': str(prev_mid),
                    'backgd0': self.float_to_string(prev_mid_data),
                    'backgd1':self.float_to_string(next_mid_data),
                    'data': self.float_to_string(spec_data[cluster_y, xs]),
                    'width0': self.float_to_string(prev_width), 'width1': self.float_to_string(next_width)}

            next_slope = list()
            for y in range(cluster_y+1, next_mid-next_dist):
                x_set_1 = np.arange(cluster_y, y+1)
                y_set_1 = spec_data[cluster_y:(y+1), xs]
                x_set_2 = np.arange(y, next_mid+1)
                y_set_2 = spec_data[y:(next_mid+1), xs]
                slope_coeff1 = np.polyfit(x_set_1, y_set_1, 1)
                slope_coeff2 = np.polyfit(x_set_2, y_set_2, 1)
                next_slope.append([y, slope_coeff1[0], slope_coeff2[0], spec_data[y, xs]])

            prev_slope = list()
            for y in range(cluster_y-1, prev_mid+prev_dist, -1):
                x_set_1 = np.arange(y, cluster_y+1)
                y_set_1 = spec_data[y:(cluster_y+1), xs]
                x_set_2 = np.arange(prev_mid, y+1)
                y_set_2 = spec_data[prev_mid:(y+1), xs]
                slope_coeff1 = np.polyfit(x_set_1, y_set_1, 1)
                slope_coeff2 = np.polyfit(x_set_2, y_set_2, 1)
                prev_slope.append([y, slope_coeff1[0], slope_coeff2[0], spec_data[y, xs]])

            cluster_width_info.append({'x': xs, 'width_info': info_at_x, 'slopes_next': next_slope,
                                       'slopes_prev': prev_slope, 'slope_coeffs': slope_coeffs_bound })

        #print(prev_widths)
        #import pdb;pdb.set_trace()
        cluster_h = poly_coeffs[cluster_no, power+4] - poly_coeffs[cluster_no, power+3]
        avg_pwidth = self.find_val_from_histogram(np.array(prev_widths), range=[0, cluster_h], bin_no=int(cluster_h//WIDTH_TH), cut_at=WIDTH_DEFAULT)
        avg_nwidth = self.find_val_from_histogram(np.array(next_widths), range=[0, cluster_h], bin_no=int(cluster_h//WIDTH_TH), cut_at=WIDTH_DEFAULT)

        self.values_at_width(avg_pwidth, avg_nwidth, cluster_points[cluster_no, center_x], center_x)

        return {'cluster_no': cluster_no,
                'width_info_all_x': cluster_width_info,
                'avg_pwidth': avg_pwidth,
                'avg_nwidth': avg_nwidth,
                'prev_widths': prev_widths,
                'next_widths': next_widths,
                'prev_centers': prev_centers,
                'next_centers': next_centers}


    def mirror_data(self, x_set, y_set, mirror_side):
        total = np.size(x_set) - 1

        if mirror_side == 0:   #left side
            x_other_side = x_set[0:total]-total
            y_other_side = np.flip(y_set[1:])
            x_new_set = np.concatenate((x_other_side, x_set))
            y_new_set = np.concatenate((y_other_side, y_set))
        else:                  # ride side
            x_other_side = x_set[1:]+total
            y_other_side = np.flip(y_set[0:total])
            x_new_set = np.concatenate((x_set, x_other_side))
            y_new_set = np.concatenate((y_set, y_other_side))

        #import pdb;pdb.set_trace()
        return x_new_set, y_new_set

    def fit_width_by_gaussian(self, x_set, y_set, center_y, xs):
        g_init = models.Gaussian1D(mean=center_y)
        gaussian_fit = FIT_G(g_init, x_set, y_set)

        #print('gaussian_fit: ', gaussian_fit, ' center_y: ', center_y)
        if abs(gaussian_fit.mean.value - center_y) <= 1.0:
            #width = gaussian_fit.stddev.value*3.0
            width = gaussian_fit.stddev.value*2.0  # 2 sigma
            gaussian_center = gaussian_fit.mean.value

            #if width >= 12.0:
            #    width = 6.0
            #    gaussian_fit = None
        else:
            gaussian_center = gaussian_fit.mean.value
            print("center offset at ", xs, ' is: ',  abs(gaussian_fit.mean.value - center_y))
            #width = 6.0
            #gaussian_fit = None

        #print('width: ', width)F
        #import pdb;pdb.set_trace()
        return gaussian_fit, width, gaussian_center

    def fit_width_by_trap(self, x_set, y_set, center_y):
        t_init = models.Trapezoid1D(x_0=center_y)
        trape_fit = FIT_G(t_init, x_set, y_set)

        #print('trape_fit: ', trape_fit, ' center_y: ', center_y)
        #import pdb;pdb.set_trace()
        if abs(trape_fit.x_0.value - center_y) <= 1.0:
            width = trape_fit.width.value*3.0
            if width >= 12.0:
                width = 6.0
                trape_fit = None
        else:
            width = 6.0
            trape_fit = None

        #print('width: ', width)
        #import pdb;pdb.set_trace()
        return trape_fit, width


    def find_trace_width(self, center_y, roots, xloc, direction = 0):
        ratio = 0.3
        default_width = 6.0

        data, nx, ny = self.get_spectral_data()
        peak = data[center_y, xloc]
        sorted_roots = np.sort(roots)
        if direction == 0:
            sorted_roots = np.flip(sorted_roots)

        if direction == 0:
            for r in sorted_roots:
                yi = math.floor(r)
                val = data[yi, xloc]
                if val < peak * ratio:
                    return (center_y - r)
        if direction == 1:
            for r in sorted_roots:
                yi = math.ceil(r)
                val = data[yi, xloc]
                if val < peak * ratio:
                    return (r - center_y)

        return default_width


    def values_at_width(self, avg_pwidth, avg_nwidth, center_y, xloc):
        data, nx, ny = self.get_spectral_data()
        y1 = max(math.floor(center_y - avg_pwidth), 0)
        y2 = min(math.ceil(center_y + avg_nwidth), ny-1)
        peak = data[center_y, xloc]
        v1 = data[y1, xloc]
        v2 = data[y2, xloc]

        print("peak: ", peak, 'p_val/ratio: ', self.float_to_string(v1), '/', self.float_to_string(v1/peak),
                        ' n_val/ratio: ', self.float_to_string(v2), '/', self.float_to_string(v2/peak))
        return




    def find_val_from_histogram(self, vals, bin_no=4, range=None, cut_at=None):
        """ pick the value based on the histogram """

        if range is None:
            r = None
        else:
            r = (range[0], range[1])

        hist, bin_edge = np.histogram(vals, bins=bin_no, range=r)

        if np.size(np.where(hist != 0)[0]) == 0:
            if cut_at is None:
                mean_val = np.mean(vals)
            else:
                mean_val = cut_at
            return mean_val

        max_hist_idx = np.argmax(hist)
        if (max_hist_idx > 1 and (hist[max_hist_idx-1] > hist[max_hist_idx]*0.6)):
            edge_min = bin_edge[max_hist_idx-1]
        else:
            edge_min = bin_edge[max_hist_idx]

        if (max_hist_idx < (len(hist)-1) and (hist[max_hist_idx+1] > hist[max_hist_idx]*0.6)):
            edge_max = bin_edge[max_hist_idx+2]
        else:
            edge_max = bin_edge[max_hist_idx+1]


        data_idx = np.where(np.logical_and(vals >= edge_min,
                                           vals <= edge_max))[0]

        mean_val = np.mean(vals[data_idx])

        if cut_at is not None and mean_val > cut_at:
            mean_val = cut_at
        return mean_val


    def find_cluster_stats_from_histogram(self, vals, bin_no=4):
        """ pick the value based on the histogram """

        hist, bin_edge = np.histogram(vals, bins=bin_no)
        cluster_data_stats = list()
        total_range = np.size(hist)

        for i in range(total_range):
            cluster_idx = np.where(np.logical_and(vals >= bin_edge[i], vals < bin_edge[i+1]))[0]
            stat = {'range': np.array([bin_edge[i], bin_edge[i+1]]),
                    'total': hist[i],
                    'clusters': cluster_idx}
            cluster_data_stats.append(stat)

        return cluster_data_stats

    def common_member(self, a, b):
        a_set = set(a)
        b_set = set(b)
        if (a_set & b_set):
            return True
        else:
            return False


    def form_cluster2(self, x, y, thres=None):
        clusters_all_y = self.collect_clusters(x, y)
        index_info = self.remove_cluster_noise(clusters_all_y, x, y)
        return index_info

    def get_cluster_size(self, c_id, index, x, y):
        crt_idx = np.where(index == c_id)[0]
        crt_x = x[crt_idx]
        crt_y = y[crt_idx]
        total_pixel = np.size(crt_idx)
        w = 0
        h = 0
        if total_pixel > 0:
            w = np.amax(crt_x) - np.amin(crt_x) + 1
            h = np.amax(crt_y) - np.amin(crt_y) + 1

        return w, h, total_pixel, crt_idx

    def remove_cluster_noise(self, clusters_endy_dict, x_index, y_index, thres=None):
        """ remove noisy cluster based on the total pixel number and the size and assign cluster number """
        _, nx, ny = self.get_spectral_data()
        w_thres = nx//100
        h_thres = ny//800
        if thres is None:
             thres = h_thres * w_thres

        index = np.zeros(x_index.size, dtype=int)
        cluster_no = 1
        for y in range(ny):
            if (y not in clusters_endy_dict) or (len(clusters_endy_dict[y]) == 0):
                continue
            clusters = clusters_endy_dict[y]

            c_idx = 0
            for one_cluster in clusters:
                total_pixel = 0;
                for y in range(one_cluster['y1'], one_cluster['y2']+1):
                    if (len(one_cluster[y]['segments']) == 0):
                        continue

                    # count stops at the segment of some y
                    for cluster_seg in one_cluster[y]['segments']:
                        total_pixel += (cluster_seg[1]-cluster_seg[0]+1)
                        if total_pixel > thres:    # keep this cluster
                            break

                    if total_pixel > thres:
                        for y in range(one_cluster['y1'], one_cluster['y2']+1):
                            for c_seg in one_cluster[y]['segments']:
                                index[c_seg[0]:(c_seg[1]+1)] = cluster_no
                        cluster_no += 1
                        break
                c_idx += 1

        # remove narrow cluster
        max_idx = np.amax(index)
        h_size_thres = ny//100
        w_size_thres = nx//100
        for c_id in np.arange(1, max_idx+1):
            w, h, t_p, crt_cluster_idx = self.get_cluster_size(c_id, index, x_index, y_index)

            if (w <= w_size_thres and h <= h_size_thres):
                index[crt_cluster_idx] = 0
                print('cluster ', c_id, ' total: ', t_p, ' w, h', w, h, ' => remove')
            else:
                print('cluster ', c_id, ' total: ', t_p, ' w, h', w, h)
        nregions = np.amax(index)+1 if np.amin(index) == 0  else  np.amax(index)

        return {'index': index, 'nregiopns': nregions}



    def collect_clusters(self, x, y):
        """ Take x and y coordinates of pixels from (nx, ny) 2D array and identify individual clusters from pixels.

        Parameters:
            x (array): x coordinates for selected pixels
            y (array): y coordinates for selected pixels
            thres (number): threshold used for determining the cluster

        Returns:
            out (dict): {y1: clusters (array), <y1+1>: clusters (array).....}, where clusters ends at y.
                        cluster(dict): 'x1', 'x2', 'y1', 'y2', <y>, where y in [y1, y2]
                        value of <y> is dict, like {'segments': <sorted_segment>(array)},
                                                    where each segment contain starting and ending index in x

        """

        if x.size != y.size:
            return np.array([0])

        _, nx, ny = self.get_spectral_data()

        if np.max(x) >= nx or np.max(x) >= ny:
            return None

        # clusters_endy_dict contains key/value, like y(cluster with max_y at y)/list of cluster
        # cluster: a dict with properties: y1, y2, x1, x2, number(y1), number(y1+1), ...number(y2),
        #                                  value of number(y1) is like {"segments": [seg1, seg2,...]}
        #                                  where segx: [idx_1, idx_2] from index in x, y

        clusters_endy_dict = dict()      # contain clusters end at y (0 to ny-1)
        nx_prev_cluster_id = [list() for cx in range(nx)]

        for cy in range(ny):
            #if cy%10 == 0:
            print(cy, ' ', end='')

            #import pdb;pdb.set_trace()
            idx_at_cy = np.where(y == cy)[0]   # idx for y at cy

            clusters_endy_dict[cy] = list()

            if idx_at_cy.size == 0:
                continue

            # segments_at_cy: segments at each y
            # seg_to_cluster_map: seg vs. connected cluster

            # find horizontal segments at current y position
            segments_at_cy = self.get_segments_from_index_list(idx_at_cy, x)

            # first y or no cluster found at previous y
            if (cy == 0) or len(clusters_endy_dict[cy-1]) == 0:
                c_idx = 0
                for seg in segments_at_cy:
                    clusters_endy_dict[cy].append({cy: {'segments':[seg]},
                                                  'y1': cy,
                                                  'y2': cy,
                                                  'x1': x[seg[0]],
                                                  'x2': x[seg[1]]})
                    x1 = max(x[seg[0]]-1, 0)
                    x2 = min(x[seg[1]]+2, nx)
                    for cx in range(x1, x2):
                        nx_prev_cluster_id[cx].append(c_idx)
                    c_idx += 1

                continue

            # segment vs. connected cluster
            # cluster vs. connected segment
            seg_to_cluster_map = {}
            cluster_to_update = list()

            # each element contains connected clusters ('cluster_idx') and connected segments ('segment_idx')
            connected_set = list()

            #import pdb;pdb.set_trace()
            # associate clusters of previous y with each segment
            clusters_at_py = clusters_endy_dict[cy-1]
            for s_idx in range(len(segments_at_cy)):
                seg_to_cluster_map[s_idx] = list()
                seg_x1 = x[segments_at_cy[s_idx][0]]
                seg_x2 = x[segments_at_cy[s_idx][1]]
                p_cluster_idx = list()
                for cx in range(seg_x1, seg_x2+1):
                    p_cluster_idx.extend(nx_prev_cluster_id[cx])
                seg_to_cluster_map[s_idx] = list(set(p_cluster_idx))

                """
                for c_idx in range(len(clusters_at_py)):
                    cluster = clusters_at_py[c_idx]
                    if seg_x2 < (cluster['x1']-1):
                        break;
                    if seg_x1 > (cluster['x2']+1):
                        continue
                    segs_at_py = cluster[cy-1]

                    for seg in segs_at_py['segments']:
                        if (seg_x2 < (x[seg[0]]-1)):
                            break
                        if (seg_x1 > (x[seg[1]]+1)):
                            continue

                        # found segment s_idx overlap with cluster c_idx

                        if s_idx in seg_to_cluster_map:
                            seg_to_cluster_map[s_idx].append(c_idx)
                        else:
                            seg_to_cluster_map[s_idx] = [c_idx]
                        break

                """

            #import pdb;pdb.set_trace()
            # create new cluster for current y from isolated segment and cluster unit containing associated segements and
            # clusters of previous y

            cluster_at_crt_y = list()
            for s_idx in range(len(segments_at_cy)):
                if len(seg_to_cluster_map[s_idx]) == 0:    # no connected cluster
                    cluster = {cy: {'segments':[segments_at_cy[s_idx]]},
                               'y1': cy,
                               'y2': cy,
                               'x1': x[segments_at_cy[s_idx][0]],
                               'x2': x[segments_at_cy[s_idx][1]]}
                    cluster_at_crt_y.append(cluster)
                else:
                    connected_clusters = seg_to_cluster_map[s_idx]

                    b_conn = -1

                    # find connected unit which has any of the clusters connected with current segment
                    if self.common_member(cluster_to_update, connected_clusters):
                        for conn_idx in range(len(connected_set)):
                            one_conn = connected_set[conn_idx]
                            if self.common_member(connected_clusters, one_conn['cluster_idx']):
                                b_conn = conn_idx
                                break

                    cluster_to_update.extend(connected_clusters)
                    if (b_conn == -1):
                        new_conn = {'segment_idx': [s_idx], 'cluster_idx': connected_clusters}
                        connected_set.append(new_conn)
                    else:
                        if s_idx not in connected_set[b_conn]['segment_idx']:
                            connected_set[b_conn]['segment_idx'].append(s_idx)
                        for c in connected_clusters:
                             if c not in connected_set[b_conn]['cluster_idx']:
                                connected_set[b_conn]['cluster_idx'].append(c)

            #import pdb;pdb.set_trace()
            # create new cluster based on each element in the cluster unit, connected_set
            for conn in connected_set:
                all_segments = dict()
                min_x = min([clusters_at_py[c_idx]['x1'] for c_idx in conn['cluster_idx']])
                max_x = max([clusters_at_py[c_idx]['x2'] for c_idx in conn['cluster_idx']])
                min_y = min([clusters_at_py[c_idx]['y1'] for c_idx in conn['cluster_idx']])
                max_y = cy
                for y_i in range(min_y, max_y+1):
                    all_segments[y_i] = list()

                for c_idx in conn['cluster_idx']:
                    conn_cluster = clusters_at_py[c_idx]
                    for y_i in range(conn_cluster['y1'], conn_cluster['y2']+1):
                        all_segments[y_i].extend(conn_cluster[y_i]['segments'])

                for s_idx in conn['segment_idx']:
                    all_segments[cy].append(segments_at_cy[s_idx])

                new_cluster = {}
                for y_i in range(min_y, max_y+1):
                    sorted_segment = self.sort_cluster_segments(all_segments[y_i])
                    new_cluster[y_i] = {'segments': sorted_segment}

                new_cluster['x1'] = min_x
                new_cluster['x2'] = max_x
                new_cluster['y1'] = min_y
                new_cluster['y2'] = max_y
                cluster_at_crt_y.append(new_cluster)

            #import pdb;pdb.set_trace()
            cluster_at_crt_y =  self.sort_cluster_on_loc(cluster_at_crt_y, 'x1')
            clusters_endy_dict[cy] = cluster_at_crt_y

            nx_prev_cluster_id = [list() for cx in range(nx)]
            for c_idx in range(len(cluster_at_crt_y)):
                cluster = cluster_at_crt_y[c_idx]
                segments = cluster[cy]['segments']
                for seg in segments:
                    x1 = max(x[seg[0]]-1, 0)
                    x2 = min(x[seg[1]]+2, nx)
                    for cx in range(x1, x2):
                        nx_prev_cluster_id[cx].append(c_idx)

            cluster_to_update = list(set(cluster_to_update))
            cluster_to_update.sort(reverse=True)
            for c in cluster_to_update:
                clusters_endy_dict[cy-1].pop(c)

            #import pdb;pdb.set_trace()
            #print("finish ", (cy))

        self.clusters_all_y = clusters_endy_dict
        print('\n')
        return clusters_endy_dict


    def extract_order_from_cluster(self, cluster_no, index, x, y, power):
        p_info, error, area= self.curve_fitting_on_one_cluster(cluster_no, index, x, y, power)

        return p_info, error, area


    def get_segments_from_index_list(self, id_list, loc):
        segments = list()

        distcont_idx = np.where((loc[id_list] - loc[np.roll(id_list, 1)]) != 1)[0]
        # collect all segments
        p_idx = id_list[0]
        segments.append([p_idx, p_idx])

        for d_idx in distcont_idx[1:]:
            if (id_list[d_idx - 1]) > p_idx:
                segments[-1][1] = id_list[d_idx-1]
            p_idx = id_list[d_idx]
            segments.append([p_idx, p_idx])

        if distcont_idx[-1] < id_list[-1]:
            segments[-1][1] =  id_list[-1]

        return segments

    def remove_noise_in_cluster(self, cluster_curves, x_index, y_index, crt_cluster_idx, power, thres=None):
        _, nx, ny = self.get_spectral_data()
        w_thres = nx//100
        h_thres = ny//800
        if thres is None:
            thres = h_thres * w_thres *2/3

        h_size_thres = ny//100
        w_size_thres = nx//100
        index = np.zeros(x_index.size, dtype=int)
        cluster_no = 0
        #print(len(cluster_curves), h_size_thres, w_size_thres, thres)
        poly_fitting_results = dict()
        crt_cluster_x = x_index[crt_cluster_idx]
        crt_cluster_y = y_index[crt_cluster_idx]

        for curve_idx in range(len(cluster_curves)):
            curve = cluster_curves[curve_idx]
            #print('curve_idx: ', curve_idx, ' w: ', (curve['start_x']-curve['crt_x']+1), ' h: ', (curve['y2'] - curve['y1']+1))
            total_pixel = 0

            if (curve['start_x']-curve['crt_x']) < w_size_thres and (curve['y2'] - curve['y1'] ) < h_size_thres:
               continue

            #total_crt_x = dict()

            x_set = list()
            y_set = list()
            print('curve: ', curve_idx, ' width: ', str(curve['start_x']-curve['crt_x']), ' height: ', str(curve['y2'] - curve['y1'] ))
            for x_loc in range(curve['crt_x'], curve['start_x']+1):
                segs_in_y = curve[x_loc]
                #total = 0
                for seg_y in segs_in_y:
                    total_pixel += (seg_y[1]-seg_y[0]+1)
                    #total += (seg_y[1]-seg_y[0]+1)
                    for s_y in range(seg_y[0], seg_y[1]+1):
                        x_set.append(x_loc)
                        y_set.append(s_y)
                #total_crt_x[x_loc] = total
            if total_pixel < thres:
                print('  total pixel: ', total_pixel, ' => less pixel')
                continue
            else:
                print('  total pixel: ', total_pixel, ' => polyfit test')
                x_ary = np.array(x_set)
                y_ary = np.array(y_set)
                sort_idx = np.argsort(x_ary)
                x_ary = x_ary[sort_idx]
                y_ary = y_ary[sort_idx]
                coeffs = np.polyfit(x_ary, y_ary, power)
                errors = math.sqrt(np.square(np.polyval(coeffs, x_ary) - y_ary).mean())
                print("  errors: ", errors)
                #if (errors > FIT_ERROR_TH and total_pixel < thres) or (errors >= 3.0):
                if errors > FIT_ERROR_TH:
                    continue

            cluster_no += 1
            #print("start reset cluster")
            for x_loc in range(curve['crt_x'], curve['start_x']+1):
                segs_in_y = curve[x_loc]
                total = 0
                for seg_y in segs_in_y:
                    #print('seg_y: ', seg_y)
                    #import pdb;pdb.set_trace()
                    y_log = np.logical_and(crt_cluster_y >= seg_y[0], crt_cluster_y <= seg_y[1])
                    set_idx = crt_cluster_idx[np.where(np.logical_and(y_log, crt_cluster_x==x_loc))[0]]
                    index[set_idx] = cluster_no
                    #y_range = np.arange(seg_y[0], (seg_y[1]+1), dtype=int)
                    #set_idx2 = np.where(np.logical_and(np.isin(y_index, y_range), x_index==x_loc))[0]
                    #total += np.size(set_idx)
                #if (total != total_crt_x[x_loc]):
                #    print(total, total_crt_x[x_loc])
                #    import pdb;pdb.set_trace()
            poly_fitting_results[cluster_no] = {'errors': errors, 'coeffs': coeffs,
                                                'area':[np.amin(x_ary), np.amax(x_ary), np.amin(y_ary), np.amax(y_ary)]}

        return index, poly_fitting_results

    def handle_noisy_cluster(self, index_t, x, y, power, num_set):
        """ handle the cluster which is not well fitted by polynomial curve

        Paramters:
            index_t (array): array containing assigned cluster index
            x (array): array containing x location
            y (array): arry containing y location
            num_set(array): the first element is the cluster to be handled

        Returns:
            new_index_t: updated version of index_x
            status (dict): {'msg': 'delete'/'change'/'split',
                            'cluster_id': <target_cluster_id>,
                            'cluster_added': [n1, n2...,] (for 'split'),
                            'poly_fitting':{<cluster_id>: {'errors': error, 'coeffs': poly_coeffs (array), 'area': <area>(array)}
                                            <added_cluster_id_1>: {'errors': .., 'coeffs': ..., 'area': ...},
                                            <added_cluster_id_n>: {'errors': .... }}}
        """

        CURVE_TH = WIDTH_TH
        crt_cluster_idx = np.where(index_t == num_set[0])[0]
        crt_cluster_x = x[crt_cluster_idx]
        crt_cluster_y = y[crt_cluster_idx]
        ymin = np.amin(crt_cluster_y)
        ymax = np.amax(crt_cluster_y)
        xmin = np.amin(crt_cluster_x)
        xmax = np.amax(crt_cluster_x)
        xi = xmax
        crt_col_idx = np.where(crt_cluster_x==xi)[0]

        pre_segments_y = self.get_segments_from_index_list(crt_col_idx, crt_cluster_y)

        curve_records = dict()
        curve_records[xi] = list()

        for seg in pre_segments_y:
            crt_seg_y = crt_cluster_y[seg]
            if (crt_seg_y[1]-crt_seg_y[0]) <= CURVE_TH:
                curve = {'start_x': xi, 'crt_x': xi, 'y1': crt_seg_y[0], 'y2': crt_seg_y[1],
                          xi: [[crt_seg_y[0], crt_seg_y[1]]]}
                curve_records[xi].append(curve)

        self.sort_cluster_on_loc(curve_records[xi], 'y1')

        # extend or add new curve
        for xi in range(xmax-1, xmin-1, -1):
            #print(xi, end=" ")
            curve_records[xi] = list()
            crt_col_idx = np.where(crt_cluster_x == xi)[0]
            if crt_col_idx.size == 0:
                continue
            crt_segments_y = self.get_segments_from_index_list(crt_col_idx, crt_cluster_y)

            pre_curves = curve_records[xi+1]
            curves_to_upgrade=list()

            pre_curves_to_crt_seg_map = dict()
            crt_seg_to_pre_curves_map = dict()

            for idx in range(len(pre_curves)):
                pre_curves_to_crt_seg_map[idx] = list()
            for idx in range(len(crt_segments_y)):
                crt_seg_to_pre_curves_map[idx] = list()

            for crt_seg_idx in range(len(crt_segments_y)):
                found_curve = False
                crt_seg = crt_segments_y[crt_seg_idx]
                crt_seg_y = crt_cluster_y[crt_seg]
                if (crt_seg_y[1] - crt_seg_y[0]) > CURVE_TH*2:
                    print('skip on long segment: x, y1, y1 => ', xi, crt_seg_y[0], crt_seg_y[1])
                    continue

                for c_idx in range(len(pre_curves)):
                    #if crt_seg_y[1] < (pre_curves[c_idx]['y1']-1):
                    if crt_seg_y[1] < (pre_curves[c_idx]['y1']):
                        break
                    #if crt_seg_y[0] > (pre_curves[c_idx]['y2']+1):
                    if crt_seg_y[0] > (pre_curves[c_idx]['y2']):
                        continue

                    p_curve_y = pre_curves[c_idx][xi+1]

                    #if crt_seg_y[0] <= (p_curve_y[-1][1]+1) and crt_seg_y[1] >= (p_curve_y[0][0]-1):
                    if crt_seg_y[0] <= (p_curve_y[-1][1]) and crt_seg_y[1] >= (p_curve_y[0][0]):
                        if crt_seg_idx >= 1 and (c_idx in crt_seg_to_pre_curves_map[crt_seg_idx-1]):
                            pre_seg_y = crt_cluster_y[crt_segments_y[crt_seg_idx-1]]
                            if (crt_seg_y[0] - pre_seg_y[1]) >= CURVE_TH or (crt_seg_y[1] - pre_seg_y[0]) >= CURVE_TH*2:
                                #print('neighboring segments are far apart: ', 'space distance: ', str(crt_seg_y[0]-pre_seg_y[1]), 'total length: ', str(crt_seg_y[1] - pre_seg_y[0]),  'at ', str(xi))
                                continue
                        found_curve = True
                        pre_curves_to_crt_seg_map[c_idx].append(crt_seg_idx)
                        crt_seg_to_pre_curves_map[crt_seg_idx].append(c_idx)

                if found_curve is False:
                    if (crt_seg_y[1] - crt_seg_y[0]) <= CURVE_TH:
                        curve = {'start_x': xi, 'crt_x': xi, 'y1': crt_seg_y[0], 'y2': crt_seg_y[1],
                                  xi: [[crt_seg_y[0], crt_seg_y[1]]]}
                        curve_records[xi].append(curve)

            # create curve unit to contain connected curves and segments
            curve_units = list()
            already_processed = set()

            for c_idx in range(len(pre_curves)):
                if (len(pre_curves_to_crt_seg_map[c_idx]) == 0):
                    continue;
                curves_to_upgrade.append(c_idx)

                if c_idx in already_processed:
                    continue
                curve_set = set([c_idx])
                curve_set_len = len(curve_set)
                segs_set = set(pre_curves_to_crt_seg_map[c_idx])
                segs_set_len = len(segs_set)
                while(True):
                    for s in segs_set:
                        curve_set.update(crt_seg_to_pre_curves_map[s])
                    if curve_set_len == len(curve_set):
                        break
                    else:
                        curve_set_len = len(curve_set)
                    for c in curve_set:
                        segs_set.update(pre_curves_to_crt_seg_map[c])
                    if (segs_set_len == len(segs_set)):
                        break
                    else:
                        segs_set_len = len(segs_set)
                curve_units.append({'p_curves': list(curve_set), 'c_segs': list(segs_set)})
                already_processed.update(curve_set)

            if len(curve_units) == 0:
                continue

            # create new curve for each curve unit
            for c_unit in curve_units:
                start_x = max([pre_curves[c]['start_x'] for c in c_unit['p_curves']])
                new_curve = {'start_x': start_x, 'crt_x': xi}

                y1_list = [pre_curves[c]['y1'] for c in c_unit['p_curves']]
                y2_list = [pre_curves[c]['y2'] for c in c_unit['p_curves']]


                all_segs = [crt_cluster_y[crt_segments_y[s]].tolist() for s in c_unit['c_segs']]
                self.sort_cluster_on_loc(all_segs, 0)

                new_curve[xi] = all_segs
                y1_list.append(all_segs[0][0])
                y2_list.append(all_segs[-1][1])
                new_curve['y1'] = min(y1_list)
                new_curve['y2'] = max(y2_list)

                # merge segment from all curves along x axis
                for xc in range(xi+1, start_x+1):
                    all_pre_segs = list()
                    for c_idx in c_unit['p_curves']:
                        if xc in pre_curves[c_idx]:
                            all_pre_segs.extend(pre_curves[c_idx][xc])
                    self.sort_cluster_on_loc(all_pre_segs, 0)
                    new_curve[xc] = all_pre_segs

                curve_records[xi].append(new_curve)


            if len(pre_curves) > 0 and len(curves_to_upgrade) > 0:
                curves_to_upgrade.sort(reverse=True)
                for c_idx in curves_to_upgrade:
                    curve_records[xi+1].pop(c_idx)

            if len(curve_records[xi]) > 0:
                self.sort_cluster_on_loc(curve_records[xi], 'y1')


        crt_last_index = np.amax(index_t)
        new_index_t = index_t.copy()
        new_index_t[crt_cluster_idx] = 0
        all_curves_in_cluster = list()
        for xi in range(xmin, xmax+1):
            all_curves_in_cluster.extend(curve_records[xi])

        print('removing noise on (', str(len(all_curves_in_cluster)), ' curves)')
        index_in_cluster, poly_fitting = self.remove_noise_in_cluster(all_curves_in_cluster, x, y, crt_cluster_idx, power)

        #print('after removal: ', index_in_cluster[crt_cluster_idx], ' num_set[0]:', num_set[0])
        max_new_index = np.amax(index_in_cluster)
        inc = max_new_index-1

        # status properties: cluster_id, poly_fitting: {<cluster_no>: {errors, coeffs}, ..},
        #                    msg, cluster_added:[...]
        status = {'cluster_id': num_set[0], 'poly_fitting': dict(), 'cluster_added': list()}

        if inc < 0:
            status['msg'] = 'delete'
        elif inc == 0:
            status['msg'] = 'change'
        else:
            status['msg'] = 'split'
            status['cluster_added'].extend(list(range(crt_last_index+1, crt_last_index+inc+1)))

        # for 'change' and 'split'
        for c in range(1,max_new_index+1):
            set_idx = np.where(index_in_cluster == c)[0]
            if c == 1:
                new_index_t[set_idx] = num_set[0]
                status['poly_fitting'][num_set[0]] = poly_fitting[c]
            else:
                added_id = crt_last_index+c-1
                new_index_t[set_idx] = added_id
                status['poly_fitting'][added_id] = poly_fitting[c]

        return new_index_t, status


    def advanced_cluster_cleaning_handler(self, index_t, x, y, power, start_cluster = None, stop_cluster=None):
        """ advanced processing in case the total number of clusters are far more than the order trace """

        index_p = index_t.copy()
        x_p = x.copy()
        y_p = y.copy()

        next_idx = 1 if start_cluster is None else start_cluster
        all_status = dict()
        original_max_idx = np.amax(index_t) if stop_cluster is None else stop_cluster

        _, nx, ny = self.get_spectral_data()

        while (True):
            p_info, errors, area = self.extract_order_from_cluster(next_idx, index_p, x_p, y_p, power)
            c_set = [next_idx]
            cluster_img = self.make_2D_data(index_p, x_p, y_p, c_set)
            xmin = area[0]
            xmax = area[1]
            ymin = area[2]
            ymax = area[3]

            if errors <= FIT_ERROR_TH:
                status = {'msg': 'same', 'cluster_id': next_idx,
                          'poly_fitting': {next_idx: {'errors': errors, 'coeffs': p_info, 'area': area}}}
                index_new = index_p.copy()
            else:
                index_new, status = self.handle_noisy_cluster(index_p, x_p, y_p, power, [next_idx])

            all_status[next_idx] = status
            print('idx: ', next_idx, ' status: ', status)

            next_idx = next_idx+1 if next_idx < original_max_idx else None
            if next_idx is not None:
                index_p = index_new.copy()
                continue
            else:
                return index_new, all_status

    def one_step_merge_cluster(self, crt_coeffs, power, crt_index, crt_x, crt_y, print_result=False):
            merge_status = self.merge_fitting_curve(crt_coeffs, power, crt_index, crt_x, crt_y, print_result=print_result)

            if merge_status['status'] != 'nochange':
                next_x, next_y, next_index, convert_map = self.reorganize_index(merge_status['index'], crt_x, crt_y, True)

                new_polys = np.zeros((np.amax(next_index)+1, np.shape(crt_coeffs)[1]))
                for c_id in convert_map.keys():
                    new_polys[convert_map[c_id], :] = crt_coeffs[c_id, :]
                return  next_index, next_x, next_y, new_polys, merge_status
            else:
                return crt_index, crt_x, crt_y, crt_coeffs, merge_status

    def merge_clusters (self, index, x, y, power):
        new_index = index.copy()
        new_x = x.copy()
        new_y = y.copy()
        new_coeffs, errors = self.curve_fitting_on_all_clusters(new_index, new_x, new_y, power)

        while(True):
            n_index, n_x, n_y, n_coeffs, merge_status = self.one_step_merge_cluster(new_coeffs, power, new_index, new_x, new_y)

            new_index = n_index.copy()
            new_x = n_x.copy()
            new_y = n_y.copy()
            new_coeffs = n_coeffs.copy()

            if merge_status['status'] == 'nochange':
                break

        m_x, m_y, m_index = self.reorganize_index(new_index, new_x, new_y)
        m_coeffs, errors = self.curve_fitting_on_all_clusters(m_index, m_x, m_y, power)

        return m_x, m_y, m_index, m_coeffs

    def find_all_cluster_widths(self, index_t, x_t, y_t, coeffs, cluster_points, power, cluster_set=None):
        new_index = index_t.copy()
        new_x = x_t.copy()
        new_y = y_t.copy()
        max_cluster_no = np.amax(new_index)
        cluster_coeffs = coeffs.copy()
        cluster_widths = list()

        if cluster_set is None:
            cluster_set = list(range(1, max_cluster_no+1))

        for n in cluster_set:
            print('cluster: ', n)
            if n < 1 or n > max_cluster_no or (np.where(index_t==n)[0]).size == 0:
                cluster_widths.append({'avg_nwidth': WIDTH_DEFAULT, 'avg_pwidth': WIDTH_DEFAULT})
                continue
            #ext_spectrum = self.get_spectrum_around_cluster(n, new_index, new_x, new_y, cluster_coeffs, power)
            #if ext_spectrum is not None:
            cluster_width_info = self.width_of_cluster_by_gaussian(n, cluster_coeffs, cluster_points, power)
            cluster_widths.append({'avg_nwidth': cluster_width_info['avg_nwidth'], 'avg_pwidth': cluster_width_info['avg_pwidth']})
            print('top width: ', cluster_width_info['avg_nwidth'], ' bottom width: ', cluster_width_info['avg_pwidth'])

        return cluster_widths


    def sort_cluster_in_y(self, cluster_coeffs, power):
        total_cluster = np.shape(cluster_coeffs)[0]-1
        _, nx, ny = self.get_spectral_data()
        c_x = nx/2
        min_x = np.amax(cluster_coeffs[1:total_cluster+1, power+1])
        max_x = np.amin(cluster_coeffs[1:total_cluster+1, power+2])

        if min_x > max_x:
            return None;

        c_x = min(max(nx//2, min_x), max_x)

        y_pos = np.zeros(total_cluster+1)
        for i in  range(1, total_cluster+1):
            y_pos[i] = np.polyval(cluster_coeffs[i, 0:power+1], c_x)

        sorted_index = np.argsort(y_pos)
        return sorted_index

    def write_cluster_info_to_csv(self, cluster_widths, cluster_coeffs, power, csvfile):
        sorted_index = self.sort_cluster_in_y(cluster_coeffs, power)

        with open(csvfile, mode='w') as result_file:
            result_writer = csv.writer(result_file)
            for i in range(1, len(sorted_index)):
                id = sorted_index[i]
                c_widths = cluster_widths[id-1]
                prev_width = c_widths['avg_pwidth']
                next_width = c_widths['avg_nwidth']

                row_data = list()
                for i in range(power, -1, -1):
                    row_data.append(cluster_coeffs[id, i])
                row_data.append(self.float_to_string(prev_width))
                row_data.append(self.float_to_string(next_width))
                row_data.append(int(cluster_coeffs[id, power+1]))
                row_data.append(int(cluster_coeffs[id, power+2]))

                result_writer.writerow(row_data)

    def extract_order_trace(self, power):
        imm_spec, nx, ny = self.get_spectral_data()
        r_v = True if 'stacked_2fiber_flat' in self.spectral_file else False

        print("locate cluster")
        # locate cluster
        cluster_xy = self.locate_clusters(remove_vertical = r_v)

        # assign cluster id and do basic cleaning
        print("assign cluster")
        cluster_info = self.collect_clusters(cluster_xy['x'], cluster_xy['y'])
        clean_cluster_info = self.remove_cluster_noise(cluster_info, cluster_xy['x'], cluster_xy['y'])
        x, y, index_c = self.reorganize_index(clean_cluster_info['index'], cluster_xy['x'], cluster_xy['y'])

        # advanced cleaning
        print("advance clean cluster")
        advanced_index, all_status = self.advanced_cluster_cleaning_handler(index_c, x, y, power)
        new_x, new_y, new_index = self.reorganize_index(advanced_index, x, y)

        # clean clusters along bottom and top border
        print("clean border")
        index_b = self.clean_clusters_on_border(new_x, new_y, new_index, 0)
        index_t = self.clean_clusters_on_border(new_x, new_y, index_b, ny-1)
        x_border, y_border, index_border =  self.reorganize_index(index_t, new_x, new_y)

        print("merge cluster")
        merge_x, merge_y, merge_index, merge_coeffs = self.merge_clusters(index_border, x_border, y_border, power)
        c_x, c_y, c_index = self.remove_broken_cluster(merge_index, merge_x, merge_y, merge_coeffs)
        cluster_coeffs, errors = self.curve_fitting_on_all_clusters(c_index, c_x, c_y, power)
        cluster_points = self.get_cluster_points(cluster_coeffs, power)

        #peak_info = self.curve_fitting_on_peaks(cluster_coeffs, power)
        #cluster_coeffs = peak_info['coeffs']
        #cluster_points = peak_info['peak_piexls']

        print("find widths")
        all_widths = self.find_all_cluster_widths(c_index, c_x, c_y, cluster_coeffs,  cluster_points, power)
        return {'cluster_index': c_index, 'cluster_x': c_x, 'cluster_y': c_y, 'widths': all_widths,
                'coeffs': cluster_coeffs, 'errors': errors}
