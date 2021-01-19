import numpy as np
import json
from scipy import linalg
import math
from astropy.modeling import models, fitting
import csv
import time
import pandas as pd
from configparser import ConfigParser
from modules.Utils.config_parser import ConfigHandler

# Pipeline dependencies
# from kpfpipe.logger import start_logger
# from kpfpipe.primitives.level0 import KPF0_Primitive
# from kpfpipe.models.level0 import KPF0

FIT_G = fitting.LevMarLSQFitter()


def is_empty_ary(ary: np.ndarray):
    return ary is None or np.size(ary) == 0


class OrderTraceAlg:
    """Order trace extraction.

    This module defines class 'OrderTraceAlg' and methods to extract order trace from 2D spectral fits image.
    The extraction steps include

        - locate clusters: smooth the image and convert image pixels to be either black or white ('1' or '0').
        - form clusters: find cluster units (each unit containing connected pixels with value '1').
        - clean the clusters: remove noisy clusters, trim noise from the clusters, split the clusters and clean the
          clusters along the top and bottom borders.
        - merge clusters: merge broken clusters to form order trace based on the closeness and polynomial curve fitting.
        - model order trace: approximate each order trace by using least square polynomial fit.
        - find top and bottom widths:  compute the top and bottom widths along the order trace by using normal
          distribution to model the distribution of the spectral data along the order trace and approximate the top and
          bottom widths based on the magnitude of standard deviation from the mean. If the width is unresolved by the
          use of normal distribution, it is either assigned by a default number or further estimated based on widths
          of the surrounding orders.

    Args:
        data (numpy.ndarray): 2D spectral data.
        config (configparser.ConfigParser): config context.
        logger (logging.Logger): Instance of logging.Logger.

    Attributes:
        logger (logging.Logger): Instance of logging.Logger.
        instrument (str): Imaging instrument.
        config_param (ConfigHandler): Related to 'PARAM' section or section associated with the instrument
            if it is defined in the config file.
        config_logger (ConfigHandler): Related to 'LOGGER' section defined in the config file.
        flat_data (numpy.ndarray): Numpy array storing 2d image data.
        debug_output (str): File path for the file that the debug information is printed to. The printing goes to
            standard output if it is an empty string or no printing is made if it is None.
        is_time_profile (bool): Print out the time status while running.
        is_debug (bool): Print out the debug information while running.

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
        TypeError: If there is type error for `data` or `config`.
        Exception: If the size of `data` is less than 20 pixels by 20 pixels.
    """

    FIT_ERROR_TH = 2.5
    UPPER = 1
    LOWER = 0

    def __init__(self, data, config=None, logger=None):
        if not isinstance(data, np.ndarray):
            raise TypeError('image data type error, cannot construct object from OrderTraceAlg')
        if not isinstance(config, ConfigParser):
            raise TypeError('config type error, cannot construct object from OrderTraceAlg')

        ny, nx = np.shape(data)
        if ny <= 20 and nx <= 20:
            raise Exception('image data size is too small for order trace extraction')

        # get data range from config file if it is defined in.
        self.data_range = [0, ny - 1, 0, nx - 1]
        self.logger = logger
        self.flat_data = data
        self.original_size = [ny, nx]

        config_h = ConfigHandler(config, 'PARAM')
        self.instrument = config_h.get_config_value('instrument', '')
        ins = self.instrument.upper()
        self.config_param = ConfigHandler(config, ins, config_h)  # section of instrument or 'PARAM'
        self.config_logger = ConfigHandler(config, 'LOGGER') # section of 'LOGGER'
        self.debug_output = None
        self.is_time_profile = False
        self.is_debug = True if self.logger else False

    def enable_debug_print(self, to_print=True):
        """Enable or disable debug printing.

        Args:
            to_print (bool, optional): Print out the debug information while running. Defaults to True.

        """
        self.is_debug = to_print or bool(self.logger)

    def enable_time_profile(self, is_time=False):
        """Enable or disable time profiling printing.

        Args:
            is_time (bool, optional): Print out the time information while running. Defaults to False.

        """

        self.is_time_profile = is_time

    def get_config_value(self, param: str, default):
        """Get defined value from the config file.

        Search the value of the specified property from config section. The default value is returned if no found.

        Args:
            param (str): Name of the parameter to be searched.
            default (str/int/float): Default value for the searched parameter.

        Returns:
            int/float/str: Value for the searched parameter.

        """
        return self.config_param.get_config_value(param, default)

    def set_data_range(self, data_range=None):
        """Set data range to be processed

        Args:
            data_range (list): Area of the data, [x1, x2, y1, y2], to be processed. The column (or the row) is counted
                            relatively from the first column (or the first row) in case the number is not less than
                            0, otherwise the column (or the row) is counted from the last one.

        Returns:
            list: Data range position relative to the first column and first row of the raw image.
        """

        _, nx, ny = self.get_spectral_data()
        if data_range is None or (not isinstance(data_range, list)) or len(data_range) != 4:
            self.data_range = [0, ny - 1, 0, nx - 1]
        else:
            x1, x2 = sorted([data_range[i] if data_range[i] >= 0 else (nx + data_range[i]) for i in [0, 1]])
            y1, y2 = sorted([data_range[i] if data_range[i] >= 0 else (ny + data_range[i]) for i in [2, 3]])
            self.data_range = [y1, y2, x1, x2]

        self.flat_data = self.flat_data[self.data_range[0]: (self.data_range[1] + 1),
                                        self.data_range[2]: (self.data_range[3] + 1)]
        return self.data_range

    def get_data_range(self):
        if self.data_range is None:
            self.set_data_range()
        return self.data_range

    def get_poly_degree(self):
        """Order of polynomial for order trace fitting.

        Returns:
            int: Order of polynomial.

        """
        return self.get_config_value('fitting_poly_degree', 3)

    def get_instrument(self):
        """Get imaging instrument.

        Returns:
            str: Instrument name.

        """
        return self.instrument

    def d_print(self, *args, end='\n', info=False):
        """Print out running status to logger or debug information to a file.

        Args:
            *args: Variable length argument list to print.
            end (str, optional): Specify what to print at the end.
            info (bool): Print out for information level, not for debug level.

        Notes:
            This function handles the print-out to the logger defined in the config file or other file as specified in
            :func:`~alg.OrderTraceAlg.add_file_logger()`.

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
        """Print time profiling information to the logger.

        Args:
             *args: Variable length argument list to print.

        """
        if self.is_time_profile and self.logger:
            out_str = ' '.join([str(item) for item in args])
            self.logger.info(out_str)

    def get_spectral_data(self):
        """Get spectral information including data and dimension.

        Returns:
            tuple: Information of spectral data,

                * (*numpy.ndarray*): 2D spectral data.
                * **nx** (*int*): Width of the data.
                * **ny** (*int*): Height of the data.

        """
        try:
            assert self.flat_data.all()
        except AssertionError:
            return None

        ny, nx = np.shape(self.flat_data)

        return self.flat_data, nx, ny

    @staticmethod
    def opt_filter(y_data: np.ndarray, par: int, weight: np.ndarray = None):
        """A smoothing filter."""

        n = y_data.size
        # Check for some input preliminaries
        try: 
            assert(y_data.ndim == 1)
            assert(par >= 0)
        except AssertionError:
            return y_data

        if weight is None:
            # a weight is not provided as input
            wgt = np.ones(n, dtype=np.float64)
        else: 
            wgt = np.reshape(weight, (1, -1))[0]
        # f (array): solution of A^t * f = y_data*weight, 
        # A is banded matrix formed based on par & weight
        r = y_data*wgt
        a = np.ones((1, n), dtype=np.float64)[0] * (-abs(par))
        b = np.hstack([[wgt[0]+abs(par)], wgt[1:n-1]+2.0*abs(par), [wgt[n-1]+abs(par)]])
        c = a.copy()
        a[0] = c[-1] = 0
        # resolve banded matrix by combining a, b, c, abc*f = r
        f = linalg.solve_banded((1, 1), np.vstack([a, b, c]), r)

        return f

    @staticmethod
    def reset_row_or_column(imm: np.ndarray, reset_ranges: list = None, row_or_column: int = 0, val: int = 0):
        """Set a value to columns or rows in 2D image array.

        Assign a value to pixels of some columns or rows.

        Args:
            imm (numpy.ndarray): Data of 2D array.
            reset_ranges (list): Range of columns or rows to be set.
            row_or_column (int, optional): Set value to rows if zero or to columns if non-zero. Defaults to 0.
            val (int, optional): Value to be set. Defaults to 0.

        Returns:
            tuple: Pixel information after resetting,

                * **imm** (*numpy.ndarray*): 2D array with reset value.
                * (*numpy.ndarray*): Array of x coordinate of pixels with value greater than 0.
                * (*numpy.ndarray*): Array of y coordinate of pixels with value greater than 0.

        """

        if reset_ranges is None:
            pos = np.where(imm > 0)
            return imm, pos[1], pos[0]

        for rg in reset_ranges:
            range_idx_set = np.arange(*rg, dtype=int)
            if row_or_column == 0:
                for r in range_idx_set:
                    imm[r, :] = val
            else:
                for c in range_idx_set:
                    imm[:, c] = val

        pos = np.where(imm > 0)
        return imm, pos[1], pos[0]

    def locate_clusters(self, img_rows_to_reset=None, img_cols_to_reset=None):
        """ Find cluster pixels from 2D data array.

        Perform smoothing method tpconvert the pixels to be 1 and 0 and find cluster pixels.
        Cluster pixels mean a set of pixels with value 1 and each pixel connects to at least one neighbor pixel
        in vertical, horizontal or in diagonal direction.

        Args:
            img_rows_to_reset (list, optional): collection of rows to be reest.
            img_cols_to_reset (list, optional): collection of columns to be reest.

        Returns:
            dict: result of formed clusters, like::

                    {
                        'x': numpy.ndarray,  # Array of x coordinates of cluster pixels.
                        'y': numpy.ndarray,  # Array of y coordinates of cluster pixels.
                        'cluster_image': numpy.ndarray
                                             # 2D image in which the cluster pixels are with
                                             # value 1 and non cluster pixels are with value 0.
                    }

        """
        def overlap_segment(a_seg, b_seg):
            if a_seg[0] >= b_seg[1] or a_seg[1] <= b_seg[0]:
                return None
            else:
                return [max(a_seg[0], b_seg[0]), min(a_seg[1], b_seg[1])]

        # flat data array and dimension
        image_data, n_col, n_row = self.get_spectral_data()

        # Parameters
        filter_par = self.get_config_value('filter_par', 20)
        noise = self.get_config_value('locate_cluster_noise', 0.0)
        mask = self.get_config_value('cluster_mask', 1)

        rows_str = self.get_config_value('rows_to_reset', '') if img_rows_to_reset is None \
            else json.dumps(img_rows_to_reset)
        cols_str = self.get_config_value('cols_to_reset', '') if img_cols_to_reset is None \
            else json.dumps(img_cols_to_reset)

        o_row, o_col = self.original_size
        s_row, s_col = self.data_range[0], self.data_range[2]

        # adjust rows and cols to reset based on the data rage
        rows_to_reset = None
        if rows_str:
            rows_list = json.loads(rows_str)
            if isinstance(rows_list, list):
                if all([isinstance(r, list) and len(r) == 2 for r in rows_list]):
                    rows_to_reset = [overlap_segment(([r[0]-s_row, r[1]-s_row])
                                                     if r[0] >= 0 else ([r[0]+o_row-s_row, r[1]+o_row-s_row]),
                                                     [0, n_row]) for r in rows_list]
                    rows_to_reset = [r for r in rows_to_reset if r is not None]

        cols_to_reset = None
        if cols_str:
            cols_list = json.loads(cols_str)
            if isinstance(cols_list, list):
                if all([isinstance(c, list) and len(c) == 2 for c in cols_list]):
                    cols_to_reset = [overlap_segment(([c[0]-s_col, c[1]-s_col])
                                                     if c[0] >= 0 else ([c[0]+o_col-s_col, c[1]+o_col-s_col]),
                                                     [0, n_col]) for c in cols_list]
                    cols_to_reset = [c for c in cols_to_reset if c is not None]

        self.d_print('rows_to_reset:', rows_to_reset)
        self.d_print('cols to reset:', cols_to_reset)
        # binary array
        imm = np.zeros((n_row, n_col), dtype=np.uint8)

        for col in range(n_col):
            mm = image_data[:, col] + noise - self.opt_filter(image_data[:, col], filter_par)
            mm_pos = np.where(mm > 0, mm, 0)
            h = 0.5*np.sort(mm_pos)[mm_pos.size//2]
            imm[:, col][mm > (h+1)] = mask

        y, x = np.where(imm > 0)  # ex: (array([4, 5, 6, 7]), array([2, 2, 2, 2]))

        # correction on filtered image (ex. for NEID flat, stacked_2fiber_flat.fits)
        if rows_to_reset is not None:
            self.d_print('pos size before row reset: ' + str(np.size(y)) + ' ' + str(np.size(x)))
            imm, x, y = self.reset_row_or_column(imm, rows_to_reset)
            self.d_print('pos size after row reset: '+str(np.size(y)) + ' ' + str(np.size(x)))

        if cols_to_reset is not None:
            self.d_print('pos size before column reset: ' + str(np.size(y)) + ' ' + str(np.size(x)))
            imm, x, y = self.reset_row_or_column(imm, cols_to_reset, row_or_column=1)
            self.d_print('pos size after column reset: ' + str(np.size(y)) + ' ' + str(np.size(x)))

        return {'x': x, 'y': y, 'cluster_image': imm}

    def collect_clusters(self, c_x: np.ndarray, c_y: np.ndarray):
        """Identify cluster units per positions of cluster pixels.

        The cluster units are identified by checking into the set of cluster pixels and there is no pixels connected
        among the resultant cluster units.

        Parameters:
            c_x (numpy.ndarray): Array of x coordinates for cluster pixels.
            c_y (numpy.ndarray): Array of y coordinates for cluster pixels.

        Returns:
            dict: identified cluster units from the image, like::

                    {
                        <y_1>: <clusters_1> list,
                        <y_2>: <clusters_2> list,...,
                        <y_n>: <clusters_n> list
                    },
                    '''
                    where
                        <y_n> is vertical location (value along y axis)
                        <clusters_n> is list of cluster units ending at <y_n>, like:
                            [ cluster_1, cluster_2, ..., cluster_n],
                            where cluster_i (dict) contains area of the cluster and horizontal
                            segments it covers, like:
                                 {
                                     'x1': int,  # left of the cluster.
                                     'x2': int,  # right of the cluster.
                                     'y1': int,  # top of the cluster.
                                     'y2': int,  # bottom of the cluster.
                                     <y_i_1>: <segments_1> dict, ..., <y_i_n>: <segments_n> dict
                                 }
                                where
                                     <y_i_t> is one of y location ranging from cluster_i['y1'] to
                                     cluster_i['y2'].
                                     <segments_i> contains horizontal segments at <y_i_t> like:
                                        {
                                            'segments': [[x_0, x_1], [x_2, x_3], ....[x_i, x_i+1]]
                                        }
                                        where x_i and x_i+1 means the starting and ending index for
                                        array c_x.
                    ex: clusters units end at y = 10 and y = 11,
                        {
                            10: [
                                    {
                                        'x1': 20, 'x2': 30, 'y1': 9,  'y2': 10,
                                        9:{'segments': [[4, 8], [12, 13]]},
                                        10:{'segments': [[100, 107], [109, 118]]}
                                    },
                                    {
                                         'x1': 50, 'x2': 77, 'y1': 7, 'y2': 10,
                                         7:{'segments': [...]},
                                         8:{'segments': [....]},
                                         9:{'segments': [....]},
                                         10:{'segments: [....]}
                                     }
                                 ],
                             11: [
                                     {<cluster unit ends at y = 11>},
                                     {<cluster unit ends at y = 11>}...
                                 ]
                         }
                    '''

        """
        x, y = c_x, c_y
        _, nx, ny = self.get_spectral_data()

        # clusters_endy_dict contains key:value, like y:[<cluster with maximum y at y>, <....>] : list of clusters
        #
        # cluster: a dict with properties: y1, y2, x1, x2, number(y1), number(y1+1), ...number(y2),
        #                                  value of number(y1) is like {"segments": [seg_1, seg_2,...]}
        #                                  where seg_i: [idx_1, idx_2] containing index for x, y

        clusters_endy_dict = dict()      # contain clusters end at y (0 to ny-1)
        nx_prev_cluster_id = [list() for _ in range(nx)]

        if self.logger:
            self.logger.info("OrderTraceAlg: collecting clusters...")

        for cy in range(ny):
            if cy % 100 == 0:
                self.d_print(cy, '', end='')

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
                    clusters_endy_dict[cy].append({cy: {'segments': [seg]},
                                                  'y1': cy, 'y2': cy, 'x1': x[seg[0]], 'x2': x[seg[1]]})
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

            # create new cluster for current y from isolated segment and cluster unit containing associated segments &
            # clusters of previous y
            cluster_at_crt_y = list()
            for s_idx in range(len(segments_at_cy)):
                if len(seg_to_cluster_map[s_idx]) == 0:    # no connected cluster
                    cluster = {cy: {'segments': [segments_at_cy[s_idx]]},
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
                    if b_conn == -1:
                        new_conn = {'segment_idx': [s_idx], 'cluster_idx': connected_clusters}
                        connected_set.append(new_conn)
                    else:
                        if s_idx not in connected_set[b_conn]['segment_idx']:
                            connected_set[b_conn]['segment_idx'].append(s_idx)
                        for c in connected_clusters:
                            if c not in connected_set[b_conn]['cluster_idx']:
                                connected_set[b_conn]['cluster_idx'].append(c)
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

            cluster_at_crt_y = self.sort_cluster_on_loc(cluster_at_crt_y, 'x1')
            clusters_endy_dict[cy] = cluster_at_crt_y
            nx_prev_cluster_id = [list() for _ in range(nx)]

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

        self.d_print('\n')
        return clusters_endy_dict

    def remove_cluster_by_size(self, clusters_endy_dict: dict, x_index: np.ndarray, y_index: np.ndarray, th=None):
        """Remove noisy clusters.

        The removal process is based on pixel number and the size of the cluster. Assign an id to non-noisy cluster.

        Args:
            clusters_endy_dict (dict): Collection of clusters collected by collect_clusters,
                please see `Returns` section of :func:`~alg.OrderTraceAlg.collect_clusters` for more detail.
            x_index (numpy.ndarray): Array of x coordinates of cluster pixels.
            y_index (numpy.ndarray): Array of y coordinates of cluster pixels.
            th (int, optional): Size threshold for removing the noisy cluster. Defaults to None.

        Returns:
            dict: cluster information containing assigned id, like::

                {
                    'index': numpy.ndarray,
                                        # array of cluster id associated with cluster pixels.
                    'n_regions': int    # total cluster.
                }

        """

        if self.logger:
            self.logger.info("OrderTraceAlg: removing clusters by size...")

        _, nx, ny = self.get_spectral_data()
        w_th = max(nx//100, 1)
        h_th = max(ny//800, 1)
        if th is None:
            th = h_th * w_th

        self.d_print('there are total ', x_index.size, ' clusters to test.')
        index = np.zeros(x_index.size, dtype=int)
        cluster_no = 1
        for y in range(ny):
            if (y not in clusters_endy_dict) or (len(clusters_endy_dict[y]) == 0):
                continue
            clusters = clusters_endy_dict[y]

            c_idx = 0
            for one_cluster in clusters:
                total_pixel = 0
                for y_n in range(one_cluster['y1'], one_cluster['y2']+1):
                    if len(one_cluster[y_n]['segments']) == 0:
                        continue

                    # count stops at the segment of some y
                    for cluster_seg in one_cluster[y_n]['segments']:
                        total_pixel += (cluster_seg[1]-cluster_seg[0]+1)
                        if total_pixel > th:    # keep this cluster
                            break

                    # assign cluster id to non-noisy cluster
                    if total_pixel > th:
                        for y_c in range(one_cluster['y1'], one_cluster['y2']+1):
                            for c_seg in one_cluster[y_c]['segments']:
                                index[c_seg[0]:(c_seg[1]+1)] = cluster_no
                        cluster_no += 1
                        break
                c_idx += 1

        # remove narrow cluster
        max_idx = np.amax(index)
        h_size_th = max(ny//100, 1)
        w_size_th = max(nx//100, 1)
        for c_id in np.arange(1, max_idx+1):
            w, h, t_p, crt_cluster_idx = self.get_cluster_size(c_id, index, x_index, y_index)

            if w <= w_size_th and h <= h_size_th:
                index[crt_cluster_idx] = 0
                self.d_print('cluster ', c_id, ' total: ', t_p, ' w, h', w, h, ' => remove')
            else:
                self.d_print('cluster ', c_id, ' total: ', t_p, ' w, h', w, h)

        n_regions = np.amax(index) + 1 if np.amin(index) == 0 else np.amax(index)

        return {'index': index, 'n_regions': n_regions}

    def form_clusters(self, c_x: np.ndarray, c_y: np.ndarray, th=None):
        """Form clusters and assign id to each formed cluster.

        Form the cluster units and remove the small size cluster units. There is no pixel connected between different
        cluster units.

        Args:
            c_x (numpy.ndarray): Array of x coordinates for cluster pixels.
            c_y (numpy.ndarray): Array of y coordinates for cluster pixels.
            th (int, optional): Size threshold used for removing noisy cluster. Defaults to None.

        Returns:
            tuple: Information of cluster pixels after cluster units are formed,

                * **new_x** (*numpy.ndarray*): Array of x coordinates of cluster pixels.
                * **new_y** (*numpy.ndarray*): Array of y coordinates of cluster pixels.
                * **new_index** (*numpy.ndarray*): Array of cluster id on cluster pixels.

        Raises:
            AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
            TypeError: If there is type error for  `c_x` or `c_y`.
            Exception: If the size of `c_x` or `c_y` are not the same.

        """

        if (not isinstance(c_x, np.ndarray)) or (not isinstance(c_y, np.ndarray)):
            raise TypeError("input c_x or c_y type error")

        if is_empty_ary(c_x) or is_empty_ary(c_y):
            return np.array([]), np.array([]), np.array([])

        if c_x.size != c_y.size:
            raise Exception("size of arrays of x and y coordinates not matched")

        clusters_all_y = self.collect_clusters(c_x, c_y)
        index_info = self.remove_cluster_by_size(clusters_all_y, c_x, c_y, th)
        new_x, new_y, new_index = self.reorganize_index(index_info['index'], c_x, c_y)
        return new_x, new_y, new_index

    def advanced_cluster_cleaning_handler(self, index: np.ndarray, x: np.ndarray, y: np.ndarray,
                                          start_cluster: int = None, stop_cluster: int = None):
        """Remove or clean noisy clusters.

        This removal process uses polynomial fitting on all or selected clusters formed by `form_clusters()`.

        Args:
            index (numpy.ndarray): Array of cluster id on cluster pixels.
            x (numpy.ndarray): Array of x coordinates on cluster pixels.
            y (numpy.ndarray): Array of y coordinates on cluster pixels.
            start_cluster (int, optional): Cluster id of the first cluster to process. Defaults to None.
            stop_cluster (int, optional): Cluster id of the last cluster to process. Defaults to None.

        Returns:
            tuple: cleaning status on clusters:

                * **index_p** (*numpy.ndarray*):  Array of cluster id on cluster pixels after cleaning.
                * **all_status** (*dict*): Cleaning status on processed clusters, like::

                    {
                       <cluster_id_i> int: <cleaning status> dict,
                                # <cluster_id_i> is cluster id of i-th cluster.
                                # <cleaning status> is cleaning status for the cluster
                                # See Returns in handle_noisy_cluster()
                       :
                    }

        Raises:
            AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
            TypeError: If there is type error for  `x`, `y` or `index`.
            Exception: If the size of `x`, `y`, or `index` are not the same.

        """
        if (not isinstance(x, np.ndarray)) or (not isinstance(y, np.ndarray) or (not isinstance(index, np.ndarray))):
            raise TypeError("input x or y or index type error")

        if is_empty_ary(x) or is_empty_ary(y) or is_empty_ary(index):
            return np.array([]), np.array([]), np.array([]), dict()

        if x.size != y.size:
            raise Exception("size of arrays of x and y coordinates not matched")
        if x.size != index.size:
            raise Exception("size of array of x and index not matched")

        index_p = index.copy()
        x_p = x.copy()
        y_p = y.copy()

        next_idx = 1 if start_cluster is None else start_cluster
        original_max_idx = np.amax(index) if stop_cluster is None else stop_cluster
        all_status = dict()
        _, nx, ny = self.get_spectral_data()

        while True:
            p_info, errors, area = self.extract_order_from_cluster(next_idx, index_p, x_p, y_p)
            if errors <= self.FIT_ERROR_TH:
                status = {'msg': 'same', 'cluster_id': next_idx,
                          'poly_fitting': {next_idx: {'errors': errors, 'coeffs': p_info, 'area': area}}}
            else:
                index_p, status = self.handle_noisy_cluster(index_p, x_p, y_p, [next_idx])

            all_status[next_idx] = status
            self.d_print('idx: ', next_idx, ' status: ', status)

            next_idx = next_idx+1 if next_idx < original_max_idx else None
            if next_idx is not None:
                continue
            else:
                new_x, new_y, new_index = self.reorganize_index(index_p, x, y)
                return new_x, new_y, new_index, all_status

    def extract_order_from_cluster(self, cluster_no: int, index: np.ndarray, x: np.ndarray, y: np.ndarray):
        """Get curve fitting result on specified cluster.

        Args:
            cluster_no (int): id of the cluster to find the curve fitting results.
            index (numpy.ndarray): Array of cluster id on cluster pixels.
            x (numpy.ndarray): Array of x coordinates of cluster pixels.
            y (numpy.ndarray): Array of y coordinates of cluster pixels.

        Returns:
            tuple: Please see `Returns` of :func:`~alg.OrderTraceAlg.curve_fitting_on_one_cluster`.

        """

        power = self.get_poly_degree()
        p_info, error, area = self.curve_fitting_on_one_cluster(cluster_no, index, x, y, power)

        return p_info, error, area

    def handle_noisy_cluster(self, index_t: np.ndarray, x: np.ndarray, y: np.ndarray, num_set: list):
        """
        Handle the cluster which is not well fitted by polynomial curve.

        Parameters:
            index_t (numpy.ndarray): Array of cluster id on cluster pixels.
            x (numpy.ndarray): Array of x coordinates on cluster pixels.
            y (numpy.ndarray): Array of y coordinates on cluster pixels.
            num_set(list): The cluster with the specified id (currently, the first member in the list) is handled.

        Returns:
            tuple: Status after processing:

                * **new_index_t** (*np.ndarray*): updated version of `index_t` after processing
                * **status** (*dict*):  One of the following possible process results is returned:

                  1. the cluster is to be deleted.
                  2. the cluster pixels is to be changed.
                  3. the cluster is to be split into multiple cluster units.
                  4. the cluster remains the same.

                  it is like::

                            {
                                'msg': 'delete'/'change'/'split'/'same',
                                'cluster_id': <target_cluster_id> int,
                                'cluster_added': [<new_id_1>, <new_id_2>,...,<new_id_i>],
                                'poly_fitting':{
                                    <cluster_id>: {
                                        'errors': float,
                                                    # Least square error by using polynomial fit.
                                        'coeffs': numpy.ndarray,
                                                    # Coefficients of polynomial fit.
                                        'area': list,     # Area of the cluster, like
                                                          # [<min_x>, <max_x>, <min_y>, <max_y>]
                                                          # for 4 borders of the cluster.
                                    }
                                    <new_id_1>: {'errors': ..., 'coeffs': ..., 'area': ...},
                                    <new_id_n>: {'errors': ..., 'coeffs': ..., 'area': ...}}
                            }
                            # <new_id_i> is the id for newly created cluster, if 'split'.

        """

        curve_th = self.get_config_value('order_width_th', 7)
        crt_cluster_idx = np.where(index_t == num_set[0])[0]
        crt_cluster_x = x[crt_cluster_idx]
        crt_cluster_y = y[crt_cluster_idx]
        x_min = np.amin(crt_cluster_x)
        x_max = np.amax(crt_cluster_x)
        xi = x_max
        crt_col_idx = np.where(crt_cluster_x == xi)[0]

        # start from the most right position of the cluster to find the segments in vertical direction column by column
        # extend the curve from right to left by adding the vertical segments to the curve in case any of the segments
        # at current x position has good size and good overlap with the those found at previous x position.
        pre_segments_y = self.get_segments_from_index_list(crt_col_idx, crt_cluster_y)

        curve_records = dict()
        curve_records[xi] = list()

        for seg in pre_segments_y:
            crt_seg_y = crt_cluster_y[seg]
            if (crt_seg_y[1]-crt_seg_y[0]) <= curve_th:
                curve = {'start_x': xi, 'crt_x': xi, 'y1': crt_seg_y[0], 'y2': crt_seg_y[1],
                         xi: [[crt_seg_y[0], crt_seg_y[1]]]}
                curve_records[xi].append(curve)

        self.sort_cluster_on_loc(curve_records[xi], 'y1')

        # extend current curve or branch out new curve
        for xi in range(x_max-1, x_min-1, -1):
            # print(xi, end=" ")
            curve_records[xi] = list()
            crt_col_idx = np.where(crt_cluster_x == xi)[0]
            if crt_col_idx.size == 0:
                continue
            crt_segments_y = self.get_segments_from_index_list(crt_col_idx, crt_cluster_y)

            pre_curves = curve_records[xi+1]
            curves_to_upgrade = list()

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
                if (crt_seg_y[1] - crt_seg_y[0]) > curve_th*2:
                    # print('skip on long segment: x, y1, y1 => ', xi, crt_seg_y[0], crt_seg_y[1])
                    continue

                for c_idx in range(len(pre_curves)):
                    # if crt_seg_y[1] < (pre_curves[c_idx]['y1']-1):
                    if crt_seg_y[1] < (pre_curves[c_idx]['y1']):
                        break
                    # if crt_seg_y[0] > (pre_curves[c_idx]['y2']+1):
                    if crt_seg_y[0] > (pre_curves[c_idx]['y2']):
                        continue

                    p_curve_y = pre_curves[c_idx][xi+1]

                    if crt_seg_y[0] <= (p_curve_y[-1][1]) and crt_seg_y[1] >= (p_curve_y[0][0]):
                        if crt_seg_idx >= 1 and (c_idx in crt_seg_to_pre_curves_map[crt_seg_idx-1]):
                            pre_seg_y = crt_cluster_y[crt_segments_y[crt_seg_idx-1]]
                            if (crt_seg_y[0] - pre_seg_y[1]) >= curve_th or (crt_seg_y[1] - pre_seg_y[0]) >= curve_th*2:
                                continue
                        found_curve = True
                        pre_curves_to_crt_seg_map[c_idx].append(crt_seg_idx)
                        crt_seg_to_pre_curves_map[crt_seg_idx].append(c_idx)

                if found_curve is False:
                    if (crt_seg_y[1] - crt_seg_y[0]) <= curve_th:
                        curve = {'start_x': xi, 'crt_x': xi, 'y1': crt_seg_y[0], 'y2': crt_seg_y[1],
                                 xi: [[crt_seg_y[0], crt_seg_y[1]]]}
                        curve_records[xi].append(curve)

            # create curve unit to contain connected curves and segments
            curve_units = list()
            already_processed = set()

            for c_idx in range(len(pre_curves)):
                if len(pre_curves_to_crt_seg_map[c_idx]) == 0:
                    continue
                curves_to_upgrade.append(c_idx)

                if c_idx in already_processed:
                    continue
                curve_set = {c_idx}
                curve_set_len = len(curve_set)
                segs_set = set(pre_curves_to_crt_seg_map[c_idx])
                segs_set_len = len(segs_set)
                while True:
                    for s in segs_set:
                        curve_set.update(crt_seg_to_pre_curves_map[s])
                    if curve_set_len == len(curve_set):
                        break
                    else:
                        curve_set_len = len(curve_set)
                    for c in curve_set:
                        segs_set.update(pre_curves_to_crt_seg_map[c])
                    if segs_set_len == len(segs_set):
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
        for xi in range(x_min, x_max+1):
            all_curves_in_cluster.extend(curve_records[xi])

        index_in_cluster, poly_fitting = self.remove_noise_in_cluster(all_curves_in_cluster, x, y, crt_cluster_idx)

        # print('after removal: ', index_in_cluster[crt_cluster_idx], ' num_set[0]:', num_set[0])
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
        for c in range(1, max_new_index+1):
            set_idx = np.where(index_in_cluster == c)[0]
            if c == 1:
                new_index_t[set_idx] = num_set[0]
                status['poly_fitting'][num_set[0]] = poly_fitting[c]
            else:
                added_id = crt_last_index+c-1
                new_index_t[set_idx] = added_id
                status['poly_fitting'][added_id] = poly_fitting[c]

        return new_index_t, status

    @staticmethod
    def get_segments_from_index_list(id_list: np.ndarray,  loc: np.ndarray):
        """Find horizontal segments at some y location.

        Horizontal segment means a segment containing continuous cluster pixels at the same y position.
        The finding is based on index list associated with an array of x coordinates.

        Args:
            id_list (numpy.ndarray): Array of index for the array of `loc`.
            loc (numpy.ndarray): Array of x coordinates of cluster pixels.

        Returns:
            list: List of horizontal segments, like::

                [[<start_idx>_i, <end_idx>_i], ..., [<start_idx>_n, <end_idx>_n]]
                '''
                where
                    <start_idx>_i and <end_idx>_i represent the starting and ending index
                    of the i-th segment and the index is associated with parameter loc.

                ex. [[1, 3], [7, 10], ..., [150, 160]] means the following segments are
                    included,
                    1st segment is from loc[1] to loc[3] along x-axis.
                    2nd segment is from loc[7] to loc[10] along x-axis.
                    last segment is from loc[150] to loc[160] along x-axis.
                '''

        """
        segments = list()

        distcont_idx = np.where((loc[id_list] - loc[np.roll(id_list, 1)]) != 1)[0]

        # collect all segments in terms of two index from loc
        p_idx = id_list[0]
        segments.append([p_idx, p_idx])

        for d_idx in distcont_idx[1:]:
            if (id_list[d_idx - 1]) > p_idx:
                segments[-1][1] = id_list[d_idx - 1]
            p_idx = id_list[d_idx]
            segments.append([p_idx, p_idx])

        if distcont_idx[-1] < id_list[-1]:
            segments[-1][1] = id_list[-1]

        return segments

    def remove_noise_in_cluster(self, cluster_curves: list, x_index: np.ndarray, y_index: np.ndarray,
                                crt_cluster_idx: np.ndarray, th=None):
        """Remove noise cluster, trim noise from the cluster, or split the cluster into another clusters.

        The removal works on the clusters collected by :func:`~alg.OrderTraceAlg.handle_noisy_cluster`.
        Whether the cluster in the collection is kept or removed depends on the size and polynomial fitting result.

        Args:
            cluster_curves (list): Array of clusters collected by :func:`~alg.OrderTraceAlg.handle_noisy_cluster` are
                tested to be kept or removed.
            x_index (numpy.ndarray): Array of x coordinates of cluster pixels.
            y_index (numpy.ndarray): Array of x coordinates of cluster pixels.
            crt_cluster_idx (numpy.ndarray): Set of index for the clusters included in `cluster_curves` and the index
                is for cluster pixels related array, like `x_index` or `y_index`.
            th (float, optional): Threshold for cluster size. Defaults to None.

        Returns:
            tuple: Polynomial fit results and cluster id for not removed clusters,

                * **index** (*np.npdarray*): Array associated with cluster pixels in which the pixels covered by any
                  not removed clusters  of `cluster_curves` are marked by a cluster no. starting from 1.
                * **poly_fitting_results** (*dict*): Polynomial fitting results for not removed clusters in
                  `cluster_curves`, like::

                        {
                            'errors': float,  # Least square errors of polynomial fitting.
                            'coeffs': numpy.ndarray,  # Coefficients of polynomial fitting.
                            'area': list              # area of the cluster, like
                                                      # [<min_x>, <max_x>, <min_y>, <max_y>]
                                                      # for 4 borders of the cluster.
                        }

        """

        power = self.get_poly_degree()
        _, nx, ny = self.get_spectral_data()

        w_th = max(nx//100, 1)
        h_th = max(ny//800, 1)
        if th is None:
            th = max(h_th * w_th * 2/3, 1)

        h_size_th = max(ny//100, 1)
        w_size_th = max(nx//100, 1)
        index = np.zeros(x_index.size, dtype=int)
        cluster_no = 0

        # print(len(cluster_curves), h_size_th, w_size_th, th)
        poly_fitting_results = dict()
        crt_cluster_x = x_index[crt_cluster_idx]
        crt_cluster_y = y_index[crt_cluster_idx]

        # index will contain new assigned index id if there is new curve split from the original cluster
        for curve_idx in range(len(cluster_curves)):
            curve = cluster_curves[curve_idx]
            total_pixel = 0

            if (curve['start_x']-curve['crt_x']) < w_size_th and (curve['y2'] - curve['y1']) < h_size_th:
                continue

            x_set = list()
            y_set = list()
            for x_loc in range(curve['crt_x'], curve['start_x']+1):
                segs_in_y = curve[x_loc]

                for seg_y in segs_in_y:
                    total_pixel += (seg_y[1]-seg_y[0]+1)
                    for s_y in range(seg_y[0], seg_y[1]+1):
                        x_set.append(x_loc)
                        y_set.append(s_y)
            if total_pixel < th:
                continue
            else:
                # print('  total pixel: ', total_pixel, ' => polyfit test')
                x_ary = np.array(x_set)
                y_ary = np.array(y_set)
                sort_idx = np.argsort(x_ary)
                x_ary = x_ary[sort_idx]
                y_ary = y_ary[sort_idx]
                coeffs = np.polyfit(x_ary, y_ary, power)
                errors = math.sqrt(np.square(np.polyval(coeffs, x_ary) - y_ary).mean())
                if errors > self.FIT_ERROR_TH:
                    continue

            cluster_no += 1
            # print("start reset cluster")
            for x_loc in range(curve['crt_x'], curve['start_x']+1):
                segs_in_y = curve[x_loc]
                for seg_y in segs_in_y:
                    y_log = np.logical_and(crt_cluster_y >= seg_y[0], crt_cluster_y <= seg_y[1])
                    set_idx = crt_cluster_idx[np.where(np.logical_and(y_log, crt_cluster_x == x_loc))[0]]
                    index[set_idx] = cluster_no

            poly_fitting_results[cluster_no] = {'errors': errors, 'coeffs': coeffs,
                                                'area': [np.amin(x_ary), np.amax(x_ary), np.amin(y_ary), np.amax(y_ary)]
                                                }

        return index, poly_fitting_results

    def clean_clusters_on_border(self, x: np.ndarray,  y: np.ndarray, index: np.ndarray, border_y: int):
        """Clean clusters crossing the top or bottom boundary based on the given border position along y axis.

        Parameters:
            x (array): Array of x coordinates of cluster pixels.
            y (array): Array of y coordinates of cluster pixels.
            index (array): Array of cluster id on cluster pixels.
            border_y (int): The vertical position (y coordinate) of the border to check.

        Returns:
            tuple: Cluster pixels after cleaning:

                * (*numpy.ndarray*): Array of x coordinates of cluster pixels after cleaning.
                * (*numpy.ndarray*): Array of y coordinates of cluster pixels after cleaning.
                * (*numpy.ndarray*): Array of cluster id on cluster pixels after cleaning.
        """

        border_cross = np.where(y == border_y)[0]  # boundary pixels (top or bottom) by checking y position
        changed = 0

        if border_cross.size > 0:
            border_cross = index[border_cross]       # cluster id of boundary pixels
            border_cross = np.unique(border_cross)   # sorted unique bottom boundary cluster number

            # cluster number at bottom (or top) boundry
            for i in range(border_cross.size):
                if border_cross[i] == 0:     # not a cluster id
                    continue
                idx_of_c_num = np.where(index == border_cross[i])[0]           # all pixels at this cluster id
                bind = idx_of_c_num[np.where(y[idx_of_c_num] == border_y)[0]]  # all pixels at border & border_corss[i]
                for ii in range(bind.size):
                    idx_to_remove = idx_of_c_num[np.where(x[idx_of_c_num] == x[bind[ii]])[0]]
                    index[idx_to_remove] = 0
                    changed = 1

        if changed == 1:
            return self.reorganize_index(index, x, y)
        else:
            return x, y, index

    def clean_clusters_on_borders(self, x: np.ndarray, y: np.ndarray, index: np.ndarray, top_border: int = None,
                                  bottom_border: int = None):
        """Clean clusters crossing the top and bottom boundaries of the image.

        Args:
            x (array): Array of x coordinates of cluster pixels.
            y (array): Array of y coordinates of cluster pixels.
            index (array): Array of cluster id on cluster pixels.
            top_border (int, optional): Top border vertical position (along y axis). Defaults to None.
            bottom_border (int, optional): Bottom border vertical position (along y axis). Defaults to None.

        Returns:
            tuple: Cluster pixels after cleaning:

                * **new_x** (*numpy.ndarray*): Array of x coordinates of cluster pixels after cleaning.
                * **new_y** (*numpy.ndarray*): Array of y coordinates of cluster pixels after cleaning.
                * **new_index** (*numpy.ndarray*): Array of cluster id on cluster pixels after cleaning.

        Raises:
            AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
            TypeError: If there is type error for  `x`, `y` or `index`.
            Exception: If the size of `x`, `y`, or `index` are not the same.

        """
        if (not isinstance(x, np.ndarray)) or (not isinstance(y, np.ndarray) or (not isinstance(index, np.ndarray))):
            raise TypeError("input x or y or index type error")

        if is_empty_ary(x) or is_empty_ary(y) or is_empty_ary(index):
            return np.array([]), np.array([]), np.array([])

        if x.size != y.size:
            raise Exception("size of arrays of x and y coordinates not matched")
        if x.size != index.size:
            raise Exception("size of array of x and index not matched")

        _, _, ny = self.get_spectral_data()

        if top_border is None:
            top_border = ny-1

        if bottom_border is None:
            bottom_border = 0
        x_b, y_b, index_b = self.clean_clusters_on_border(x, y, index, bottom_border)
        new_x, new_y, new_index = self.clean_clusters_on_border(x_b, y_b, index_b, top_border)

        return new_x, new_y, new_index

    def merge_clusters_and_clean(self, index: np.ndarray, x: np.ndarray, y: np.ndarray):
        """Merge clusters and remove the clusters with big opening in the center.

        Args:
            index (numpy.ndarray): Array of cluster id on cluster pixels.
            x (numpy.ndarray): Array of x coordinates of cluster pixels.
            y (numpy.ndarray): Array of y coordinates of cluster pixels.

        Returns:
            tuple: Information of cluster pixels after merging,

                * **new_x** (*numpy.ndarray*): Array of x coordinates of cluster pixels after processing.
                * **new_y** (*numpy.ndarray*): Array of y coordinates of cluster pixels after processing.
                * **new_index** (*numpy.ndarray*): Array of cluster id of cluster pixels after processing.

        Raises:
            AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
            TypeError: If there is type error for  'x`, `y` or `index`.
            Exception: If the size of `x`, `y`, or `index` are not the same.

        """
        if (not isinstance(x, np.ndarray)) or (not isinstance(y, np.ndarray) or (not isinstance(index, np.ndarray))):
            raise TypeError("input x or y or index type error")

        if is_empty_ary(x) or is_empty_ary(y) or is_empty_ary(index):
            return np.array([]), np.array([]), np.array([])

        if x.size != y.size:
            raise Exception("size of arrays of x and y coordinates not matched")
        if x.size != index.size:
            raise Exception("size of array of x and index not matched")

        m_x, m_y, m_index, m_coeffs = self.merge_clusters(index, x, y)
        new_x, new_y, new_index = self.remove_broken_cluster(m_index, m_x, m_y)
        return new_x, new_y, new_index

    def merge_clusters(self, index: np.ndarray, x: np.ndarray, y: np.ndarray):
        """Merge clusters based on the closeness between the clusters and the fitting quality by the same polynomial.

        Parameters:
            index (numpy.ndarray): Array of cluster id on cluster pixels.
            x (numpy.array): Array of x coordinates of cluster pixels.
            y (numpy.array): Array of y coordinates of cluster pixels.

        Returns:
             tuple: Information of cluster pixels after processing,
                * **new_x** (*numpy.ndarray*): Array of x coordinates of cluster pixels after processing.
                * **new_y** (*numpy.ndarray*): Array of y coordinates of cluster pixels after processing.
                * **new_index** (*numpy.ndarray*): Array of cluster id of cluster pixels.
                * **m_coeffs** (*numpy.ndarray*): Array containing polynomial fitting coefficients and the area of
                  the clusters. Each row of the array has the data for one cluster.

        """
        new_index = index.copy()
        new_x = x.copy()
        new_y = y.copy()
        new_coeffs, errors = self.curve_fitting_on_all_clusters(new_index, new_x, new_y)
        t = 1

        while True:
            self.d_print('merge time: ', t)
            t += 1
            n_index, n_x, n_y, n_coeffs, merge_status = self.one_step_merge_cluster(new_coeffs,
                                                                                    new_index, new_x, new_y)

            new_index = n_index.copy()
            new_x = n_x.copy()
            new_y = n_y.copy()
            new_coeffs = n_coeffs.copy()

            if merge_status['status'] == 'nochange':
                break

        m_x, m_y, m_index = self.reorganize_index(new_index, new_x, new_y)
        m_coeffs, errors = self.curve_fitting_on_all_clusters(m_index, m_x, m_y)

        sorted_index = self.sort_cluster_in_y(m_coeffs)
        new_index_sort = np.zeros(np.size(m_index), dtype=int)
        new_coeffs_sort = np.zeros(np.shape(m_coeffs))
        for i, v_sort in enumerate(sorted_index):
            if i != 0:
                idx = np.where(new_index == v_sort)[0]
                new_index_sort[idx] = i
                new_coeffs_sort[i] = m_coeffs[v_sort]

        m_index = new_index_sort
        m_coeffs = new_coeffs_sort
        return m_x, m_y, m_index, m_coeffs

    def one_step_merge_cluster(self, crt_coeffs: np.ndarray, crt_index: np.ndarray,
                               crt_x: np.ndarray, crt_y: np.ndarray):
        """Single step of cluster merging, at most one pair of clusters is merged.

        Parameters:
            crt_coeffs (numpy.ndarray): Coefficients of polynomial fit and the area of the clusters.
            crt_index (numpy.ndarray): Array of cluster id on cluster pixels.
            crt_x (numpy.ndarray): Array of x coordinates of cluster pixels.
            crt_y (numpy.ndarray): Array of y coordinates of cluster pixels.

        Returns:
            tuple: Information of cluster pixels after merge and merge status:

                * (*numpy.ndarray*): Array of cluster id of cluster pixels after merge.
                * (*numpy.ndarray*): Array of x coordinates of cluster pixels after merge.
                * (*numpy.ndarray*): Array of  y coordinates of cluster pixels after merge.
                * (*numpy.ndarray*): Coefficients of polynomial fit and the area of the clusters after the merge.
                * **merge_status** (*dict*): merge status, please see :func:`~alg.OrderTraceAlg.merge_fitting_curve()`
                  for the detail.
        """
        merge_status = self.merge_fitting_curve(crt_coeffs, crt_index, crt_x, crt_y)

        if merge_status['status'] != 'nochange':
            next_x, next_y, next_index, convert_map = self.reorganize_index(merge_status['index'], crt_x, crt_y,
                                                                            True)

            new_polys = np.zeros((np.amax(next_index)+1, np.shape(crt_coeffs)[1]))
            for c_id, m_id in convert_map.items():    # ???
                new_polys[m_id, :] = crt_coeffs[c_id, :]
            return next_index, next_x, next_y, new_polys, merge_status
        else:
            return crt_index, crt_x, crt_y, crt_coeffs, merge_status

    def merge_fitting_curve(self, poly_curves: np.ndarray, index: np.ndarray, x: np.ndarray, y: np.ndarray,
                            threshold=FIT_ERROR_TH):

        """Merge the cluster to the closest neighbor.

        The merge iterates on cluster pairs and stops when one merge is made or all paris are tested.

        Parameters:
            poly_curves (numpy.ndarray): Array containing coefficients of polynomial fitting to all clusters and
                the area of the clusters. Each row contains the coefficients and the area for one cluster.
            index (numpy.ndarray): Array of cluster id on cluster pixels.
            x (numpy.ndarray): Array of x coordinates of cluster pixels.
            y (numpy.ndarray): Array of y coordinates of cluster pixels.
            threshold (float): error threshold to determine the polynomial fitting quality.

        Returns:
            dict: merge status, like::

                    {
                        'status': 'changed'|'nochange'.
                        'index': numpy.ndarray,
                                        # Array of cluster id on cluster pixels after merge.
                        'kept_curves': list,     # Array of cluster id of unchanged clusters.
                        'log': <messge>.
                    }

                    # 'status' means if clusters are 'changed' (if merge happens) or 'nochange'.
                    # 'log' contains the message regarding any merge action if there is,
                    # like 'remove id' or 'merge id_1 and id_2'.

        """
        power = self.get_poly_degree()

        x_min_c = power+1
        x_max_c = power+2
        y_min_c = power+3
        y_max_c = power+4

        _, nx, ny = self.get_spectral_data()

        max_order = np.amax(index)

        sort_idx_on_miny = np.argsort(poly_curves[:, y_min_c])
        new_polys = poly_curves[sort_idx_on_miny]

        cluster_changed = 0
        non_exist = -1
        short_curve = nx/2
        reserve_curve = nx//20

        m_height = np.median(poly_curves[:, y_max_c] - poly_curves[:, y_min_c])
        log = ''

        kept_curves = []

        c1 = 1
        while True:
            if c1 > max_order:
                break
            if cluster_changed >= 1:       # stop at when the number of cluster changed is made
                break

            # if print_result is True:
            #    print("current test curve: c1: "+ str(c1) + " o_c1: "+ str(sort_idx_on_miny[c1]))
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

                # skip the curve which is horintally overlapped

                if new_polys[c1, x_min_c] < new_polys[c2, x_min_c]:
                    h_overlap = new_polys[c1, x_max_c] - new_polys[c2, x_min_c]
                else:
                    h_overlap = new_polys[c2, x_max_c] - new_polys[c1, x_min_c]

                if h_overlap < nx/20:        # not overlap too much
                    v_neighbors.append(c2)
                    # print('add ', c2, sort_idx_on_miny[c2], new_polys[c2, x_min_c:y_max_c+1])

            o_c1 = sort_idx_on_miny[c1]
            v_neighbors = np.asarray(v_neighbors)
            errors = np.full(v_neighbors.size, ny*ny, dtype=float)
            merged_poly_info = dict()

            # no vertical neighbor, set the cluster to be 0
            if v_neighbors.size > 0:
                v_neighbors = np.sort(v_neighbors)

                # for i in range(v_neighbors.size):   # try for i, v_neighbor in enumerate(v_neighbors): ???
                #    o_c2 = sort_idx_on_miny[v_neighbors[i]]
                #    merged_poly_info[o_c2], errors[i] = self.merge_two_clusters(np.array([o_c1, o_c2]),
                #                                                                x, y, index, power)
                for i, v_neighbor in enumerate(v_neighbors):   # try for i, v_neighbor in enumerate(v_neighbors):
                    o_c2 = sort_idx_on_miny[v_neighbor]
                    merged_poly_info[o_c2], errors[i] = self.merge_two_clusters(np.array([o_c1, o_c2]),
                                                                                x, y, index, power)

            # if print_result is True:
            #    print('neighbors: ', v_neighbors, 'neighbors errors: ', errors)

            # no neighbors or no neighbors qualified to merge
            if v_neighbors.size == 0 or (v_neighbors.size > 0 and np.amin(errors) > threshold):
                curve_width = new_polys[c1, x_max_c] - new_polys[c1, x_min_c]
                if curve_width > reserve_curve:
                    pass_center = 1 if new_polys[c1, x_min_c] < short_curve < new_polys[c1, x_max_c] else 0
                    # print('no neighbor, width: ', curve_width,  ' pass center: ', pass_center)
                    if pass_center == 1:
                        kept_curves.append(sort_idx_on_miny[c1])
                        c1 += 1
                        continue

                index = np.where(index == o_c1, 0, index)
                new_polys[c1, x_min_c] = non_exist
                self.d_print("remove: ", c1, ' from: ', o_c1)
                log += 'remove '+str(o_c1)
                cluster_changed += 1
                c1 += 1
                continue

            c_neighbors = v_neighbors[np.where(errors < threshold)[0]]
            c_neighbors_distance = np.zeros(c_neighbors.size)
            x_dists = np.zeros(c_neighbors.size)
            y_dists = np.zeros(c_neighbors.size)

            cross_neighbor = np.zeros(c_neighbors.size)

            for i, c2 in enumerate(c_neighbors):
                o_c2 = sort_idx_on_miny[c2]
                cluster_nos = np.array([o_c1, o_c2]) if new_polys[c1, x_min_c] < new_polys[c2, x_min_c] else \
                    np.array([o_c2, o_c1])
                dist_x, dist_y = self.distance_between_clusters(cluster_nos, x, y, index)
                c_neighbors_distance[i] = dist_x + dist_y
                x_dists[i] = dist_x
                y_dists[i] = dist_y

                if self.cross_other_cluster(new_polys, np.array([c1, c2]), np.array([o_c1, o_c2]), x, y, index, power,
                                            merged_poly_info[o_c2]):
                    cross_neighbor[i] = 1

            neighbor_idx = np.where(np.logical_and(x_dists < nx/2, cross_neighbor == 0))[0]

            if neighbor_idx.size == 0:
                curve_width = new_polys[c1, x_max_c] - new_polys[c1, x_min_c]
                if curve_width > reserve_curve:
                    pass_center = 1 if new_polys[c1, x_min_c] < short_curve < new_polys[c1, x_max_c] else 0
                    # print('no neighbor, width: ', curve_width,  ' pass center: ', pass_center)
                    if pass_center == 1:
                        kept_curves.append(sort_idx_on_miny[c1])
                        c1 += 1
                        continue
                index = np.where(index == o_c1, 0, index)
                new_polys[c1, x_min_c] = non_exist
                self.d_print("remove: ", c1, ' from: ', o_c1)
                log += 'remove '+str(o_c1)
                cluster_changed += 1
                c1 += 1
                continue

            c_neighbors_distance = c_neighbors_distance[neighbor_idx]
            best_neighbors = c_neighbors[neighbor_idx]
            best_neighbor = best_neighbors[np.argsort(c_neighbors_distance)][0]
            o_c2 = sort_idx_on_miny[best_neighbor]
            index = np.where(index == o_c2, o_c1, index)
            self.d_print('merge: ', c1, best_neighbor, ' from: ', o_c1, o_c2)
            log += 'merge '+str(o_c1) + ' and ' + str(o_c2)

            new_polys[c1, x_min_c] = min(new_polys[c1, x_min_c], new_polys[best_neighbor, x_min_c])
            new_polys[c1, x_max_c] = max(new_polys[c1, x_max_c], new_polys[best_neighbor, x_max_c])
            new_polys[c1, y_min_c] = min(new_polys[c1, y_min_c], new_polys[best_neighbor, y_min_c])
            new_polys[c1, y_max_c] = max(new_polys[c1, y_max_c], new_polys[best_neighbor, y_max_c])
            new_polys[best_neighbor, x_min_c] = non_exist
            poly_curves[o_c1, x_min_c:y_max_c+1] = new_polys[c1, x_min_c:y_max_c+1]
            poly_curves[o_c1, 0:power+1] = merged_poly_info[o_c2][0:power+1]
            cluster_changed += 1

        return {'status': 'changed' if cluster_changed >= 1 else 'nochange',
                'index': index, 'kept_curves': kept_curves, 'log': log}

    @staticmethod
    def merge_two_clusters(cluster_nos: np.ndarray,  x: np.ndarray, y: np.ndarray, index: np.ndarray, power: int):
        """ Calculate the polynomial fitting error and distance in case two clusters are merged.

        Parameters:
            cluster_nos (numpy.ndarray): Two cluster id included and the first is the cluster located leftmost.
            x (numpy.ndarray): Array of x coordinates of cluster pixels.
            y (numpy.ndarray): Array of y coordinates of cluster pixels.
            index (numpy.ndarray): Array of cluster id on cluster pixels.
            power (int): Degree of polynomial to fit two clusters.

        Returns:
            tuple: Information of polynomial fit to two clusters,

                * **poly_info** (*numpy.ndarray*): Array contains coefficients of fitting polynomial and area of
                  the cluster after the merge.
                * **errors** (*float*): Least square error of polynomial fitting.
        """

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
        errors = math.sqrt(np.square(np.polyval(poly_info[0:power + 1], x_set) - y_set).mean())
        return poly_info, errors

    @staticmethod
    def distance_between_clusters(cluster_nos: np.ndarray, x: np.ndarray, y: np.ndarray, index: np.ndarray):
        """Find the horizontal and vertical distance between the clusters, the first cluster has smaller x position.

        Args:
            cluster_nos (numpy.ndarray): Array contains the cluster id of two clusters.
            x (numpy.ndarray): Array of x coordinates of cluster pixels.
            y (numpy.ndarray): Array of y coordinates of cluster pixels.
            index (numpy.ndarray): Array of cluster id on cluster pixels.

        Returns:
            tuple: tuple containing:

                * **dist_x** (*float*): The horizontal gap between two clusters.
                  The distance is 0 if there is horizontal overlap between two clusters.
                * **dist_y** (*float*): The vertical gap between two clusters.
                  The distance is 0 if there is vertical overlap between two clusters.

        """
        end_x = np.zeros(2)
        end_y = np.zeros(2)

        for i in range(2):
            idx_c = np.where(index == cluster_nos[i])[0]
            all_x_c = x[idx_c]
            all_y_c = y[idx_c]
            end_idx = np.argsort(all_x_c)[-1] if i == 0 else np.argsort(all_x_c)[0]
            end_x[i] = all_x_c[end_idx]
            end_y[i] = all_y_c[end_idx]

        dist_x = end_x[1] - end_x[0] if end_x[1] > end_x[0] else 0
        dist_y = abs(end_y[0]-end_y[1])
        return dist_x, dist_y

    def cross_other_cluster(self, polys: np.ndarray, cluster_nos_for_polys: np.ndarray, cluster_nos: np.ndarray,
                            x: np.ndarray, y: np.ndarray, index: np.ndarray, power: int,
                            merged_coeffs: np.ndarray):
        """Detect if there is another cluster that will prevent the merge of two given clusters.

        Args:
            polys (numpy.ndarray):  Array contains coefficients of polynomial fit to the clusters and
                the area of the clusters. Each row contains the coefficients and the area for one cluster.
            cluster_nos_for_polys (numpy.ndarray): The map between the `polys` and cluster no.
                Value of cluster_nos_for_polys points to the row index for `polys`.
            cluster_nos (numpy.ndarray): Array containing the cluster id of two clustered to have the merge test.
            x (numpy.ndarray): Array of x coordinates of cluster pixels.
            y (numpy.ndarray): Array of y coordinates of cluster pixels.
            index (numpy.ndarray): Array of cluster id on cluster pixels.
            power (int): Degree of the polynomial to fit the cluster.
            merged_coeffs (numpy.ndarray): The coefficients of polynomial fit to and the area of the clusters in case
                the two clusters of `cluster_nos` are merged.

        Returns:
            bool: The merge is blocked by other cluster if True, or the merge is safe if False.

        """

        width_th = self.get_config_value('order_width_th', 7)
        # merge_coeffs contains the coeffs and range in case the two cluster get merged
        min_x = int(merged_coeffs[power+1])
        max_x = int(merged_coeffs[power+2])
        min_y = int(merged_coeffs[power+3])
        max_y = int(merged_coeffs[power+4])

        cluster_idx = np.where(np.logical_or(index == cluster_nos[0], index == cluster_nos[1]))[0]
        cluster_x = x[cluster_idx]
        cluster_y = y[cluster_idx]

        # x1 of cluster_nos_for_polys[0] is smaller than that of cluster_nos_for_polys[1]
        two_curve_x1 = polys[cluster_nos_for_polys[0], power+2]
        two_curve_x2 = polys[cluster_nos_for_polys[1], power+1]

        _, nx, ny = self.get_spectral_data()

        all_x = list()
        all_y = list()

        # find x belonging to curves intended to be merged
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
                return False
            elif abs(y_max_1-y_min_2) < width_th or abs(y_min_1-y_max_2) < width_th:
                return False

        total_c = np.shape(polys)[0]
        # self.d_print('in cross_other_cluster test for: ', cluster_nos_for_polys, ' from ', cluster_nos)

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

            # cross_point is used to record if the tested cluster is vertically above or below or the same as the
            # as the merged clusters at x locations including ends of the overlap to the merged clusters and
            # the overlap to the gaps of the merged clusters
            cross_point = dict()
            com_min_x = com_max_x = -1

            zero_above = 0
            zero_below = 0
            # find if tested cluster horizontally overlaps out of gap ends of two merged clusters
            if polys[c_idx, power+1] <= max_x and polys[c_idx, power+2] >= min_x:
                com_min_x = int(max(polys[c_idx, power+1], min_x))
                com_max_x = int(min(polys[c_idx, power+2], max_x))
                com_list = [com_min_x, com_max_x] if com_min_x != com_max_x else [com_min_x]

                for curve_end in com_list:
                    if curve_end in np.array(all_x)[gap_x_idx]:   # the end point of overlap meet the gap ends
                        continue

                    # mark if end point of overlap vertically connected to the two merged clusters
                    one_y_val = np.polyval(polys[c_idx, 0:power+1], curve_end)
                    merged_y = np.polyval(merged_coeffs[0:power+1], curve_end)

                    # compare the y location of the tested curve and the merged curves
                    if abs(one_y_val - merged_y) < 1:             # within one pixel range
                        cross_point[int(curve_end)] = 0
                        if one_y_val > merged_y:
                            zero_above += 1
                        elif one_y_val < merged_y:
                            zero_below += 1
                    else:
                        cross_point[int(curve_end)] = (one_y_val - merged_y)/abs(one_y_val-merged_y)

            vals = np.array([v for v in cross_point.values()])

            # check if tested curve has short horizontal overlap with merged curves and vertically meets with the merged
            # curves at all gap ends
            if np.size(vals) != 0:      # when overlap ends not at the gap ends
                same_y_count = np.size(np.where(vals == 0)[0])
                com_dist = abs(com_max_x - com_min_x)

                # overlap ends are the same and same y or overlap with short range and same y at two ends of overlap,
                # meaning no chance to intersect the merged clusters
                if (same_y_count == 1 and com_dist == 0) or (same_y_count == 2 and com_dist < 10):
                    continue

            # check the y location at every gap overlapping with the test curve, cross_point records the y position
            # at selected x positions

            in_gap = 0
            for n_idx in range(0, len(gap_x_idx), 2):
                gap1 = gap_x_idx[n_idx]
                gap2 = gap_x_idx[n_idx+1]

                if (polys[c_idx, power+1] < (all_x[gap2] + offset)) and \
                        (polys[c_idx, power+2] > (all_x[gap1] - offset)):            # overlap or close to the gap area
                    if all_x[gap2] < polys[c_idx, power+1]:                          # no overlap, at the right of gap
                        two_y_val = np.polyval(polys[c_idx, 0:power+1],
                                               np.array([polys[c_idx, power+1], polys[c_idx, power+1]]))
                    elif all_x[gap1] > polys[c_idx, power+2]:                       # no overlap, at the left of gap
                        two_y_val = np.polyval(polys[c_idx, 0:power+1],
                                               np.array([polys[c_idx, power+2], polys[c_idx, power+2]]))
                    else:
                        end1 = max(all_x[gap1], polys[c_idx, power+1])              # overlap with the gap
                        end2 = min(all_x[gap2], polys[c_idx, power+2])
                        two_y_val = np.polyval(polys[c_idx, 0:power+1], np.array([end1, end2]))
                    in_gap = 1

                    for i in [0, 1]:
                        if abs(two_y_val[i] - all_y[gap_x_idx[n_idx+i]]) < 1:
                            cross_point[all_x[gap_x_idx[n_idx+i]]] = 0
                        else:
                            cross_point[all_x[gap_x_idx[n_idx+i]]] = \
                                (two_y_val[i] - all_y[gap_x_idx[n_idx+i]])/abs(two_y_val[i] - all_y[gap_x_idx[n_idx+i]])

            vals = np.array([v for v in cross_point.values()])
            positive_zero_total = np.size(np.where(np.logical_or(vals == 1, vals == 0))[0])
            negative_zero_total = np.size(np.where(np.logical_or(vals == -1, vals == 0))[0])

            # self.d_print('test ', c_idx, ' from ', sort_map[c_idx], ' merged original index: ',
            #             cluster_nos, ' vals: ', vals, ' at points: ', cross_point.keys(), ' x: ',
            #             polys[c_idx, power+1], polys[c_idx, power+2])

            # in case the cluster is not above or below the merged clusters at all overlap ends or gap ends
            if positive_zero_total >= 1 and negative_zero_total >= 1:

                if in_gap == 0:
                    if 0 < zero_above == negative_zero_total and zero_below == 0:
                        continue
                    if 0 < zero_below == positive_zero_total and zero_above == 0:
                        continue

                # self.d_print('  ', cluster_nos, ' cross ', c_idx, ' from ', sort_map[c_idx])
                return True

        return False

    def remove_broken_cluster(self, index: np.ndarray, x: np.ndarray, y: np.ndarray):
        """Remove the cluster which has big opening around the center of the image.

        Parameters:
            index (numpy.ndarray): Array of cluster id on cluster pixels.
            x (numpy.ndarray): Array of x coordinates of cluster pixels.
            y (numpy.ndarray): Array of y coordinates of cluster pixels.

        Returns:
            tuple: Information of cluster pixels after processing,

            * **new_x** (*numpy.ndarray*): Array of  x coordinates of cluster pixels after processing.
            * **new_y** (*numpy.ndarray*): Array of  y coordinates of cluster pixels after processing.
            * **new_index** (*numpy.ndarray*): Array of cluster id on cluster pixels after processing.

        """
        _, nx, _ = self.get_spectral_data()
        gap = max(nx//200, 1)
        # gap = nx//200
        data_x_center = nx//2
        max_idx = np.amax(index)
        changed = 0

        for c in range(1, max_idx+1):
            border_idx = np.where(index == c)[0]
            x_border_set = x[border_idx]
            x_before_center = x_border_set[np.where(x_border_set <= data_x_center)[0]]
            x_after_center = x_border_set[np.where(x_border_set > data_x_center)[0]]
            x_before = np.amax(x_before_center) if x_before_center.size > 0 else 0
            x_after = np.amin(x_after_center) if x_after_center.size > 0 else (nx - 1)
            if (x_after - x_before) > gap:
                index[border_idx] = 0
                changed = 1

        new_x = x.copy()
        new_y = y.copy()
        new_index = index.copy()

        if changed == 1:
            new_x, new_y, new_index = self.reorganize_index(new_index, new_x, new_y)

        return new_x, new_y, new_index

    def find_all_cluster_widths(self, index_t: np.ndarray, new_x: np.ndarray, new_y: np.ndarray,
                                power_for_width_estimation: int = 3,
                                cluster_set: list = None):

        """Compute the top and bottom widths along the order trace.

        Parameters:
            index_t (numpy.ndarray): Array of cluster id on cluster pixels.
            new_x (numpy.ndarray): Array of x coordinates of cluster pixels.
            new_y (numpy.ndarray): Array of y coordinates of cluster pixels.
            power_for_width_estimation (int, optional): Degree of polynomial fit for width estimation,
                degree 2 or 3 is suggested. Defaults to 3. The estimation step skips if it is less than 0.
            cluster_set (list, optional): Set of selected cluster id for width finding. Defaults to None.
                Widths of all clusters are computed if None.

        Returns:
            list: a list of width information for each order trace. Each element in the list is like::

                                    {
                                        'top_edge': float,   # top width along the trace.
                                        'bottom_edge': float # bottom width along the trace.
                                    }

        Raises:
            AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
            TypeError: If there is type error for  `new_x`, `new_y` or `index_t`.
            Exception: If the size of `new_x`, `new_y`, or `index_t` are not the same.

        """
        if (not isinstance(new_x, np.ndarray)) or (not isinstance(new_y, np.ndarray) or
                                                   (not isinstance(index_t, np.ndarray))):
            raise TypeError("input new_x or new_y or index_t type error")

        if new_x is None or new_y is None or index_t is None or new_x.size == 0 or new_y.size == 0 or index_t.size == 0:
            return list(), np.array([])

        if new_x.size != new_y.size:
            raise Exception("size of arrays of x and y coordinates not matched")
        if new_x.size != index_t.size:
            raise Exception("size of array of x and index not matched")

        coeffs, errors = self.curve_fitting_on_all_clusters(index_t, new_x, new_y)
        cluster_points = self.get_cluster_points(coeffs)

        width_default = self.get_config_value('width_default', 6)
        new_index = index_t.copy()
        cluster_coeffs = coeffs.copy()
        max_cluster_no = np.amax(new_index)
        cluster_widths = list()

        if cluster_set is None:
            cluster_set = list(range(1, max_cluster_no+1))

        for n in cluster_set:
            self.d_print('cluster: ', n)
            if n < 1 or n > max_cluster_no or (np.where(index_t == n)[0]).size == 0:
                cluster_widths.append({'top_edge': width_default, 'bottom_edge': width_default})
                continue

            cluster_width_info = self.find_cluster_width_by_gaussian(n, cluster_coeffs, cluster_points)
            cluster_widths.append({'top_edge': cluster_width_info['avg_nwidth'],
                                   'bottom_edge': cluster_width_info['avg_pwidth']})
            self.d_print('top edge: ', cluster_width_info['avg_nwidth'],
                         ' bottom edge: ', cluster_width_info['avg_pwidth'])

        if power_for_width_estimation > 0:
            cluster_widths = self.approximate_width_of_default(cluster_widths, cluster_points, cluster_coeffs,
                                                               power_for_width_estimation)
            self.d_print('after estimation: \n', '\n'.join([str(index+1)+': '+str(w)
                                                            for index, w in enumerate(cluster_widths)]))

        return cluster_widths, coeffs

    def find_cluster_width_by_gaussian(self, cluster_no: int, poly_coeffs: np.ndarray, cluster_points: np.ndarray):
        """Find the width of the cluster uisng Gaussian to approximate the distribution of collected spectral data.

        Parameters:
            cluster_no (int): Cluster id.
            poly_coeffs (numpy.ndarray): Polynomial fitting information and the covered area of all clusters.
            cluster_points (numpy.ndarray): Pixel position (y values) along the polynomial fit of every cluster.

        Returns:
            dict: cluster width information like::

                {
                    'cluster_no': int,   # cluster id.
                    'avg_pwidth': float, # bottom width of cluster.
                    'avg_nwidth': float  # top width of cluster.
                }
        """

        power = self.get_poly_degree()
        width_default = self.get_config_value('width_default', 6)
        width_th = self.get_config_value('order_width_th', 7)
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

        # compute the width along x direction every step
        step = 100
        x_loc1 = np.arange(center_x, int(x_range[1])+1, step)
        x_loc2 = np.arange(center_x-step, int(x_range[0])-1, -step)
        x_loc = np.concatenate((np.flip(x_loc2), x_loc1))
        # cluster_width_info = list()
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

            # finding width at both sides
            x_set = np.arange(prev_mid, cluster_y+1)
            y_set = spec_data[prev_mid:(cluster_y+1), xs]
            new_x_set, new_y_set = self.mirror_data(x_set, y_set, 1)
            if new_x_set.size >= 3:
                gaussian_fit_prev, prev_width, prev_center = \
                    self.fit_width_by_gaussian(new_x_set, new_y_set, cluster_y, xs)
                prev_widths.append(prev_width)
                prev_centers.append(prev_center)

            x_set = np.arange(cluster_y, next_mid+1)
            y_set = spec_data[cluster_y:(next_mid+1), xs]
            new_x_set, new_y_set = self.mirror_data(x_set, y_set, 0)
            if new_x_set.size >= 3:
                gaussian_fit_next, next_width, next_center = \
                    self.fit_width_by_gaussian(new_x_set, new_y_set, cluster_y, xs)
                next_widths.append(next_width)
                next_centers.append(next_center)

        cluster_h = poly_coeffs[cluster_no, power+4] - poly_coeffs[cluster_no, power+3]
        avg_pwidth = self.find_mean_from_histogram(np.array(prev_widths), c_range=[0, cluster_h],
                                                   bin_no=max(int(cluster_h//width_th), 1), cut_at=width_default)
        avg_nwidth = self.find_mean_from_histogram(np.array(next_widths), c_range=[0, cluster_h],
                                                   bin_no=max(int(cluster_h//width_th), 1), cut_at=width_default)

        # self.values_at_width(avg_pwidth, avg_nwidth, cluster_points[cluster_no, center_x], center_x)

        return {'cluster_no': cluster_no,
                'avg_pwidth': avg_pwidth,
                'avg_nwidth': avg_nwidth}

    def find_background_around(self, cluster_no: int, poly_coeffs: np.ndarray, cluster_points: np.ndarray,
                               sorted_idx_per_ypos: dict):
        """Find the background data below and above the specified cluster.

        Args:
            cluster_no (int): cluster id of the cluster to find.
            poly_coeffs (numpy.ndarray): Polynomial fitting information and the covered area of all clusters.
            cluster_points (numpy.ndarray): Cluster points along the trace based on the polynomial fitting.
            sorted_idx_per_ypos (dict): Sorted index of cluster id based on y position.

        Returns:
            numpy.ndarray: background data above and below the cluster along x direction, like::

                [[<background_value_below_trace>_i, <background_value_above_trace>_i], .., ]
                where
                    <background_value_below_trace>_i (float):
                                # background value below the trace at i-th x location.
                    <background_value_abobe_trace>_i (float):
                                # background value above the trace at i-th x location.

        """

        curve_width = self.get_config_value('order_width_th', 7)
        data, nx, ny = self.get_spectral_data()
        total_cluster = np.shape(poly_coeffs)[0]-1
        power = self.get_poly_degree()

        # index_pos = self.get_sorted_index(poly_coeffs, cluster_no, power, nx//2)  ???
        crt_idx = sorted_idx_per_ypos['idx']
        sorted_index = sorted_idx_per_ypos['index_v_pos']

        # background before peak and after peak
        backgrounds = np.zeros((2, nx))
        prev_idx = crt_idx - 1 if crt_idx > 1 else crt_idx
        next_idx = crt_idx + 1 if crt_idx < total_cluster else crt_idx
        three_clusters = np.array([cluster_no, sorted_index[prev_idx], sorted_index[next_idx]])
        min_x = int(np.amax(poly_coeffs[three_clusters, power+1]))
        max_x = int(np.amin(poly_coeffs[three_clusters, power+2]))

        crt_peak_y = cluster_points[cluster_no]

        prev_peak_y = cluster_points[sorted_index[crt_idx - 1]] if crt_idx > 1 \
            else np.where((cluster_points[sorted_index[1]] - 2 * curve_width) < 0, 0,
                          (cluster_points[sorted_index[1]] - 2 * curve_width))
        next_peak_y = cluster_points[sorted_index[crt_idx + 1]] if crt_idx < total_cluster \
            else np.where((cluster_points[total_cluster] + 2 * curve_width) > (ny-1), ny-1,
                          (cluster_points[total_cluster] + 2 * curve_width))

        prev_mid = ((crt_peak_y+prev_peak_y)//2).astype(int)
        total_prev_data = (crt_peak_y - prev_peak_y) + 1
        collect_prev_no = (total_prev_data * 0.2).astype(int)//2
        collect_prev_no = np.where(collect_prev_no >= 1, collect_prev_no, 1)

        next_mid = ((crt_peak_y+next_peak_y)//2).astype(int)
        total_next_data = (next_peak_y - crt_peak_y) + 1
        collect_next_no = (total_next_data * 0.2).astype(int)//2
        collect_next_no = np.where(collect_next_no >= 1, collect_next_no, 1)

        for x in range(min_x, max_x+1):
            data_collected = data[max(prev_mid[x]-collect_prev_no[x], 0):min(prev_mid[x]+collect_prev_no[x]+1, ny), x]

            hist, bin_edge = np.histogram(data_collected, bins=4)
            max_hist_idx = np.argmax(hist)
            data_idx = np.where(np.logical_and(data_collected >= bin_edge[max_hist_idx],
                                               data_collected <= bin_edge[max_hist_idx+1]))[0]
            backgrounds[0, x] = np.mean(data_collected[data_idx])

            data_collected = data[max(0, next_mid[x]-collect_next_no[x]):min(next_mid[x]+collect_next_no[x]+1, ny), x]
            hist, bin_edge = np.histogram(data_collected, bins=4)
            max_hist_idx = np.argmax(hist)
            data_idx = np.where(np.logical_and(data_collected >= bin_edge[max_hist_idx],
                                               data_collected <= bin_edge[max_hist_idx+1]))[0]
            backgrounds[1, x] = np.mean(data_collected[data_idx])

        for i in range(0, 2):
            if min_x > 0:
                backgrounds[i, 0:min_x] = backgrounds[i, min_x]
            if max_x < nx-1:
                backgrounds[i, max_x+1:] = backgrounds[i, max_x]
        return backgrounds

    @staticmethod
    def mirror_data(x_set: np.ndarray, y_set: np.ndarray, mirror_side: int):
        """Mirror y value to the left or right side of x_set.

        Args:
            x_set (numpy.ndarray): Array of x values.
            y_set (numpy.ndarray): Array of y values paired to each of  `x_set`.
            mirror_side (int): Mirror direction. Mirror to the left side of `x_set` at if 0, or to the right side of
                `x_set` if 1.

        Returns:
            tuple: Data after mirroring,

                * **x_new_set** (*numpy.ndarray*): Array containing x coordinates from left to the right
                  after mirroring.
                * **y_new_set** (*numpy.ndarray*): Array containing y coordinates relevant to `x_new_set`.

        """
        total = np.size(x_set) - 1

        if mirror_side == 0:   # left side
            x_other_side = x_set[0:total]-total
            y_other_side = np.flip(y_set[1:])
            x_new_set = np.concatenate((x_other_side, x_set))
            y_new_set = np.concatenate((y_other_side, y_set))
        else:                  # ride side
            x_other_side = x_set[1:]+total
            y_other_side = np.flip(y_set[0:total])
            x_new_set = np.concatenate((x_set, x_other_side))
            y_new_set = np.concatenate((y_set, y_other_side))

        return x_new_set, y_new_set

    @staticmethod
    def fit_width_by_gaussian(x_set: np.ndarray, y_set: np.ndarray, center_y: float, xs: int,
                              sigma: float = 3.0):
        """Find the width using Gaussian fitting.

        Fit the x, y set of data using Gaussian and find the width by looking at sigma of the Gaussian fit.

        Parameters:
            x_set (numpy.ndarray): x data set.
            y_set (numpy.ndarray): y data set.
            center_y (float): Estimation of y value at the center from `x_set`.
            xs (int): x location for `center_y`.
            sigma (float, optional): Magnitude of standard deviation to get the width. Defaults to 3.0.

        Returns:
            tuple: Gaussian fit results:

                * **gaussian_fit**: Gaussian fit object.
                * **width** (*float*): Width found by Gaussian fit.
                * **gaussian_center** (*float*): Mean of Gaussian fit.

        """

        g_init = models.Gaussian1D(mean=center_y)

        gaussian_fit = FIT_G(g_init, x_set, y_set)

        if abs(gaussian_fit.mean.value - center_y) <= 1.0:
            width = gaussian_fit.stddev.value*sigma
            gaussian_center = gaussian_fit.mean.value
        else:
            gaussian_center = gaussian_fit.mean.value
            width = gaussian_fit.stddev.value * sigma

        return gaussian_fit, width, gaussian_center

    @staticmethod
    def find_mean_from_histogram(vals: np.ndarray, bin_no: int = 4, c_range: list = None, cut_at: float = None):
        """Find the mean value based on the histogram of the data set.

        Calculate the mean of the data selected from the given data set based on the histogram of the set.

        Args:
            vals (numpy.ndarray): Array of values.
            bin_no (int): Bin number for the histogram.
            c_range (list, optional): Range for making histogram.
            cut_at (float, optional): Upper limit of the mean value. Defaults to None.

        Returns:
           float: Mean value of the data set.

        """

        if c_range is None:
            r = None
        else:
            r = (c_range[0], c_range[1])

        hist, bin_edge = np.histogram(vals, bins=bin_no, range=r)

        if np.size(np.where(hist != 0)[0]) == 0:
            if cut_at is None:
                mean_val = np.mean(vals)
            else:
                mean_val = cut_at
            return mean_val

        max_hist_idx = np.argmax(hist)
        if max_hist_idx > 1 and (hist[max_hist_idx-1] > hist[max_hist_idx]*0.6):
            edge_min = bin_edge[max_hist_idx-1]
        else:
            edge_min = bin_edge[max_hist_idx]

        if max_hist_idx < (len(hist)-1) and (hist[max_hist_idx+1] > hist[max_hist_idx]*0.6):
            edge_max = bin_edge[max_hist_idx+2]
        else:
            edge_max = bin_edge[max_hist_idx+1]

        data_idx = np.where(np.logical_and(vals >= edge_min,
                                           vals <= edge_max))[0]

        mean_val = np.mean(vals[data_idx])

        if cut_at is not None and mean_val > cut_at:
            mean_val = cut_at
        return mean_val

    def approximate_width_of_default(self, cluster_widths: list, cluster_points: np.ndarray,
                                     cluster_coeffs: np.ndarray, poly_fit_power: int = 2):
        """Approximate unresolved width using least square polynomial fit to determined widths.

        Parameters:
            cluster_widths (list): Top and bottom widths of all clusters, like
                                  [{'top_edge': <number>, 'bottom_edge': <number>},...].
            cluster_points (numpy.ndarray): Arrays contains cluster points (y values) along the trace based on
                the polynomial fitting. Each row includes y values along x axis of one cluster.
            cluster_coeffs (numpy.ndarray): Coefficients of Polynomial fit and area of all order traces.
            poly_fit_power (int, optional): Degree of polynomial fit for width estimation, degree 2 or 3 is suggested.
                Defaults to 2.

        Returns:
            list: top and bottom widths of all clusters after using polynomial approximation, like::

                        [
                            {
                                'top_edge': float,     # top width of first cluster,
                                'bottom_edge': float   # bottom width of first cluster
                            },
                            :
                            {
                                'top_edge': float,     # top width of last cluster,
                                'bottom_edge': float   # bottom width of last cluster
                            }
                        ]

        """
        _, nx, ny = self.get_spectral_data()
        power = self.get_poly_degree()

        h_center = nx//2
        total_cluster = np.shape(cluster_points)[0] - 1

        y_middle_list = np.zeros(total_cluster)
        for c in range(1, np.shape(cluster_points)[0]):
            if cluster_coeffs[c, power+1] <= h_center <= cluster_coeffs[c, power+2]:
                y_middle_list[c-1] = cluster_points[c, h_center]
            else:
                y_middle_list[c-1] = np.polyval(cluster_coeffs[c, 0:power+1], h_center)

        widths_all = list()   # [ <np.array of bottom_width>, <np.array of top widths> ]
        width_default = self.get_config_value('width_default', 6)

        widths_all.append(np.array([c_widths['bottom_edge'] for c_widths in cluster_widths]))
        widths_all.append(np.array([c_widths['top_edge'] for c_widths in cluster_widths]))
        for widths in widths_all:
            c_idx = np.where(widths != width_default)[0]    # index set of non-cut widths
            s_idx = np.where(widths == width_default)[0]    # index set of cut widths

            if np.size(s_idx) == 0 or (np.size(c_idx) <= (poly_fit_power+1)):
                continue
            coeffs = np.polyfit(y_middle_list[c_idx], widths[c_idx], poly_fit_power)  # poly fit on all non-cut width
            w_sel = np.polyval(coeffs, y_middle_list[s_idx])   # approximate the widths by poly fit
            widths[s_idx] = w_sel

        new_cluster_widths = [{'top_edge': widths_all[self.UPPER][i], 'bottom_edge': widths_all[self.LOWER][i]}
                              for i in range(total_cluster)]

        return new_cluster_widths

    def get_cluster_points(self, polys_coeffs: np.ndarray):
        """Compute cluster points (y values) along the fitting curve within x range of the cluster.

        Args:
            polys_coeffs (numpy.ndarray): Polynomial fit coefficients and area on clusters.

        Returns:
            numpy.ndarray: Arrays contains cluster points (y values) along the trace based on
            the polynomial fitting. Each row includes y values along x axis of one cluster.

        """
        power = self.get_poly_degree()
        _, nx, ny = self.get_spectral_data()
        s_coeffs = np.shape(polys_coeffs)
        cluster_points = np.zeros((s_coeffs[0], nx), dtype=int)
        for c in range(1, s_coeffs[0]):
            s_x = int(max(0, polys_coeffs[c, power+1]))
            e_x = int(min(nx, polys_coeffs[c, power+2]+1))
            x_val = np.arange(s_x, e_x, dtype=int)
            pt_vals = np.round(np.polyval(polys_coeffs[c, 0:power+1], x_val))
            pt_vals = np.where(pt_vals < 0, 0, pt_vals)
            pt_vals = np.where(pt_vals >= ny, ny-1, pt_vals)
            cluster_points[c, s_x:e_x] = pt_vals

        return cluster_points

    @staticmethod
    def curve_fitting_on_one_cluster(cluster_no: int, index: np.ndarray, x: np.ndarray, y: np.ndarray, power: int,
                                     poly_info: np.ndarray = None):
        """Finding polynomial to fit the cluster pixels.

        Args:
            cluster_no (int): cluster id
            index (numpy.ndarray): Array of cluster id of cluster pixels.
            x (numpy.ndarray): Array of x coordinates of cluster pixels.
            y (numpy.ndarray): Array of y coordinates of cluster pixels.
            power (int): Degree of fitting polynomial.
            poly_info (numpy.ndarray, optional): Array contains the coefficients of polynomial fit and the area of
                the cluster. Defaults to None.

        Returns:
            tuple: Coefficients and errors from polynomial fit:

            * **poly_info** (*numpy.ndarray*): Array contains coefficients of fitting polynomial from higher degree
              to the lower and the area enclosing the cluster, minimum x, maximum x, minimum y and maximum y.
            * **error** (*float*): Polynomial fitting error.
            * **area** (*list*): Cluster area, like [min_x, max_x, min_y, max_y].

        """

        if poly_info is None:
            poly_info = np.zeros(power+5)   # containing polynomial coefficients, power+1, and cluster area range

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

    def curve_fitting_on_all_clusters(self, index: np.ndarray, x: np.ndarray, y: np.ndarray):
        """Do polynomial fitting on cluster pixels for all clusters.

        Args:
            index (numpy.ndarray): Array of cluster id on cluster pixels.
            x (numpy.ndarray): Array of x coordinates on cluster pixels.
            y (numpy.ndarray): Array of y coordinates on cluster pixels.

        Returns:
            tuple: Coefficients and errors from polynomial fit:

                * **poly_all** (*numpy.ndarray*): Array contains coefficients of polynomial fit and the area of
                  all clusters. Each row contains the coefficients and the area for one cluster. Please see `Returns`
                  in :func:`~alg.OrderTraceAlg.curve_fitting_on_one_cluster`  for the detail of  each row.
                * **errors** (*numpy.ndarray*): Array contains least square errors of polynomial fit to all clusters.

        """

        power = self.get_poly_degree()
        max_index = np.amax(index)
        poly_all = np.zeros((max_index+1, power+5))
        errors = np.zeros(max_index+1)

        for c in range(1, max_index+1):
            poly, error, area = self.curve_fitting_on_one_cluster(c, index, x, y, power)
            poly_all[c, ] = poly
            errors[c] = error

        return poly_all, errors

    def curve_fitting_on_peaks(self, crt_coeffs: np.ndarray):
        """
        Polynomial fit on the peaks of spectral data along the cluster.

        Args:
            crt_coeffs (numpy.ndarray): Array contains coefficients of polynomial fit on cluster pixels and the area
                of all clusters.

        Returns:
            dict: containing polynomial fit on cluster peaks, like::

                {
                    'coeffs': numpy.ndarray,
                                        # Coefficients of polynomial fit on cluster peaks
                                        # and the area of the clusters.
                    'peak_pixels': numpy.ndarray,
                                        # Peak points (y) of all clusters along x axis.
                    'cluster_pixels': numpy.ndarray,
                                        # Cluster points (y) of all clusters along x axis.
                                        # Please see Results of get_cluster_points().
                    'errors': numpy.ndarray
                                        # Arrays contains the error from the polynomial
                                        # fit on cluster peaks.
                }

        """

        power = self.get_poly_degree()
        all_cluster_points = self.get_cluster_points(crt_coeffs)
        cluster_pixels_at_peaks = self.get_cluster_peak_pixels(all_cluster_points, crt_coeffs)
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

    def get_cluster_peak_pixels(self, cluster_pixels: np.ndarray, poly_coeffs: np.ndarray):
        """Get the peak location for every cluster along x direction

        Args:
            cluster_pixels (numpy.ndarray):  Cluster points (y values) of all clusters along x axis. Please see
                `Returns` of :func:`~alg.OrderTraceAlg.get_cluster_points()`.
            poly_coeffs (numpy.ndarray): Array contains coefficients of polynomial fit on cluster pixels and the area
                of all clusters.

        Returns:
            numpy.ndarray: Peak points (y values) of all clusters along x axis. Each row includes peak points
            for one cluster.

        """
        power = self.get_poly_degree()
        spectral_data, nx, ny = self.get_spectral_data()

        # get distance between two fitting curves (from cluster_pixels)
        size = np.shape(cluster_pixels)
        v_dists = self.get_cluster_distance_at_x(cluster_pixels, poly_coeffs)
        all_peak_pixels = np.zeros((size[0], size[1]), dtype=int)

        # get peaks for every cluster within min_x and max_x range
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

    def get_cluster_distance_at_x(self, cluster_pixels: np.ndarray, coeffs: np.array, x_loc: int = None):
        """Vertical distance between every two clusters at x location, x_loc.

        Calculate the vertical distance of every neighboring clusters based on the given cluster pixels
        at x location, `x_loc`. The cluster is sorted first based on the y position at `x_loc`.

        Args:
            cluster_pixels (numpy.ndarray): Cluster position (y values) along x axis. It could be cluster points from
                the polynomial fit per cluster pixels or cluster peaks.
            coeffs (numpy.ndarray): Coeffients of polynomial fit and area of the clusters.
            x_loc (int, optional): x position for distance calculation. Defaults to None.

        Returns:
            numpy.ndarray: Distances between the neighboring clusters, like::

                 '''
                 Assume peak_width is the return, then
                 peak_width[1] is distance between the first cluster and second cluster.
                 peak_width[2] is distance between the second cluster and third cluster.
                 etc.

                 The order of the clusters is based on y position of the clusters.
                 '''
        """
        power = self.get_poly_degree()
        _, nx, _ = self.get_spectral_data()

        x_center = x_loc if x_loc is not None else nx//2
        y_at_center = cluster_pixels[:, x_center]

        s = np.shape(coeffs)
        x_exist = np.zeros(s[0], dtype=bool)
        x_exist[np.where(np.logical_and(x_center >= coeffs[:, power+1], x_center <= coeffs[:, power+2]))[0]] = 1

        y_sort_idx = np.argsort(y_at_center)
        y_at_center_sorted = y_at_center[y_sort_idx]
        x_exist_sorted = x_exist[y_sort_idx]
        curve_width = self.get_config_value('order_width_th', 7.0)
        peak_width = np.ones(s[0], dtype=float) * curve_width

        # distance between (1st, 2nd), (2nd, 3rd) ... (last to the 2nd, last) clusters
        for c in range(1, s[0]-1):
            if x_exist_sorted[c] and x_exist_sorted[c+1]:
                peak_width[y_sort_idx[c]] = abs(y_at_center_sorted[c+1] - y_at_center_sorted[c])//2

        peak_width[0] = peak_width[1]
        peak_width[-1] = peak_width[-2]

        return peak_width

    @staticmethod
    def common_member(a: list, b: list):
        """ Find if there is common elements of two lists.

        Args:
            a (list): First list.
            b (list): Second list.

        Returns:
            bool: True if there is common element, or False.

        """
        a_set = set(a)
        b_set = set(b)
        if a_set & b_set:
            return True
        else:
            return False

    @staticmethod
    def sort_cluster_on_loc(clusters: list, loc):
        """ Sort the clusters based on the specified location key.

        Args:
            clusters (list): List of clusters to be sorted.
            loc (str/int): The key that the sorting is based on.

        Returns:
            list: Sorted result.

        """
        clusters.sort(key=lambda c: c[loc])
        return clusters

    @staticmethod
    def sort_cluster_segments(segments: list):
        """Sort a set of segments based on the first number contained in each segment.

        Args:
            segments (list): Array of segments. Each element in `segments` is list-like type.

        Returns:
            list: Sorted result.

        """

        segments.sort(key=lambda s: s[0])
        return segments

    @staticmethod
    def get_sorted_index(poly_coeffs: np.ndarray, cluster_no: int, power: int, x_loc: int):
        """Get sorted index for a cluster.

        Do sorting on the list with cluster id based on the cluster's position (y values) at `x_loc` and find the
        index of the cluster with `cluster_no` in the newly sorted list.

        Args:
            poly_coeffs (numpy.ndarray): Array contains coefficients of polynomial fit and the area of the clusters.
            cluster_no (int): id of the cluster to get the index from the sorted list.
            power (int): Degree of the polynomial fit to the clusters.
            x_loc (int): x position for the sorting.

        Returns:
            dict: contains the sorted information, like::

                {
                    'idx': int,
                            # index of the cluster `cluster_no` from the new sorted list.
                    'index_v_pos': numpy.ndarray
                            # sorted list of cluster id based on the y position at `x_loc`.
                }

        """
        max_idx = np.shape(poly_coeffs)[0]-1

        centers = np.zeros(max_idx+1)
        for c in range(1, max_idx+1):
            centers[c] = np.polyval(poly_coeffs[c, 0:power+1], x_loc)

        center_index = np.argsort(centers)
        idx = np.where(center_index == cluster_no)[0]
        return {'idx': idx[0], 'index_v_pos': center_index}

    def sort_cluster_in_y(self, cluster_coeffs: np.ndarray):
        """Sort cluster based on vertical position.

        Args:
            cluster_coeffs (np.ndarray): Array contains coefficients of polynomial fit and the area of the clusters.

        Returns:
            np.ndarray: Sorted list of cluster id based on the vertical position of the clusters.

        """
        total_cluster = np.shape(cluster_coeffs)[0]-1
        _, nx, ny = self.get_spectral_data()
        power = self.get_poly_degree()

        min_x = np.amax(cluster_coeffs[1:total_cluster+1, power+1])
        max_x = np.amin(cluster_coeffs[1:total_cluster+1, power+2])

        if min_x > max_x:
            return np.arange(0, total_cluster+1, dtype=int)

        c_x = min(max(nx//2, min_x), max_x)

        sorted_info = self.get_sorted_index(cluster_coeffs, 1, power, c_x)
        return sorted_info['index_v_pos']

    @staticmethod
    def get_cluster_size(c_id: int, index: np.ndarray, x: np.ndarray, y: np.ndarray):
        """Compute the width, height, total pixels and pixel index collection per specified cluster id.

        Args:
            c_id (int): Cluster id.
            index (np.ndarray): Array of cluster id on cluster pixels.
            x (np.ndarray): Array of x coordinates of cluster pixels.
            y (np.ndarray): Array of y coordinates of cluster pixels.

        Returns:
            tuple: Size information of the cluster,

                * **w** (*int*): width of the cluster `c_id`.
                * **h** (*int*): height of the cluster `c_id`.
                * **total_pixel** (*int*): total pixel contained in the cluster `c_id`.
                * **crt_idx** (*numpy.ndarray*): Array contains the index from `index` for all pixels belonging
                  to cluster `c_id`.


        """
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

    def reorganize_index(self, index: np.ndarray, x: np.ndarray, y: np.ndarray, return_map: bool = False):
        """Remove cluster pixels with unsigned cluster no. and reorder the cluster.

        Remove the cluster pixels with cluster number less than 1 and re-assign the cluster id to existing cluster
        pixels.

        Args:
            index (numpy.ndarray): Array of cluster id on cluster pixels.
            x (numpy.ndarray): Array of x coordinates of cluster pixels.
            y (numpy.ndarray): Array of y coordinates of cluster pixels.
            return_map (bool, optional): Return map between old cluster id and new cluster id if True.

        Returns:
            tuple: Information of cluster pixels after processing,

                * **new_x** (*numpy.ndarray*): Array of x coordinates of cluster pixels after processing.
                * **new_y** (*numpy.ndarray*): Array of y coordinates of cluster pixels after processing.
                * **new_index** (*numpy.ndarray*): Array of cluster id on cluster pixels after processing.
                * **return_map** (*dict*): Map between old cluster id and new cluster id like::

                    {
                        <old cluster id> int: <new cluster id> int
                    }
        """

        new_x, new_y, new_index = self.remove_unassigned_cluster(x, y, index)
        if new_index is None:
            return new_x, new_y, new_index

        max_index = np.amax(new_index)
        unique_index = np.sort(np.unique(new_index))
        full_index = np.arange(1, max_index+2)

        not_exist_idx = full_index[np.where(np.logical_not(np.isin(full_index, unique_index)))[0]]
        if len(not_exist_idx) == 0:
            if return_map is False:
                return new_index, new_x, new_y
            else:
                return new_index, new_x, new_y, dict(zip(unique_index, unique_index))

        rnt_index = np.copy(new_index)

        offset = 0
        bound1 = -1

        for b in not_exist_idx:
            bound2 = b
            inbound_idx = np.where(np.logical_and(new_index > bound1, new_index < bound2))[0]
            if np.size(inbound_idx) != 0:
                rnt_index[inbound_idx] = new_index[inbound_idx] - offset
            offset += 1
            bound1 = bound2

        unique_result = np.sort(np.unique(rnt_index))

        if return_map is False:
            return new_x, new_y, rnt_index
        else:
            return new_x, new_y, rnt_index, dict(zip(unique_index,  unique_result))

    @staticmethod
    def remove_unassigned_cluster(x: np.ndarray, y: np.ndarray, index: np.ndarray):
        """Remove the cluster pixels which has no cluster number assigned.

        Args:
            x (numpy.ndarray): Array of x coordinates of cluster pixels.
            y (numpy.ndarray): Array of y coordinates of cluster pixels.
            index (numpy.ndarray): Array of cluster id on cluster pixels.

        Returns:
            tuple: Information of cluster pixels after processing,

                * **x_r** (*numpy.ndarray*): Array of x coordinates of cluster pixels after processing.
                * **y_r** (*numpy.ndarray*): Array of y coordinates of cluster pixels after processing.
                * **index_r** (*numpy.ndarray*): Array of cluster id on cluster pixels after processing.
        """

        idx_cluster = np.where(index > 0)[0]   # the pixel which is assigned cluster number
        x_r = None
        y_r = None
        index_r = None
        if idx_cluster.size != 0:
            x_r = x[idx_cluster]                   # x, y coordinate of pixel which is assigned cluster number
            y_r = y[idx_cluster]
            index_r = index[idx_cluster]
        return x_r, y_r, index_r

    def make_2d_data(self, index: np.ndarray, x: np.ndarray, y: np.ndarray, selected_clusters: np.ndarray = None):
        """Create 2D data based on cluster pixels related information.

        Args:
            x (numpy.ndarray): Array of x coordinates of cluster pixels.
            y (numpy.ndarray): Array of y coordinates of cluster pixels.
            index (numpy.ndarray): Array of cluster id on cluster pixels.
            selected_clusters (numpy.ndarray, optional) : Make 2D data based on selected clusters only.
                Defaults to None.

        Returns:
            numpy.ndarray: 2D data with pixels set as 1 on cluster pixels, or 0 on non cluster pixels.

        """

        _, nx, ny = self.get_spectral_data()

        imm = np.zeros((ny, nx), dtype=np.uint8)
        if selected_clusters is None:
            selected_clusters = np.arange(1, np.amax(index) + 1, dtype=int)

        for idx in selected_clusters:
            crt_idx = np.where(index == idx)[0]
            imm[y[crt_idx], x[crt_idx]] = 1

        return imm

    @staticmethod
    def rms_of_polys(poly_coeff1: np.ndarray, poly_coeff2: np.ndarray, power: int):
        """Root mean square of difference between two polynomial fitting.

        Args:
            poly_coeff1 (numpy.ndarray): Coefficients of polynomial fit and area information of first cluster.
            poly_coeff2 (numpy.ndarray): Coefficients of polynomial fit and area information of second cluster.
            power (int): Degree of polynomial fit to the cluster.

        Returns:
            numpy.ndarray: Root mean square of difference between two polynomial fitting on all clusters.

        """

        total_cluster = np.shape(poly_coeff1)[0]-1
        rms = np.zeros(total_cluster+1)
        for c in range(1, total_cluster+1):
            x_set = np.arange(int(poly_coeff1[c, power+1]), int(poly_coeff1[c, power+2])+1)
            y1_clusters = np.polyval(poly_coeff1[c, 0:power+1], x_set)
            y2_clusters = np.polyval(poly_coeff2[c, 0:power+1], x_set)
            rms[c] = np.sqrt(np.mean((y1_clusters - y2_clusters)**2))
        return rms

    def write_cluster_info_to_csv(self, cluster_widths: list, cluster_coeffs: np.ndarray, csv_file: str):
        """Write the coefficients of  polynomial fit, area and top/bottom widths of clusters to a csv file.

        Args:
            cluster_widths (list): Array contains the top and bottom widths of clusters, like::

                [
                    {
                        'top edge': float,      # top width of first cluster,
                        'bottom edge': float    # bottom width of first cluster
                    }, ....,
                    {
                        'top edge': float,     # top width of last cluster,
                        'bottom edge': float   # bottom width of last cluster
                    }
                ]

            cluster_coeffs (numpy.ndarray): Array contains coefficients of polynomial fit and the area of the clusters.
            csv_file (str): Filename of csv file to write to.

        Returns:
            None.

        """
        power = self.get_poly_degree()
        sorted_index = self.sort_cluster_in_y(cluster_coeffs)

        with open(csv_file, mode='w') as result_file:
            result_writer = csv.writer(result_file)
            for i in range(1, len(sorted_index)):
                idx = sorted_index[i]           # cluster id
                c_widths = cluster_widths[idx-1]
                prev_width = c_widths.get('bottom_edge')
                next_width = c_widths.get('top_edge')

                row_data = list()
                for t in range(power, -1, -1):  # from lower degree to higher degree
                    row_data.append(cluster_coeffs[idx, t])
                row_data.append(self.float_to_string(prev_width))    # bottom width
                row_data.append(self.float_to_string(next_width))    # top width
                row_data.append(int(cluster_coeffs[idx, power+1]))    # left x
                row_data.append(int(cluster_coeffs[idx, power+2]))    # right x

                result_writer.writerow(row_data)

    def write_cluster_info_to_dataframe(self, cluster_widths: list, cluster_coeffs: np.ndarray):
        """Write the coefficients of polynomial fit, area and top/bottom widths of order trace to DataFrame object.

        Args:
            cluster_widths (list): Array contains the top and bottom widths of clusters, like::

                [
                    {
                        'top edge': float,     # top width of first cluster
                        'bottom edge': float   # bottom width of first cluster
                    }, ....,
                    {
                        'top edge': float,     # top width of last cluster
                        'bottom edge': float   # bottom width of last cluster
                    }
                ]

            cluster_coeffs (numpy.ndarray): Array contains coefficients of polynomial fit and the area of the clusters.

        Returns:
            Pandas.DataFrame: Instance of DataFrame containing columns (for polynomial of degree 3) like,

                *Coeff0*, *Coeff1*, *Coeff2*, *Coeff3*, *BottomEdge*, *TopEdge*, *X1*, *X2*

                to contain coefficients of polynomial fit from lower order to higher, bottom and top widths, and the
                left and right boundary of the orders.

        """
        if cluster_widths is None or cluster_coeffs is None:
            return None

        total_row = np.shape(cluster_coeffs)[0]
        if total_row <= 1:
            return None

        power = self.get_poly_degree()
        trace_table = {}
        column_names = ['Coeff'+str(i) for i in range(power+1)]
        for i in range(power+1):
            trace_table[column_names[i]] = cluster_coeffs[1:, power - i]

        trace_table['BottomEdge'] = np.zeros(total_row-1)
        trace_table['TopEdge'] = np.zeros(total_row-1)
        for i in range(total_row-1):
            trace_table['BottomEdge'][i] = self.float_to_string(cluster_widths[i]['bottom_edge'])
            trace_table['TopEdge'][i] = self.float_to_string(cluster_widths[i]['top_edge'])
        trace_table['X1'] = cluster_coeffs[1:, power+1].astype(int)
        trace_table['X2'] = cluster_coeffs[1:, power+2].astype(int)

        df = pd.DataFrame(trace_table)
        df.attrs['STARTROW'] = self.data_range[0]
        df.attrs['ENDROW'] = self.data_range[1]
        df.attrs['STARTCOL'] = self.data_range[2]
        df.attrs['ENDCOL'] = self.data_range[3]
        df.attrs['POLY_DEG'] = self.get_poly_degree()
        return df

    @staticmethod
    def float_to_string(afloat):
        """Convert float to string by taking 4 decimal digits.

        Args:
            afloat (float): A float number.

        Returns:
            str: String of a float number with 4 decimal digits.

        """
        new_str = f"{afloat:.4f}"
        return new_str

    def time_check(self, t_start, step_msg):
        """Count the time and display the span of the time.

        Args:
            t_start (float): Start time to count.
            step_msg (str): Message to print.

        Returns:
            float: End of time.

        """
        t_end = time.time()
        self.t_print(step_msg, (t_end - t_start), 'sec.')
        return t_end

    def add_file_logger(self, filename: str = None):
        """Add file to log debug information.

        Args:
            filename (str, optional): Filename of the log file. Defaults to None.

        Returns:
            None.

        """
        self.enable_debug_print(filename is not None)
        self.debug_output = filename

    def extract_order_trace(self, power_for_width_estimation: int = -1, data_range = None, show_time: bool = False,
                            print_debug: str = None, rows_to_reset = None, cols_to_reset = None):
        """ Order trace extraction.

        The order trace extraction includes the steps to smooth the image, locate the clusters, form clusters,
        remove and trim noisy clusters, merge the clusters to form order traces, model the order trace using polynomial
        fit and find the top and bottom widths along the traces.

        Args:
            power_for_width_estimation (int): Degree of polynomial fit for trace width estimation. Defaults to -1.
            data_range (list): Area of the data, `x1, x2, y1, y2`, to be processed,
                where *x1*, *y1* and *x2*, *y2* are the corner coordinates of the area.
                *x1*, *x2* or *y1*, *y2* respectively represents the horizontal or vertical position relative to
                the first column or row of the image when it is greater than or equal to 0, otherwise
                the position relative to the last column or the last row.
            show_time (bool, optional): Show running time if True. Defaults to False.
            print_debug (str, optional): Print debug information to stdout if it is provided as empty string,
                a file with path `print_debug` if it is non empty string, or no print if it is None.
                Defaults to None.
            rows_to_reset (list, optional): Collection of rows to reset. Default to None.
            cols_to_reset (list, optional): Collection of columns to reset. Default to None.

        Returns:
            dict: order trace extraction and analysis result, like::

                {
                    'order_trace_result': Padas.DataFrame,
                                          # table storing coefficients of polynomial
                                          # fit, bottom/top width, and left/right boundary.
                    'cluster_index': numpy.ndarray, # Array of cluster id of cluster pixels.
                    'cluster_x': numpy.ndarray, # Array of x coordinates of cluster pixels.
                    'cluster_y': numpy.ndarray  # Array of y coordinates of cluster pixels.
                }

        """
        imm_spec, nx, ny = self.get_spectral_data()
        self.enable_time_profile(show_time)
        self.add_file_logger(print_debug)

        self.set_data_range(data_range)

        t_start = time.time()
        # locate cluster
        self.d_print("*** locate cluster", info=True)
        cluster_xy = self.locate_clusters(rows_to_reset, cols_to_reset)
        t_start = self.time_check(t_start, '*** locate cluster: ')

        # assign cluster id and do basic cleaning
        self.d_print("*** form cluster", info=True)
        x, y, index_r = self.form_clusters(cluster_xy['x'], cluster_xy['y'])
        t_start = self.time_check(t_start, "*** assign cluster: ")

        # advanced cleaning
        self.d_print("*** advanced clean cluster", info=True)
        new_x, new_y, new_index, all_status = self.advanced_cluster_cleaning_handler(index_r, x, y)
        t_start = self.time_check(t_start,  "*** advanced clean cluster: ")

        # clean clusters along bottom and top border
        self.d_print("*** clean border", info=True)
        new_x, new_y, new_index = self.clean_clusters_on_borders(new_x, new_y, new_index, top_border=ny-1,
                                                                 bottom_border=0)
        t_start = self.time_check(t_start, "*** clean border: ")

        # merge clusters & remove broken cluster
        self.d_print("*** merge cluster and remove cluster with big opening in the center ", info=True)
        new_x, new_y, new_index = self.merge_clusters_and_clean(new_index, new_x, new_y)
        t_start = self.time_check(t_start,
                                  "*** merge cluster and remove cluster with big opening in the center: ")
        # find width
        self.d_print("*** find widths", info=True)
        all_widths, cluster_coeffs = self.find_all_cluster_widths(new_index, new_x, new_y,
                                                                  power_for_width_estimation=power_for_width_estimation)
        self.time_check(t_start, "*** find widths: ")

        self.d_print("*** write result to Pandas Dataframe", info=True)
        df = self.write_cluster_info_to_dataframe(all_widths, cluster_coeffs)
        return {'order_trace_result': df, 'cluster_index': new_index, 'cluster_x': new_x, 'cluster_y': new_y}
