
import configparser
import numpy as np
import json
from scipy import linalg, ndimage
import sys, os

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.models.level0 import KPF0


class OrderTraceAlg:

    def __init__(self, data, config=None):
        self.flat_data = data
        self.config = config

    def get_config_value(self, property: str, default=''):
        if self.config is not None:
            if isinstance(default, int):
                return self.config.getint(property, default)
            elif isinstance(default, float):
                return self.config.getfloat(property, default)
            else:
                return self.config.get(property, default)
        else:
            return default

    def get_poly_degree(self):
        return self.get_config_value('fitting_poly_degree', 3)

    def get_spectral_data(self):
        """
        get spectral information including data and size
        """

        try:
            assert self.flat_data
        except AssertionError:
            return None

        ny, nx = np.shape(self.flat_data.data)
        return self.flat_data.data, nx, ny

    @staticmethod
    def opt_filter(y_data: np.ndarray, par: int, weight: np.ndarray = None):
        """
        A smoothing filter
        """

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
        """
        set value of columns or rows by val, the setting is made on row if row_or_column is 0 or on column
        """

        if reset_ranges is None:
            pos = np.where(imm > 0)
            return imm, pos[1], pos[0]

        for range in reset_ranges:
            range_idx_set = np.arange(range[0], range[1], dtype=int)
            if row_or_column == 0:
                for r in range_idx_set:
                    imm[r, :] = val
            else:
                for c in range_idx_set:
                    imm[:, c] = val

        pos = np.where(imm > 0)
        return imm, pos[1], pos[0]

    def locate_clusters(self):
        """ 
        Find cluster pixels from 2D data array

        Returns:
            cluster_info (tuple): result of cluster, (x, y, imm) where x is width of imm, y is height of imm
                                                                 and imm is 2D data after processing.
        """
        # flat data array and dimension
        image_data, n_col, n_row = self.get_spectral_data()

        # Parameters
        filter_par = self.get_config_value('filter_par', 20)
        # filter_par = self.config.getint('filter_par', 20)
        noise = self.get_config_value('locate_cluster_noise', 0.0)
        # noise = self.config.getfloat('locate_cluster_noise', 0.0)
        mask = self.get_config_value('cluster_mask', 1)
        # mask = self.config.getint('cluster_mask', 1)

        rows_str = self.get_config_value('rows_to_reset', '')
        cols_str = self.get_config_value('cols_to_reset', '')

        rows_to_reset = None
        if rows_str:
            rows_list = json.loads(rows_str)
            if isinstance(rows_list, list):
                if all([isinstance(r, list) and len(r) == 2 for r in rows_list]):
                    rows_to_reset = [r if r[0] >= 0 else [r[0]+n_row, r[1]+n_row] for r in rows_list]

        cols_to_reset = None
        if cols_str:
            cols_list = json.loads(cols_str)
            if isinstance(cols_list, list):
                if all([isinstance(c, list) and len(c) == 2 for c in cols_list]):
                    cols_to_reset = [c if c[0] >= 0 else [c[0]+n_col, c[1]+n_col] for c in cols_list]

        print('rows_to_reset:', rows_to_reset)
        print('cols to reset:', cols_to_reset)
        # binary array
        imm = np.zeros((n_row, n_col), dtype=np.uint8)

        for col in range(n_col):
            mm = image_data[:, col] + noise - self.opt_filter(image_data[:, col], filter_par)
            mm_pos = np.where(mm > 0, mm, 0)
            h = 0.5*np.sort(mm_pos)[mm_pos.size//2]
            imm[:, col][mm > (h+1)] = mask

        y, x = np.where(imm > 0)  # ex: (array([4, 5, 6, 7]), array([2, 2, 2, 2]))
        print('total cluster pixel: ', y.size)

        # correction on filtered image (ex. for NEID flat, stacked_2fiber_flat.fits)
        if rows_to_reset is not None:
            imm, x, y = self.reset_row_or_column(imm, rows_to_reset)
            print('pos size after row reset: '+str(np.size(y)) + ' ' + str(np.size(x)))

        if cols_to_reset is not None:
            imm, x, y = self.reset_row_or_column(imm, cols_to_reset, row_or_column=1)
            print('pos size after column reset: ' + str(np.size(y)) + ' ' + str(np.size(x)))

        return {'x': x, 'y': y, 'cluster_image': imm}

    def collect_clusters(self, c_x: np.ndarray, c_y: np.ndarray):
        """ identify cluster unit from c_x and c_y meaning the x, y position of all cluster pixels
        Parameters:
            c_x (array): x coordinates for all cluster pixels
            c_y (array): y coordinates for all cluster pixels

        Returns:
            out (dict): {<y1>: clusters (array), <y1+1>: clusters (array).....},
                               where the value of each key representing clusters which vertical position stop at y2.
                               clusters (array) is a list of cluster like:
                                    cluster(dict): 'x1', 'x2', 'y1', 'y2', <y>, for all y in [y1, y2]
                                    value of <y> is dict, like {'segments': <sorted_segment>(array)},
                                                    where each segment contain starting and ending index of c_x
                        ex: {10: [{'x1': 20, 'x2': 30, 'y1': 9,  'y2': 10, 9:{'segments': [[4, 8], [12, 13]]},
                                                                           10:{'segments': [[100, 107], [109, 118]]}},
                                  {'x1': 50, 'x2': 77, 'y1': 5, 'y2': 10, 5:{'segments': [...]}, 6:{....} ...., 10:{}}],
                             11: [{<cluster ends at 11>}, {<cluster ends at 11>}...]}

        """
        x, y = c_x, c_y
        _, nx, ny = self.get_spectral_data()

        try:
            assert(x.size == y.size)
            assert(np.max(x) < nx)
            assert(np.max(x) < ny)
        except AssertionError:
            return None

        # clusters_endy_dict contains key:value, like y:[<cluster with max_y at y>, <....>] : list of cluster
        #
        # cluster: a dict with properties: y1, y2, x1, x2, number(y1), number(y1+1), ...number(y2),
        #                                  value of number(y1) is like {"segments": [seg_1, seg_2,...]}
        #                                  where seg_i: [idx_1, idx_2] containing index for x, y

        clusters_endy_dict = dict()      # contain clusters end at y (0 to ny-1)
        nx_prev_cluster_id = [list() for cx in range(nx)]   # check ???

        for cy in range(ny):
            # if cy%10 == 0:
            print(cy, ' ', end='')

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

            nx_prev_cluster_id = [list() for cx in range(nx)]    # check ???
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

        return clusters_endy_dict

    def remove_cluster_noise(self, clusters_endy_dict: dict, x_index: np.ndarray, y_index: np.ndarray, thres=None):
        """ remove noisy cluster per pixel number and the size of the cluster. Id is assigned to non-noisy cluster
        Parameters:
            clusters_endy_dict (dict): collection of clusters collected by collect_clusters,
                                      please see 'Returns' of collect_clusters for more detail
            x_index (array): x coordinates of cluster pixels
            y_index (array): y coordinates of cluster pixels
            thres (int): optional size threshold for removing the noisy cluster

         Returns:
               out (dict): index: array of cluster id associated with each cluster pixel
                           n_regions: total clusters (??? check here)
        """
        _, nx, ny = self.get_spectral_data()
        w_th = nx//100
        h_th = ny//800
        if thres is None:
            thres = h_th * w_th

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
                        if total_pixel > thres:    # keep this cluster
                            break

                    # assign cluster id to non-noisy cluster
                    if total_pixel > thres:
                        for y_c in range(one_cluster['y1'], one_cluster['y2']+1):
                            for c_seg in one_cluster[y_c]['segments']:
                                index[c_seg[0]:(c_seg[1]+1)] = cluster_no
                        cluster_no += 1
                        break
                c_idx += 1

        # remove narrow cluster
        max_idx = np.amax(index)
        h_size_th = ny//100
        w_size_th = nx//100
        for c_id in np.arange(1, max_idx+1):
            w, h, t_p, crt_cluster_idx = self.get_cluster_size(c_id, index, x_index, y_index)

            if w <= w_size_th and h <= h_size_th:
                index[crt_cluster_idx] = 0

        n_regions = np.amax(index) + 1 if np.amin(index) == 0 else np.amax(index)

        return {'index': index, 'n_regions': n_regions}

    def form_cluster(self, c_x: np.ndarray, c_y: np.ndarray, thres=None):
        """ form the cluster with assigned id
        Parameters:
            c_x (array): x coordinates for all cluster pixels
            c_y (array): y coordinates for all cluster pixels
            thres (int): size threshold used for removing noisy cluster

        Returns:
            out (tuple): array of x coordinates of cluster pixels
                         array of y coordinates of cluster pixels
                         array of cluster id associated each clusters
        """
        clusters_all_y = self.collect_clusters(c_x, c_y)
        index_info = self.remove_cluster_noise(clusters_all_y, c_x, c_y, thres)
        new_x, new_y, new_index = self.reorganize_index(index_info['index'], c_x, c_y)
        return new_x, new_y, new_index

    @staticmethod
    def common_member(a: list, b: list):
        """ find if there is common element from two list """
        a_set = set(a)
        b_set = set(b)
        if a_set & b_set:
            return True
        else:
            return False

    @staticmethod
    def sort_cluster_on_loc(clusters: list, loc: str):
        """
        sort the clusters base on the specified location key
        """
        clusters.sort(key=lambda c: c[loc])
        return clusters

    @staticmethod
    def get_segments_from_index_list(id_list: np.ndarray,  loc: np.ndarray):
        """
        collect segments based on location list, loc,  and the index set for the location list
        """
        segments = list()

        distcont_idx = np.where((loc[id_list] - loc[np.roll(id_list, 1)]) != 1)[0]
        # collect all segments
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

    @staticmethod
    def sort_cluster_segments(segments: list):
        """
        sort the segment based on the first location number
        """

        segments.sort(key=lambda s: s[0])
        return segments

    @staticmethod
    def get_cluster_size(c_id: int, index: np.ndarray, x: np.ndarray, y: np.ndarray):
        """
        compute the width, height, total pixels and pixel index collection of specified cluster
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
        """
        remove pixels with unsigned cluster no and reorder the cluster pixel and id
        """
        new_x, new_y, new_index = self.remove_unassigned_cluster(x, y, index)
        max_index = np.amax(new_index)
        unique_index = np.sort(np.unique(new_index[1:]))
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

        unique_result = np.sort(np.unique(rnt_index[1:]))

        if return_map is False:
            return new_x, new_y, rnt_index
        else:
            return new_x, new_y, rnt_index, dict(zip(unique_index,  unique_result))

    @staticmethod
    def remove_unassigned_cluster(x: np.ndarray, y: np.ndarray, index: np.ndarray):
        """
        remove the pixel which has no cluster number assigned
        """

        idx_cluster = np.where(index > 0)[0]   # the pixel which is assigned cluster number
        x_r = x[idx_cluster]                   # x, y coordinate of pixel which is assigned cluster number
        y_r = y[idx_cluster]
        index_r = index[idx_cluster]
        return x_r, y_r, index_r

    def make_2d_data(self, index: np.ndarray, x: np.ndarray, y: np.ndarray, selected_clusters: np.ndarray = None):
        """ make 2D data based on cluster number and location

        Parameters:
            x (array): x coordinates of cluster pixels
            y (array): y coordinates of cluster pixels
            index (array): cluster number on pixels which x, y stand for
            selected_clusters (array) : make 2D data based on selected clusters only

        Returns:
            out (array): 2D data with pixel set as 1 on the selected clusters
        """

        _, nx, ny = self.get_spectral_data()

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
                break
            y_cond = np.where(y == cy)[0]
            if selected_clusters is None:
                nz_idx_at_cy = y_cond[np.where(index[y_cond] != 0)[0]]
            else:
                nz_idx_at_cy = y_cond[np.where(np.isin(index[y_cond], selected_clusters))[0]]
            imm[cy, x[nz_idx_at_cy]] = 1
        return imm

