
import configparser
import numpy as np

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.models.level0 import KPF0


class OrderTraceAlg:

    def __init__(self, data: KPF0, config = configparser.ConfigParser):
        self.flat_data = data
        self.config = config

    def opt_filter(self, y_data: np.ndarray, 
                         par: int, 
                         weight: np.ndarray=None):
        """ 
        A smoothing filter
        """

        n = y_data.size
        # Check for some input preliminaries
        try: 
            assert(y_data.ndim == 1)
            assert(par > 0)
        except AssertionError:
            return y_data

        if weight is not none:
            # a weight is not provided as input
            wgt = np.ones((1, n), dtype=np.float64)[0]
        else: 
            wgt = np.reshape(weight, (1, -1))
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

    def remove_vertical_line(self, pos: tuple, imm: np.ndarray, run:bool = False):

        if run: 
            # What are these?
            v_idx = np.array([439, 440, 446, 447])
            for cx in v_idx:
                imm[:, cx] = 0
            pos = np.where(imm>0)
            return pos, imm
        else: 
            # do nothing
            return pos, imm

    def locate_clusters(self):
        """ 
        Find cluster pixels from 2D data array

        Parameters:
            filename (str): spectral file name or use the one already loaded
            filter (number): the width of the filter for detection of pixels that belong to local maxima

        Returns:
            cluster_info (dict): result of cluster, like
                                        {'x': array([2, 2, 2, 2],
                                        'y': array([4, 5, 6, 7])}
        """
        # flat data array and dimension
        image_data = self.flat_data.data
        n_row, n_col = image_data.shape
        # Parameters
        filter_par = self.config.get('filter_par', 20)
        noise = self.config.get('locate_cluster_noise', 0.0)
        mask = self.config('mask', 1.0)
        remove_vertical = self.config('remove_vertical', False)

        # binary array
        imm = np.zeros((n_row, n_col), dtype=np.uint8)

        for col in range(n_col):
            mm = image_data[:, col] + noise - self.opt_filter(image_data[:, col], filter_par)
            mm_pos = np.where(mm>0, mm, 0)
            h = 0.5*np.sort(mm_pos)[mm_pos.size//2]
            imm[:, col][mm>(h+1)] = mask

        pos = np.where(imm>0)  # ex: (array([4, 5, 6, 7]), array([2, 2, 2, 2]))
        pos imm = self.remove_vertical_line(pos, imm, run=remove_vertical)
        
        # (x, y, im_map) tuple
        return (pos[1], pos[0], imm)

    def collect_clusters(self, cluster_info):
        '''
        Take x and y coordinates of pixels from (nx, ny) 2D array and identify 
        individual clusters from pixels.
        --TODO-- Finish this
        '''
        x, y, im_map = cluster_info
        nx, ny = self.flat_data.data.shape
        try:
            assert(a.size == y.size)
            assert(np.max(x) >= nx)
            assert(np.max(x) >= ny)
        except AssertionError:
            return None

        # clusters_endy_dict contains key/value, 
        # ike y(cluster with max_y at y)/list of cluster
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

        self.clusters_all_y = clusters_endy_dict
        return clusters_endy_dict

