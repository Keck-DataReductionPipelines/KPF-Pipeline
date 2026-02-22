"""
Base class for KPF Masters modules.
"""
from astropy.stats import mad_std
import numpy as np
import warnings

from kpfpipe import DEFAULTS
from kpfpipe.constants import NROW, NCOL
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.utils.kpf_parse import get_datecode, fetch_filepath
from kpfpipe.utils.stats import flag_outliers

DEFAULTS.update({
    'nframe_stream': 5,
    'stack_sigma':, 5.0
})


class BaseMastersModule:
    def __init__(self, l0_file_list):
        self.l0_file_list = l0_file_list

        for k in DEFAULTS.keys():
            self.__setattr__(k, config.get(k,DEFAULTS[k]))


    @staticmethod
    def load_frame(fn):
        l0_obj = KPF0.from_fits(fn)
        return l0_obj


    @staticmethod
    def assemble_frame(l0_obj):
        l1_obj = ImageAssembly(l0_obj).perform()
        return l1_obj


    def stack_frames(self, l0_file_list=None, nframe_stream=None, sigma=None):
        """
        Stacks full frame images and computes clipped mean and variance
          * For N <= 5, statistics are computed directly
          * For N > 5, computation uses streaming Welford's algorithm

          Note: need to cache direct frames as e.g. self.direct_frames to avoid re-reading
          Note: make sure that files are clearing from memory after read (in KPF0?)
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if nframe_stream is None:
            nframe_stream = self.nframe_stream
        if sigma is None:
            sigma = self.stack_sigma

        if len(l0_file_list) <= nframe_stream:
            mean, var = self.compute_direct_statistics(sigma = sigma)
        else:
            mean, var = self.compute_streaming_statistics(sigma = sigma)

        return mean, var


    def compute_direct_statistics(self, l0_file_list=None, nframe_stack=None, nframe_cache=None, sigma=None):
        """
        nframe_stack : maximum number of frames to stack for computing statistics
        nframe_cache : maximum number of L1 objects to cache

        Notes
        -----
        This function will automatically cache a subset of L1 objects for downstream use.
        The maximum number of cached items allowed is set by nframe_cached, but the actual number
        of cached items will be the smallest of len(l0_file_list), nframe_stack, and nframe_cache.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if nframe_stack is None:
            nframe_stack = self.nframe_stream
        if nframe_cache is None:
            nframe_cache = self.nframe_stream
        if sigma is None:
            sigma = self.stack_sigma
        
        nframe = np.min([nframe_stack,len(l0_file_list)])
        ncache = np.min([nframe, nframe_cache])

        if not hasattr(self, '_l1_obj_cache'):
            self._l1_obj_cache = {}
        
        data_cube = {}

        for chip in self.chips:
            data_cube[f'{chip}_CCD'] = np.zeros((nframe,NROW,NCOL),dtype=np.float32)
            data_cube[f'{chip}_VAR'] = np.zeros((nframe,NROW,NCOL),dtype=np.float32)

        i = 0
        failure = 0

        for fn in l0_file_list:
            if i >= nframe:
                break

            if fn in self._l1_obj_cache.keys():
                for chip in self.chips:
                    data_cube[f'{chip}_CCD'] = self._l1_obj_cache[fn].data[f'{chip}_CCD']
                    data_cube[f'{chip}_VAR'] = self._l1_obj_cache[fn].data[f'{chip}_VAR']
                i += 1

            else:
                try:
                    l0_obj = self.load_frame(fn)
                    l1_obj = self.assemble_frame(l0_obj)

                    for chip in self.chips:
                        data_cube[f'{chip}_CCD'][i] = l1_obj.data[f'{chip}_CCD']
                        data_cube[f'{chip}_VAR'][i] = l1_obj.data[f'{chip}_VAR']
    
                    if len(self._l1_obj_cache.keys()) < ncache:
                        self._l1_obj_cache[fn] = l1_obj

                    i += 1

                except FileNotFoundError as e:
                    warnings.warn(f"Skipping {fn} in compute_direct_statistics: {e}")
                    failure += 1
                    continue

        if failure / nframe > 0.2:
            raise ValueError(f"more than 20% of frames in stack failed to load")

        stats = {}

        for chip in chips:
            stats[f'{chip}_CCD'] = {}
            stats[f'{chip}_VAR'] = {}

            # TODO: streamline quality control
            if np.any(np.isnan(data_cube[f'{chip}_CCD'])):
                raise ValueError(f"NaN values detected in {chip}_CCD data cube")
            if np.any(np.isnan(data_cube[f'{chip}_VAR'])):
                raise ValueError(f"NaN values detected in {chip}_VAR data cube")

            out = flag_outliers(data_cube[f'{chip}_CCD'], axis=0) | flag_outliers(data_cube[f'{chip}_VAR'], axis=0)

            for suffix in ['CCD', 'VAR']:
                ext_name = f'{chip}_{suffix}'

                stats[ext_name]['nframe'] = np.sum(~out, axis=0)

                S = np.sum(np.where(out, 0, data_cube[ext_name]), axis=0)
                stats[ext_name]['mean'] = S / stats[ext_name]['nframe']

                S2 = np.sum(np.where(out, 0, (data_cube[ext_name] - stats[ext_name]['mean'])**2), axis=0)
                stats[ext_name]['rms'] = np.sqrt(S2 / stats[ext_name]['nframe'])

        return stats


    def compute_streaming_statistics(self, sigma_clip=5.0):
        """
        Computes mean and variance using Welford's algorithm
        Optimized to reduce RAM usage at the expense of compute speed
        Estimates approximate mean and variance directly to perform outlier rejection
        """
        if sigma_clip:
            approx_mean, approx_var = self.compute_direct_mean_and_variance(sigma_clip = sigma_clip, nframe_max = 5)
            lower = approx_mean - sigma_clip * np.sqrt(approx_var)
            upper = approx_mean + sigma_clip * np.sqrt(approx_var)
        else:
            lower = -np.inf
            upper = np.inf
            
        S = np.zeros((NROW,NCOL), dtype=float)
        S2 = np.zeros((NROW,NCOL), dtype=float)
        count = np.zeros((NROW,NCOL), dtype=int)
        failure = 0

        for i, obs_id in enumerate(self.obs_ids):
            # TODO: scale by exposure time
            try:
                l0_obj = self.load_frame(obs_id)
                frame = self.assemble_frame(l0_obj)
            except Exception as e:
                logger.warning(f"Skipping {obs_id} in compute_streaming_mean_and_variance: {e}")
                failure += 1
                continue

            if failure / len(self.obs_ids) > 0.2:
                raise ValueError(f"more than 20% of frames in stack failed to load")

            mask = (frame >= lower) & (frame <= upper)
            S += frame * mask
            S2 += frame ** 2 * mask
            count += mask.astype(int)
    
        bad = count <= 0.5 * len(self.obs_ids)
        count = np.where(bad, 1, count)

        clipped_mean = np.where(bad, np.nan, S / count)
        clipped_var = np.where(bad, np.nan, S2 / count - clipped_mean ** 2)

        return clipped_mean, clipped_var