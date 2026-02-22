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
    'stack_sigma': 5.0
})

# TODO: scale stacks by exposure time
# TODO: double-check variance calculations for correctness
# TODO: check consistency of nframe_stream and nframe_direct keywords
# TODO: line profile and remove uneccessary array allocations
# TODO: stack frames into L1 object


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
          * For nframe_stream <= 5, statistics are computed directly
          * For nframe_stream > 5, computation uses streaming Welford's algorithm

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
            stats = self.compute_direct_statistics(sigma = sigma)
        else:
            stats = self.compute_streaming_statistics(sigma = sigma)

        return None


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

        for chip in self.chips:
            stats[f'{chip}_CCD'] = {}
            stats[f'{chip}_VAR'] = {}

            # TODO: replace with robust quality control
            if np.any(np.isnan(data_cube[f'{chip}_CCD'])):
                raise ValueError(f"NaN values detected in {chip}_CCD data cube")
            if np.any(np.isnan(data_cube[f'{chip}_VAR'])):
                raise ValueError(f"NaN values detected in {chip}_VAR data cube")

            out = (
                flag_outliers(data_cube[f'{chip}_CCD'], sigma, axis=0) | 
                flag_outliers(data_cube[f'{chip}_VAR'], sigma, axis=0)
            )

            for suffix in ['CCD', 'VAR']:
                ext_name = f'{chip}_{suffix}'

                N = np.sum(~out, axis=0)
                stats[ext_name]['nframe'] = N

                X = np.sum(np.where(out, 0, data_cube[ext_name]), axis=0)
                stats[ext_name]['mean'] = np.where(N >= 1, X / N, np.nan)

                X2 = np.sum(np.where(out, 0, (data_cube[ext_name] - stats[ext_name]['mean'])**2), axis=0)
                stats[ext_name]['rms'] = np.where(N >= 2, np.sqrt(X2 / N), np.nan)

        return stats


    def compute_streaming_statistics(self, l0_file_list=None, nframe_direct=None, sigma=None):
        """
        Computes mean and variance using Welford's algorithm
        
        Note that the variance computed by Welford's algorithm is the frame-to-frame
        variance NOT the photon noise variance, which is the weighted sum of variance
        across all frames in the stack.


        Optimized to reduce RAM usage at the expense of compute speed
        Estimates approximate mean and variance directly to perform outlier rejection

        nframe_direct : maximum number of frames pass to compute_direct_statistics
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if nframe_direct is None:
            nframe_direct = self.nframe_stream
        if sigma is None:
            sigma = self.stack_sigma

        approx_stats = self.compute_direct_statistics(l0_file_list=l0_file_list,
                                                     nframe_stack=nframe_direct, 
                                                     nframe_cache=nframe_direct, 
                                                     sigma=sigma
                                                     )
        
        if len(l0_file_list) <= nframe_direct:
            return approx_stats  # exact_stats = approx_stats

        exact_stats = {}

        for chip in self.chips:
            for suffix in ['CCD', 'VAR']:
                ext_name = f'{chip}_{suffix}'

                exact_stats[ext_name] = {}
                exact_stats[ext_name]['nframe'] = np.zeros((NROW,NCOL),dtype=int)
                exact_stats[ext_name]['mean'] = np.zeros((NROW,NCOL),dtype=np.float32)
                exact_stats[ext_name]['X2'] = np.zeros((NROW,NCOL),dtype=np.float32)
                
                approx_stats[ext_name]['lower'] = approx_stats[ext_name]['mean'] - sigma * approx_stats[ext_name]['rms']
                approx_stats[ext_name]['upper'] = approx_stats[ext_name]['mean'] + sigma * approx_stats[ext_name]['rms']

        failure = 0

        for fn in l0_file_list:
            if fn in self._l1_obj_cache.keys():
                l1_obj = self._l1_obj_cache[fn]
                
            else:
                try:
                    l0_obj = self.load_frame(fn)
                    l1_obj = self.assemble_frame(l0_obj)
                except FileNotFoundError as e:
                    warnings.warn(f"Skipping {fn} in compute_streaming_statistics: {e}")
                    failure += 1
                    continue

            if failure / len(l0_file_list) > 0.2:
                raise ValueError(f"more than 20% of frames in stack failed to load")
        
            for chip in self.chips:
                for suffix in ['CCD', 'VAR']:
                    ext_name = f'{chip}_{suffix}'

                    frame = l1_obj.data[ext_name]
                    lower = approx_stats[ext_name]['lower']
                    upper = approx_stats[ext_name]['upper']
                    mask = (frame >= lower) & (frame <= upper)

                    exact_stats[ext_name]['nframe'] += mask.astype(int)

                    valid = mask & (exact_stats[ext_name]['nframe'] > 0)

                    delta = np.where(valid, frame - exact_stats[ext_name]['mean'], 0)

                    exact_stats[ext_name]['mean'] += (
                        np.where(valid, delta / exact_stats[ext_name]['nframe'], 0)
                    )

                    delta2 = np.where(valid, frame - exact_stats[ext_name]['mean'], 0)

                    exact_stats[ext_name]['X2'] += np.where(valid, delta * delta2, 0)
    
        for chip in self.chips:
            for suffix in ['CCD', 'VAR']:
                ext_name = f'{chip}_{suffix}'
                
                N = exact_stats[ext_name]['nframe']
                X2 = exact_stats[ext_name]['X2']

                exact_stats[ext_name]['rms'] = np.where(N >= 2, np.sqrt(X2 / N), np.nan)

        exact_stats.pop('X2')

        return exact_stats