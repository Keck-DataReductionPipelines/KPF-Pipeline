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
    'stack_sigma': 5.0,
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

        Returns
            dict
                sum (2D array; total photons collected, after rejecting outliers)
                mean (2D array; frame-to-frame mean, after normalizing exptime)
                rms (2D array; frame-to-frame rms, after normalizing exptime)
            
            exptime (float; integrated exposure time across full stack)

            Note that due to outlier rejection sum = mean * exptime only for pixels
            which are valid in every frame in the stack

        IMG = CCD['mean']
        SNR = CCD['sum'] / np.sqrt(VAR['sum'])
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


    def _compute_direct_statistics(self, l0_file_list=None, nframe_stack=None, nframe_cache=None, sigma=None):
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
        exptime = np.zeros(nframe,dtype=np.float32)

        for chip in self.chips:
            data_cube[f'{chip}_CCD'] = np.zeros((nframe,NROW,NCOL),dtype=np.float32)
            data_cube[f'{chip}_VAR'] = np.zeros((nframe,NROW,NCOL),dtype=np.float32)

        i = 0
        failure = 0

        for fn in l0_file_list:
            if i >= nframe:
                break

            if fn in self._l1_obj_cache:
                l1_obj = self._l1_obj_cache[fn]

            else:
                try:
                    l0_obj = self.load_frame(fn)
                    l1_obj = self.assemble_frame(l0_obj)

                    if len(self._l1_obj_cache) < ncache:
                        self._l1_obj_cache[fn] = l1_obj

                except FileNotFoundError as e:
                    warnings.warn(f"Skipping {fn} in compute_direct_statistics: {e}")
                    failure += 1
                    if failure / nframe > 0.2:
                        raise ValueError(f"more than 20% of frames in stack failed to load")
                    continue

            exptime[i] = l1_obj.header['PRIMARY']['ELAPSED']
            
            for chip in self.chips:
                data_cube[f'{chip}_CCD'][i] = l1_obj.data[f'{chip}_CCD']
                data_cube[f'{chip}_VAR'][i] = l1_obj.data[f'{chip}_VAR']

            i += 1

        if i < 2:
            raise ValueError(f"At least two frames neeed for frame stacking, got {i}")

        if i < nframe:
            exptime = exptime[:i]
            for k in data_cube.keys():
                data_cube[k] = data_cube[k][:i]

        if np.any(exptime < 0):
            raise ValueError(f"Exposure times cannot be negative; exptime = {exptime}")
        
        if np.all(exptime > 0):
            T = exptime[:, None, None]
        elif np.all(exptime == 0):
            T = np.ones_like(exptime)[:, None, None]
        else:
            raise ValueError(f"Exposure times must be all zero or all non-zero; exptime = {exptime}")

        stats = {}

        for chip in self.chips:
            stats[f'{chip}_CCD'] = {}
            stats[f'{chip}_VAR'] = {}

            out = (
                flag_outliers(data_cube[f'{chip}_CCD'] / T, sigma, axis=0) | 
                flag_outliers(data_cube[f'{chip}_VAR'] / T, sigma, axis=0)
            )

            N = np.sum(~out, axis=0)
            valid = ~out
            
            for suffix in ['CCD', 'VAR']:
                ext_name = f'{chip}_{suffix}'

                D = data_cube[ext_name]
                S0 = np.sum(D, axis=0, where=valid)        
                S1 = np.sum(D / T, axis=0, where=valid)
                S2 = np.sum((D / T)**2, axis=0, where=valid)

                good = N > 0
                
                mean = np.zeros_like(S1)
                mean[good] = S1[good] / N[good]

                var = np.zeros_like(S2)
                var[good] = S2[good] / N[good] - mean[good]**2
                
                rms = np.sqrt(np.maximum(var,0))

                stats[ext_name]['clipped_sum'] = S0
                stats[ext_name]['norm_mean'] = mean
                stats[ext_name]['norm_rms'] = rms

        return stats, np.sum(exptime)


    def _compute_streaming_statistics(self, l0_file_list=None, nframe_direct=None, sigma=None):
        """
        Computes mean and rms using Welford's algorithm
        
        Note that the variance computed by Welford's algorithm is the frame-to-frame
        rms NOT the photon noise variance.

        Optimized to reduce RAM usage at the expense of compute speed
        Estimates approximate mean and rms directly to perform outlier rejection

        nframe_direct : maximum number of frames pass to _compute_direct_statistics
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if nframe_direct is None:
            nframe_direct = self.nframe_stream
        if sigma is None:
            sigma = self.stack_sigma

        approx_stats, exptime_direct = (
            self._compute_direct_statistics(
                l0_file_list=l0_file_list,         
                nframe_stack=nframe_direct, 
                nframe_cache=nframe_direct, 
                sigma=sigma
            )
        )

        zero_exptime = exptime_direct == 0
        
        if len(l0_file_list) <= nframe_direct:
            return approx_stats  # exact_stats = approx_stats

        exact_stats = {}
        exptime = 0.0

        for chip in self.chips:
            for suffix in ['CCD', 'VAR']:
                ext_name = f'{chip}_{suffix}'

                exact_stats[ext_name] = {}
                exact_stats[ext_name]['clipped_sum'] = np.zeros((NROW,NCOL),dtype=np.float32)
                exact_stats[ext_name]['norm_mean'] = np.zeros((NROW,NCOL),dtype=np.float32)
                exact_stats[ext_name]['norm_rms'] = np.zeros((NROW,NCOL),dtype=np.float32)

                mean = approx_stats[ext_name]['norm_mean']
                dev = approx_stats[ext_name]['norm_rms'] * sigma
                
                approx_stats[ext_name]['norm_lower'] = mean - dev
                approx_stats[ext_name]['norm_upper'] = mean + dev

        failure = 0

        for fn in l0_file_list:
            if fn in self._l1_obj_cache:
                l1_obj = self._l1_obj_cache[fn]
                
            else:
                try:
                    l0_obj = self.load_frame(fn)
                    l1_obj = self.assemble_frame(l0_obj)
                except FileNotFoundError as e:
                    warnings.warn(f"Skipping {fn} in compute_streaming_statistics: {e}")
                    failure += 1
                    if failure / len(l0_file_list) > 0.2:
                        raise ValueError(f"more than 20% of frames in stack failed to load")
                    continue

            exptime_total += l1_obj.header['PRIMARY']['ELAPSED']

            if (exptime_total == 0) != zero_exptime:
                raise ValueError(f"Exposure times must be all zero or all non-zero")
            
            for chip in self.chips:
                for suffix in ['CCD', 'VAR']:
                    ext_name = f'{chip}_{suffix}'
                    
                    frame = l1_obj.data[ext_name]
                    exptime = l1_obj.header['PRIMARY']['ELAPSED']

                    lower = approx_stats[ext_name]['norm_lower']
                    upper = approx_stats[ext_name]['norm_upper']
                    valid = (frame / exptime >= lower) & (frame / exptime <= upper)

                    delta = frame - exact_stats[ext_name]['norm_mean']
                    delta[~valid] = 0

                    mean = exact_stats['exptime']['']


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
                exact_stats[ext_name].pop('X2')

        return exact_stats