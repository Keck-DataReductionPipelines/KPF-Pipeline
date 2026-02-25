"""
Base class for KPF Masters modules.
"""
from astropy.stats import mad_std
import numpy as np
import warnings

from kpfpipe import DEFAULTS, DETECTOR
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.utils.kpf_parse import get_datecode, fetch_filepath
from kpfpipe.utils.stats import flag_outliers

DEFAULTS.update({
    'nframe_stream': 6,
    'stack_sigma': 5.0,
})

DEFAULTS.update(DETECTOR)
NROW = DETECTOR['ccd']['nrow']
NCOL = DETECTOR['ccd']['ncol']

# TODO: move frame-caching logic into load_frame
# TODO: double-check variance calculations for correctness
# TODO: check consistency of nframe_stream and nframe_direct keywords
# TODO: line profile and remove uneccessary array allocations
# TODO: build output object
# TODO: decide how to handle ImageAssembly config


class BaseMastersModule:
    def __init__(self, l0_file_list, config={}):
        self.l0_file_list = l0_file_list

        for k in DEFAULTS.keys():
            self.__setattr__(k, config.get(k,DEFAULTS[k]))


    def load_frame(self, fn, ncache=None):
        """
        Loads and L0 file from disk and performs image assembly

        fn (str): filename 

        Returns
        -------
            l1_obj (KPF1)
            exit_code (bool) : 1 if file was successfully loaded and processed, 0 otherwise

        Notes
        -----
        This function will automatically cache a subset of L1 objects for downstream use.
        The maximum number of cached items allowed is set by ncache.
        """
        if ncache is None:
            ncache = self.nframe_stream - 1

        if not hasattr(self, '_l1_obj_cache'):
            self._l1_obj_cache = {}

        success = True
        failure = False

        if fn in self._l1_obj_cache:
            l1_obj = self._l1_obj_cache[fn]

        else:
            try:
                l0_obj = KPF0.from_fits(fn)
                l1_obj = ImageAssembly(l0_obj).perform()

                if len(self._l1_obj_cache) < ncache:
                    self._l1_obj_cache[fn] = l1_obj

            except FileNotFoundError as e:
                warnings.warn(f"Failed to load {fn}: {e}")
                return None, failure
        
        return l1_obj, success


    def stack_frames(self, l0_file_list=None, nstream=None, sigma=None):
        """
        Stacks full frame images and computes clipped mean and variance
        
         - if len(l0_file_list) < nstream, statistics are computed directly
         - if len(l0_file_list) >= nstream, statistics are computed using
           Welford's algorithm for streaming mean and variance

        Returns
            dict
                sum (2D array; total photons collected, after rejecting outliers)
                mean (2D array; frame-to-frame mean, after normalizing exptime)
                rms (2D array; frame-to-frame rms, after normalizing exptime)
            
            exptime (float; integrated exposure time across full stack)

            Note that due to outlier rejection, sum = mean * exptime only for pixels
            which are valid in every frame in the stack

        IMG: CCD['mean']
        SNR: CCD['sum'] / np.sqrt(VAR['sum'])
        MASK: bool, pixels where at least 50% of frames in stack are good
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if nstream is None:
            nstream = self.nframe_stream
        if sigma is None:
            sigma = self.stack_sigma

        nframe = len(l0_file_list)

        if nframe < 2:
            raise ValueError(f"Stacking requires at least two frames, got {nframe}")

        if nframe < nstream:
            stats, exptime = self._compute_direct_statistics(l0_file_list, nstream - 1, sigma)
        else:
            stats, exptime = self._compute_streaming_statistics(l0_file_list, nstream - 1, sigma)

        l1_arrays = {}
        for chip in self.chips:
            img = stats[f'{chip}_CCD']['norm_mean']
            tot = stats[f'{chip}_CCD']['raw_sum']
            var = stats[f'{chip}_VAR']['raw_sum']

            good = var > 0
            for suffix in ['CCD','VAR']:
                ext_name = f'{chip}_{suffix}'
                good &= stats[ext_name]['nframe'] > 0.5 * nframe

            snr = np.zeros_like(img)
            snr[good] = tot[good] / np.sqrt(var[good])

            l1_arrays[f'{chip}_IMG'] = img
            l1_arrays[f'{chip}_SNR'] = snr
            l1_arrays[f'{chip}_MASK'] = good

        return l1_arrays


    def _compute_direct_statistics(self, l0_file_list=None, nframe=None, sigma=None):
        """
        nframe : maximum number of frames to stack for computing statistics

        Notes
        -----
        Passing nframe in addition to l0_file_list helps robust statistics to be
        computed by enabling extra fallback files if some frames fail to load
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if nframe is None:
            nframe = self.nframe_stream - 1
        if sigma is None:
            sigma = self.stack_sigma
        
        nframe = np.min([nframe,len(l0_file_list)])

        if nframe < 2:
            raise ValueError(f"Stacking requires at least two frames, got {nframe}")

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

            l1_obj, success = self.load_frame(fn)
            
            if not success:
                failure += 1
                if failure / len(l0_file_list) > 0.2:
                    raise ValueError(f"more than 20% of frames in stack failed to load")
                continue

            exptime[i] = l1_obj.header['PRIMARY']['ELAPSED']
            
            for chip in self.chips:
                data_cube[f'{chip}_CCD'][i] = l1_obj.data[f'{chip}_CCD']
                data_cube[f'{chip}_VAR'][i] = l1_obj.data[f'{chip}_VAR']

            i += 1

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
        exptime_total = np.sum(exptime)

        for chip in self.chips:
            stats[f'{chip}_CCD'] = {}
            stats[f'{chip}_VAR'] = {}

            out = (
                flag_outliers(data_cube[f'{chip}_CCD'] / T, sigma, axis=0) | 
                flag_outliers(data_cube[f'{chip}_VAR'] / T, sigma, axis=0)
            )

            valid = ~out
            N = np.sum(~out, axis=0)
            good = N > 1
            
            for suffix in ['CCD', 'VAR']:
                ext_name = f'{chip}_{suffix}'

                D = data_cube[ext_name]
                S0 = np.sum(D, axis=0, where=valid)        
                S1 = np.sum(D / T, axis=0, where=valid)
                S2 = np.sum((D / T)**2, axis=0, where=valid)

                mean = np.zeros_like(S1)
                mean[good] = S1[good] / N[good]

                var = np.zeros_like(S2)
                var[good] = (S2[good] - S1[good]**2 / N[good]) / (N[good] - 1)
                rms = np.sqrt(np.maximum(var,0))

                stats[ext_name]['nframe'] = N
                stats[ext_name]['raw_sum'] = S0
                stats[ext_name]['norm_mean'] = mean
                stats[ext_name]['norm_rms'] = rms

        return stats, exptime_total


    def _compute_streaming_statistics(self, l0_file_list=None, ndirect=None, sigma=None):
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
        if ndirect is None:
            ndirect = self.nframe_stream - 1
        if sigma is None:
            sigma = self.stack_sigma

        approx_stats, exptime_direct = (
            self._compute_direct_statistics(
                l0_file_list=l0_file_list,         
                nframe=ndirect, 
                sigma=sigma
            )
        )

        if len(l0_file_list) <= ndirect:
            return approx_stats, exptime_direct

        exact_stats = {}
        exptime_total = 0.0
        zero_exptime = exptime_direct == 0
        
        for chip in self.chips:
            for suffix in ['CCD', 'VAR']:
                ext_name = f'{chip}_{suffix}'

                exact_stats[ext_name] = {}
                exact_stats[ext_name]['N'] = np.zeros((NROW,NCOL),dtype=np.int32)
                exact_stats[ext_name]['S0'] = np.zeros((NROW,NCOL),dtype=np.float32)
                exact_stats[ext_name]['S1'] = np.zeros((NROW,NCOL),dtype=np.float32)
                exact_stats[ext_name]['S2'] = np.zeros((NROW,NCOL),dtype=np.float32)

                approx_mean = approx_stats[ext_name]['norm_mean']
                approx_rms = approx_stats[ext_name]['norm_rms']
                
                approx_stats[ext_name]['norm_lower'] = approx_mean - approx_rms * sigma
                approx_stats[ext_name]['norm_upper'] = approx_mean + approx_rms * sigma

        failure = 0

        for fn in l0_file_list:
            l1_obj, success = self.load_frame(fn)

            if not success:
                failure += 1
                if failure / len(l0_file_list) > 0.2:
                    raise ValueError(f"more than 20% of frames in stack failed to load")
                continue

            exptime = l1_obj.header['PRIMARY']['ELAPSED']

            if zero_exptime != (exptime == 0):
                raise ValueError(f"Exposure times must be all zero or all non-zero")

            if exptime < 0:
                raise ValueError("Exposure times cannot be negative")
            elif exptime == 0:
                T = 1.0
            else:
                T = exptime
            
            exptime_total += exptime                
            
            for chip in self.chips:
                for suffix in ['CCD', 'VAR']:
                    ext_name = f'{chip}_{suffix}'
                    
                    D = l1_obj.data[ext_name]

                    lower = approx_stats[ext_name]['norm_lower']
                    upper = approx_stats[ext_name]['norm_upper']
                    valid = (D / T >= lower) & (D / T <= upper)

                    N = exact_stats[ext_name]['N']
                    N[valid] += 1
    
                    S0 = exact_stats[ext_name]['S0']
                    S0[valid] += D[valid]

                    S1 = exact_stats[ext_name]['S1']
                    delta1 = D / T - S1
                    delta1[~valid] = 0
                    S1[valid] += delta1[valid] / N[valid]
                    delta2 = D / T - S1
                    S2 = exact_stats[ext_name]['S2']
                    S2[valid] += delta1[valid] * delta2[valid]

                    exact_stats[ext_name]['S0'] = S0
                    exact_stats[ext_name]['S1'] = S1
                    exact_stats[ext_name]['S2'] = S2

    
        for chip in self.chips:
            for suffix in ['CCD', 'VAR']:
                ext_name = f'{chip}_{suffix}'

                N = exact_stats[ext_name]['N']
                S0 = exact_stats[ext_name]['S0']
                S1 = exact_stats[ext_name]['S1']
                S2 = exact_stats[ext_name]['S2']

                mean = S1
                good = N > 1

                var = np.zeros_like(S2)
                var[good] = S2[good] / (N[good] - 1)
                rms = np.sqrt(np.maximum(var,0))

                exact_stats[ext_name]['nframe'] = exact_stats[ext_name].pop('N')
                exact_stats[ext_name]['raw_sum'] = exact_stats[ext_name].pop('S0')
                exact_stats[ext_name]['norm_mean'] = exact_stats[ext_name].pop('S1')
                exact_stats[ext_name]['norm_rms'] = rms

                del exact_stats[ext_name]['S2']

        return exact_stats, exptime_total