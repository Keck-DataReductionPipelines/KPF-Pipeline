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
    'exptime_tolerance': 0.1,
})

DEFAULTS.update(DETECTOR)
NROW = DETECTOR['ccd']['nrow']
NCOL = DETECTOR['ccd']['ncol']

# TODO: line profile and remove uneccessary array allocations
# TODO: build output object
# TODO: decide how to handle ImageAssembly config
# TODO: throw out first frame in stack?
# TODO: use start, middle, end of stack for initial datacube


class BaseMasterModule:
    """
    Base class for KPF masters generation.
    The class should not be called directly, but is used for inheritance
    of masters subclasses: Bias, Dark, Flat, WLS.

    Masters modules read a stack of L0 files from disk and output
    a masters L1 object.
    """
    def __init__(self, l0_file_list, config={}):
        self.l0_file_list = l0_file_list

        for k in DEFAULTS.keys():
            self.__setattr__(k, config.get(k,DEFAULTS[k]))


    def stack_frames(self, l0_file_list=None, nstream=None, sigma=None):
        """
        Stack full-frame images to produce masters L1.

        Parameters
        ----------
        l0_file_list : list of str, optional
            List of L0 FITS filenames to stack.
        nstream : int, optional
            Threshold number of frames above which streaming statistics are used.
        sigma : float, optional
            Sigma threshold for frame-to-frame outlier rejection.

        Returns
        -------
        l1_arrays : dict
            Dictionary containing per-chip stacked products:
            - '{chip}_IMG'  : mean count rate FFI
            - '{chip}_SNR'  : signal-to-noise ratio FFI
            - '{chip}_MASK' : boolean bad pixel mask (1 = good, 0 = bad)

        Notes
        -----
        If number of frames is less than `nstream`, statistics are computed
        directly from a full data cube. Otherwise, streaming Welford statistics
        are used to reduce memory usage.

        An initial subset of frames is processed using the direct data cube
        method to estimate approximate mean and rms. These estimates define
        per-pixel clipping bounds for the streaming pass.
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
            stats, exptime = self._compute_stats_from_datacube(l0_file_list, nstream - 1, sigma)
        else:
            stats, exptime = self._compute_stats_from_stream(l0_file_list, nstream - 1, sigma)

        # TODO: add check that nframe is consistent between CCD and VAR
        for chip in self.chips:
            if np.any(stats[f'{chip}_CCD']['nframe'] != stats[f'{chip}_VAR']['nframe']):
                raise ValueError(f"mismatched frame count between {chip}_CCD and {chip}_VAR")

        l1_arrays = {}
        for chip in self.chips:

            img = stats[f'{chip}_CCD']['rate_mean']
            tot = stats[f'{chip}_CCD']['total_sum']
            var = stats[f'{chip}_VAR']['total_sum']

            good = var > 0

            for suffix in ['CCD','VAR']:
                ext = f'{chip}_{suffix}'
                good &= stats[ext]['nframe'] > 0.5 * nframe

            snr = np.zeros_like(img)
            snr[good] = np.abs(tot[good]) / np.sqrt(var[good])

            l1_arrays[f'{chip}_IMG'] = img
            l1_arrays[f'{chip}_SNR'] = snr
            l1_arrays[f'{chip}_MASK'] = good

        return l1_arrays


    def _load_frame(self, fn, ncache=None, exptime_tolerance=None):
        """
        Load an L0 file and perform image assembly to produce an L1 object.

        Parameters
        ----------
        fn : str
            Path to L0 FITS file.
        ncache : int, optional
            Maximum number of L1 objects to retain in internal cache.

        Returns
        -------
        l1_obj : KPF1 or None
            Assembled L1 data object if successful, otherwise None.
        success : bool
            True if file was successfully loaded and processed, False otherwise.

        Notes
        -----
        Successfully processed frames may be cached to reduce redundant I/O and
        recomputation. Cache size is limited by `ncache`, which defaults to
        `nframe_stream - 1`.
        """
        print(f"loading {fn}")

        if ncache is None:
            ncache = self.nframe_stream - 1
        if exptime_tolerance is None:
            exptime_tolerance = self.exptime_tolerance

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

        self._check_exptime_vs_elapsed(l1_obj, exptime_tolerance)
        
        return l1_obj, success


    @staticmethod
    def _check_exptime_vs_elapsed(l1_obj, exptime_tolerance):
        exptime = l1_obj.headers['PRIMARY']['EXPTIME']
        elapsed = l1_obj.headers['PRIMARY']['ELAPSED']

        delta = elapsed - exptime
        if delta < 0:
            raise ValueError("premature frame readout detected")
        if delta > exptime_tolerance:
            raise ValueError(f"elapsed time - requested time > {exptime_tolerance}")


    def _compute_stats_from_datacube(self, l0_file_list=None, nframe=None, sigma=None):
        """
        Compute stacked statistics using an in-memory data cube.

        Parameters
        ----------
        l0_file_list : list of str, optional
            List of L0 FITS filenames to process.
        nframe : int, optional
            Maximum number of successfully loaded frames to include.
        sigma : float, optional
            Sigma threshold for outlier rejection across frames.

        Returns
        -------
        stats : dict
            Per-extension statistics including:
            - 'nframe'     : number of valid frames per pixel
            - 'total_sum'  : summed counts across valid frames
            - 'rate_mean'  : exposure-time-normalized mean
            - 'rate_rms'   : frame-to-frame sample RMS
        exptime_total : float
            Total integrated exposure time across included frames.

        Notes
        -----
        Outlier rejection is performed jointly on CCD and VAR extensions.
        Exposure times must be either all zero or all strictly positive.
        Raises an error if more than 20% of frames fail to load.
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

            l1_obj, success = self._load_frame(fn)
            
            if not success:
                failure += 1
                if failure / len(l0_file_list) > 0.2:
                    raise ValueError(f"more than 20% of frames in stack failed to load")
                continue

            exptime[i] = l1_obj.headers['PRIMARY']['EXPTIME']
            
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
                ext = f'{chip}_{suffix}'
                D = data_cube[ext]
                R = D / T

                total_sum = np.sum(D, axis=0, where=valid)

                rate_mean = np.zeros_like(R[0])
                rate_mean[good] = np.sum(R, axis=0, where=valid)[good] / N[good]
                
                diff2 = (R - rate_mean)**2
                ssd = np.sum(diff2, axis=0, where=valid)

                rate_rms = np.zeros_like(R[0])
                rate_rms[good] = np.sqrt(ssd[good] / (N[good] - 1))

                stats[ext]['nframe'] = N
                stats[ext]['total_sum'] = total_sum
                stats[ext]['rate_mean'] = rate_mean
                stats[ext]['rate_rms'] = rate_rms

        return stats, exptime_total


    def _compute_stats_from_stream(self, l0_file_list=None, ndirect=None, sigma=None):
        """
        Compute stacked statistics using streaming Welford accumulation.

        Parameters
        ----------
        l0_file_list : list of str, optional
            List of L0 FITS filenames to process.
        ndirect : int, optional
            Number of initial frames used to estimate approximate statistics
            for defining clipping thresholds.
        sigma : float, optional
            Sigma threshold for outlier rejection.

        Returns
        -------
        exact_stats : dict
            Per-extension statistics including:
            - 'nframe'     : number of valid frames per pixel
            - 'total_sum'  : summed counts across valid frames
            - 'rate_mean'  : per-pixel rate mean, normalized by exposure time
            - 'rate_rms'   : frame-to-frame rate rms deviation
        exptime_total : float
            Total integrated exposure time across included frames.

        Notes
        -----
        An initial subset of frames is processed using the direct data cube
        method to estimate approximate mean and RMS. These estimates define
        per-pixel clipping bounds for the streaming pass.

        Optimized to reduce memory usage at the expense of compute speed.
        Raises an error if more than 20% of frames fail to load or if exposure
        times are inconsistent.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if ndirect is None:
            ndirect = self.nframe_stream - 1
        if sigma is None:
            sigma = self.stack_sigma

        approx_stats, exptime_direct = (
            self._compute_stats_from_datacube(
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
                ext = f'{chip}_{suffix}'

                exact_stats[ext] = {}
                exact_stats[ext]['nframe'] = np.zeros((NROW,NCOL),dtype=np.int32)
                exact_stats[ext]['total_sum'] = np.zeros((NROW,NCOL),dtype=np.float64)
                exact_stats[ext]['rate_mean'] = np.zeros((NROW,NCOL),dtype=np.float64)
                exact_stats[ext]['rate_M2'] = np.zeros((NROW,NCOL),dtype=np.float64)

                approx_mean = approx_stats[ext]['rate_mean']
                approx_rms = approx_stats[ext]['rate_rms']
                
                approx_stats[ext]['rate_lower'] = approx_mean - approx_rms * sigma
                approx_stats[ext]['rate_upper'] = approx_mean + approx_rms * sigma

        failure = 0
        valid = np.ones((NROW, NCOL), dtype=bool)

        for fn in l0_file_list:
            l1_obj, success = self._load_frame(fn)

            if not success:
                failure += 1
                if failure / len(l0_file_list) > 0.2:
                    raise ValueError(f"more than 20% of frames in stack failed to load")
                continue

            exptime = l1_obj.headers['PRIMARY']['EXPTIME']

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
                valid[:] = True
                R = {}

                for suffix in ['CCD', 'VAR']:
                    ext = f'{chip}_{suffix}'
                    D = l1_obj.data[ext]
                    R[ext] = D / T

                    lower = approx_stats[ext]['rate_lower']
                    upper = approx_stats[ext]['rate_upper']
                    valid &= (R[ext] >= lower) & (R[ext] <= upper)

                for suffix in ['CCD', 'VAR']:
                    ext = f'{chip}_{suffix}'
                    D = l1_obj.data[ext]
                    rate = R[ext]

                    N = exact_stats[ext]['nframe']
                    N += valid

                    total_sum = exact_stats[ext]['total_sum']
                    total_sum += D * valid

                    # Welford algorithm accumulation begins
                    mean = exact_stats[ext]['rate_mean']
                    safe_N = np.maximum(N, 1)
                    delta = (rate - mean) * valid
                    mean += delta / safe_N
                    delta2 = (rate - mean) * valid
                    M2 = exact_stats[ext]['rate_M2']
                    M2 += delta * delta2
                    # Welford algorithm accumulation ends

                    exact_stats[ext]['total_sum'] = total_sum
                    exact_stats[ext]['rate_mean'] = mean
                    exact_stats[ext]['rate_M2'] = M2

        for chip in self.chips:
            for suffix in ['CCD', 'VAR']:
                ext = f'{chip}_{suffix}'

                N = exact_stats[ext]['nframe']
                mean = exact_stats[ext]['rate_mean']
                M2 = exact_stats[ext]['rate_M2']
                rms = np.sqrt(np.where(N > 1, M2 / (N - 1), 0))

                exact_stats[ext]['rate_rms'] = rms
                del exact_stats[ext]['rate_M2']

        return exact_stats, exptime_total