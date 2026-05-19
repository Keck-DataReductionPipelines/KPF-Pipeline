"""
KPF Master Wavelength Solution construction module.
"""
import warnings

from astropy.stats import mad_std
import h5py
import numpy as np
from numpy.polynomial import legendre
import pandas as pd

from kpfpipe import DEFAULTS, REPO_ROOT
from kpfpipe.data_models.masters import KPFMasterL2
from kpfpipe.modules.masters.base import BaseMasterModule
from kpfpipe.modules.calibration_association import CalibrationAssociation
from kpfpipe.modules.image_processing import ImageProcessing
from kpfpipe.modules.spectral_extraction import SpectralExtraction
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import optimize_lsq

DEFAULTS.update({
    'linelist': f'{REPO_ROOT}/reference/thar_line_list.csv',
    'lineprofile': 'gaussian',
    'polyorder_x': 6,
    'polyorder_m': 3,
    'polyorder_f': 2,
})

_ROUGH_WLS_FILE = f'{REPO_ROOT}/reference/rough_wls_fallback.csv'


class WLS(BaseMasterModule):
    """
    Construct a master wavelength solution from a stack of WLS L0 exposures.

    Each frame is processed individually through the L0→L2 pipeline. Fitted
    line positions across the stack are combined to derive per-fiber
    wavelength solutions.

    Parameters
    ----------
    l0_file_list : list of str
        Sorted list of L0 FITS file paths to process.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: linelist, lineprofile,
        polyorder_x, polyorder_m, polyorder_f, chips, fibers,
        KPF_DATA_INPUT.
    """
    def __init__(self, l0_file_list, config=None):
        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["DATA_DIRS", "KPFPIPE", "WLS"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")
        super().__init__(l0_file_list, params)
        self._data_root = params.get('KPF_DATA_INPUT')
        self.rough_wls_file = _ROUGH_WLS_FILE

        self._load_rough_wls()
        self._load_linelist()

        self._results = None  # populated by make_master_l2()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_linelist(self, linelist=None):
        """
        Return the cached line wavelength array, loading or reloading if needed.

        Loads from disk when no array is cached yet, or when `linelist`
        (a file path) differs from the cached `self.linelist`. Otherwise
        returns the cache unchanged. The cache is transparently kept in
        sync with the most recently passed-in file path.
        """
        needs_load = not hasattr(self, '_linelist_array')
        if linelist is not None and linelist != self.linelist:
            self.linelist = linelist
            needs_load = True
        if needs_load:
            self._linelist_array = pd.read_csv(self.linelist)['Wavelength'].values
        return self._linelist_array


    def _load_rough_wls(self, rough_wls_file=None):
        """
        Return the cached rough WLS dict, loading or reloading if needed.

        Loads from disk when no rough WLS is cached yet, or when
        `rough_wls_file` (a file path) differs from the cached
        `self.rough_wls_file`. Otherwise returns the cache unchanged. The
        cache is transparently kept in sync with the most recently
        passed-in file path.
        """
        needs_load = not hasattr(self, 'rough_wls')
        if rough_wls_file is not None and rough_wls_file != self.rough_wls_file:
            self.rough_wls_file = rough_wls_file
            needs_load = True
        if needs_load:
            df = pd.read_csv(self.rough_wls_file)

            ncol = self.ccd['ncol']
            self.rough_wls = {}

            for chip in self.chips:
                norder = self.norder[chip]
                for fiber in self.fibers:
                    channel_ext = f'{chip}_{fiber}_WAVE'
                    self.rough_wls[channel_ext] = np.zeros((norder, ncol))

                    for o in range(norder):
                        use = (df.CHIP == chip) & (df.ORDER == o)
                        self.rough_wls[channel_ext][o] = np.linspace(
                            df.loc[use, 'WAVE_MIN'].values[0],
                            df.loc[use, 'WAVE_MAX'].values[0],
                            ncol,
                        )

        return self.rough_wls
    

    def _extract_frame(self, l1_obj, verbose=True):
        calibration_association = CalibrationAssociation(l1_obj, {'KPF_DATA_INPUT': self._data_root})
        l1_obj = calibration_association.perform(['bias'])

        image_processing = ImageProcessing(l1_obj)
        l1_obj = image_processing.perform()

        spectral_extraction = SpectralExtraction(l1_obj)
        l2_obj = spectral_extraction.perform(verbose=verbose)

        return l2_obj


    @staticmethod
    def _package_stacks_hdf5(coeffs_by_chip, lines_by_chip):
        """
        Package per-chip WLS stacks into an in-memory HDF5 file.

        Layout: /<chip>/coeffs_stack as a dataset, /<chip>/lines_stack/
        frame_<NNN>/<key> as per-frame subgroups, with every key from
        the per-frame `lines` dict (e.g. wav, pix, ord, fib, bad, plus
        diagnostics like std, amp, rms).
        """
        f = h5py.File('wls_stacks.h5', 'w', driver='core', backing_store=False)
        str_dt = h5py.string_dtype(encoding='utf-8')

        for chip, coeffs_stack in coeffs_by_chip.items():
            chip_group = f.create_group(chip)
            chip_group.create_dataset('coeffs_stack', data=np.asarray(coeffs_stack))

            lines_group = chip_group.create_group('lines_stack')
            for i, lines in enumerate(lines_by_chip[chip]):
                frame_group = lines_group.create_group(f'frame_{i:03d}')
                for key, value in lines.items():
                    arr = np.asarray(value)
                    if arr.dtype.kind in ('U', 'S', 'O'):
                        frame_group.create_dataset(key, data=arr.astype(object), dtype=str_dt)
                    else:
                        frame_group.create_dataset(key, data=arr)

        return f


    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def process_stack_l0_to_l2(self, l0_file_list=None, verbose=True):
        """
        Run each L0 frame in the stack through the L0→L2 pipeline.

        Parameters
        ----------
        l0_file_list : list of str, optional
            L0 files to process. Defaults to self.l0_file_list.
        verbose : bool, optional
            If True (default), emit progress prints and per-frame warnings
            from the underlying L0 → L1 → L2 calls.

        Returns
        -------
        list of KPF2
            Extracted L2 objects for all successfully processed frames.

        Notes
        -----
        Raises ValueError if more than 20% of frames fail to load.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list

        if len(l0_file_list) == 0:
            raise ValueError("Empty l0_file_list; must supply at least one valid file")

        failure = 0

        for fn in l0_file_list:
            l1_obj, success = self._load_frame(fn, ncache=0, verbose=verbose)

            if not success:
                failure += 1
                if failure / len(l0_file_list) > 0.2:
                    raise ValueError(f"more than 20% of frames in stack failed to load")
                continue

            l2_obj = self._extract_frame(l1_obj, verbose=verbose)
            
            if not hasattr(self, '_l2_obj_cache'):
                self._l2_obj_cache = []
            
            self._l2_obj_cache.append(l2_obj)

        return self._l2_obj_cache
    

    def fit_line_positions_1D(self,
                              flux1d,
                              wave1d,
                              linelist=None,
                              lineprofile='gaussian',
                              window=5,
                              qc_sigma=2.5,
                              ):
        """
        Fit line positions (in pixel space) along a 1D extracted order.

        For each linelist entry within the wavelength range of `wave1d`,
        fits a 1D line model over a ±`window` pixel neighborhood around
        the rough wavelength match. Fits whose sigma or normalized RMS
        deviate from the median by more than `qc_sigma` are rejected, as
        are negative-amplitude fits.

        Parameters
        ----------
        flux1d : ndarray
            1D extracted flux for a single order.
        wave1d : ndarray
            1D rough wavelength grid for the same order.
        linelist : str, optional
            Path to a CSV line list. If different from the currently
            cached `self.linelist`, the file is reloaded and the cache
            is updated. Defaults to `self.linelist` (no reload).
        lineprofile : str, optional
            Line profile model name. See kpfpipe.utils.stats._FUNCTIONS
            for supported values.
        window : int, optional
            Half-width of the fit window, in pixels.
        qc_sigma : float, optional
            Outlier rejection threshold (in MAD-stds) applied to the
            line-by-line sigma and normalized RMS.

        Returns
        -------
        lines : dict of ndarray
            Per-line arrays of equal length. All entries are retained
            regardless of QC status; the caller is responsible for
            filtering on 'bad' before downstream use. Lines whose fit
            window contains any non-finite flux are dropped silently
            before fitting. If the input order is entirely non-finite
            (e.g. an extraction-failed orderlet filled with NaN), all
            arrays are returned empty. Keys:
              'wav' - reference line wavelength
              'pix' - fitted pixel position
              'std' - fitted line sigma (Gaussian width)
              'amp' - fitted line amplitude
              'rms' - normalized fit residual RMS
              'bad' - boolean QC flag (True = line failed QC)
        """
        linelist_array = self._load_linelist(linelist)

        if len(flux1d) != len(wave1d):
            raise ValueError("length of flux and wave arrays are mismatched")
        ncol = len(flux1d)

        lines = {k: np.zeros(0, dtype='float') for k in ['wav', 'pix', 'std', 'amp', 'rms']}
        lines['bad'] = np.zeros(0, dtype=bool)

        if not np.isfinite(flux1d).any():
            return lines

        candidate_wavs = np.sort(linelist_array[(linelist_array > wave1d.min()) & (linelist_array < wave1d.max())])

        keep = np.zeros(len(candidate_wavs), dtype=bool)
        for i, lw in enumerate(candidate_wavs):
            loc = np.argmin(np.abs(wave1d - lw))
            cols = np.arange(loc - window, loc + window + 1)
            cols = cols[(cols >= 0) & (cols < ncol)]
            keep[i] = np.isfinite(flux1d[cols]).all()

        lines['wav'] = candidate_wavs[keep]
        nlines = len(lines['wav'])

        if nlines == 0:
            return lines

        for key in ['pix', 'std', 'amp', 'rms']:
            lines[key] = np.zeros(nlines, dtype='float')

        for i, lw in enumerate(lines['wav']):
            loc = np.argmin(np.abs(wave1d - lw))
            cols = np.arange(loc - window, loc + window + 1)
            cols = cols[(cols >= 0) & (cols < ncol)]

            x = cols
            y = flux1d[cols]
            theta, rms = optimize_lsq(x, y, lineprofile)

            if lineprofile == 'gaussian':
                # gaussian_dist theta convention: [b, a, mu, sigma]
                lines['pix'][i] = theta[2]
                lines['std'][i] = theta[3]
                lines['amp'][i] = theta[1]
                lines['rms'][i] = rms / np.abs(theta[1] * np.sqrt(2*np.pi) * theta[3])
            else:
                raise ValueError(f"Unsupported lineprofile: {lineprofile}")

        if lineprofile == 'gaussian':
            lines['bad'] = np.abs(lines['rms'] - np.median(lines['rms'])) / mad_std(lines['rms']) > qc_sigma
            lines['bad'] |= np.abs(lines['std'] - np.median(lines['std'])) / mad_std(lines['std']) > qc_sigma
            lines['bad'] |= lines['amp'] < 0
        else:
            raise ValueError(f"Unsupported lineprofile: {lineprofile}")

        return lines
    

    def fit_line_positions_ffi(self,
                               l2_obj,
                               chip,
                               fibers,
                               linelist=None,
                               lineprofile=None,
                               window=5,
                               qc_sigma=2.5,
                               verbose=True,
                               ):
        """
        Fit line positions across all orders and fibers of one chip.

        Loops over the requested fibers, calling `fit_line_positions_1D`
        on each (order, fiber) extracted spectrum, and concatenates the
        surviving lines into flat arrays tagged with their order number
        and fiber name.

        Parameters
        ----------
        l2_obj : KPF2
            Extracted L2 object containing per-fiber FLUX arrays of shape
            (norder, ncol).
        chip : str
            Chip identifier ('GREEN' or 'RED').
        fibers : list of str
            Fiber identifiers (e.g. ['SCI1', 'SCI2', 'SCI3']).
        linelist : str, optional
            Path to a CSV line list. If different from the currently
            cached `self.linelist`, the file is reloaded and the cache
            is updated. Defaults to `self.linelist` (no reload).
        lineprofile : str, optional
            Line profile model name. See kpfpipe.utils.stats._FUNCTIONS.
        window : int, optional
            Half-width of the per-line fit window, in pixels.
        qc_sigma : float, optional
            Outlier rejection threshold passed through to
            `fit_line_positions_1D`.
        verbose : bool, optional
            If True, print a progress line for each fiber.

        Returns
        -------
        lines : dict of ndarray
            Flat 1D arrays, all of equal length. All lines are retained
            regardless of QC status; the caller is responsible for
            filtering on 'bad' before downstream use. All per-line keys
            produced by `fit_line_positions_1D` are carried through, plus
            'ord' and 'fib' which tag each line with its source order
            and fiber. Keys:
              'wav' - reference line wavelength
              'pix' - fitted pixel position
              'std' - fitted line sigma (Gaussian width)
              'amp' - fitted line amplitude
              'rms' - normalized fit residual RMS
              'bad' - boolean QC flag (True = line failed QC)
              'ord' - 1-indexed order number
              'fib' - fiber name
        """
        self._load_linelist(linelist)

        if lineprofile is None:
            lineprofile = self.lineprofile

        norder = self.norder[chip]
        # keys mirror fit_line_positions_1D's output plus per-line provenance tags
        keys = ('wav', 'pix', 'std', 'amp', 'rms', 'bad', 'ord', 'fib')
        lines = {k: [[None] * norder for _ in fibers] for k in keys}

        for i, fiber in enumerate(fibers):
            if verbose:
                print(f"fitting {chip} {fiber} line positions")

            flux_arr = l2_obj.data[f'{chip}_{fiber}_FLUX']
            wave_arr = self.rough_wls[f'{chip}_{fiber}_WAVE']

            if np.shape(flux_arr) != np.shape(wave_arr):
                raise ValueError("shape mismatch between flux array and rough WLS")

            for o in range(norder):
                line_dict = self.fit_line_positions_1D(
                    flux_arr[o],
                    wave_arr[o],
                    linelist=linelist,
                    lineprofile=lineprofile,
                    window=window,
                    qc_sigma=qc_sigma,
                )

                nlines = len(line_dict['wav'])
                if nlines == 0 and verbose:
                    warnings.warn(
                        f"{chip} {fiber} order {o + 1}: orderlet skipped "
                        f"(no fittable lines; flux likely NaN-filled)"
                    )
                line_dict['ord'] = (o + 1) * np.ones(nlines, dtype=int)
                line_dict['fib'] = np.full(nlines, fiber)

                for k in keys:
                    lines[k][i][o] = line_dict[k]

            for k in lines:
                lines[k][i] = np.hstack(lines[k][i])

            n_total = len(lines['wav'][i])
            n_good = int(np.sum(~lines['bad'][i]))
            if verbose:
                print(f"  {chip} {fiber}: {n_good}/{n_total} good lines")
            if n_good == 0 and verbose:
                warnings.warn(
                    f"{chip} {fiber}: no good lines retained "
                    f"({n_total} attempted; all rejected or NaN-filled)"
                )

        for k in lines:
            lines[k] = np.hstack(lines[k])

        return lines


    def calculate_wls_coeffs(self,
                             lines,
                             norder,
                             polyorder_x=None,
                             polyorder_m=None,
                             polyorder_f=None,
                             ):
        """
        Fit a multivariate Legendre polynomial wavelength solution to
        fitted line positions.

        Treats wavelength as a smooth function of pixel position (x), order
        number (m), and (optionally) fiber index (f). All variables are
        rescaled to [-1, 1] before fitting.

        Parameters
        ----------
        lines : dict of ndarray
            Flat 1D arrays as produced by `fit_line_positions_ffi`. Lines
            with `lines['bad']` set are excluded from the fit. Required
            keys: 'wav', 'pix', 'ord', 'fib', 'bad'.
        norder : int
            Total number of spectral orders on the chip. Used to rescale
            the order axis to [-1, 1].
        polyorder_x : int, optional
            Polynomial degree along the pixel axis.
        polyorder_m : int, optional
            Polynomial degree along the order axis.
        polyorder_f : int, optional
            Polynomial degree along the fiber axis (only used for 3-fiber fits).

        Returns
        -------
        coeffs : ndarray
            Legendre coefficient array. Shape is
            (polyorder_x+1, polyorder_m+1) for a single-fiber fit, or
            (polyorder_x+1, polyorder_m+1, polyorder_f+1) for a 3-fiber fit.

        Notes
        -----
        Raises ValueError if `lines['fib']` contains anything other than
        one fiber, all five fibers, or three SCI fibers (SCI1, SCI2, SCI3).
        """
        if polyorder_x is None:
            polyorder_x = self.polyorder_x
        if polyorder_m is None:
            polyorder_m = self.polyorder_m
        if polyorder_f is None:
            polyorder_f = self.polyorder_f

        good = ~lines['bad']
        wav = lines['wav'][good]
        pix = lines['pix'][good]
        ord = lines['ord'][good]
        fib = lines['fib'][good]

        fibers = list(set(fib))

        if (len(fibers) != 1) and (len(fibers) != 3) and (len(fibers) != 5):
            raise ValueError(f"expected 1, 3, or 5 fibers, got {len(fibers)}")

        if len(fibers) == 3:
            expected_fibers = ['SCI1', 'SCI2', 'SCI3']
        elif len(fibers) == 5:
            expected_fibers = ['SKY', 'SCI1', 'SCI2', 'SCI3', 'CAL']

        if len(fibers) != 1 and not (
            np.all(np.isin(fibers, expected_fibers)) and
            np.all(np.isin(expected_fibers, fibers))
        ):
            raise ValueError(f"unexpected fibers input: {fibers}")

        # guard against degenerate / underconstrained fits
        n_params = (polyorder_x + 1) * (polyorder_m + 1)
        if len(fibers) != 1:
            n_params *= (polyorder_f + 1)
        if len(wav) < n_params:
            raise ValueError(
                f"WLS fit underconstrained: {len(wav)} good lines < "
                f"{n_params} free parameters "
                f"(polyorder_x={polyorder_x}, polyorder_m={polyorder_m}, "
                f"polyorder_f={polyorder_f}, fibers={sorted(fibers)})"
            )

        ncol = self.ccd['ncol']

        # rescale position variables to [-1,1] for Legendre fitting
        _x = 2*pix/(ncol - 1) - 1
        _m = 2*(ord - 1)/(norder - 1) - 1

        if len(fibers) != 1:
            # map fibers to their positional rank then rescale to [-1, 1]
            canonical = sorted(expected_fibers, key=lambda f: self.fiber_positions[f])
            fiber_pos = {f: i for i, f in enumerate(canonical)}
            _f = np.array([fiber_pos[f] for f in fib], dtype=int)
            _f = 2*_f/(len(canonical) - 1) - 1


        if len(fibers) == 1:
            V = legendre.legvander2d(_x, _m, deg=[polyorder_x, polyorder_m])

            coeffs, *_ = np.linalg.lstsq(V, wav, rcond=None)
            coeffs = coeffs.reshape(polyorder_x+1, polyorder_m+1)

        else:
            V = legendre.legvander3d(_x, _m, _f, deg=[polyorder_x, polyorder_m, polyorder_f])

            coeffs, *_ = np.linalg.lstsq(V, wav, rcond=None)
            coeffs = coeffs.reshape(polyorder_x+1, polyorder_m+1, polyorder_f+1)


        return coeffs
    

    @staticmethod
    def evaluate_wls_coeffs(coeffs, ncol, norder, nfiber):
        """
        Evaluate a Legendre wavelength solution onto a regular grid.

        Parameters
        ----------
        coeffs : ndarray
            Legendre coefficient array from `calculate_wls_coeffs`. Either
            2D (single-fiber) or 3D (three-fiber).
        ncol : int
            Number of detector columns at which to evaluate.
        norder : int
            Number of spectral orders at which to evaluate.
        nfiber : int
            Number of fibers at which to evaluate. Ignored for 2D `coeffs`.

        Returns
        -------
        W : ndarray
            Wavelength array of shape (norder, ncol) for 2D `coeffs`, or
            (norder, ncol, nfiber) for 3D `coeffs`.
        """
        _x = np.linspace(-1, 1, ncol)
        _y = np.linspace(-1, 1, norder)
        _z = np.linspace(-1, 1, nfiber)

        if coeffs.ndim == 2:
            X, Y = np.meshgrid(_x, _y)
            W = legendre.legval2d(X, Y, coeffs)
        
        elif coeffs.ndim == 3:
            X, Y, Z = np.meshgrid(_x, _y, _z)
            W = legendre.legval3d(X, Y, Z, coeffs)

        else:
            raise ValueError(f"coeffs.ndim expected to be 2 or 3, got {coeffs.ndim}")

        return W
    

    def compute_wls_from_stack(self,
                               chip,
                               fibers,
                               linelist=None,
                               lineprofile=None,
                               window=5,
                               qc_sigma=2.5,
                               polyorder_x=None,
                               polyorder_m=None,
                               polyorder_f=None,
                               verbose=True,
                               return_stacks=True,
                               ):
        """
        Compute a master wavelength solution from a stack of extracted L2 frames.

        For each L2 frame in `self._l2_obj_cache` (populated by
        `process_stack_l0_to_l2`), fits line positions across the requested
        fibers, fits a Legendre WLS to those line positions, then combines
        the per-frame coefficient sets via per-coefficient outlier-rejected
        averaging. The averaged coefficients are evaluated to produce a
        master wavelength array.

        Parameters
        ----------
        chip : str
            Chip identifier, i.e. 'GREEN' or 'RED'.
        fibers : list of str
            Fiber identifiers, e.g. ['SCI1', 'SCI2', 'SCI3'] or a single fiber.
        linelist : str, optional
            Path to a CSV line list. If different from the currently
            cached `self.linelist`, the file is reloaded and the cache
            is updated. Defaults to `self.linelist` (no reload).
        lineprofile : str, optional
            Line profile model name. Defaults to self.lineprofile.
        window : int, optional
            Half-width of the per-line fit window, in pixels.
        qc_sigma : float, optional
            Outlier rejection threshold applied to per-frame Legendre
            coefficients, and passed through to the per-line QC in
            `fit_line_positions_1D`.
        polyorder_x : int, optional
            Polynomial degree along the pixel axis.
        polyorder_m : int, optional
            Polynomial degree along the order axis.
        polyorder_f : int, optional
            Polynomial degree along the fiber axis (only used for 3-fiber fits).
        verbose : bool, optional
            If True, print progress for each frame and fiber.
        return_stacks : bool, optional
            If True, also return the per-frame line and coefficient stacks.

        Returns
        -------
        W : ndarray
            Master wavelength array, shape (norder, ncol) or
            (norder, ncol, nfiber).
        coeffs_mean : ndarray
            Outlier-rejected mean Legendre coefficient array.
        coeffs_stack : ndarray, optional
            Per-frame coefficient arrays, returned only if `return_stacks=True`.
        lines_stack : list of dict, optional
            Per-frame line dicts from `fit_line_positions_ffi`, returned only
            if `return_stacks=True`.

        Notes
        -----
        Raises ValueError if `self._l2_obj_cache` is empty (i.e.,
        `process_stack_l0_to_l2` has not been run).
        """
        self._load_linelist(linelist)
        if lineprofile is None:
            lineprofile = self.lineprofile
        if polyorder_x is None:
            polyorder_x = self.polyorder_x
        if polyorder_m is None:
            polyorder_m = self.polyorder_m
        if polyorder_f is None:
            polyorder_f = self.polyorder_f

        
        if hasattr(self, '_l2_obj_cache'):
            l2_obj_list = self._l2_obj_cache
        else:
            raise ValueError(
                "No L2 objects found; please run process_stack_l0_to_l2"
            )
        
        nobs = len(l2_obj_list)

        lines_stack = [None]*nobs
        coeffs_stack = [None]*nobs

        for i, l2_obj in enumerate(l2_obj_list):
            if verbose:
                print(f"\n{i+1} of {nobs}")
            
            lines_stack[i] = self.fit_line_positions_ffi(l2_obj, 
                                                         chip, 
                                                         fibers, 
                                                         linelist = linelist,
                                                         lineprofile = lineprofile,
                                                         window = window, 
                                                         qc_sigma = qc_sigma,
                                                         verbose = verbose,
                                                         )



            coeffs_stack[i] = self.calculate_wls_coeffs(lines_stack[i],
                                                        self.norder[chip],
                                                        polyorder_x = polyorder_x,
                                                        polyorder_m = polyorder_m,
                                                        polyorder_f = polyorder_f,
                                                        )

        coeffs_stack = np.array(coeffs_stack)
        diff = np.abs(coeffs_stack - np.median(coeffs_stack, axis=0))
        sigma = mad_std(coeffs_stack, axis=0)
        bad = diff > qc_sigma * sigma
        coeffs_mean = np.sum(coeffs_stack * ~bad, axis=0)/np.sum(~bad, axis=0)

        W = self.evaluate_wls_coeffs(coeffs_mean, self.ccd['ncol'], self.norder[chip], len(fibers))

        if return_stacks:
            return W, coeffs_mean, coeffs_stack, lines_stack

        return W, coeffs_mean

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def make_master_l2(self,
                       l0_file_list=None,
                       linelist=None,
                       lineprofile=None,
                       polyorder_x=None,
                       polyorder_m=None,
                       polyorder_f=None,
                       return_stacks=False,
                       verbose=True,
                      ):
        """
        Build a master wavelength solution from a stack of L0 frames.

        Processes each input L0 frame through the L0-to-L2 pipeline, then
        computes per-chip Legendre wavelength solutions using
        `compute_wls_from_stack`. The resulting wavelength arrays are
        written to the per-fiber _WAVE extensions of a KPFMasterL2 object,
        which is returned and cached on `self.ml2_obj`.

        Parameters
        ----------
        l0_file_list : list of str, optional
            L0 files to process. Defaults to self.l0_file_list.
        linelist : str, optional
            Path to a CSV line list. If different from the currently
            cached `self.linelist`, the file is reloaded and the cache
            is updated. Defaults to `self.linelist` (no reload).
        lineprofile : str, optional
            Line profile model name. Defaults to self.lineprofile.
        polyorder_x : int, optional
            Polynomial degree along the pixel axis. Defaults to self.polyorder_x.
        polyorder_m : int, optional
            Polynomial degree along the order axis. Defaults to self.polyorder_m.
        polyorder_f : int, optional
            Polynomial degree along the fiber axis (used for 3- and 5-fiber fits).
            Defaults to self.polyorder_f.
        return_stacks : bool, optional
            If True, also return an in-memory HDF5 file containing per-frame
            coefficient and line stacks for every chip.
        verbose : bool, optional
            If True (default), emit progress prints and informational
            warnings from frame loading, spectral extraction, and the
            per-frame WLS fit. Hard failures still raise.

        Returns
        -------
        KPFMasterL2
            Master L2 object with per-fiber _WAVE extensions populated for
            every chip in self.chips, INPUT_FILES recording the stacked
            L0 files, and a 'master_wls' receipt entry.
        h5py.File, optional
            In-memory HDF5 file (core driver, no backing store) packaging
            per-frame WLS diagnostics for every chip. Layout:
              /<chip>/coeffs_stack                    per-frame Legendre coefficients
              /<chip>/lines_stack/frame_<NNN>/wav     reference line wavelength
              /<chip>/lines_stack/frame_<NNN>/pix     fitted pixel position
              /<chip>/lines_stack/frame_<NNN>/std     fitted line sigma
              /<chip>/lines_stack/frame_<NNN>/amp     fitted line amplitude
              /<chip>/lines_stack/frame_<NNN>/rms     normalized fit residual RMS
              /<chip>/lines_stack/frame_<NNN>/bad     boolean QC flag
              /<chip>/lines_stack/frame_<NNN>/ord     1-indexed order number
              /<chip>/lines_stack/frame_<NNN>/fib     fiber name
            Returned only if `return_stacks=True`.

        Notes
        -----
        Resets self._l2_obj_cache before processing so repeat calls do not
        carry stale frames forward.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list
        if lineprofile is None:
            lineprofile = self.lineprofile
        if polyorder_x is None:
            polyorder_x = self.polyorder_x
        if polyorder_m is None:
            polyorder_m = self.polyorder_m
        if polyorder_f is None:
            polyorder_f = self.polyorder_f

        self._load_linelist(linelist)

        self._l2_obj_cache = []
        self.process_stack_l0_to_l2(l0_file_list=l0_file_list, verbose=verbose)

        self.ml2_obj = KPFMasterL2()

        coeffs_by_chip = {}
        lines_by_chip = {}
        self._results = {}

        for chip in self.chips:
            # Always request stacks internally so we can populate self._results
            # for info(); only package them into HDF5 when the caller asks.
            result = self.compute_wls_from_stack(
                chip=chip,
                fibers=self.fibers,
                lineprofile=lineprofile,
                polyorder_x=polyorder_x,
                polyorder_m=polyorder_m,
                polyorder_f=polyorder_f,
                return_stacks=True,
                verbose=verbose,
            )
            W, coeffs, coeffs_stack, lines_stack = result

            self._results[chip] = {
                'n_total': sum(len(frame['wav']) for frame in lines_stack),
                'n_fit':   sum(int(np.sum(~frame['bad'])) for frame in lines_stack),
            }

            if return_stacks:
                coeffs_by_chip[chip] = coeffs_stack
                lines_by_chip[chip] = lines_stack

            for i, fiber in enumerate(self.fibers):
                if W.ndim == 2:
                    self.ml2_obj.data[f'{chip}_{fiber}_WAVE'] = W
                else:
                    self.ml2_obj.data[f'{chip}_{fiber}_WAVE'] = W[:, :, i]

            coeffs_ext = f'{chip}_WLS_COEFFS'
            if coeffs_ext not in self.ml2_obj.extensions:
                self.ml2_obj.create_extension(coeffs_ext, 'ImageHDU')
            self.ml2_obj.set_data(coeffs_ext, coeffs)

            coeffs_hdr = self.ml2_obj.headers[coeffs_ext]
            coeffs_hdr['POLYORDX'] = (polyorder_x, 'WLS polynomial degree, pixel axis')
            coeffs_hdr['POLYORDM'] = (polyorder_m, 'WLS polynomial degree, order axis')
            coeffs_hdr['POLYORDF'] = (polyorder_f, 'WLS polynomial degree, fiber axis')

        self.ml2_obj.set_input_files(l0_file_list)

        primary = self.ml2_obj.headers['PRIMARY']
        primary['ROUGHWLS'] = (self.rough_wls_file, 'Rough WLS reference file')
        primary['LINELIST'] = (self.linelist, 'Line list reference file')
        primary['LINEPROF'] = (lineprofile, 'Line profile model used in WLS fit')
        primary['POLYORDX'] = (polyorder_x, 'WLS polynomial degree, pixel axis')
        primary['POLYORDM'] = (polyorder_m, 'WLS polynomial degree, order axis')
        primary['POLYORDF'] = (polyorder_f, 'WLS polynomial degree, fiber axis')
        primary['CHIPS'] = (','.join(self.chips), 'Chips included in master WLS')
        primary['FIBERS'] = (','.join(self.fibers), 'Fibers included in master WLS')

        self.ml2_obj.receipt_add_entry('master_wls', 'PASS')

        if return_stacks:
            stacks_hdf5 = self._package_stacks_hdf5(coeffs_by_chip, lines_by_chip)
            return self.ml2_obj, stacks_hdf5

        return self.ml2_obj

    def info(self):
        """Print a summary of the module configuration and WLS results."""
        print("WLS")
        print(f"  l0_file_list:")
        for fn in self.l0_file_list:
            print(f"    {fn}")
        print(f"  chips:           {self.chips}")
        print(f"  fibers:          {self.fibers}")
        print(f"  linelist:        {self.linelist}")
        print(f"  rough_wls_file:  {self.rough_wls_file}")
        print(f"  lineprofile:     {self.lineprofile}")
        print(f"  polyorder:       x={self.polyorder_x}, m={self.polyorder_m}, f={self.polyorder_f}")

        if self._results is None:
            print("  make_master_l2() has not been called")
            return

        print(f"\n  {'chip':<8s} {'n lines fit/total'}")
        print("  " + "-" * 40)
        for chip, stats in self._results.items():
            n_fit, n_total = stats['n_fit'], stats['n_total']
            pct = 100.0 * n_fit / n_total if n_total else 0.0
            print(f"  {chip:<8s} {n_fit} / {n_total} ({pct:.1f}%)")
