"""
KPF Master Wavelength Solution construction module.
"""
from astropy.stats import mad_std
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
    'rough_wls_file': f'{REPO_ROOT}/reference/rough_wls_fallback.csv',
    'line_list_file': f'{REPO_ROOT}/reference/thar_line_list.csv',
    'lineprofile': 'gaussian',
    'polyorder_x': 6,
    'polyorder_m': 3,
    'polyorder_f': 2,
})

FIBER_INDEX_MAP = {'SKY':0, 'SCI1':1, 'SCI2':2, 'SCI3':3, 'CAL':4}

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
        Module configuration. Recognized keys: KPF_DATA_INPUT.
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

        self._load_rough_wls()
        self._load_linelist()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_linelist(self):
        self.linelist = pd.read_csv(self.line_list_file)['Wavelength'].values
        return self.linelist
        
    def _load_rough_wls(self):
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
    
    def _load_frame(self, fn, ncache=0, exptime_tolerance=None):
        # load and assemble a single raw image frame from L0 --> L1
        return super()._load_frame(fn, ncache=ncache, exptime_tolerance=exptime_tolerance)

    def _extract_frame(self, l1_obj):
        # process and extract single image frame from L1 --> L2
        calibration_association = CalibrationAssociation(l1_obj, {'KPF_DATA_INPUT': self._data_root})
        l1_obj = calibration_association.perform(['bias'])

        image_processing = ImageProcessing(l1_obj)
        l1_obj = image_processing.perform()

        spectral_extraction = SpectralExtraction(l1_obj)
        l2_obj = spectral_extraction.perform()

        return l2_obj
    

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def process_stack_l0_to_l2(self, l0_file_list=None):
        """
        Run each L0 frame in the stack through the L0→L2 pipeline.

        Parameters
        ----------
        l0_file_list : list of str, optional
            L0 files to process. Defaults to self.l0_file_list.

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
            l1_obj, success = self._load_frame(fn, ncache=0)

            if not success:
                failure += 1
                if failure / len(l0_file_list) > 0.2:
                    raise ValueError(f"more than 20% of frames in stack failed to load")
                continue
            
            # extract L1 --> L2
            l2_obj = self._extract_frame(l1_obj)
            
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
        linelist : ndarray, optional
            Reference line wavelengths (Angstroms). Defaults to
            self.linelist.
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
        line_w : ndarray
            Reference wavelengths of surviving lines.
        line_x : ndarray
            Fitted pixel positions of surviving lines.
        """
        if linelist is None:
            linelist = self.linelist

        assert len(flux1d) == len(wave1d), "length of flux and wave arrays are mismatched"
        ncol = len(flux1d)

        lines = {}
        lines['wav'] = np.sort(linelist[(linelist > wave1d.min()) & (linelist < wave1d.max())])
        nlines = len(lines['wav'])

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

        line_x = lines['pix'][~lines['bad']]
        line_w = lines['wav'][~lines['bad']]

        return line_w, line_x
    

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
        linelist : ndarray, optional
            Reference line wavelengths. Defaults to self.linelist.
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
            Flat 1D arrays, all of equal length, with keys:
              'w' — reference wavelength of each surviving line
              'x' — fitted pixel position
              'm' — 1-indexed order number
              'f' — fiber name
        """
        if linelist is None:
            linelist = self.linelist

        if lineprofile is None:
            lineprofile = self.lineprofile

        lines = {}
        for k in ['w', 'x', 'm', 'f']:
            lines[k] = [None] * len(fibers)

        for i, fiber in enumerate(fibers):
            if verbose:
                print(f"fitting {chip} {fiber} line positions")

            flux_arr = l2_obj.data[f'{chip}_{fiber}_FLUX']
            wave_arr = self.rough_wls[f'{chip}_{fiber}_WAVE']

            assert np.shape(flux_arr) == np.shape(wave_arr), "shape mismatch between flux array and rough WLS"

            norder = np.shape(flux_arr)[0]

            for k in ['w', 'x', 'm', 'f']:
                lines[k][i] = [None] * norder

            for o in range(norder):
                line_w, line_x = self.fit_line_positions_1D(
                    flux_arr[o],
                    wave_arr[o],
                    linelist=linelist,
                    lineprofile=lineprofile,
                    window=window,
                    qc_sigma=qc_sigma,
                )

                lines['w'][i][o] = line_w
                lines['x'][i][o] = line_x
                lines['m'][i][o] = (o + 1) * np.ones_like(line_x, dtype=int)
                lines['f'][i][o] = np.array([fiber] * len(line_x))

            for k in lines.keys():
                lines[k][i] = np.hstack(lines[k][i])

        for k in lines.keys():
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
        number (m), and (optionally) fiber index (f). Pixel and order
        variables are rescaled to [-1, 1] before fitting, using the
        canonical detector ranges (pixel in [0, ncol] and order in
        [1, norder]) so that the fit and `evaluate_wls_coeffs` share the
        same parameter space. The 3-fiber case assumes the three KPF
        science fibers (SCI1, SCI2, SCI3) and maps them to {-1, 0, 1}.

        Parameters
        ----------
        lines : dict of ndarray
            Flat 1D arrays as produced by `fit_line_positions_ffi`, with keys:
              'w' - reference line wavelengths
              'x' - fitted pixel positions
              'm' - 1-indexed order numbers
              'f' - fiber names
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
        Raises ValueError if `lines['f']` contains anything other than one
        fiber or the three science fibers (SCI1, SCI2, SCI3).
        """
        # sanitize inputs
        if polyorder_x is None:
            polyorder_x = self.polyorder_x
        if polyorder_m is None:
            polyorder_m = self.polyorder_m
        if polyorder_f is None:
            polyorder_f = self.polyorder_f

        fibers = list(set(lines['f']))

        if (len(fibers) != 1) and (len(fibers) != 3) and (len(fibers) != 5):
            raise ValueError(f"expected 1, 3, or 5 fibers, got {len(fibers)}")
        
        if len(fibers) == 3:
            expected_fibers = ['SCI1', 'SCI2', 'SCI3']
        elif len(fibers) == 5:
            expected_fibers = ['SKY', 'SCI1', 'SCI2', 'SCI3', 'CAL']

        if not (
            np.all(np.isin(fibers, expected_fibers)) and 
            np.all(np.isin(expected_fibers, fibers))
        ):
            raise ValueError(f"unexpected fibers input: {fibers}")

        ncol = self.ccd['ncol']

        # rescale position variables to [-1,1] for Legendre fitting; use
        # canonical detector ranges so this matches evaluate_wls_coeffs.
        _x = 2*lines['x']/ncol - 1
        _m = 2*(lines['m'] - 1)/(norder - 1) - 1

        if len(fibers) != 1:
            _f = np.array([FIBER_INDEX_MAP[f] for f in lines['f']], dtype=int)
            _f = (2*_f)/(len(fibers) - 1) - 1


        # fit Legendre polynomials
        if len(fibers) == 1:
            V = legendre.legvander2d(_x, _m, deg=[polyorder_x, polyorder_m])

            coeffs, *_ = np.linalg.lstsq(V, lines['w'], rcond=None)
            coeffs = coeffs.reshape(polyorder_x+1, polyorder_m+1)
        
        else:
            V = legendre.legvander3d(_x, _m, _f, deg=[polyorder_x, polyorder_m, polyorder_f])

            coeffs, *_ = np.linalg.lstsq(V, lines['w'], rcond=None)
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
        linelist : ndarray, optional
            Reference line wavelengths. Defaults to self.linelist.
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
        if linelist is None:
            linelist = self.linelist
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
        bad = np.abs(coeffs_stack - np.median(coeffs_stack, axis=0)) / mad_std(coeffs_stack, axis=0) > qc_sigma
        coeffs_mean = np.sum(coeffs_stack * ~bad, axis=0)/np.sum(~bad, axis=0)

        W = self.evaluate_wls_coeffs(coeffs_mean, self.ccd['ncol'], self.norder[chip], len(fibers))

        if return_stacks:
            return W, coeffs_mean, coeffs_stack, lines_stack

        return W, coeffs_mean

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def make_master_l2(self, l0_file_list=None, lineprofile=None,
                       polyorder_x=None, polyorder_m=None, polyorder_f=None):
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
        lineprofile : str, optional
            Line profile model name. Defaults to self.lineprofile.
        polyorder_x : int, optional
            Polynomial degree along the pixel axis. Defaults to self.polyorder_x.
        polyorder_m : int, optional
            Polynomial degree along the order axis. Defaults to self.polyorder_m.
        polyorder_f : int, optional
            Polynomial degree along the fiber axis (used for 3- and 5-fiber fits).
            Defaults to self.polyorder_f.

        Returns
        -------
        KPFMasterL2
            Master L2 object with per-fiber _WAVE extensions populated for
            every chip in self.chips, INPUT_FILES recording the stacked
            L0 files, and a 'master_wls' receipt entry.

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

        self._l2_obj_cache = []
        self.process_stack_l0_to_l2(l0_file_list=l0_file_list)

        self.ml2_obj = KPFMasterL2()

        for chip in self.chips:
            W, coeffs = self.compute_wls_from_stack(
                chip=chip,
                fibers=self.fibers,
                lineprofile=lineprofile,
                polyorder_x=polyorder_x,
                polyorder_m=polyorder_m,
                polyorder_f=polyorder_f,
                return_stacks=False,
            )

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
        primary['LINELIST'] = (self.line_list_file, 'Line list reference file')
        primary['LINEPROF'] = (lineprofile, 'Line profile model used in WLS fit')
        primary['POLYORDX'] = (polyorder_x, 'WLS polynomial degree, pixel axis')
        primary['POLYORDM'] = (polyorder_m, 'WLS polynomial degree, order axis')
        primary['POLYORDF'] = (polyorder_f, 'WLS polynomial degree, fiber axis')
        primary['CHIPS'] = (','.join(self.chips), 'Chips included in master WLS')
        primary['FIBERS'] = (','.join(self.fibers), 'Fibers included in master WLS')

        self.ml2_obj.receipt_add_entry('master_wls', 'PASS')

        return self.ml2_obj
