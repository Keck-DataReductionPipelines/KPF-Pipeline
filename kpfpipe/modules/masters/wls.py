"""
KPF Master Wavelength Solution construction module.
"""
import numpy as np
import pandas as pd
from astropy.stats import mad_std

from kpfpipe import DEFAULTS, REPO_ROOT
from kpfpipe.modules.masters.base import BaseMasterModule
from kpfpipe.modules.calibration_association import CalibrationAssociation
from kpfpipe.modules.image_processing import ImageProcessing
from kpfpipe.modules.spectral_extraction import SpectralExtraction
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import optimize_lsq

DEFAULTS.update({
    'rough_wls_file': f'{REPO_ROOT}/reference/rough_wls_fallback.csv',
    'linelist_file': f'{REPO_ROOT}/reference/thar_line_list.csv',
})

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
        self.linelist = pd.read_csv(self.linelist_file)['Wavelength'].values
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

        Raises
        ------
        ValueError
            If more than 20% of frames fail to load.
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
                              linemodel='gaussian',
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
        linemodel : str, optional
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
            theta, rms = optimize_lsq(x, y, linemodel)

            if linemodel == 'gaussian':
                # gaussian_dist theta convention: [b, a, mu, sigma]
                lines['pix'][i] = theta[2]
                lines['std'][i] = theta[3]
                lines['amp'][i] = theta[1]
                lines['rms'][i] = rms / np.abs(theta[1] * np.sqrt(2*np.pi) * theta[3])
            else:
                raise ValueError(f"Unsupported linemodel: {linemodel}")

        if linemodel == 'gaussian':
            lines['bad'] = np.abs(lines['rms'] - np.median(lines['rms'])) / mad_std(lines['rms']) > qc_sigma
            lines['bad'] |= np.abs(lines['std'] - np.median(lines['std'])) / mad_std(lines['std']) > qc_sigma
            lines['bad'] |= lines['amp'] < 0
        else:
            raise ValueError(f"Unsupported linemodel: {linemodel}")

        line_x = lines['pix'][~lines['bad']]
        line_w = lines['wav'][~lines['bad']]

        return line_w, line_x
    

    def fit_line_positions_ffi(self,
                               l2_obj,
                               chip,
                               fibers,
                               linelist=None,
                               linemodel=None,
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
        linemodel : str, optional
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

        if linemodel is None:
            linemodel = self.linemodel

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
                    linemodel=linemodel,
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
