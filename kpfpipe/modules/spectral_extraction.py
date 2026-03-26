"""
KPF Spectral Extraction module.
"""
import numpy as np
import pandas as pd
from numpy.polynomial import polynomial

from kpfpipe import REPO_ROOT, DEFAULTS
from kpfpipe.exceptions import KPFError
from kpfpipe.utils.config_parser import ConfigHandler
from kpfpipe.utils.quality_control import validate_array

DEFAULTS.update({'extraction_method': 'box'})

class SpectralExtraction:
    """
    This class performs spectral extraction of the 1D spectrum.
    Processes data from KPF1 to KPF2.

    Notes
    -----
    Single-letter variable names for 2D images in this class follow 
    Horne 1986 optimal extraction, with small modifcations:
      - D = data
      - V = variance
      - S = sky / scattered light
      - F = flat
      - P = profile
      - M = mask
      - W = weight
    """
    def __init__(self, l1_obj, config=None):
        self.l1_obj = l1_obj

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(
                ["DATA_DIRS", "KPFPIPE", "MODULE_SPECTRAL_EXTRACTION"]
            )
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in DEFAULTS.items():
            setattr(self, k, params.get(k, v))


    def _read_order_trace_reference(self, chip):
        """
        Load and cache the order trace reference table for a given chip.

        Parameters
        ----------
        chip : str
            Chip identifier (e.g., 'GREEN', 'RED').

        Notes
        -----
        The trace reference file is read from the repository reference
        directory and cached in `self.order_trace` to avoid repeated I/O.
        """
        if not hasattr(self, 'order_trace'):
            self.order_trace = {}

        filepath = f'{REPO_ROOT}/reference/order_trace_{chip.lower()}.csv'
        with open(filepath, 'r') as f:
            self.order_trace[chip.upper()] = (
                pd.read_csv(f, index_col=0)
                .set_index(['Fiber', 'Order'])
                .sort_index()
            )


    def _get_orderlet_pixels(self, chip, fiber, order, return_coords=False):
        """
        Extract the 2D pixel region corresponding to a single orderlet.

        Parameters
        ----------
        chip : str
            Chip identifier, i.e. 'GREEN' or 'RED'
        fiber : str
            Fiber identifier, e.g. 'SCI2'
        order : int
            Spectral order number.
        return_coords : bool, optional
            If True, also return the detector row bounds of the extracted box.

        Returns
        -------
        D : ndarray
            2D array of data values within the bounding box.
        V : ndarray
            2D array of variance values within the bounding box.
        W : ndarray
            2D weight array accounting for fractional pixel coverage at
            order boundaries.
        row_min : int, optional
            Lower detector row index of the bounding box (if return_coords=True).
        row_max : int, optional
            Upper detector row index of the bounding box (if return_coords=True).

        Notes
        -----
        The bounding region fully encloses the traced orderlet. Due to order
        tilt and curvature, the box may include pixels from adjacent orders.
        Weights are assigned as:
        - 1 for fully enclosed pixels,
        - 0 for pixels outside the orderlet,
        - fractional values at the top and bottom trace edges.
        """
        chip = chip.upper()
        fiber = fiber.upper()

        data_image = self.l1_obj.data[f'{chip}_CCD']
        var_image = self.l1_obj.data[f'{chip}_VAR']
        nrow, ncol = data_image.shape

        if not hasattr(self, 'order_trace') or chip not in self.order_trace:
            self._read_order_trace_reference(chip)

        try:
            trace = self.order_trace[chip].loc[(fiber, order)]
        except KeyError:
            raise KPFError(f"No trace found for {chip} {fiber} Order {order}")

        if trace.ndim != 1:
            raise KPFError(
                f"Expected exactly one row for {chip} {fiber} Order {order} "
                f"but found {trace.shape[0]}"
        )

        # track the trace position
        coeffs = np.array(trace[[f'Coeff{i}' for i in range(4)]], dtype=np.float32)

        trace_center = polynomial.polyval(np.arange(ncol), coeffs)
        trace_top    = trace_center + trace.TopEdge
        trace_bottom = trace_center - trace.BottomEdge

        off_detector = (trace_top > nrow-1) | (trace_bottom < 0)

        if np.any(off_detector):
            trace_top[off_detector] = np.minimum(trace_top, nrow-1)[off_detector]
            trace_center[off_detector] = np.minimum(trace_center, nrow-1)[off_detector]
            trace_bottom[off_detector] = np.minimum(trace_bottom, nrow-1)[off_detector]
    
            trace_top[off_detector] = np.maximum(trace_top, 0)[off_detector]
            trace_center[off_detector] = np.maximum(trace_center, 0)[off_detector]
            trace_bottom[off_detector] = np.maximum(trace_bottom, 0)[off_detector]

        box_zeropt = int(np.floor(trace_bottom.min()))
        box_height = int(np.ceil(trace_top.max())) - box_zeropt

        edge_pixel_top = np.array(np.floor(trace_top - box_zeropt), dtype=int)
        edge_pixel_bottom = np.array(np.floor(trace_bottom - box_zeropt), dtype=int)

        # broadcast vectors            
        _row = np.arange(box_height)[:,None]
        _edge_pixel_top = edge_pixel_top[None,:]
        _edge_pixel_bottom = edge_pixel_bottom[None,:]
        _trace_top = trace_top[None,:]
        _trace_bottom = trace_bottom[None,:]

        # make data, variance, and weight arrays
        D = data_image[box_zeropt:box_zeropt + box_height]
        V = var_image[box_zeropt:box_zeropt + box_height]        
        
        W = np.zeros_like(D, dtype=np.float32)
        W[(_row > _edge_pixel_bottom) & (_row < _edge_pixel_top)] = 1

        mask_top = _row == _edge_pixel_top
        frac_top = np.tile((_trace_top - box_zeropt - _edge_pixel_top), (box_height,1))
        W[mask_top] = frac_top[mask_top]

        mask_bot = _row == _edge_pixel_bottom
        frac_bot = np.tile((1 - (_trace_bottom - box_zeropt - _edge_pixel_bottom)), (box_height,1))
        W[mask_bot] = frac_bot[mask_bot]
        
        if return_coords:
            return D, V, W, box_zeropt, box_zeropt+box_height
        return D, V, W


    @staticmethod
    def _box_extraction(D, V, *, S=None, M=None, W=None):
        """
        Perform simple box (summation) extraction of a 2D spectral trace.

        Parameters
        ----------
        D : ndarray
            2D data array.
        V : ndarray
            2D variance array.
        S : ndarray, optional
            2D sky/scattered light model.
        M : ndarray, optional
            Binary bad-pixel mask (1 = good, 0 = bad).
        W : ndarray, optional
            Pixel weights accounting for trace geometry.

        Returns
        -------
        flux_1d : ndarray
            Extracted 1D flux spectrum.
        var_1d : ndarray
            Corresponding 1D variance spectrum.
        """
        if S is None:
            S = np.zeros_like(D)
        if M is None:
            M = np.ones_like(D)
        if W is None:
            W = np.ones_like(D)

        if np.any((M * W).sum(axis=0) == 0):
            raise ValueError("Fully masked columns detected in trace")

        M = M * (M.shape[0] / M.sum(0))

        flux_1d = np.sum((D - S) * M * W, axis=0)
        var_1d = np.sum(V * (M * W) ** 2, axis=0)
                        
        return flux_1d, var_1d


    @staticmethod
    def _optimal_extraction(D, V, *, S=None, M=None, W=None, P=None):
        """
        Perform optimal extraction of a 2D spectral trace.

        Parameters
        ----------
        D : ndarray
            2D data array.
        V : ndarray
            2D variance array.
        S : ndarray, optional
            2D sky/scattered light model.
        M : ndarray, optional
            Binary bad-pixel mask (1 = good, 0 = bad).
        W : ndarray, optional
            Pixel weights accounting for trace geometry.
        P : ndarray, optional
            Spatial profile model of the spectral trace.

        Returns
        -------
        flux_1d : ndarray
            Extracted 1D flux spectrum.
        var_1d : ndarray
            Corresponding 1D variance spectrum.

        Notes
        -----
        Follows Horne (1986) optimal extraction algorithm.
        """
        raise NotImplementedError("optimal extraction not yet implemented")


    @staticmethod
    def _flat_relative_extraction(D, V, *, S=None, M=None, W=None, F=None):
        """
        Perform flat-relative spectral extraction.

        Parameters
        ----------
        D : ndarray
            2D data array.
        V : ndarray
            2D variance array.
        S : ndarray, optional
            2D sky/scattered light model.
        M : ndarray, optional
            Binary bad-pixel mask (1 = good, 0 = bad).
        W : ndarray, optional
            Pixel weights accounting for trace geometry.
        F : ndarray, optional
            Flat-field reference image.

        Returns
        -------
        flux_1d : ndarray
            Extracted 1D flux spectrum.
        var_1d : ndarray
            Corresponding 1D variance spectrum.

        Notes
        -----
        Follows Zechmeister et al. (2014) flat-relative extraction algorithm.
        """
        raise NotImplementedError("flat relative extraction not yet implemented")


    def extract_orderlet(self, chip, fiber, order, method=None):
        """
        Extract a single orderlet as a 1D spectrum.

        Parameters
        ----------
        chip : str
            Chip identifier, i.e. 'GREEN' or 'RED'
        fiber : str
            Fiber identifier, e.g. 'SCI2'
        order : int
            Spectral order number.
        method : str, optional
            Extraction method ('box', 'optimal', or 'flat_relative').

        Returns
        -------
        flux_1d : ndarray
            Extracted 1D flux spectrum for the specified orderlet.
        var_1d : ndarray
            Corresponding 1D variance spectrum.

        Notes
        -----
        Retrieves the orderlet pixel region and dispatches to the selected
        extraction method.
        """
        if method is None:
            method = self.extraction_method

        try:
            extraction_fxn = self.__getattribute__(f'_{method}_extraction')
        except AttributeError:
            raise AttributeError(f"Unsupported extraction method: '{method}'")

        D, V, W, row_min, row_max = self._get_orderlet_pixels(chip, fiber, order, return_coords=True)

        # TODO: add sky background
        # TODO: add bad pixel masking
        flux_1d, var_1d = extraction_fxn(D, V, W=W)

        validate_array(flux_1d, context=f'flux_1d array: {chip} {fiber} {order}', response='warn')
        validate_array(var_1d, context=f'var_1d array: {chip} {fiber} {order}', response='warn')

        return flux_1d, var_1d


    def extract_ffi(self, chip, fibers=None, method=None):
        """
        Extract all spectral orders from a full-frame image (FFI).

        Parameters
        ----------
        chip : str
            Chip identifier, i.e. 'GREEN' or 'RED'
        fibers : list of str, optional
            Fibers identifiers, e.g. 'SCI2'
        method : str, optional
            Extraction method ('box', 'optimal', or 'flat_relative').

        Returns
        -------
        dict
            Dictionary containing 2D arrays of shape (norder, ncol) for
            extracted flux and variance. Keys follow standard KPF name
            conventions, e.g. 'GREEN_SCI2_FLUX'.

        Notes
        -----
        Loops over all spectral orders and requested fibers, performing
        order-by-order extraction.
        """
        if fibers is None:
            fibers = self.fibers
        if method is None:
            method = self.extraction_method

        chip = chip.upper()
        fibers = [f.upper() for f in fibers]

        norder = self.norder[chip]
        nrow, ncol = self.l1_obj.data[f'{chip}_CCD'].shape

        l2_arrays = {}
        for fiber in fibers:
            l2_arrays[f'{chip}_{fiber}_FLUX'] = np.empty((norder,ncol))
            l2_arrays[f'{chip}_{fiber}_VAR'] = np.empty((norder,ncol))

        failure = 0
        for order in range(1,norder+1):
            for fiber in fibers:
                try:
                    flux_1d, var_1d = self.extract_orderlet(chip, fiber, order, method)
                except KPFError:
                    failure += 1
                    flux_1d = np.full(ncol, np.nan, dtype=np.float32)
                    var_1d  = np.full(ncol, np.nan, dtype=np.float32)

                l2_arrays[f'{chip}_{fiber}_FLUX'][order-1] = flux_1d
                l2_arrays[f'{chip}_{fiber}_VAR'][order-1] = var_1d

        # During some KPF eras one of the traces does not fall on the detector.
        # In this case a single failure is expected from this method. Allowing
        # the loop to continue through all orders provides useful diagnostic
        # information for cases where the algorithm truly fails.
        if failure > 1:
            raise KPFError(f"Failed to extract {failure} orders from the {chip} CCD")
        
        return l2_arrays


    def perform(self, chips=None, fibers=None, method=None):
        """
        Execute spectral extraction. Optional keyword arguments
        default to config settings.

        Parameters
        ----------
        chips : list of str, optional
            Chip identifiers, i.e. 'GREEN' or 'RED'
        fibers : list of str, optional
            Fiber identifiers, e.g. 'SCI2'
        method : str, optional
            Extraction method ('box', 'optimal', or 'flat_relative').

        Returns
        -------
        l2_obj : KPF2
            L2 data object containing extracted 1D flux and variance arrays.

        Notes
        -----
        Creates a KPF2 object from the input KPF1 object and populates it
        with extracted spectra for all requested chips and fibers.
        """
        if chips is None:
            chips = self.chips
        if fibers is None:
            fibers = self.fibers
        if method is None:
            method = self.extraction_method

        l2_obj = self.l1_obj.to_kpf2()

        for chip in chips:
            l2_arrays = self.extract_ffi(chip, fibers, method)
            for fiber in fibers:
                l2_obj.set_data(f'{chip}_{fiber}_FLUX', l2_arrays[f'{chip}_{fiber}_FLUX'])
                l2_obj.set_data(f'{chip}_{fiber}_VAR',  l2_arrays[f'{chip}_{fiber}_VAR'])

        l2_obj.receipt_add_entry('spectral_extraction', 'PASS')
        return l2_obj
