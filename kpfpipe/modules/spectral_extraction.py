"""
KPF Spectral Extraction module.

Extracts per-order 1D spectra from an assembled L1 frame into a KPF2 (L2),
populating the per-fiber FLUX and VAR arrays.
"""

import glob
import logging
import os

import numpy as np
import pandas as pd
from numpy.polynomial import polynomial

from kpfpipe import DEFAULTS, REPO_ROOT
from kpfpipe.utils.config import ConfigHandler

logger = logging.getLogger(__name__)

_DEFAULTS = {**DEFAULTS, "extraction_method": "box"}


class SpectralExtraction:
    """
    Extract per-order 1D spectra from a KPF1, producing a KPF2.

    Parameters
    ----------
    l1_obj : KPF1
        Assembled L1 frame carrying the per-chip ``{CHIP}_CCD`` / ``{CHIP}_VAR``
        full-frame images to extract from.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: extraction_method.
    """

    def __init__(self, l1_obj, config=None):
        self.l1_obj = l1_obj

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(
                ["DATA_DIRS", "TRACES", "MODULE_SPECTRAL_EXTRACTION"]
            )
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in _DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        self._order_trace = None
        self._order_trace_path = None
        self._instera = None
        self._info = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer_instrument_era(self):
        """Infer this frame's instrument era from ``JD_UTC``, caching its tag in
        ``self._instera`` and returning its ``kpf_instrument_eras`` row with the
        frame's observation time.

        A frame whose ``INSTERA`` disagrees is warned about and restamped -- the
        timestamp wins, since references are keyed to it.

        Raises ``ValueError`` when the frame cannot be dated or falls outside
        every era. Deliberately not a ``LookupError``: ``extract_ffi`` catches
        that to fill an absent orderlet with NaN, and would bury this."""
        primary = self.l1_obj.headers["PRIMARY"]
        jd_utc = primary.get("JD_UTC")
        obs_time = pd.to_datetime(jd_utc, unit="D", origin="julian")
        if pd.isna(obs_time):
            raise ValueError(
                f"Cannot infer the instrument era of {self.l1_obj.obs_id}: its "
                f"JD_UTC is {jd_utc!r}"
            )

        eras = pd.read_csv(
            f"{REPO_ROOT}/reference/kpf_instrument_eras.csv",
            parse_dates=["UT_start_date", "UT_end_date"],
        )
        in_era = eras[
            (eras["UT_start_date"] <= obs_time) & (obs_time <= eras["UT_end_date"])
        ]
        if in_era.empty:
            raise ValueError(
                f"No KPF instrument era covers {obs_time}; the eras of "
                f"reference/kpf_instrument_eras.csv do not span it"
            )

        era = in_era.iloc[0]
        self._instera = str(era["INSTERA"])

        if str(primary.get("INSTERA")) != self._instera:
            logger.warning(
                "header INSTERA %s disagrees with instrument era %s inferred from "
                "JD_UTC; restamping INSTERA as %s",
                primary.get("INSTERA"),
                self._instera,
                self._instera,
            )
            self.l1_obj.set_keyword("INSTERA", self._instera)

        return era, obs_time

    def _read_order_trace_reference(self):
        """Load the vetted order trace for this frame, caching the per-chip
        tables in ``self._order_trace`` and the file in ``self._order_trace_path``.

        The reference is the most recent one measured before the frame within
        the frame's instrument era, read from ``reference/order_traces``."""
        era, obs_time = self._infer_instrument_era()

        # Trace geometry moves whenever the instrument is opened, so only a
        # reference measured earlier in this era describes this frame.
        latest = min(era["UT_end_date"], obs_time)
        in_era = {}
        for path in glob.glob(f"{REPO_ROOT}/reference/order_traces/order_trace_*.csv"):
            datecode = os.path.basename(path)[len("order_trace_") : -len(".csv")]
            measured = pd.Timestamp(datecode)
            if era["UT_start_date"] <= measured <= latest:
                in_era[measured] = path
        if not in_era:
            raise FileNotFoundError(
                f"No order trace measured before {obs_time} within KPF instrument "
                f"era {self._instera}"
            )

        filepath = in_era[max(in_era)]
        logger.info("reading order trace from %s", filepath)

        table = pd.read_csv(filepath)
        table = table[table["Status"] != "missing"]
        self._order_trace = {
            chip: rows.set_index(["Fiber", "Order"]).sort_index()
            for chip, rows in table.groupby("Chip")
        }
        self._order_trace_path = filepath

    def _get_orderlet_pixels(self, chip, fiber, order, return_coords=False):
        """Slice the 2D bounding box (data ``D``, variance ``V``, weights ``W``)
        for a single orderlet, optionally with its detector row bounds.

        The box encloses the traced orderlet; order tilt/curvature can pull in
        adjacent-order pixels. ``W`` is 1 inside, 0 outside, fractional at the
        top/bottom trace edges, and 0 beyond the trace's ``X1``..``X2`` span."""
        chip = chip.upper()
        fiber = fiber.upper()

        data_image = self.l1_obj.data[f"{chip}_CCD"]
        var_image = self.l1_obj.data[f"{chip}_VAR"]
        nrow, ncol = data_image.shape

        if self._order_trace is None:
            self._read_order_trace_reference()

        try:
            trace = self._order_trace[chip].loc[(fiber, order)]
        except KeyError:
            raise LookupError(
                f"No trace found for {chip} {fiber} Order {order}"
            ) from None

        if trace.ndim != 1:
            raise ValueError(
                f"Expected exactly one row for {chip} {fiber} Order {order} "
                f"but found {trace.shape[0]} (likely duplicate entries in the "
                f"order trace reference file)"
            )

        # Evaluate the polynomial trace so each column gets its own row center.
        coefficient_columns = sorted(
            (column for column in trace.index if column.startswith("Coeff")),
            key=lambda column: int(column.removeprefix("Coeff")),
        )
        coeffs = np.array(trace[coefficient_columns], dtype=np.float32)

        trace_center = polynomial.polyval(np.arange(ncol, dtype=np.float32), coeffs)
        trace_top = (trace_center + trace.TopEdge).astype(np.float32)
        trace_bottom = (trace_center - trace.BottomEdge).astype(np.float32)

        off_detector = (trace_top > nrow - 1) | (trace_bottom < 0)

        if np.any(off_detector):
            trace_top[off_detector] = np.minimum(trace_top, nrow - 1)[off_detector]
            trace_center[off_detector] = np.minimum(trace_center, nrow - 1)[
                off_detector
            ]
            trace_bottom[off_detector] = np.minimum(trace_bottom, nrow - 1)[
                off_detector
            ]

            trace_top[off_detector] = np.maximum(trace_top, 0)[off_detector]
            trace_center[off_detector] = np.maximum(trace_center, 0)[off_detector]
            trace_bottom[off_detector] = np.maximum(trace_bottom, 0)[off_detector]

        box_zeropt = int(np.floor(trace_bottom.min()))
        box_height = int(np.ceil(trace_top.max())) - box_zeropt

        edge_pixel_top = np.array(np.floor(trace_top - box_zeropt), dtype=int)
        edge_pixel_bottom = np.array(np.floor(trace_bottom - box_zeropt), dtype=int)

        # Reshape the per-column edge vectors so they broadcast against the box
        # rows when building the weight array below.
        _row = np.arange(box_height)[:, None]
        _edge_pixel_top = edge_pixel_top[None, :]
        _edge_pixel_bottom = edge_pixel_bottom[None, :]
        _trace_top = trace_top[None, :]
        _trace_bottom = trace_bottom[None, :]

        # Slice the bounding box out of the full detector for data and variance;
        # the weight array is then filled per-pixel for fractional edge coverage.
        D = data_image[box_zeropt : box_zeropt + box_height]
        V = var_image[box_zeropt : box_zeropt + box_height]

        W = np.zeros_like(D, dtype=np.float32)
        W[(_row > _edge_pixel_bottom) & (_row < _edge_pixel_top)] = 1

        mask_top = _row == _edge_pixel_top
        frac_top = np.tile((_trace_top - box_zeropt - _edge_pixel_top), (box_height, 1))
        W[mask_top] = frac_top[mask_top]

        mask_bot = _row == _edge_pixel_bottom
        frac_bot = np.tile(
            (1 - (_trace_bottom - box_zeropt - _edge_pixel_bottom)), (box_height, 1)
        )
        W[mask_bot] = frac_bot[mask_bot]

        detector_columns = np.arange(ncol)
        W[:, (detector_columns < trace.X1) | (detector_columns > trace.X2)] = 0

        if return_coords:
            return D, V, W, box_zeropt, box_zeropt + box_height
        return D, V, W

    @staticmethod
    def _box_extraction(D, V, *, S=None, M=None, W=None):
        """Box (summation) extraction of a 2D trace, returning 1D flux/variance.

        Single-letter array names (D, V, S, M, W) follow the Horne (1986)
        optimal-extraction convention."""
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
        """Optimal extraction of a 2D trace (not yet implemented).

        Follows the Horne (1986) optimal-extraction algorithm; single-letter array
        names follow the ``_box_extraction`` convention (``P`` adds the spatial
        profile)."""
        raise NotImplementedError("optimal extraction not yet implemented")

    @staticmethod
    def _flat_relative_extraction(D, V, *, S=None, M=None, W=None, F=None):
        """Flat-relative extraction of a 2D trace (not yet implemented).

        Follows the Zechmeister et al. (2014) flat-relative extraction algorithm;
        single-letter array names follow the ``_box_extraction`` convention
        (``F`` adds the flat)."""
        raise NotImplementedError("flat relative extraction not yet implemented")

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def extract_orderlet(self, chip, fiber, order, extraction_method=None):
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
        extraction_method : str, optional
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
        if extraction_method is None:
            extraction_method = self.extraction_method

        try:
            extraction_fxn = getattr(self, f"_{extraction_method}_extraction")
        except AttributeError:
            raise AttributeError(
                f"Unsupported extraction method: '{extraction_method}'"
            ) from None

        D, V, W, row_min, row_max = self._get_orderlet_pixels(
            chip, fiber, order, return_coords=True
        )

        # A column whose whole aperture has left the detector carries no flux.
        on_detector = W.sum(axis=0) > 0
        if on_detector.all():
            flux_1d, var_1d = extraction_fxn(D, V, W=W)
        else:
            flux_1d = np.full(W.shape[1], np.nan, dtype=np.float32)
            var_1d = np.full(W.shape[1], np.nan, dtype=np.float32)
            flux_1d[on_detector], var_1d[on_detector] = extraction_fxn(
                D[:, on_detector], V[:, on_detector], W=W[:, on_detector]
            )

        for arr, name in ((flux_1d, "flux_1d"), (var_1d, "var_1d")):
            n_bad = int(np.sum(~np.isfinite(arr)))
            n_neg = int(np.sum(arr < 0))
            if n_bad or n_neg:
                logger.debug(
                    "%s array: %s %s %s has %d non-finite, %d negative values",
                    name,
                    chip,
                    fiber,
                    order,
                    n_bad,
                    n_neg,
                )

        return flux_1d, var_1d

    def extract_ffi(self, chip, fibers=None, extraction_method=None):
        """
        Extract all spectral orders from a full-frame image (FFI).

        Parameters
        ----------
        chip : str
            Chip identifier, i.e. 'GREEN' or 'RED'
        fibers : list of str, optional
            Fibers identifiers, e.g. 'SCI2'
        extraction_method : str, optional
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
        if extraction_method is None:
            extraction_method = self.extraction_method

        chip = chip.upper()
        fibers = [f.upper() for f in fibers]

        norder = self.norder[chip]
        nrow, ncol = self.l1_obj.data[f"{chip}_CCD"].shape

        l2_arrays = {}
        for fiber in fibers:
            l2_arrays[f"{chip}_{fiber}_FLUX"] = np.empty(
                (norder, ncol), dtype=np.float32
            )
            l2_arrays[f"{chip}_{fiber}_VAR"] = np.empty(
                (norder, ncol), dtype=np.float32
            )

        failure = 0
        for order in range(norder):
            for fiber in fibers:
                try:
                    flux_1d, var_1d = self.extract_orderlet(
                        chip, fiber, order, extraction_method
                    )
                except LookupError:
                    failure += 1
                    flux_1d = np.full(ncol, np.nan, dtype=np.float32)
                    var_1d = np.full(ncol, np.nan, dtype=np.float32)

                l2_arrays[f"{chip}_{fiber}_FLUX"][order] = flux_1d
                l2_arrays[f"{chip}_{fiber}_VAR"][order] = var_1d

        # During some KPF eras one of the traces does not fall on the detector.
        # In this case a single failure is expected from this method. Allowing
        # the loop to continue through all orders provides useful diagnostic
        # information for cases where the algorithm truly fails.
        if failure == 1:
            logger.warning(
                "1 orderlet failed to extract from the %s CCD; filled with NaN.", chip
            )
        elif failure > 1:
            raise LookupError(
                f"Failed to extract {failure} orderlets from the {chip} CCD"
            )

        return l2_arrays

    # ------------------------------------------------------------------
    # Private helpers - module execution
    # ------------------------------------------------------------------

    def _track_info(self, chips, fibers):
        """Build and cache the info() summary text from instance attributes."""
        lines = [
            "SpectralExtraction",
            f"  obs_id:            {self.l1_obj.obs_id}",
            f"  extraction_method: {self.extraction_method}",
            f"  order trace:       {self._order_trace_path}",
            f"\n  {'CHIP':<8s} {'FIBERS':<30s} {'NORDER'}",
            "  " + "-" * 46,
        ]
        fibers_str = " ".join(fibers)
        for chip in chips:
            lines.append(f"  {chip:<8s} {fibers_str:<30s} {self.norder[chip.upper()]}")
        self._info = "\n\n" + "\n".join(lines) + "\n\n"

    def _set_headers(self, l2_obj):
        """Write the order trace the spectra were extracted with and the era it
        was chosen for (inferred after ``to_kpf2`` copied the L1 PRIMARY)."""
        if self._order_trace_path is not None:
            l2_obj.set_keyword("TRACFILE", self._order_trace_path)
        if self._instera is not None:
            l2_obj.set_keyword("INSTERA", self._instera)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(self, chips=None, fibers=None, *, extraction_method=None):
        """
        Execute spectral extraction. Optional keyword arguments
        default to config settings.

        Parameters
        ----------
        chips : list of str, optional
            Chip identifiers, i.e. 'GREEN' or 'RED'
        fibers : list of str, optional
            Fiber identifiers, e.g. 'SCI2'
        extraction_method : str, optional
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
        if extraction_method is None:
            extraction_method = self.extraction_method

        l2_obj = self.l1_obj.to_kpf2()

        for chip in chips:
            l2_arrays = self.extract_ffi(chip, fibers, extraction_method)
            for fiber in fibers:
                l2_obj.set_data(
                    f"{chip}_{fiber}_FLUX", l2_arrays[f"{chip}_{fiber}_FLUX"]
                )
                l2_obj.set_data(f"{chip}_{fiber}_VAR", l2_arrays[f"{chip}_{fiber}_VAR"])

        self._set_headers(l2_obj)
        self._track_info(chips, fibers)
        l2_obj.receipt_add_entry("spectral_extraction", "", "PASS")

        logger.info("%s", self._info)
        return l2_obj

    def info(self):
        """Print a summary of the module configuration and extraction results."""
        if self._info is None:
            print(f"{type(self).__name__}: perform() has not been called")
        else:
            print(self._info)
