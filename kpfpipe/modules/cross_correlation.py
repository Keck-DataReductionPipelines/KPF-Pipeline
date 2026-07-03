"""
KPF Cross-Correlation module.

Cross-correlates each order of an extracted, wavelength-calibrated,
barycentric-corrected KPF L2 against a line mask to produce per-order
cross-correlation functions (CCFs). Produces a KPF4 (L4) holding the per-order
CCFs, their per-bin photon variances, and the per-order metadata table the
radial-velocity step later fills with fitted RVs.

Each fiber's mask, barycentric handling, and CCF grid center are dispatched from
its illumination source (SCI-OBJ/SKY-OBJ/CAL-OBJ in INSTRUMENT_HEADER):

  source   mask                 barycorr  grid center
  target   TARGTEFF-lookup      yes       TARGRADV (systemic)
  sky      G2_espresso (solar)  yes       0
  thar     ThAr list (unit wt)  no        0
  etalon   / lfc                                skipped (no CCF; not implemented)
  none     not illuminated -> skipped (no CCF)

All reference wavelengths (stellar line masks, ThAr line list) are in vacuum;
no air/vacuum conversion is performed.
"""

import warnings

import astropy.units as u
import numpy as np
import pandas as pd

from kpfpipe import DEFAULTS, REPO_ROOT
from kpfpipe.utils.astro import compute_redshift
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.validation import strictly_increasing

_DEFAULTS = {
    **DEFAULTS,
    "ccf_mask_width": 1.0,
    "ccf_step_size": 0.25,
    "ccf_window": [-100.0, 100.0],
}


class CrossCorrelation:
    """
    Compute per-order cross-correlation functions from a KPF2.

    Parameters
    ----------
    l2_obj : KPF2
        Extracted L2 frame. Must have per-fiber FLUX and WAVE arrays populated
        (by SpectralExtraction and WavelengthCalibration) and the per-order
        barycentric correction extensions populated by BarycentricCorrection.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: chips, fibers, ccf_mask_width,
        ccf_step_size, ccf_window.
    """

    def __init__(self, l2_obj, config=None):
        self.l2_obj = l2_obj

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(
                ["DATA_DIRS", "KPFPIPE", "MODULE_CROSS_CORRELATION"]
            )
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in _DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        # Lazily-populated caches; the per-orderlet ones are keyed by f'{chip}_{fiber}'.
        self._illumination_source = {}  # set by _resolve_illumination_source()
        self._line_mask = {}  # set by _build_line_mask()
        self._velocity_grid = {}  # set by _build_velocity_grid()
        self._ccf = {}  # CCF cube, set by compute_ccfs()
        self._ccf_var = {}  # per-bin CCF variance cube, set by compute_ccfs()
        self._ccf_mask_width = self.ccf_mask_width  # width behind the cached CCFs
        self._ccf_step_size = self.ccf_step_size  # step behind the cached grids
        self._order_weights = None  # order-weight table, loaded by _get_order_weights()
        self._chips_done = []  # chips processed, for _set_headers/_track_info
        self._fibers_done = []  # illuminated fibers written, for _set_headers
        self._info = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # Fiber -> the INSTRUMENT_HEADER keyword giving its illumination source.
    _OBJ_KEYWORD = {
        "SCI1": "SCI-OBJ",
        "SCI2": "SCI-OBJ",
        "SCI3": "SCI-OBJ",
        "SKY": "SKY-OBJ",
        "CAL": "CAL-OBJ",
    }

    def _resolve_illumination_source(self, chip, fiber):
        """
        Resolve (and cache) the illumination source and its CCF settings for one
        orderlet, from its SCI-OBJ/SKY-OBJ/CAL-OBJ keyword in INSTRUMENT_HEADER.

        Returns a dict with keys 'object' (the normalized source), 'mask_name',
        'apply_barycorr', and 'vel_grid_center'. An unilluminated fiber ('none')
        has None mask/barycorr/center. Sources whose CCF path is not yet built
        (etalon, lfc) are skipped the same way, with a warning.
        """
        key = f"{chip.upper()}_{fiber.upper()}"
        if key in self._illumination_source:
            return self._illumination_source[key]
        try:
            keyword = self._OBJ_KEYWORD[fiber.upper()]
        except KeyError:
            raise ValueError(
                f"unknown fiber {fiber!r}; expected one of {sorted(self._OBJ_KEYWORD)}"
            ) from None
        inst = self.l2_obj.headers.get("INSTRUMENT_HEADER", {})
        if keyword not in inst:
            raise ValueError(
                f"illumination keyword {keyword!r} not in INSTRUMENT_HEADER; "
                f"cannot dispatch a mask for fiber {fiber}"
            )

        # Map the raw keyword to the source object and its CCF settings (mask,
        # barycorr flag, grid center: systemic RV for a star, 0 for sky/cal).
        raw = inst.get(keyword)
        v = str(raw).strip().lower()
        if v == "target":
            source = {
                "object": "target",
                "mask_name": self._resolve_stellar_mask(),
                "apply_barycorr": True,
                "vel_grid_center": self._get_systemic_rv(),
            }
        elif v == "sky":
            source = {
                "object": "sky",
                "mask_name": "G2_espresso",
                "apply_barycorr": True,
                "vel_grid_center": 0.0,
            }
        elif v in ("th_gold", "th_daily"):
            source = {
                "object": "thar",
                "mask_name": "thar",
                "apply_barycorr": False,
                "vel_grid_center": 0.0,
            }
        elif v == "none":
            source = {
                "object": "none",
                "mask_name": None,
                "apply_barycorr": None,
                "vel_grid_center": None,
            }
        elif v == "lfcfiber":
            source = {
                "object": "lfc",
                "mask_name": None,
                "apply_barycorr": None,
                "vel_grid_center": None,
            }
            warnings.warn(
                f"{fiber.upper()} is lfc-illuminated; CCF is not implemented. "
                "Skipping this fiber.",
                UserWarning,
                stacklevel=2,
            )
        elif "etalon" in v:
            source = {
                "object": "etalon",
                "mask_name": None,
                "apply_barycorr": None,
                "vel_grid_center": None,
            }
            warnings.warn(
                f"{fiber.upper()} is etalon-illuminated; CCF is not implemented. "
                "Skipping this fiber.",
                UserWarning,
                stacklevel=2,
            )
        else:
            raise ValueError(f"unrecognized illumination source {raw!r}")

        self._illumination_source[key] = source
        return source

    def _resolve_stellar_mask(self):
        """Select the stellar line-mask name from TARGTEFF via line_mask_lookup.csv."""
        inst = self.l2_obj.headers.get("INSTRUMENT_HEADER", {})
        try:
            teff = float(inst.get("TARGTEFF"))
        except (TypeError, ValueError):
            teff = None
        if teff is None or not np.isfinite(teff) or teff <= 0:
            raise ValueError(
                "target effective temperature (TARGTEFF) not available in "
                "INSTRUMENT_HEADER; cannot select a stellar line mask"
            )
        line_map = pd.read_csv(f"{REPO_ROOT}/reference/line_masks/line_mask_lookup.csv")
        row = line_map[(line_map["TEFF_MIN"] <= teff) & (teff < line_map["TEFF_MAX"])]
        return row["DEFAULT_MASK"].iloc[0]

    def _get_systemic_rv(self):
        """Target systemic RV (TARGRADV) [km/s] — the stellar CCF grid center."""
        inst = self.l2_obj.headers.get("INSTRUMENT_HEADER", {})
        try:
            star_rv = float(inst.get("TARGRADV"))
        except (TypeError, ValueError):
            star_rv = None
        if star_rv is None or not np.isfinite(star_rv):
            raise ValueError(
                "target radial velocity (TARGRADV) not available in "
                "INSTRUMENT_HEADER; cannot center the CCF velocity grid"
            )
        return star_rv

    def _build_line_mask(self, chip, fiber, mask_width=None):
        """
        Build (and cache) the orderlet's CCF line mask: vacuum line centers,
        weights, and per-line top-hat holes of full width `mask_width` about
        each center (relativistic Doppler). The mask is selected from the orderlet's
        illumination source.

        Stellar masks load from reference/line_masks/stellar_masks/; the 'thar'
        mask is built from the ThAr line list with uniform weights.

        Returns
        -------
        dict
            Mask with keys 'center', 'weight', 'start', 'end', each a 1D ndarray
            of length n_line; wavelengths are vacuum [Å].
        """
        if mask_width is None:
            mask_width = self.ccf_mask_width
        key = f"{chip.upper()}_{fiber.upper()}"
        if key in self._line_mask:
            return self._line_mask[key]

        mask_name = self._resolve_illumination_source(chip, fiber)["mask_name"]
        if mask_name == "thar":
            df = pd.read_csv(f"{REPO_ROOT}/reference/thar_line_list.csv")
            # Deduplicate: lines recur across overlapping orders and would
            # otherwise be double-counted.
            centers = np.unique(df["WAVE"].to_numpy(dtype=float))
            weights = np.ones(centers.size)
        else:
            mask_path = (
                f"{REPO_ROOT}/reference/line_masks/stellar_masks/{mask_name}.txt"
            )
            centers, weights = np.loadtxt(mask_path, unpack=True)  # vacuum wavelengths

        half_width = mask_width / 2.0  # hole spans +/- half_width about each center
        mask = {
            "center": centers,
            "weight": weights,
            "start": centers * (1.0 + compute_redshift(-half_width * u.km / u.s)),
            "end": centers * (1.0 + compute_redshift(+half_width * u.km / u.s)),
        }
        self._line_mask[key] = mask
        return mask

    def _build_velocity_grid(self, chip, fiber, step_size=None, window=None):
        """
        Build (and cache) the orderlet's CCF velocity grid: evenly spaced over
        `window` about the orderlet's grid center in `step_size` increments.

        The [min, max] `window` is converted to an integer number of `step_size`
        steps (so the step is exact), then shifted by the center — the systemic
        RV (TARGRADV) for stellar fibers, 0 for sky/cal fibers.
        """
        if step_size is None:
            step_size = self.ccf_step_size
        if window is None:
            window = self.ccf_window
        key = f"{chip.upper()}_{fiber.upper()}"
        if key in self._velocity_grid:
            return self._velocity_grid[key]

        center = self._resolve_illumination_source(chip, fiber)["vel_grid_center"]
        lo_kms, hi_kms = window
        lo = int(round(lo_kms / step_size))
        hi = int(round(hi_kms / step_size))
        grid = np.arange(lo, hi + 1) * step_size + center
        self._velocity_grid[key] = grid
        return grid

    def _get_order_weights(self, chip, fiber):
        """
        Per-order CCF-combination weights for one orderlet, from
        reference/ccf_order_weights.csv (column selected by the orderlet's mask).
        Returns a 1D ndarray of length norder_chip, ordered by ORDER.
        """
        if self._order_weights is None:
            self._order_weights = pd.read_csv(
                f"{REPO_ROOT}/reference/ccf_order_weights.csv"
            )
        df = self._order_weights
        mask_name = self._resolve_illumination_source(chip, fiber)["mask_name"]
        if mask_name not in df.columns:
            raise KeyError(
                f"no CCF order-weight column for mask {mask_name!r} in "
                f"ccf_order_weights.csv; have "
                f"{[c for c in df.columns if c not in ('CHIP', 'ORDER')]}"
            )
        rows = df[df["CHIP"] == chip.upper()].sort_values("ORDER")
        return rows[mask_name].to_numpy(dtype=float)

    @staticmethod
    def _compute_ccf_1d(wave, flux, var, line_mask, velocity_grid, barycorr_z):
        """
        Cross-correlate one order's spectrum against the mask over the velocity
        grid, folding in the order's barycentric redshift z.

        Parameters
        ----------
        wave : ndarray
            1D wavelength solution for the order [Å].
        flux : ndarray
            1D extracted flux for the order.
        var : ndarray
            1D per-pixel variance for the order (TRACE_VAR); sets the CCF photon
            variance.
        line_mask : dict
            Line mask (keys 'start', 'end', 'weight') from _build_line_mask.
        velocity_grid : ndarray
            CCF velocity steps [km/s].
        barycorr_z : float
            Barycentric redshift for the order.

        Returns
        -------
        ccf : ndarray
            CCF value at each velocity step (all zeros if the order is unusable
            or no mask lines fall fully within it).
        ccf_var : ndarray
            Per-velocity-bin photon variance sum(w**2 * var), where w is the
            per-pixel mask weight (all zeros in the same unusable cases).

        Raises
        ------
        ValueError
            If the WAVE array is descending; an ascending (blue->red) solution
            is expected, so a reversed order signals an upstream orientation
            error rather than something to silently correct.
        """
        wave = np.asarray(wave, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)
        var = np.asarray(var, dtype=np.float64)
        if wave[0] > wave[-1]:
            raise ValueError(
                f"WAVE array is descending (wave[0]={wave[0]:.4f} > "
                f"wave[-1]={wave[-1]:.4f}); expected ascending blue->red "
                f"orientation. This signals an upstream orientation error."
            )

        ccf = np.zeros(velocity_grid.size)
        ccf_var = np.zeros(velocity_grid.size)
        n_pix = wave.size
        if n_pix < 3 or not strictly_increasing(wave):
            return ccf, ccf_var

        # Wavelength bin edges (length n+1) and widths at the pixel midpoints.
        edges = np.empty(n_pix + 1)
        edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
        edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
        edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
        widths = np.diff(edges)
        # Relativistic mask shift per velocity step, de-redshifting the barycorr.
        shift = (1.0 + compute_redshift(velocity_grid * u.km / u.s)) / (
            1.0 + barycorr_z
        )

        # Keep only mask lines that stay fully inside the order across the whole
        # scan, so the same lines contribute at every step (flat CCF baseline).
        smin, smax = shift.min(), shift.max()
        keep = (line_mask["start"] * smin >= wave[0]) & (
            line_mask["end"] * smax <= wave[-1]
        )
        if not np.any(keep):
            return ccf, ccf_var
        l_start, l_end = line_mask["start"][keep], line_mask["end"][keep]
        l_weight = line_mask["weight"][keep]

        # NaN-clean once: overlap weights are finite, so flux_clean * overlap_frac
        # matches np.nansum without a per-step NaN mask.
        flux_clean = np.nan_to_num(flux)
        var_clean = np.nan_to_num(var)

        # Shifted line edges and their covering-pixel indices for all velocity
        # steps at once (batched searchsorted over the (nv, nline) grid).
        line_lo_all = shift[:, None] * l_start[None, :]
        line_hi_all = shift[:, None] * l_end[None, :]
        idx_lo_all = np.clip(
            np.searchsorted(edges, line_lo_all, side="right") - 1, 0, n_pix - 1
        )
        idx_hi_all = np.clip(
            np.searchsorted(edges, line_hi_all, side="right") - 1, 0, n_pix - 1
        )

        for vi in range(velocity_grid.size):
            line_lo = line_lo_all[vi]
            line_hi = line_hi_all[vi]
            idx_lo = idx_lo_all[vi]
            idx_hi = idx_hi_all[vi]

            # Fractional overlap of each (narrow) line with the pixels it covers.
            overlap_frac = np.zeros(n_pix)
            for offset in range(int((idx_hi - idx_lo).max()) + 1):
                pix = idx_lo + offset
                still_spanning = pix <= idx_hi
                pix_sel = pix[still_spanning]
                overlap = np.minimum(
                    edges[pix_sel + 1], line_hi[still_spanning]
                ) - np.maximum(edges[pix_sel], line_lo[still_spanning])
                np.maximum(overlap, 0.0, out=overlap)
                np.add.at(
                    overlap_frac,
                    pix_sel,
                    l_weight[still_spanning] * overlap / widths[pix_sel],
                )

            ccf[vi] = np.sum(flux_clean * overlap_frac)
            ccf_var[vi] = np.sum(var_clean * overlap_frac**2)

        return ccf, ccf_var

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def compute_ccfs(
        self,
        chip,
        fiber,
        mask_width=None,
        step_size=None,
        window=None,
        clip_edge_pixels=(500, 500),
    ):
        """
        Cross-correlate every order of one chip/fiber against the line mask.

        Parameters
        ----------
        chip : str
            Chip identifier, i.e. 'GREEN' or 'RED'.
        fiber : str
            Fiber identifier, e.g. 'SCI1'.
        mask_width : float, optional
            Per-line mask top-hat width [km/s]. Defaults to the configured
            value.
        step_size : float, optional
            CCF velocity step size [km/s]. Defaults to the configured value.
        window : list of float, optional
            CCF velocity grid range [km/s] as [min, max] about the grid center.
            Defaults to the configured value.
        clip_edge_pixels : tuple of int, optional
            Number of pixels to drop from the (short_wavelength_end,
            long_wavelength_end) of each order before correlating, removing the
            blaze-faint, low-S/N order edges. Defaults to (500, 500).

        Returns
        -------
        dict or None
            {'velocity', 'ccf'}: the CCF velocity grid [km/s] and the CCF with
            shape (norder_chip, n_velocity_step). The CCF is also cached under
            f'{chip}_{fiber}'. Returns None if the fiber is not illuminated
            (source 'none').

        Raises
        ------
        ValueError
            If BARYCORR_Z is required (astronomical source) but not populated.
        NotImplementedError
            If the fiber's illumination source has no CCF path yet (etalon, lfc).
        """
        chip = chip.upper()
        fiber = fiber.upper()
        if mask_width is None:
            mask_width = self.ccf_mask_width
        if step_size is None:
            step_size = self.ccf_step_size
        if window is None:
            window = self.ccf_window

        source = self._resolve_illumination_source(chip, fiber)
        if source["object"] == "none":
            return None  # fiber not illuminated; caller skips
        apply_barycorr = source["apply_barycorr"]

        line_mask = self._build_line_mask(chip, fiber, mask_width)
        velocity_grid = self._build_velocity_grid(chip, fiber, step_size, window)

        flux = np.asarray(self.l2_obj.data[f"{chip}_{fiber}_FLUX"], dtype=np.float64)
        wave = np.asarray(self.l2_obj.data[f"{chip}_{fiber}_WAVE"], dtype=np.float64)
        var = np.asarray(self.l2_obj.data[f"{chip}_{fiber}_VAR"], dtype=np.float64)

        # Drop the blaze-faint, low-S/N order edges (they inject CCF noise).
        # clip_edge_pixels is [short_wavelength_end, long_wavelength_end]; map to
        # the pixel axis via the measured dispersion direction.
        n_short, n_long = int(clip_edge_pixels[0]), int(clip_edge_pixels[1])
        if n_short or n_long:
            ncol = flux.shape[1]
            if n_short + n_long >= ncol:
                raise ValueError(
                    f"clip_edge_pixels {list(clip_edge_pixels)} removes all "
                    f"{ncol} pixels of {chip}_{fiber}"
                )
            if np.nanmedian(np.diff(wave, axis=1)) >= 0:  # pixel 0 = short wavelength
                cols = slice(n_short, ncol - n_long)
            else:
                cols = slice(n_long, ncol - n_short)
            flux = flux[:, cols]
            wave = wave[:, cols]
            var = var[:, cols]

        norder = flux.shape[0]

        # Per-order barycentric redshift — only for astronomical sources (target,
        # sky). Calibration sources (thar) stay in the instrument frame (z = 0).
        if apply_barycorr:
            if np.size(self.l2_obj.data.get("BARYCORR_Z", np.array([]))) == 0:
                raise ValueError(
                    "per-order barycentric redshift (BARYCORR_Z) not populated; "
                    "run BarycentricCorrection first"
                )
            barycorr_z = np.asarray(
                self.l2_obj.data[f"{chip}_BARYCORR_Z"], dtype=np.float64
            )
        else:
            barycorr_z = np.zeros(norder)

        ccf = np.zeros((norder, velocity_grid.size))
        ccf_var = np.zeros((norder, velocity_grid.size))

        # A zero CCF for a single order is legitimate (no mask lines in coverage,
        # or no usable flux) and skipped; a whole-orderlet zero is a failure
        # (caught below).
        for o in range(norder):
            if not np.all(np.isfinite(wave[o])) or not np.any(np.isfinite(flux[o])):
                continue
            ccf[o], ccf_var[o] = self._compute_ccf_1d(
                wave[o], flux[o], var[o], line_mask, velocity_grid, barycorr_z[o]
            )

        # Fail loudly: an all-zero CCF means the orderlet produced no usable
        # signal (unpopulated wave/flux, or a mask that misses the data).
        if not np.any(ccf):
            raise RuntimeError(
                f"CCF for {chip}_{fiber} is identically zero across all {norder} "
                "orders; cross-correlation produced no usable signal. Check that "
                f"{chip}_{fiber}_WAVE and {chip}_{fiber}_FLUX are populated and "
                "finite, and that the line mask overlaps the data wavelengths."
            )

        self._ccf[f"{chip}_{fiber}"] = ccf
        self._ccf_var[f"{chip}_{fiber}"] = ccf_var
        self._ccf_mask_width = mask_width

        return {"velocity": velocity_grid, "ccf": ccf}

    # ------------------------------------------------------------------
    # Private helpers - module execution
    # ------------------------------------------------------------------

    def _track_info(self, chips, fibers):
        """Populate _info (the info() summary) from instance attributes."""
        chips = [c.upper() for c in chips]
        self._info = {}
        for fiber in fibers:
            fiber = fiber.upper()
            src = self._illumination_source.get(f"{chips[0]}_{fiber}")
            source = src["object"] if src else None
            if fiber not in self._fibers_done:
                self._info[fiber] = {"source": source, "nccf": {}}
                continue
            # Count of orders with a non-zero CCF, per chip.
            nccf = {}
            for chip in chips:
                ccf = self._ccf.get(f"{chip}_{fiber}")
                nccf[chip] = (
                    int(np.sum(np.any(ccf != 0, axis=1))) if ccf is not None else 0
                )
            grid = self._velocity_grid[f"{chips[-1]}_{fiber}"]
            self._info[fiber] = {
                "source": source,
                "grid_span": (float(grid[0]), float(grid[-1])),
                "nccf": nccf,
            }

    def _set_headers(self, l4_obj):
        """
        Write all CCF/RV extension headers, the single place this module writes
        headers, called just before the receipt entry. Reads the per-orderlet
        caches populated by perform()/compute_ccfs (velocity grid, illumination
        source, step, mask width). Each keyword is an EPRV per-extension card, so
        it is routed with an explicit ext= to this orderlet's CCF{n}/RV{n}. CCF
        axes are (velocity, order); RV axes are (columns, order).
        """
        for fiber in self._fibers_done:
            key = f"{self._chips_done[-1]}_{fiber}"
            grid = self._velocity_grid[key]
            mask_name = self._illumination_source[key]["mask_name"]

            ccf_ext = f"{fiber}_CCF"
            l4_obj.set_keyword("CTYPE1", "Velocity", ext=ccf_ext)
            l4_obj.set_keyword("CTYPE2", "Order-N", ext=ccf_ext)
            l4_obj.set_keyword("VELSTART", float(grid[0]), ext=ccf_ext)
            l4_obj.set_keyword("VELSTEP", float(self._ccf_step_size), ext=ccf_ext)
            l4_obj.set_keyword("VELNSTEP", int(grid.size), ext=ccf_ext)
            l4_obj.set_keyword("CCFMASK", mask_name, ext=ccf_ext)
            l4_obj.set_keyword("VELMASK", float(self._ccf_mask_width), ext=ccf_ext)

            rv_ext = f"{fiber}_RV"
            l4_obj.set_keyword("CTYPE1", "Columns", ext=rv_ext)
            l4_obj.set_keyword("CTYPE2", "Order-N", ext=rv_ext)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def perform(
        self,
        chips=None,
        fibers=None,
        *,
        ccf_mask_width=None,
        ccf_step_size=None,
        ccf_window=None,
        clip_edge_pixels=(500, 500),
    ):
        """
        Compute per-order CCFs and package them in a KPF4.

        For each illuminated orderlet (fiber), the per-order CCFs of both chips
        are written to the orderlet's CCF cube ({fiber}_CCF, green+red
        concatenated) and their per-bin photon variances to {fiber}_CCF_VAR (same
        shape). The orderlet's RV table ({fiber}_RV) is seeded with its per-order
        metadata (ORDER_INDEX/ORDER_ID/ECHELLE_ORDER/BJD_TDB/BERV/WAVE_START/
        WAVE_END/WEIGHT); the RV/RV_ERR columns are left NaN for RadialVelocity to
        fill.

        Parameters
        ----------
        chips : list of str, optional
            Chip identifiers, i.e. 'GREEN' or 'RED'. Defaults to the configured
            chips.
        fibers : list of str, optional
            Fiber identifiers, e.g. ['SCI1', 'SCI2']. Defaults to all configured
            fibers (SCI, CAL, and SKY).
        ccf_mask_width : float, optional
            Per-line mask top-hat width [km/s]. Overrides the configured value.
        ccf_step_size : float, optional
            CCF velocity step size [km/s]. Overrides the configured value.
        ccf_window : list of float, optional
            CCF velocity grid range [km/s] as [min, max] about the grid center.
            Overrides the configured value.
        clip_edge_pixels : tuple of int, optional
            Pixels to drop from the (short_wavelength_end, long_wavelength_end)
            of each order before correlating. Defaults to (500, 500).

        Returns
        -------
        l4_obj : KPF4
            L4 with a CCF cube, a per-bin CCF variance cube, and a metadata-seeded
            per-order RV table per illuminated orderlet. Each CCF extension carries
            CTYPE1/CTYPE2/VELSTART/VELSTEP/VELNSTEP/CCFMASK/VELMASK; each RV
            extension carries CTYPE1/CTYPE2. Unilluminated ('none') or
            not-yet-implemented (etalon, lfc) fibers are skipped (empty
            extensions).
        """
        if chips is None:
            chips = self.chips
        if fibers is None:
            fibers = self.fibers
        if ccf_mask_width is None:
            ccf_mask_width = self.ccf_mask_width
        if ccf_step_size is None:
            ccf_step_size = self.ccf_step_size
        if ccf_window is None:
            ccf_window = self.ccf_window

        chips = [c.upper() for c in chips]
        fibers = [f.upper() for f in fibers]
        self._chips_done = chips
        self._ccf_step_size = ccf_step_size

        norder_green = self.norder["GREEN"]
        norder = norder_green + self.norder["RED"]

        l4_obj = self.l2_obj.to_kpf4()

        # Per-order barycentric metadata, shared by every orderlet's RV table.
        bjd_tdb = np.asarray(self.l2_obj.data["BJD_TDB"], dtype=np.float64)
        berv = np.asarray(self.l2_obj.data["BARYCORR_KMS"], dtype=np.float64)

        self._fibers_done = []
        for fiber in fibers:
            weight = np.full(norder, np.nan)

            # Unilluminated or not-yet-implemented sources have no mask -> skip.
            source = self._resolve_illumination_source(chips[0], fiber)
            if source["mask_name"] is None:
                print(
                    f"  {fiber}: illumination source {source['object']!r}; "
                    "skipping (no CCF)"
                )
                continue

            for chip in chips:
                ccf = self.compute_ccfs(
                    chip,
                    fiber,
                    ccf_mask_width,
                    ccf_step_size,
                    ccf_window,
                    clip_edge_pixels=clip_edge_pixels,
                )["ccf"]
                l4_obj.set_data(f"{chip}_{fiber}_CCF", ccf)
                l4_obj.set_data(
                    f"{chip}_{fiber}_CCF_VAR", self._ccf_var[f"{chip}_{fiber}"]
                )
                rows = (
                    slice(0, norder_green)
                    if chip == "GREEN"
                    else slice(norder_green, norder)
                )
                weight[rows] = self._get_order_weights(chip, fiber)

            # Per-orderlet RV table, one row per order (green then red). ORDER_ID
            # is 1-based per chip; ECHELLE_ORDER is the physical grating order
            # (detector.toml, blue->red); WEIGHT is the per-order CCF weight
            # (ccf_order_weights.csv). RV/RV_ERR are NaN placeholders RadialVelocity
            # fills from the CCFs.
            order_id = np.array(
                [
                    f"{chip}_{fiber}_{order}"
                    for chip in ("GREEN", "RED")
                    for order in range(1, self.norder[chip] + 1)
                ]
            )
            echelle_order = np.concatenate(
                [
                    np.linspace(
                        self.echelle_orders[chip][0],
                        self.echelle_orders[chip][1],
                        self.norder[chip],
                    )
                    .round()
                    .astype(np.int64)
                    for chip in ("GREEN", "RED")
                ]
            )
            wave = np.asarray(self.l2_obj.data[f"{fiber}_WAVE"], dtype=np.float64)
            l4_obj.set_data(
                f"{fiber}_RV",
                pd.DataFrame(
                    {
                        "ORDER_INDEX": np.arange(norder, dtype=np.int64),
                        "ORDER_ID": order_id,
                        "ECHELLE_ORDER": echelle_order,
                        "BJD_TDB": bjd_tdb,
                        "BERV": berv,
                        "WAVE_START": wave[:, 0],
                        "WAVE_END": wave[:, -1],
                        "RV": np.full(norder, np.nan),
                        "RV_ERR": np.full(norder, np.nan),
                        "WEIGHT": weight,
                    }
                ),
            )
            self._fibers_done.append(fiber)

        self._set_headers(l4_obj)
        self._track_info(chips, fibers)
        l4_obj.receipt_add_entry("cross_correlation", "", "PASS")
        return l4_obj

    def info(self):
        """Print a summary of the module configuration and CCF results."""
        print("CrossCorrelation")
        obs_id = self.l2_obj.headers.get("RECEIPT", {}).get("ORIGID", "unknown")
        print(f"  obs_id:         {obs_id}")
        print(f"  ccf_mask_width: {self.ccf_mask_width} km/s")
        print(f"  ccf_step_size:  {self.ccf_step_size} km/s")
        print(f"  ccf_window:     {self.ccf_window} km/s")

        if self._info is None:
            print("  perform() has not been called")
            return

        # CCF velocity grid: per-fiber center, shared step/span.
        print(
            f"\n  CCF velocity grid: {self.ccf_window[0]:+.1f} to "
            f"{self.ccf_window[1]:+.1f} km/s "
            f"about each fiber's center, step {self.ccf_step_size} km/s"
        )

        # Per-CCD, per-orderlet summary. SOURCE is the illumination source; NCCF
        # is the number of orders with a non-zero CCF on that chip.
        fiber_order = [
            f for f in ("SCI1", "SCI2", "SCI3", "SKY", "CAL") if f in self._info
        ]
        fiber_order += [f for f in self._info if f not in fiber_order]

        print(f"\n  {'CHIP':<8s}{'FIBER':<8s}{'SOURCE':<10s}{'NCCF':>8s}")
        print("  " + "-" * 34)
        for chip in ("GREEN", "RED"):
            for fiber in fiber_order:
                res = self._info[fiber]
                nccf = res.get("nccf", {}).get(chip)
                if not nccf:
                    continue
                print(f"  {chip:<8s}{fiber:<8s}{res.get('source', ''):<10s}{nccf:>8d}")
