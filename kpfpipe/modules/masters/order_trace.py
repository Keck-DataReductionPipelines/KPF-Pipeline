"""KPF spectral order tracing from one master-flat image."""

import logging
import os
import tempfile

import numpy as np
import pandas as pd
from astropy.stats import mad_std
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from kpfpipe import DEFAULTS, REPO_ROOT
from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.kpf import get_timestamp, kpf_timestamp_to_datetime

logger = logging.getLogger(__name__)

_REFERENCE_COEFFICIENT_COLUMNS = [f"Coeff{i}" for i in range(4)]
_TRACE_GEOMETRY_COLUMNS = [
    "BottomEdge",
    "TopEdge",
    "X1",
    "X2",
]
_TRACE_ID_COLUMNS = [
    "Fiber",
    "Order",
]
_REFERENCE_TRACE_COLUMNS = (
    _REFERENCE_COEFFICIENT_COLUMNS + _TRACE_GEOMETRY_COLUMNS + _TRACE_ID_COLUMNS
)

_ERA_DEFINITIONS_PATH = f"{REPO_ROOT}/reference/kpf_instrument_eras.csv"
_ORDER_TRACE_PATHS = {
    "GREEN": f"{REPO_ROOT}/reference/order_trace_green.csv",
    "RED": f"{REPO_ROOT}/reference/order_trace_red.csv",
}

_DEFAULTS = {
    **DEFAULTS,
    "poly_degree": 3,
}


class OrderTrace:
    """
    Measure spectral traces from one vNext KPF master flat and write trace CSVs.

    Existing trace references supply only approximate locations and ``(Fiber,
    Order)`` identities; all output geometry is measured from the input flat.
    Standalone rather than a ``BaseMasterModule`` subclass because it consumes
    one completed L1 master instead of stacking L0 exposures.

    Parameters
    ----------
    master_flat_filename : str or pathlib.Path
        vNext ``KPFMasterL1`` flat FITS filename.
    config : None | dict | ConfigHandler
        Module configuration. ``ConfigHandler`` values are read from TRACES and
        ORDER_TRACE, whose only module-specific parameter is ``poly_degree``.
    """

    def __init__(self, master_flat_filename, config=None):
        self.master_flat_filename = str(master_flat_filename)

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["TRACES", "ORDER_TRACE"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in _DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        self._master_flat = None
        self._instrument_era = None
        self._anchor_shifts = {}
        self._fit_rms = {}
        self._results = None
        self._output_paths = {}
        self._info = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _coefficient_columns(self):
        """Return output coefficient columns for the configured fit degree."""
        degree = int(self.poly_degree)
        if degree < 0:
            raise ValueError(f"poly_degree must be non-negative, got {degree!r}")
        coefficient_count = max(len(_REFERENCE_COEFFICIENT_COLUMNS), degree + 1)
        return [f"Coeff{i}" for i in range(coefficient_count)]

    def _validate_manual_anchors(self, chips, cal_order3_y):
        """Return uppercase, finite manual anchors for the requested chips."""
        if cal_order3_y is None:
            return None
        if not isinstance(cal_order3_y, dict):
            raise TypeError("cal_order3_y must be None or a dict keyed by chip")

        anchors = {str(chip).upper(): value for chip, value in cal_order3_y.items()}
        unknown = set(anchors).difference({"GREEN", "RED"})
        if unknown:
            raise ValueError(f"cal_order3_y has unsupported chips: {sorted(unknown)}")
        missing = set(chips).difference(anchors)
        if missing:
            raise ValueError(
                f"cal_order3_y is missing requested chips: {sorted(missing)}"
            )
        for chip in chips:
            try:
                anchors[chip] = float(anchors[chip])
            except (TypeError, ValueError) as e:
                raise TypeError(f"cal_order3_y[{chip!r}] must be numeric") from e
            if not np.isfinite(anchors[chip]):
                raise ValueError(f"cal_order3_y[{chip!r}] must be finite")
        return anchors

    def _load_seed_table(self, chip, manual_anchor=None):
        """Load and optionally translate one chip's approximate trace table."""
        path = _ORDER_TRACE_PATHS[chip]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Order-trace reference not found: {path}")
        table = pd.read_csv(path, index_col=0)
        if list(table.columns) != _REFERENCE_TRACE_COLUMNS:
            raise ValueError(
                f"{path} has incompatible columns; expected {_REFERENCE_TRACE_COLUMNS}"
            )
        if table.duplicated(["Fiber", "Order"]).any():
            raise ValueError(f"{path} contains duplicate Fiber/Order rows")

        table = table.copy()
        numeric = _REFERENCE_COEFFICIENT_COLUMNS + _TRACE_GEOMETRY_COLUMNS
        table[numeric] = table[numeric].apply(pd.to_numeric, errors="raise")
        table["Order"] = pd.to_numeric(table["Order"], errors="raise").astype(int)

        shift = 0.0
        if manual_anchor is not None:
            anchor = table.loc[(table["Fiber"] == "CAL") & (table["Order"] == 3)]
            if len(anchor) != 1:
                raise ValueError(
                    f"{path} must contain exactly one CAL/order-3 anchor row"
                )
            shift = float(manual_anchor) - float(anchor.iloc[0]["Coeff0"])
            table["Coeff0"] += shift
        self._anchor_shifts[chip] = shift
        return table

    def _chip_image(self, chip):
        """Return one finite, two-dimensional master-flat CCD image."""
        extension = f"{chip}_IMG"
        if extension not in self._master_flat.data:
            raise KeyError(f"{extension} extension is not available")
        image = np.asarray(self._master_flat.data[extension])
        if image.ndim != 2:
            raise ValueError(f"{extension} must be a 2D image")
        if not np.issubdtype(image.dtype, np.number):
            raise TypeError(f"{extension} must contain numeric data")
        return image

    def _sample_columns(self, ncol, sample_count=65):
        """Return evenly spaced, unique detector columns including both edges."""
        return np.unique(np.linspace(0, ncol - 1, sample_count, dtype=int))

    def _column_profile(self, image, column, col_half_window=3):
        """Return a robust cross-dispersion profile around one detector column."""
        c0 = max(0, int(column) - col_half_window)
        c1 = min(image.shape[1], int(column) + col_half_window + 1)
        profile = np.nanmedian(image[:, c0:c1], axis=1)
        finite = np.isfinite(profile)
        if not finite.any():
            return np.zeros(image.shape[0], dtype=float)
        fill = np.nanmedian(profile[finite])
        return np.where(finite, profile, fill).astype(float, copy=False)

    def _candidate_rows(
        self,
        image,
        column,
        background_smoothing_sigma=20.0,
        profile_smoothing_sigma=1.0,
        candidate_distance_pixels=5,
        candidate_prominence_sigma=4.0,
    ):
        """Locate illuminated trace candidates in one median column strip."""
        profile = self._column_profile(image, column)
        background = gaussian_filter1d(
            profile, sigma=background_smoothing_sigma, mode="nearest"
        )
        residual = profile - background
        noise = mad_std(residual, ignore_nan=True)
        if not np.isfinite(noise) or noise <= 0:
            noise = np.nanstd(residual)
        if not np.isfinite(noise) or noise <= 0:
            noise = np.finfo(float).eps

        smoothed = gaussian_filter1d(
            np.clip(residual, 0.0, None),
            sigma=profile_smoothing_sigma,
            mode="nearest",
        )
        peaks, _ = find_peaks(
            smoothed,
            distance=candidate_distance_pixels,
            prominence=candidate_prominence_sigma * noise,
        )
        return peaks.astype(float)

    def _local_peak_center(
        self,
        image,
        column,
        guess,
        candidates,
        row_half_window=7,
        profile_smoothing_sigma=1.0,
        winsor_percentile=90.0,
    ):
        """Measure a subpixel center for a peaked CAL-fiber profile."""
        if not np.isfinite(guess):
            return np.nan

        nearby = candidates[np.abs(candidates - guess) <= row_half_window]
        if nearby.size:
            guess = nearby[np.argmin(np.abs(nearby - guess))]

        r0 = max(0, int(np.floor(guess)) - row_half_window)
        r1 = min(image.shape[0], int(np.ceil(guess)) + row_half_window + 1)
        if r1 - r0 < 3:
            return np.nan

        profile = self._column_profile(image, column)[r0:r1]
        rows = np.arange(r0, r1, dtype=float)
        background = np.nanpercentile(profile, 20.0)
        signal = np.clip(profile - background, 0.0, None)
        smoothed = gaussian_filter1d(
            np.nan_to_num(signal),
            sigma=profile_smoothing_sigma,
            mode="nearest",
        )
        if not np.any(smoothed > 0):
            return np.nan

        peak = int(np.argmax(smoothed))
        lo = max(0, peak - 3)
        hi = min(signal.size, peak + 4)
        weights = signal[lo:hi]
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            return np.nan
        limit = np.nanpercentile(weights, winsor_percentile)
        weights = np.minimum(weights, limit)
        if weights.sum() <= 0:
            return np.nan
        return float(np.average(rows[lo:hi], weights=weights))

    @staticmethod
    def _threshold_crossing(rows, values, index, direction, threshold):
        """Interpolate one threshold crossing away from an illuminated core."""
        current = int(index)
        adjacent = current + int(direction)
        while 0 <= adjacent < values.size and values[adjacent] >= threshold:
            current = adjacent
            adjacent = current + int(direction)
        if adjacent < 0 or adjacent >= values.size:
            return np.nan

        value_inside = values[current]
        value_outside = values[adjacent]
        denominator = value_inside - value_outside
        if not np.isfinite(denominator) or denominator <= 0.0:
            return np.nan
        fraction = (value_inside - threshold) / denominator
        return float(rows[current] + fraction * (rows[adjacent] - rows[current]))

    def _local_edge_center(
        self,
        image,
        column,
        guess,
        row_half_window=7,
        background_smoothing_sigma=20.0,
        profile_smoothing_sigma=1.0,
        edge_levels=(0.25, 0.40, 0.55, 0.70),
        edge_min_width_pixels=3.0,
        edge_max_center_spread=0.75,
    ):
        """Measure the geometric center of a flat-topped orderlet from its edges."""
        if not np.isfinite(guess):
            return np.nan

        r0 = max(0, int(np.floor(guess)) - row_half_window)
        r1 = min(image.shape[0], int(np.ceil(guess)) + row_half_window + 1)
        if r1 - r0 < 5:
            return np.nan

        full_profile = self._column_profile(image, column)
        background = gaussian_filter1d(
            full_profile,
            sigma=background_smoothing_sigma,
            mode="nearest",
        )
        values = gaussian_filter1d(
            full_profile[r0:r1] - background[r0:r1],
            sigma=profile_smoothing_sigma,
            mode="nearest",
        )
        rows = np.arange(r0, r1, dtype=float)
        if not np.isfinite(values).all():
            return np.nan

        baseline = np.nanpercentile(values, 20.0)
        amplitude = np.nanpercentile(values, 90.0) - baseline
        if not np.isfinite(amplitude) or amplitude <= 0.0:
            return np.nan

        centers = []
        guess_index = int(np.argmin(np.abs(rows - guess)))
        for level in edge_levels:
            threshold = baseline + level * amplitude
            illuminated = np.flatnonzero(values >= threshold)
            if illuminated.size == 0:
                continue
            core = illuminated[np.argmin(np.abs(illuminated - guess_index))]
            lower = self._threshold_crossing(rows, values, core, -1, threshold)
            upper = self._threshold_crossing(rows, values, core, 1, threshold)
            width = upper - lower
            if (
                np.isfinite(lower)
                and np.isfinite(upper)
                and width >= edge_min_width_pixels
            ):
                centers.append((lower + upper) / 2.0)

        minimum = max(2, int(np.ceil(len(edge_levels) / 2)))
        if len(centers) < minimum:
            return np.nan
        centers = np.asarray(centers, dtype=float)
        center = float(np.nanmedian(centers))
        spread = mad_std(centers, ignore_nan=True)
        if spread > edge_max_center_spread:
            return np.nan
        return center

    def _local_center(self, image, column, guess, candidates, fiber):
        """Dispatch to the profile estimator appropriate for one fiber."""
        if fiber == "CAL":
            return self._local_peak_center(image, column, guess, candidates)
        return self._local_edge_center(image, column, guess)

    def _trace_centers(self, image, seed_coeffs, sample_x, candidate_rows, fiber):
        """Follow one orderlet from the detector center toward both edges."""
        seed_y = np.polynomial.polynomial.polyval(sample_x, seed_coeffs)
        centers = np.full(sample_x.size, np.nan, dtype=float)
        anchor = int(np.argmin(np.abs(sample_x - image.shape[1] // 2)))
        centers[anchor] = self._local_center(
            image,
            sample_x[anchor],
            seed_y[anchor],
            candidate_rows[anchor],
            fiber,
        )

        directions = (
            range(anchor + 1, sample_x.size),
            range(anchor - 1, -1, -1),
        )
        for indices in directions:
            previous = anchor
            for i in indices:
                if np.isfinite(centers[previous]):
                    guess = centers[previous] + seed_y[i] - seed_y[previous]
                else:
                    guess = seed_y[i]
                measured = self._local_center(
                    image, sample_x[i], guess, candidate_rows[i], fiber
                )
                centers[i] = measured
                if np.isfinite(measured):
                    previous = i
        return centers

    def _robust_polynomial_fit(
        self,
        x,
        y,
        min_valid_fraction=0.5,
        fit_max_iterations=8,
        fit_sigma=4.0,
    ):
        """Fit a polynomial with iterative median/MAD residual rejection."""
        degree = int(self.poly_degree)
        coefficient_columns = self._coefficient_columns()
        keep = np.isfinite(x) & np.isfinite(y)
        minimum = max(degree + 1, int(np.ceil(x.size * min_valid_fraction)))
        if keep.sum() < minimum:
            raise ValueError(
                f"only {keep.sum()} of {x.size} trace centers are valid; "
                f"at least {minimum} are required"
            )

        for _ in range(fit_max_iterations):
            coeffs = np.polynomial.polynomial.polyfit(x[keep], y[keep], degree)
            residual = y - np.polynomial.polynomial.polyval(x, coeffs)
            center = np.nanmedian(residual[keep])
            scale = mad_std(residual[keep], ignore_nan=True)
            limit = max(0.25, fit_sigma * scale)
            updated = np.isfinite(y) & (np.abs(residual - center) <= limit)
            if updated.sum() < minimum:
                break
            if np.array_equal(updated, keep):
                keep = updated
                break
            keep = updated

        coeffs = np.polynomial.polynomial.polyfit(x[keep], y[keep], degree)
        padded = np.zeros(len(coefficient_columns), dtype=float)
        padded[: coeffs.size] = coeffs
        residual = y[keep] - np.polynomial.polynomial.polyval(x[keep], padded)
        rms = float(np.sqrt(np.mean(residual**2)))
        return padded, keep, rms

    def _width_at_column(
        self,
        image,
        column,
        center,
        width_half_window=14,
        winsor_percentile=90.0,
        width_sigma=2.8,
    ):
        """Estimate lower and upper Gaussian widths at one trace sample."""
        r0 = max(0, int(np.floor(center)) - width_half_window)
        r1 = min(image.shape[0], int(np.ceil(center)) + width_half_window + 1)
        if r1 - r0 < 5:
            return np.nan, np.nan

        profile = self._column_profile(image, column)[r0:r1]
        rows = np.arange(r0, r1, dtype=float)
        signal = np.clip(profile - np.nanpercentile(profile, 20.0), 0.0, None)
        if not np.any(signal > 0):
            return np.nan, np.nan
        signal = np.minimum(signal, np.nanpercentile(signal, winsor_percentile))
        offsets = rows - center

        widths = []
        for side in (offsets <= 0, offsets >= 0):
            weights = signal[side]
            distance = np.abs(offsets[side])
            if weights.size < 2 or weights.sum() <= 0:
                widths.append(np.nan)
                continue
            # A Gaussian's second moment is sigma. Estimating it on each half
            # preserves aperture asymmetry without the unstable unconstrained
            # half-Gaussian fits used by the legacy implementation.
            sigma = np.sqrt(np.sum(weights * distance**2) / np.sum(weights))
            widths.append(width_sigma * sigma)
        return tuple(widths)

    def _estimate_widths(
        self, image, coeffs, x, keep, width_half_window=14, width_default=11.0
    ):
        """Return robust lower/upper aperture widths from accepted samples."""
        lower = []
        upper = []
        kept_x = x[keep]
        stride = max(1, kept_x.size // 24)
        for column in kept_x[::stride]:
            center = np.polynomial.polynomial.polyval(column, coeffs)
            bottom, top = self._width_at_column(image, int(column), center)
            if np.isfinite(bottom) and bottom > 0:
                lower.append(bottom)
            if np.isfinite(top) and top > 0:
                upper.append(top)

        if len(lower) < 3 or len(upper) < 3:
            raise ValueError(
                "fewer than three valid samples for trace-width estimation"
            )
        maximum = min(width_half_window, width_default)
        bottom = float(np.clip(np.nanmedian(lower), 1.0, maximum))
        top = float(np.clip(np.nanmedian(upper), 1.0, maximum))
        return bottom, top

    def _constrain_neighbor_widths(self, rows, ncol, orderlet_gap_pixels=2.0):
        """Keep adjacent apertures separated by the required orderlet gap."""
        coefficient_columns = self._coefficient_columns()
        x_mid = (ncol - 1) / 2.0
        for lower, upper in zip(rows[:-1], rows[1:], strict=False):
            lower_y = np.polynomial.polynomial.polyval(
                x_mid, [lower[column] for column in coefficient_columns]
            )
            upper_y = np.polynomial.polynomial.polyval(
                x_mid, [upper[column] for column in coefficient_columns]
            )
            separation = upper_y - lower_y
            available = separation - orderlet_gap_pixels
            requested = lower["TopEdge"] + upper["BottomEdge"]
            if available <= 0:
                raise ValueError(
                    "neighboring fitted traces cross or have no aperture gap"
                )
            if requested > available:
                scale = available / requested
                lower["TopEdge"] *= scale
                upper["BottomEdge"] *= scale

    def _validate_trace_table(self, chip, table, nrow, ncol, row_half_window=7):
        """Validate output schema, geometry, labels, and detector coverage."""
        coefficient_columns = self._coefficient_columns()
        trace_columns = (
            coefficient_columns + _TRACE_GEOMETRY_COLUMNS + _TRACE_ID_COLUMNS
        )
        if list(table.columns) != trace_columns:
            raise ValueError(f"{chip} output has incompatible columns")
        if table.empty:
            raise ValueError(f"{chip} produced no trace rows")
        if table.duplicated(["Fiber", "Order"]).any():
            raise ValueError(f"{chip} output has duplicate Fiber/Order rows")

        numeric = table[coefficient_columns + _TRACE_GEOMETRY_COLUMNS].to_numpy(
            dtype=float
        )
        if not np.isfinite(numeric).all():
            raise ValueError(f"{chip} output contains non-finite geometry")
        if not ((table["BottomEdge"] > 0) & (table["TopEdge"] > 0)).all():
            raise ValueError(f"{chip} output contains non-positive widths")
        if not ((table["X1"] >= 0) & (table["X2"] < ncol)).all():
            raise ValueError(f"{chip} output contains out-of-range column bounds")
        if not (table["X1"] <= table["X2"]).all():
            raise ValueError(f"{chip} output contains reversed column bounds")

        test_x = np.linspace(0, ncol - 1, 17)
        coeffs = table[coefficient_columns].to_numpy(dtype=float)
        centers = np.array(
            [np.polynomial.polynomial.polyval(test_x, coeff) for coeff in coeffs]
        )
        if np.any(np.diff(centers, axis=0) <= 0):
            raise ValueError(f"{chip} fitted traces cross or are out of detector order")
        covered = (centers >= -row_half_window) & (
            centers <= nrow - 1 + row_half_window
        )
        if not covered.any(axis=1).all():
            raise ValueError(f"{chip} output contains a wholly off-detector trace")

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def _trace_chip(self, chip, seed_table, row_half_window=7):
        """Measure all seeded orderlets on one CCD and return the output table."""
        image = self._chip_image(chip)
        nrow, ncol = image.shape
        coefficient_columns = self._coefficient_columns()
        sample_x = self._sample_columns(ncol)
        candidate_rows = [self._candidate_rows(image, column) for column in sample_x]

        rows = []
        rms_values = []
        for seed in seed_table.itertuples(index=False):
            seed_coeffs = np.array(
                [getattr(seed, column) for column in _REFERENCE_COEFFICIENT_COLUMNS],
                dtype=float,
            )
            predicted = np.polynomial.polynomial.polyval(sample_x, seed_coeffs)
            if not np.any(
                (predicted >= -row_half_window)
                & (predicted <= nrow - 1 + row_half_window)
            ):
                logger.warning(
                    "%s %s order %d is wholly off detector; omitting trace",
                    chip,
                    seed.Fiber,
                    seed.Order,
                )
                continue

            measured = self._trace_centers(
                image, seed_coeffs, sample_x, candidate_rows, seed.Fiber
            )
            try:
                coeffs, keep, rms = self._robust_polynomial_fit(
                    sample_x.astype(float), measured
                )
            except ValueError as e:
                raise ValueError(f"{chip} {seed.Fiber} order {seed.Order}: {e}") from e

            try:
                bottom, top = self._estimate_widths(
                    image, coeffs, sample_x.astype(float), keep
                )
            except ValueError as e:
                raise ValueError(f"{chip} {seed.Fiber} order {seed.Order}: {e}") from e

            kept_x = sample_x[keep]
            rows.append(
                {
                    **dict(zip(coefficient_columns, coeffs, strict=True)),
                    "BottomEdge": bottom,
                    "TopEdge": top,
                    "X1": int(kept_x.min()),
                    "X2": int(kept_x.max()),
                    "Fiber": seed.Fiber,
                    "Order": int(seed.Order),
                }
            )
            rms_values.append(rms)

        self._constrain_neighbor_widths(rows, ncol)
        trace_columns = (
            coefficient_columns + _TRACE_GEOMETRY_COLUMNS + _TRACE_ID_COLUMNS
        )
        table = pd.DataFrame(rows, columns=trace_columns)
        self._validate_trace_table(chip, table, nrow, ncol)
        self._fit_rms[chip] = np.asarray(rms_values, dtype=float)
        return table

    def _write_results(self, results, output_dir, overwrite):
        """Stage and atomically install all requested CSV outputs."""
        os.makedirs(output_dir, exist_ok=True)
        paths = {
            chip: os.path.join(output_dir, f"order_trace_{chip.lower()}.csv")
            for chip in results
        }
        existing = [path for path in paths.values() if os.path.exists(path)]
        if existing and not overwrite:
            raise FileExistsError(f"Order-trace output already exists: {existing}")

        temporary = {}
        try:
            for chip, table in results.items():
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".csv.tmp",
                    prefix=f"order_trace_{chip.lower()}_",
                    dir=output_dir,
                    delete=False,
                ) as stream:
                    temporary[chip] = stream.name
                    table.to_csv(stream, lineterminator="\n")

            for chip, path in paths.items():
                os.replace(temporary[chip], path)
                temporary.pop(chip)
        finally:
            for path in temporary.values():
                if os.path.exists(path):
                    os.remove(path)

        self._output_paths = paths

    # ------------------------------------------------------------------
    # Private helpers - module execution
    # ------------------------------------------------------------------

    def _track_info(self, chips):
        """Build and cache the interactive execution summary."""
        era = (
            "manual/unregistered"
            if self._instrument_era is None
            else self._instrument_era
        )
        lines = [
            "OrderTrace",
            f"  master flat: {self.master_flat_filename}",
            f"  INSTERA:  {era}",
            "",
            f"  {'chip':<8s} {'traces':>7s} {'seed shift':>12s} "
            f"{'median RMS [pix]':>18s}  output",
            "  " + "-" * 90,
        ]
        for chip in chips:
            median_rms = np.nanmedian(self._fit_rms[chip])
            lines.append(
                f"  {chip:<8s} {len(self._results[chip]):>7d} "
                f"{self._anchor_shifts[chip]:>12.3f} "
                f"{median_rms:>18.4f}  {self._output_paths[chip]}"
            )
        self._info = "\n\n" + "\n".join(lines) + "\n\n"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def make_master_order_trace(
        self, chips=None, *, output_dir, cal_order3_y=None, overwrite=False
    ):
        """
        Build order-trace calibrations from the requested master-flat CCDs.

        Parameters
        ----------
        chips : list of str, optional
            CCDs to trace. Defaults to configured GREEN and RED.
        output_dir : str or pathlib.Path
            Directory receiving ``order_trace_<chip>.csv``. Required.
        cal_order3_y : dict, optional
            Manual CAL/order-3 row at assembled detector column zero, keyed by
            chip. Required for an observation outside every defined instrument
            era; when supplied for a known era, explicitly overrides its seed
            positions by vertically translating the reference geometry.
        overwrite : bool
            Replace existing output CSVs when True. Defaults to False.

        Returns
        -------
        dict
            Mapping of uppercase chip name to the DataFrame written for it.

        Raises
        ------
        FileNotFoundError
            If the era table or trace reference is absent.
        ValueError
            If the era/anchor, detected geometry, or output table is invalid.
        FileExistsError
            If a requested output exists and ``overwrite`` is False.
        """
        if chips is None:
            chips = self.chips
        anchors = self._validate_manual_anchors(chips, cal_order3_y)

        self._master_flat = KPFMasterL1.from_fits(self.master_flat_filename)
        timestamp = get_timestamp(self.master_flat_filename)
        observation_time = kpf_timestamp_to_datetime(timestamp)
        eras = pd.read_csv(_ERA_DEFINITIONS_PATH, skipinitialspace=True)
        in_era = (pd.to_datetime(eras["UT_start_date"]) <= observation_time) & (
            observation_time <= pd.to_datetime(eras["UT_end_date"])
        )
        era = eras.loc[in_era, "INSTERA"]
        self._instrument_era = None if era.empty else float(era.iloc[0])
        if self._instrument_era is None and anchors is None:
            raise ValueError(
                f"master timestamp {timestamp} is outside the defined instrument eras; "
                "provide cal_order3_y for every requested chip"
            )

        results = {}
        for chip in chips:
            manual_anchor = None if anchors is None else anchors[chip]
            seed_table = self._load_seed_table(chip, manual_anchor)
            results[chip] = self._trace_chip(chip, seed_table)

        self._results = results
        self._write_results(results, output_dir, bool(overwrite))
        self._track_info(chips)
        logger.info("%s", self._info)
        return results

    def info(self):
        """Print a summary of the module configuration and tracing results."""
        if self._info is None:
            print(
                f"{type(self).__name__}: make_master_order_trace() has not been called"
            )
        else:
            print(self._info)
