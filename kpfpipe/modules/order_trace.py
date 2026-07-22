"""KPF spectral order tracing from a single raw wideflat exposure."""

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from kpfpipe import DEFAULTS, REPO_ROOT
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.modules.calibration_association import CalibrationAssociation
from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.modules.image_processing import ImageProcessing
from kpfpipe.utils.config import ConfigHandler

logger = logging.getLogger(__name__)

_TRACE_COLUMNS = [
    "Coeff0",
    "Coeff1",
    "Coeff2",
    "Coeff3",
    "BottomEdge",
    "TopEdge",
    "X1",
    "X2",
    "Fiber",
    "Order",
]

_ERA_DEFINITIONS_PATH = REPO_ROOT / "reference/kpf_instrument_eras.csv"
_ORDER_TRACE_PATHS = {
    "GREEN": REPO_ROOT / "reference/order_trace_green.csv",
    "RED": REPO_ROOT / "reference/order_trace_red.csv",
}

_DEFAULTS = {
    **DEFAULTS,
    "poly_degree": 3,
}


class OrderTrace:
    """
    Measure spectral traces from one raw KPF wideflat and write trace CSVs.

    The raw exposure is assembled and bias-subtracted with the standard vNext
    modules. Existing trace references provide approximate locations and
    ``(Fiber, Order)`` identities only; all output geometry is measured from the
    input wideflat.

    Parameters
    ----------
    wideflat_filename : str or pathlib.Path
        Raw L0 wideflat FITS filename.
    config : None | dict | ConfigHandler
        Module configuration. ``ConfigHandler`` values are read from DATA_DIRS,
        TRACES, and MODULE_ORDER_TRACE, whose only module-specific parameter is
        ``poly_degree``. The same configuration is forwarded to image assembly,
        calibration association, and image processing.
    """

    def __init__(self, wideflat_filename, config=None):
        self.wideflat_filename = str(wideflat_filename)
        self._config = config

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["DATA_DIRS", "TRACES", "MODULE_ORDER_TRACE"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")

        for k, v in _DEFAULTS.items():
            setattr(self, k, params.get(k, v))

        self._l1_obj = None
        self._instrument_era = None
        self._bias_path = None
        self._anchor_shifts = {}
        self._fit_rms = {}
        self._results = None
        self._output_paths = {}
        self._info = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_chips(chips):
        """Return validated uppercase chip names without changing their order."""
        normalised = [str(chip).upper() for chip in chips]
        if not normalised:
            raise ValueError("chips must contain at least one CCD")
        if len(set(normalised)) != len(normalised):
            raise ValueError("chips must not contain duplicates")
        unknown = [chip for chip in normalised if chip not in {"GREEN", "RED"}]
        if unknown:
            raise ValueError(
                f"unsupported chip(s) {unknown}; expected GREEN and/or RED"
            )
        return normalised

    def _validate_parameters(self):
        """Validate the configured polynomial degree before reading the wideflat."""
        if not 1 <= int(self.poly_degree) <= 3:
            raise ValueError("poly_degree must be between 1 and 3")

    def _preprocess(self, chips):
        """Load, assemble, associate a bias, and bias-subtract the wideflat."""
        path = Path(self.wideflat_filename)
        if not path.is_file():
            raise FileNotFoundError(f"Wideflat file not found: {path}")

        l0_obj = KPF0.from_fits(str(path))
        l1_obj = ImageAssembly(l0_obj, self._config).perform(chips=chips)
        l1_obj = CalibrationAssociation(l1_obj, self._config).perform(["bias"])
        self._bias_path = l1_obj.headers["RECEIPT"]["BIASFILE"]
        l1_obj = ImageProcessing(l1_obj, self._config).perform(
            chips=chips, bias=True, dark=False, flat=False
        )
        return l1_obj

    def _resolve_instrument_era(self, date_obs):
        """Return the unique INSTERA containing ``date_obs``, or None."""
        path = _ERA_DEFINITIONS_PATH
        if not path.is_file():
            raise FileNotFoundError(f"Instrument-era table not found: {path}")

        eras = pd.read_csv(path, skipinitialspace=True)
        required = {"INSTERA", "UT_start_date", "UT_end_date"}
        missing = required.difference(eras.columns)
        if missing:
            raise ValueError(
                f"Instrument-era table is missing columns: {sorted(missing)}"
            )

        observation_time = pd.to_datetime(date_obs, utc=True).tz_localize(None)
        starts = pd.to_datetime(eras["UT_start_date"], utc=True).dt.tz_localize(None)
        ends = pd.to_datetime(eras["UT_end_date"], utc=True).dt.tz_localize(None)
        matches = eras.loc[(starts <= observation_time) & (observation_time <= ends)]
        if len(matches) > 1:
            raise ValueError(f"DATE-OBS {date_obs} matches multiple instrument eras")
        if matches.empty:
            return None
        return float(matches.iloc[0]["INSTERA"])

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

    @staticmethod
    def _reference_path(chip):
        """Return the repository trace-reference path for one chip."""
        return _ORDER_TRACE_PATHS[chip]

    def _load_seed_table(self, chip, manual_anchor=None):
        """Load and optionally translate one chip's approximate trace table."""
        path = self._reference_path(chip)
        if not path.is_file():
            raise FileNotFoundError(f"Order-trace reference not found: {path}")
        table = pd.read_csv(path, index_col=0)
        if list(table.columns) != _TRACE_COLUMNS:
            raise ValueError(
                f"{path} has incompatible columns; expected {_TRACE_COLUMNS}"
            )
        if table.duplicated(["Fiber", "Order"]).any():
            raise ValueError(f"{path} contains duplicate Fiber/Order rows")

        table = table.copy()
        numeric = _TRACE_COLUMNS[:8]
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
        """Return one finite, two-dimensional bias-subtracted CCD image."""
        extension = f"{chip}_CCD"
        if extension not in self._l1_obj.data:
            raise KeyError(f"{extension} extension is not available")
        image = np.asarray(self._l1_obj.data[extension])
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
        residual_center = np.nanmedian(residual)
        noise = 1.4826 * np.nanmedian(np.abs(residual - residual_center))
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
        spread = 1.4826 * np.nanmedian(np.abs(centers - center))
        if spread > edge_max_center_spread:
            return np.nan
        return center

    def _local_center(self, image, column, guess, candidates, fiber):
        """Dispatch to the profile estimator appropriate for one fiber."""
        if fiber == "CAL":
            return self._local_peak_center(image, column, guess, candidates)
        return self._local_edge_center(image, column, guess)

    def _trace_centers(
        self, image, seed_coeffs, sample_x, candidate_rows, fiber
    ):
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
            scale = 1.4826 * np.nanmedian(np.abs(residual[keep] - center))
            limit = max(0.25, fit_sigma * scale)
            updated = np.isfinite(y) & (np.abs(residual - center) <= limit)
            if updated.sum() < minimum:
                break
            if np.array_equal(updated, keep):
                keep = updated
                break
            keep = updated

        coeffs = np.polynomial.polynomial.polyfit(x[keep], y[keep], degree)
        padded = np.zeros(4, dtype=float)
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
        signal = np.minimum(
            signal, np.nanpercentile(signal, winsor_percentile)
        )
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
        x_mid = (ncol - 1) / 2.0
        for lower, upper in zip(rows[:-1], rows[1:], strict=False):
            lower_y = np.polynomial.polynomial.polyval(
                x_mid, [lower[f"Coeff{i}"] for i in range(4)]
            )
            upper_y = np.polynomial.polynomial.polyval(
                x_mid, [upper[f"Coeff{i}"] for i in range(4)]
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
        if list(table.columns) != _TRACE_COLUMNS:
            raise ValueError(f"{chip} output has incompatible columns")
        if table.empty:
            raise ValueError(f"{chip} produced no trace rows")
        if table.duplicated(["Fiber", "Order"]).any():
            raise ValueError(f"{chip} output has duplicate Fiber/Order rows")

        numeric = table[_TRACE_COLUMNS[:8]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{chip} output contains non-finite geometry")
        if not ((table["BottomEdge"] > 0) & (table["TopEdge"] > 0)).all():
            raise ValueError(f"{chip} output contains non-positive widths")
        if not ((table["X1"] >= 0) & (table["X2"] < ncol)).all():
            raise ValueError(f"{chip} output contains out-of-range column bounds")
        if not (table["X1"] <= table["X2"]).all():
            raise ValueError(f"{chip} output contains reversed column bounds")

        test_x = np.linspace(0, ncol - 1, 17)
        coeffs = table[[f"Coeff{i}" for i in range(4)]].to_numpy(dtype=float)
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
        sample_x = self._sample_columns(ncol)
        candidate_rows = [self._candidate_rows(image, column) for column in sample_x]

        rows = []
        rms_values = []
        for seed in seed_table.itertuples(index=False):
            seed_coeffs = np.array(
                [getattr(seed, f"Coeff{i}") for i in range(4)], dtype=float
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
                    "Coeff0": coeffs[0],
                    "Coeff1": coeffs[1],
                    "Coeff2": coeffs[2],
                    "Coeff3": coeffs[3],
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
        table = pd.DataFrame(rows, columns=_TRACE_COLUMNS)
        self._validate_trace_table(chip, table, nrow, ncol)
        self._fit_rms[chip] = np.asarray(rms_values, dtype=float)
        return table

    def _write_results(self, results, output_dir, overwrite):
        """Stage and atomically install all requested CSV outputs."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            chip: output_dir / f"order_trace_{chip.lower()}.csv" for chip in results
        }
        existing = [str(path) for path in paths.values() if path.exists()]
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
                    temporary[chip] = Path(stream.name)
                    table.to_csv(stream, lineterminator="\n")

            for chip, path in paths.items():
                os.replace(temporary[chip], path)
                temporary.pop(chip)
        finally:
            for path in temporary.values():
                path.unlink(missing_ok=True)

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
            f"  wideflat: {self.wideflat_filename}",
            f"  bias:     {self._bias_path}",
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

    def perform(self, chips=None, *, output_dir, cal_order3_y=None, overwrite=False):
        """
        Trace requested CCDs and always write their compatible CSV tables.

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
            If the wideflat, bias master, era table, or trace reference is absent.
        ValueError
            If the era/anchor, detected geometry, or output table is invalid.
        FileExistsError
            If a requested output exists and ``overwrite`` is False.
        """
        if chips is None:
            chips = self.chips
        chips = self._normalise_chips(chips)
        self._validate_parameters()
        anchors = self._validate_manual_anchors(chips, cal_order3_y)

        self._l1_obj = self._preprocess(chips)
        date_obs = self._l1_obj.headers["PRIMARY"]["DATE-OBS"]
        self._instrument_era = self._resolve_instrument_era(date_obs)
        if self._instrument_era is None and anchors is None:
            raise ValueError(
                f"DATE-OBS {date_obs} is outside the defined instrument eras; "
                "provide cal_order3_y for every requested chip"
            )

        results = {}
        for chip in chips:
            manual_anchor = None if anchors is None else anchors[chip]
            seed_table = self._load_seed_table(chip, manual_anchor)
            results[chip] = self._trace_chip(chip, seed_table)

        self._results = results
        self._write_results(results, output_dir, bool(overwrite))
        self._l1_obj.receipt_add_entry("order_trace", "", "PASS")
        self._track_info(chips)
        logger.info("%s", self._info)
        return results

    def info(self):
        """Print a summary of the module configuration and tracing results."""
        if self._info is None:
            print(f"{type(self).__name__}: perform() has not been called")
        else:
            print(self._info)
