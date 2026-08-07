"""
KPF Order Trace module.

Locates the spectral traces on one completed vNext master flat -- where the
illuminated pixels lie on each detector -- and writes a low-order polynomial
describing every trace centerline, plus its aperture edges, to a single CSV
covering all CCDs.

Trace geometry is measured from the master flat alone. No pre-existing
order-trace table is consulted, so the module never inherits the geometry it
exists to determine. Trace identity is established before any polynomial is
fitted: the CAL orderlet is both narrower and several times brighter than the
SKY and science orderlets, which fixes the phase of the repeating
SKY-SCI1-SCI2-SCI3-CAL group, and each CAL closes one echelle order. Because
each order is anchored on its own CAL, an order truncated by a detector edge
shifts no other trace's label; any other departure from the pattern is an error.

No orderlet spacing is treated as regular. Spacing between orders varies
several-fold across a detector, and identity rests on the fiber pattern alone.

OrderTrace applies no calibrations of its own; it consumes the master flat as
delivered by the Flat module.

Axis convention
---------------
Row and column here are numpy's, which is the *transpose* of the KPF physical
convention -- a KPF scientist's "row" is this module's column. Throughout:

- ``row`` (axis 0, ``nrow``, ``y``) is **cross-dispersion**: a trace is a few
  pixels thick along it, and the fitted polynomial returns a row.
- ``column`` (axis 1, ``ncol``, ``x``) is **dispersion**: a trace runs the
  length of it, and the fitted polynomial takes a column as its argument.

"""

import logging
import os
from itertools import permutations

import numpy as np
import pandas as pd
from astropy.stats import mad_std
from numpy.polynomial import polynomial
from scipy.linalg import solve_banded
from scipy.ndimage import gaussian_filter1d, label

from kpfpipe import DEFAULTS
from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.stats import robust_polyfit

logger = logging.getLogger(__name__)


_DEFAULTS = {
    **DEFAULTS,
    "poly_degree": 3,
}


class OrderTrace:
    """
    Measure spectral traces from one vNext KPF master flat and write one trace
    CSV covering all CCDs.

    Operations include:
      - reducing each detector column to a mask of illuminated pixels
      - collecting touching mask pixels into clusters (8-connectivity)
      - curating clusters (rejecting artifacts, rejoining split traces)
      - identifying each cluster's fiber and echelle order index
      - fitting a polynomial centerline, row against column, to each trace
      - estimating aperture edges and enforcing the gap between neighbors

    Output always contains one row per expected trace of every CCD (``norder``
    x five fibers per chip), ordered by chip then order index then fiber, each
    row keyed by a leading ``Chip`` field. A trace no cluster was detected for
    -- the one an echelle order lying partly off the detector can cost -- is
    written with NaN geometry and a ``Status`` of 'missing', so it is reported
    rather than silently dropped. Every detected trace is measured or the run
    fails: one measured over only part of the detector is written 'partial',
    one spanning essentially the whole detector 'full', and a trace that cannot
    be fitted or given an aperture raises rather than reaching the CSV.

    Standalone rather than a ``BaseMasterModule`` subclass because it consumes
    one completed L1 master instead of stacking L0 exposures.

    Parameters
    ----------
    master_flat_filename : str or pathlib.Path
        vNext ``KPFMasterL1`` flat FITS filename.
    config : None | dict | ConfigHandler
        Module configuration. ``ConfigHandler`` values are read from TRACES and
        ORDER_TRACE, whose only module-specific parameter is ``poly_degree``.
        Detection and curation thresholds are tuned to the instrument rather
        than to a run, and are default arguments of the methods that use them.
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

        self._master_flat = self._load_master_flat()
        self._image = {
            chip: self._master_flat.data[f"{chip}_IMG"] for chip in self.chips
        }

        self._clusters = {}
        self._metadata = {}
        self._profiles = {}
        self._trace_tables = {}
        self._output_table = None
        self._output_path = None
        self._info = None

    # ------------------------------------------------------------------
    # Private helpers - input
    # ------------------------------------------------------------------

    def _load_master_flat(self):
        """Load and validate the vNext L1 master-flat product.

        Caches each CCD image as ``self._image[chip]``, the one place trace
        measurement reads detector pixels from.
        """
        if not os.path.isfile(self.master_flat_filename):
            raise FileNotFoundError(
                f"Master flat not found: {self.master_flat_filename}"
            )

        master_flat = KPFMasterL1.from_fits(self.master_flat_filename)
        master_type = master_flat.headers["PRIMARY"].get("MASTYPE")
        if str(master_type).lower() != "flat":
            raise ValueError(
                f"{self.master_flat_filename} is not a vNext flat master "
                f"(MASTYPE={master_type!r})"
            )

        return master_flat

    def _trace_fields(self, which="all"):
        """Return output fields of one trace table, in their written order.

        ``which`` selects the whole schema ('all'), the aperture and column
        bounds ('bounds'), the polynomial coefficients ('coeffs'), whose count
        follows the configured fit degree, or both of the last two together
        ('geometry'), which is every field a measured trace fills in.
        """
        degree = int(self.poly_degree)
        if degree < 0:
            raise ValueError(f"poly_degree must be non-negative, got {degree!r}")

        _LABELS = ["Chip", "Fiber", "Order"]
        _COEFFS = [f"Coeff{i}" for i in range(degree + 1)]
        _BOUNDS = ["BottomEdge", "TopEdge", "X1", "X2"]
        _QUALITY = ["PolyfitRMS", "Status"]

        if which == "all":
            return _LABELS + _COEFFS + _BOUNDS + _QUALITY
        if which == "bounds":
            return _BOUNDS
        elif which == "coeffs":
            return _COEFFS
        elif which == "geometry":
            return _COEFFS + _BOUNDS
        else:
            raise ValueError(
                "which must be one of 'all', 'coeffs', 'bounds', or "
                f"'geometry'; got {which}"
            )

    # ------------------------------------------------------------------
    # Private helpers - trace detection
    # ------------------------------------------------------------------

    def _detect_illuminated_pixels(self, chip, smoothing_weight=20.0, trace_ratio=0.5):
        """Return a boolean mask of illuminated pixels, one column at a time.

        Each column is reduced independently, so the threshold is local along
        dispersion. Subtracting the column's inter-order flux isolates the
        orderlet peaks, setting the threshold by orderlet contrast rather than
        by the absolute lamp level.

        That flux is a Whittaker smoother -- the f minimizing
        ``sum (f - pixels)**2 + smoothing_weight * sum (df/drow)**2`` -- whose
        normal equations are symmetric and tridiagonal, so ``solve_banded``
        recovers it in O(nrow) from one sub- and one super-diagonal. Symmetry
        lets one off-diagonal array serve as both, its two out-of-matrix corner
        entries zeroed and never read. The default weight rides over the
        orderlets (5-11 pixels thick, spaced 16-20) while still following the
        far broader order-to-order flux envelope.
        """
        nrow, ncol = self.ccd["nrow"], self.ccd["ncol"]
        filled = np.nan_to_num(self._image[chip], nan=0.0, posinf=0.0, neginf=0.0)

        off_diagonal = np.full(nrow, -smoothing_weight)
        diagonal = np.full(nrow, 1.0 + 2.0 * smoothing_weight)
        diagonal[[0, -1]] = 1.0 + smoothing_weight
        bands = np.vstack([off_diagonal, diagonal, off_diagonal])
        bands[0, 0] = bands[2, -1] = 0.0

        illuminated = np.zeros((nrow, ncol), dtype=bool)
        for j in range(ncol):
            column_pixels = filled[:, j]
            residual = column_pixels - solve_banded((1, 1), bands, column_pixels)
            positive_residual = np.clip(residual, 0.0, None)
            threshold = 0.5 * np.quantile(
                positive_residual, trace_ratio, method="lower"
            )
            illuminated[:, j] = residual > threshold + 1.0
        return illuminated

    def _detect_clusters(self, illuminated):
        """Collect touching illuminated pixels into clusters.

        The 3x3 structure treats all eight surrounding pixels as adjacent
        (8-connectivity), so pixels meeting only at a corner still join one
        cluster.
        """
        labels, cluster_count = label(illuminated, structure=np.ones((3, 3), dtype=int))
        row_indices, col_indices = np.nonzero(labels)
        cluster_ids = labels[row_indices, col_indices]

        ordering = np.argsort(cluster_ids, kind="stable")
        row_indices, col_indices = row_indices[ordering], col_indices[ordering]
        # One contiguous run of pixels per cluster id, so a single searchsorted
        # gives every cluster's slice without revisiting the label image.
        bounds = np.searchsorted(cluster_ids[ordering], np.arange(1, cluster_count + 2))
        return [
            {
                "row_indices": row_indices[start:stop],
                "col_indices": col_indices[start:stop],
                "npixel": int(stop - start),
            }
            for start, stop in zip(bounds[:-1], bounds[1:], strict=True)
        ]

    def _pixels_near_mid_dispersion(self, cluster, width=41):
        """Return one cluster's pixel row and column indices at mid-dispersion.

        Trace identity is measured over ``width`` columns centered on the
        detector, where every order is on the detector and the orderlets of an
        order are cleanly separated across dispersion.
        """
        center = self.ccd["ncol"] // 2
        half_width = width // 2
        near_center = np.abs(cluster["col_indices"] - center) <= half_width
        return cluster["row_indices"][near_center], cluster["col_indices"][near_center]

    @staticmethod
    def _log_rejection(cluster, reason):
        """Record one curated-away cluster and why it was discarded."""
        logger.debug(
            "rejecting cluster at rows %d-%d, columns %d-%d (%s)",
            cluster["row_indices"].min(),
            cluster["row_indices"].max(),
            cluster["col_indices"].min(),
            cluster["col_indices"].max(),
            reason,
        )

    def _reject_small_clusters(self, clusters, min_cluster_pixels=500):
        """Drop clusters too small to be any part of a trace.

        Runs before fragments are rejoined, so it tests only pixel count: a
        genuine fragment is short and may sit anywhere on the detector.
        """
        kept = []
        for cluster in clusters:
            if cluster["npixel"] < min_cluster_pixels:
                self._log_rejection(cluster, f"only {cluster['npixel']} pixels")
            else:
                kept.append(cluster)
        return kept

    def _reject_malformed_clusters(
        self, clusters, min_spanned_columns=200, min_rows_per_column=3.0
    ):
        """Drop rejoined clusters whose shape is not that of a trace.

        A trace runs the length of dispersion and is several pixel rows thick
        at mid-dispersion, where its shape is measured. A cluster falling short
        of either, or missing from mid-dispersion altogether, is discarded
        however trace-like its full-frame pixel count looks.
        """
        kept = []
        for cluster in clusters:
            spanned_columns = int(np.ptp(cluster["col_indices"])) + 1
            peak_rows, peak_columns = self._pixels_near_mid_dispersion(cluster)
            occupied_columns = np.unique(peak_columns).size
            rows_per_column = (
                peak_rows.size / occupied_columns if occupied_columns else 0.0
            )
            if spanned_columns < min_spanned_columns:
                reason = f"spans only {spanned_columns} pixel columns"
            elif occupied_columns == 0:
                reason = "no pixels at mid-dispersion"
            elif rows_per_column < min_rows_per_column:
                reason = f"only {rows_per_column:.1f} pixel rows thick at mid-detector"
            else:
                kept.append(cluster)
                continue
            self._log_rejection(cluster, reason)
        return kept

    @staticmethod
    def _mean_row_per_column(rows, columns):
        """Collapse cluster pixels to one mean row per occupied pixel column.

        Returns the occupied columns and their mean rows -- the (x, y) pair a
        centerline is fitted through.
        """
        row_total = np.bincount(columns, weights=rows)
        pixel_count = np.bincount(columns)
        occupied = np.flatnonzero(pixel_count)
        return occupied.astype(float), row_total[occupied] / pixel_count[occupied]

    def _find_mergeable_pair(
        self,
        clusters,
        max_gap_columns=512,
        max_row_offset=10.0,
        max_residual_pixels=2.0,
    ):
        """Return the indices of the first two clusters lying on one curve.

        Fragments are compared as one mean row per column rather than as
        individual pixels, whose scatter is the trace's cross-dispersion
        thickness and would swamp the misalignment being measured. Only pixels
        within one gap width of either side are used, and the fit is linear:
        the test asks whether the pieces line up locally, over which a trace's
        curvature is negligible.
        """
        rows = [cluster["row_indices"] for cluster in clusters]
        columns = [cluster["col_indices"] for cluster in clusters]
        first_column = [int(column.min()) for column in columns]
        last_column = [int(column.max()) for column in columns]
        row_at_first = [
            row[column == edge].mean()
            for row, column, edge in zip(rows, columns, first_column, strict=True)
        ]
        row_at_last = [
            row[column == edge].mean()
            for row, column, edge in zip(rows, columns, last_column, strict=True)
        ]

        for host, candidate in permutations(range(len(clusters)), 2):
            gap = first_column[candidate] - last_column[host]
            if not 0 < gap <= max_gap_columns:
                continue
            if abs(row_at_first[candidate] - row_at_last[host]) > max_row_offset:
                continue

            host_near_gap = columns[host] >= last_column[host] - gap
            candidate_near_gap = columns[candidate] <= first_column[candidate] + gap
            host_columns, host_centerline = self._mean_row_per_column(
                rows[host][host_near_gap], columns[host][host_near_gap]
            )
            candidate_columns, candidate_centerline = self._mean_row_per_column(
                rows[candidate][candidate_near_gap],
                columns[candidate][candidate_near_gap],
            )
            if host_columns.size < 2 or candidate_columns.size == 0:
                continue

            coeffs = polynomial.polyfit(host_columns, host_centerline, 1)
            extrapolated = polynomial.polyval(candidate_columns, coeffs)
            residual = np.median(np.abs(candidate_centerline - extrapolated))
            if residual <= max_residual_pixels:
                return host, candidate
        return None

    def _merge_fragmented_clusters(self, clusters):
        """Rejoin clusters that are separated pieces of a single trace."""
        clusters = list(clusters)
        while True:
            pair = self._find_mergeable_pair(clusters)
            if pair is None:
                return clusters
            host, candidate = (clusters[index] for index in pair)
            logger.info(
                "merging cluster fragments spanning columns %d-%d and %d-%d",
                host["col_indices"].min(),
                host["col_indices"].max(),
                candidate["col_indices"].min(),
                candidate["col_indices"].max(),
            )
            row_indices = np.concatenate(
                [host["row_indices"], candidate["row_indices"]]
            )
            col_indices = np.concatenate(
                [host["col_indices"], candidate["col_indices"]]
            )
            combined = {
                "row_indices": row_indices,
                "col_indices": col_indices,
                "npixel": int(row_indices.size),
            }
            clusters = [
                cluster for index, cluster in enumerate(clusters) if index not in pair
            ]
            clusters.append(combined)

    # ------------------------------------------------------------------
    # Private helpers - trace identity
    # ------------------------------------------------------------------

    def _track_cluster_metadata(self, chip):
        """Measure row, mask thickness, and flux for each cluster at mid-detector.

        The row is the cluster's cross-dispersion position there, and is what
        the clusters are subsequently ordered by: sorting on it walks the
        orderlets of the detector in the order they physically appear.
        """
        records = []
        for index, cluster in enumerate(self._clusters[chip]):
            row_indices, col_indices = self._pixels_near_mid_dispersion(cluster)
            records.append(
                {
                    "cluster": index,
                    "row": float(row_indices.mean()),
                    "thickness": row_indices.size / np.unique(col_indices).size,
                    "flux": float(
                        np.median(self._image[chip][row_indices, col_indices])
                    ),
                }
            )
        metadata = pd.DataFrame(records)
        return metadata.sort_values("row").reset_index(drop=True)

    def _flag_cal_clusters(self, metadata, max_thickness=6.0, min_flux_ratio=1.8):
        """Add the boolean ``is_cal`` column, flagging each order's CAL orderlet.

        A CAL is thinner and brighter than the orderlets around it. Brightness
        is judged against the mean of the two clusters bracketing each
        candidate. Lamp flux varies by an order of magnitude across the
        detector, and bracketing cancels that gradient where a running median
        of the neighborhood does not.

        One CAL closes each order, so the flagged count is checked against the
        cluster count. An order lying partly off the detector contributes
        orderlets without its CAL, or a CAL whose order is otherwise
        incomplete, which moves the count by one either way.
        """
        flux = metadata["flux"].to_numpy()
        flux_below = np.concatenate([flux[1:2], flux[:-1]])
        flux_above = np.concatenate([flux[1:], flux[-2:-1]])
        metadata = metadata.copy()
        metadata["is_cal"] = (metadata["thickness"].to_numpy() <= max_thickness) & (
            flux / ((flux_below + flux_above) / 2.0) > min_flux_ratio
        )

        cal_count = int(metadata["is_cal"].sum())
        expected_cal_count = len(metadata) // len(self.fibers)
        if abs(cal_count - expected_cal_count) > 1:
            raise ValueError(
                f"{cal_count} CAL orderlets identified among {len(metadata)} "
                f"clusters, expected {expected_cal_count}; cannot phase the "
                "fiber pattern"
            )
        return metadata

    def _assign_fiber_identities(self, chip, metadata):
        """Give every cluster its fiber name, phasing the pattern on the CALs.

        Every order is four non-CAL orderlets closed by a CAL, so a cluster's
        fiber is fixed by its rank relative to the CAL of its own order and no
        orderlet spacing need be assumed. Only the orders at the detector edges
        may depart from the pattern: the lowest can lose the orderlets that open
        it and the highest those that close it, and either edge can clip one
        orderlet of the order beyond, leaving it too faint to be recognized as
        the CAL it usually is. That clipped orderlet is left unnamed, and so
        drops out with the rest of the order it belongs to. Any other departure
        means the clusters cannot be trusted to be one orderlet each.
        """
        metadata = self._flag_cal_clusters(metadata)
        cluster_rows = metadata["row"].to_numpy()
        cal_position = len(self.fibers) - 1
        cal_indices = np.flatnonzero(metadata["is_cal"].to_numpy())

        fiber_names = np.empty(len(metadata), dtype=object)
        previous_cal = -1
        for group, cal_index in enumerate(cal_indices):
            orderlets_below = cal_index - previous_cal - 1
            if orderlets_below != cal_position and (
                group or orderlets_below > cal_position + 1
            ):
                raise ValueError(
                    f"{chip}: {orderlets_below} orderlets below the CAL at row "
                    f"{cluster_rows[cal_index]:.0f}, expected {cal_position}"
                )
            for offset in range(min(orderlets_below, cal_position) + 1):
                fiber_names[cal_index - offset] = self.fibers[cal_position - offset]
            previous_cal = cal_index

        last_cal = cal_indices[-1]
        orderlets_above = len(metadata) - last_cal - 1
        if orderlets_above > cal_position + 1:
            raise ValueError(
                f"{chip}: {orderlets_above} orderlets above the CAL at row "
                f"{cluster_rows[last_cal]:.0f}, expected at most {cal_position}"
            )
        for position in range(min(orderlets_above, cal_position)):
            fiber_names[last_cal + 1 + position] = self.fibers[position]

        metadata = metadata.copy()
        metadata["Fiber"] = fiber_names
        for row in metadata.loc[metadata["Fiber"].isna(), "row"]:
            logger.warning(
                "%s: discarding an orderlet at row %.0f, clipped by the detector "
                "edge and belonging to no expected order",
                chip,
                row,
            )
        return metadata

    def _assign_order_indexes(self, chip, metadata):
        """Number the orders and return one row per expected trace of the CCD.

        Orders are counted off the CALs, bottom to top, and given their 0-based
        ``Order`` index on the CCD. An order lying mostly off the detector shows a
        single orderlet beyond the expected count, which is discarded -- the
        shorter of the two edge orders goes, counting only the orderlets that
        were named. A trace that was never detected leaves an empty row, so the
        result always holds one row per fiber of every expected order.
        """
        norder = self.norder[chip]
        is_cal = metadata["is_cal"].to_numpy()
        index = np.cumsum(is_cal) - is_cal

        named = metadata["Fiber"].notna().to_numpy()
        order_sizes = np.bincount(index, weights=named).astype(int)
        if order_sizes.size > norder:
            discarded = 0 if order_sizes[0] <= order_sizes[-1] else index.max()
            kept = index != discarded
            logger.warning(
                "%s: %d orders detected but %d expected; discarding %d orderlets at "
                "row %.0f",
                chip,
                order_sizes.size,
                norder,
                order_sizes[discarded],
                metadata["row"].to_numpy()[~kept].mean(),
            )
            metadata = metadata[kept]
            index = index[kept] - (1 if discarded == 0 else 0)

        traces = pd.MultiIndex.from_product(
            [range(norder), self.fibers], names=["Order", "Fiber"]
        ).to_frame(index=False)

        detected = pd.DataFrame(
            {
                "Order": index,
                "Fiber": metadata["Fiber"].to_numpy(),
                "cluster": metadata["cluster"].to_numpy(),
            }
        )
        return traces.merge(detected, on=["Order", "Fiber"], how="left")

    # ------------------------------------------------------------------
    # Private helpers - trace measurement
    # ------------------------------------------------------------------

    def _sample_profiles(self, chip, sample_count=65, col_half_window=3):
        """Return everything one CCD is measured at its sampled columns.

        The columns span the detector, both edges included: a centerline is
        fitted through one measurement per sampled column, so these are the
        dispersion positions the polynomial is constrained at. Their profiles
        follow, one row of the whole cross-dispersion each, sampled column for
        sampled column with ``columns`` itself.

        Profiles rather than raw pixels: neighboring columns are median
        combined, which suppresses noise without smearing, because a trace
        moves only slightly across dispersion over so short a run along it.

        Every trace on the CCD is measured at these same few dozen columns, so
        all the profiles are built once here instead of being recomputed by
        each trace in turn.

        A window overhanging a detector edge is padded with NaN, which
        ``nanmedian`` then ignores, giving exactly the truncated median a
        clamped per-column slice would.

        The result is cached as ``self._profiles[chip]``, the one place the CCD's
        measurements at these columns are kept: ``columns`` and ``profiles``
        here, and the ``centers`` measured per trace with the ``good`` mask of
        the samples its fit accepted, both filled in by the fitting step and
        left None until it runs. The cache is keyed by CCD alone, so the
        sampling arguments are the defaults every caller uses.
        """
        if chip in self._profiles:
            return self._profiles[chip]

        image = self._image[chip]
        nrow, ncol = self.ccd["nrow"], self.ccd["ncol"]

        # Determine spatial profile over median of several columns to reduce noise
        columns = np.unique(np.linspace(0, ncol - 1, sample_count, dtype=int))
        offsets = np.arange(-col_half_window, col_half_window + 1)
        cols_in_window = columns[:, None] + offsets[None, :]
        on_detector = (cols_in_window >= 0) & (cols_in_window < ncol)

        pixels = np.full((nrow, *cols_in_window.shape), np.nan, dtype=image.dtype)
        pixels[:, on_detector] = image[:, cols_in_window[on_detector]]
        profiles = np.nanmedian(pixels, axis=2)

        # Fill non-finite rows with the profile's own median, as a wholly
        # unmeasurable column has no scale of its own to fall back on.
        finite = np.isfinite(profiles)
        fill = np.zeros(columns.size)
        measured = finite.any(axis=0)
        fill[measured] = np.nanmedian(
            np.where(finite, profiles, np.nan)[:, measured], axis=0
        )
        profiles = np.where(finite, profiles, fill[None, :]).astype(float, copy=False)

        self._profiles[chip] = {
            "columns": columns,
            "profiles": profiles.T,
            "centers": None,
            "good": None,
        }
        return self._profiles[chip]

    def _local_peak_center(
        self,
        column_profile,
        row_guess,
        row_half_window=7,
        signal_smoothing_sigma=1.0,
        winsor_percentile=90.0,
    ):
        """Measure a subpixel row center for a peaked CAL-fiber profile."""
        if not np.isfinite(row_guess):
            return np.nan

        first_i = max(0, int(np.floor(row_guess)) - row_half_window)
        last_i = min(column_profile.size, int(np.ceil(row_guess)) + row_half_window + 1)
        if last_i - first_i < 3:
            return np.nan

        profile = column_profile[first_i:last_i]
        rows = np.arange(first_i, last_i, dtype=float)
        background = np.nanpercentile(profile, 20.0)
        signal = np.clip(profile - background, 0.0, None)
        smoothed = gaussian_filter1d(
            np.nan_to_num(signal),
            sigma=signal_smoothing_sigma,
            mode="nearest",
        )
        if not np.any(smoothed > 0):
            return np.nan

        peak_index = int(np.argmax(smoothed))
        core_start = max(0, peak_index - 3)
        core_stop = min(signal.size, peak_index + 4)
        weights = signal[core_start:core_stop]
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            return np.nan
        winsor_limit = np.nanpercentile(weights, winsor_percentile)
        weights = np.minimum(weights, winsor_limit)
        if weights.sum() <= 0:
            return np.nan
        return float(np.average(rows[core_start:core_stop], weights=weights))

    def _local_edge_center(
        self,
        column_profile,
        row_guess,
        row_half_window=7,
        background_smoothing_sigma=20.0,
        signal_smoothing_sigma=1.0,
        edge_levels=(0.25, 0.40, 0.55, 0.70),
        edge_min_width_pixels=3.0,
        edge_max_center_spread=0.75,
    ):
        """Measure a flat-topped orderlet's geometric row center from its edges."""
        if not np.isfinite(row_guess):
            return np.nan

        first_i = max(0, int(np.floor(row_guess)) - row_half_window)
        last_i = min(column_profile.size, int(np.ceil(row_guess)) + row_half_window + 1)
        if last_i - first_i < 5:
            return np.nan

        background = gaussian_filter1d(
            column_profile,
            sigma=background_smoothing_sigma,
            mode="nearest",
        )
        signal = gaussian_filter1d(
            column_profile[first_i:last_i] - background[first_i:last_i],
            sigma=signal_smoothing_sigma,
            mode="nearest",
        )
        rows = np.arange(first_i, last_i, dtype=float)
        if not np.isfinite(signal).all():
            return np.nan

        baseline = np.nanpercentile(signal, 20.0)
        amplitude = np.nanpercentile(signal, 90.0) - baseline
        if not np.isfinite(amplitude) or amplitude <= 0.0:
            return np.nan

        centers = []
        row_guess_index = int(np.argmin(np.abs(rows - row_guess)))
        for level in edge_levels:
            threshold = baseline + level * amplitude
            bright = np.flatnonzero(signal >= threshold)
            if bright.size == 0:
                continue
            core_index = bright[np.argmin(np.abs(bright - row_guess_index))]

            # Walk out of the illuminated core to either side and interpolate
            # the row the signal falls back through the threshold at.
            crossings = []
            for direction in (-1, 1):
                current = int(core_index)
                adjacent = current + direction
                while 0 <= adjacent < signal.size and signal[adjacent] >= threshold:
                    current = adjacent
                    adjacent = current + direction
                if not 0 <= adjacent < signal.size:
                    crossings.append(np.nan)
                    continue
                fall = signal[current] - signal[adjacent]
                if not np.isfinite(fall) or fall <= 0.0:
                    crossings.append(np.nan)
                    continue
                fraction = (signal[current] - threshold) / fall
                crossings.append(
                    float(rows[current] + fraction * (rows[adjacent] - rows[current]))
                )

            bottom, top = crossings
            if (
                np.isfinite(bottom)
                and np.isfinite(top)
                and top - bottom >= edge_min_width_pixels
            ):
                centers.append((bottom + top) / 2.0)

        min_levels = max(2, int(np.ceil(len(edge_levels) / 2)))
        if len(centers) < min_levels:
            return np.nan
        centers = np.asarray(centers, dtype=float)
        if mad_std(centers, ignore_nan=True) > edge_max_center_spread:
            return np.nan
        return float(np.nanmedian(centers))

    def _trace_centers(self, chip, fiber, order):
        """Measure a subpixel row center at every sampled column of one trace.

        The trace's cluster and the CCD's sampled columns and profiles are read
        from the caches, and the measured row is written into
        ``self._profiles[chip]["centers"]``, which the fitting step preallocates.

        The cluster's own mask pixels at a column give the starting row, so no
        prior trace geometry is needed. The CAL orderlet is peaked and centers
        well on its brightest pixels; SKY and the science orderlets are
        flat-topped, with no meaningful peak, and are centered from their two
        cross-dispersion edges instead.
        """
        position = order * len(self.fibers) + self.fibers.index(fiber)
        sampled = self._profiles[chip]
        columns, profiles = sampled["columns"], sampled["profiles"]
        cluster = self._clusters[chip][
            int(self._metadata[chip]["cluster"].iloc[position])
        ]

        measure_center = (
            self._local_peak_center if fiber == "CAL" else self._local_edge_center
        )
        centers = sampled["centers"][position]
        for sample, j in enumerate(columns):
            in_column = cluster["col_indices"] == j
            if not in_column.any():
                continue
            row_guess = float(cluster["row_indices"][in_column].mean())
            centers[sample] = measure_center(profiles[sample], row_guess)
        return centers

    def _trace_width(
        self,
        chip,
        fiber,
        order,
        width_half_window=7,
        winsor_percentile=90.0,
        width_sigma=2.8,
    ):
        """Return one trace's robust bottom/top widths from its accepted samples.

        The trace's fitted centerline is read from ``self._trace_tables[chip]``
        and its sampled columns, profiles, and accepted samples from
        ``self._profiles[chip]``, so the fitting step must have run.

        Below and above are cross-dispersion, so the two widths become the
        ``BottomEdge`` and ``TopEdge`` output fields. Each sampled column
        contributes the second moment of the flux on one side of the centerline,
        which is a Gaussian's sigma; estimating the two halves separately
        preserves aperture asymmetry without the unstable unconstrained
        half-Gaussian fits used by the legacy implementation.
        """
        position = order * len(self.fibers) + self.fibers.index(fiber)
        sampled = self._profiles[chip]
        columns, profiles = sampled["columns"], sampled["profiles"]
        coeffs = (
            self._trace_tables[chip]
            .loc[position, self._trace_fields(which="coeffs")]
            .to_numpy(dtype=float)
        )

        bottom_widths = []
        top_widths = []
        samples = np.flatnonzero(sampled["good"][position])
        stride = max(1, samples.size // 24)
        for sample in samples[::stride]:
            center = polynomial.polyval(columns[sample], coeffs)
            column_profile = profiles[sample]
            first_i = max(0, int(np.floor(center)) - width_half_window)
            last_i = min(
                column_profile.size, int(np.ceil(center)) + width_half_window + 1
            )
            if last_i - first_i < 5:
                continue

            profile = column_profile[first_i:last_i]
            rows = np.arange(first_i, last_i, dtype=float)
            signal = np.clip(profile - np.nanpercentile(profile, 20.0), 0.0, None)
            if not np.any(signal > 0):
                continue
            signal = np.minimum(signal, np.nanpercentile(signal, winsor_percentile))
            offsets = rows - center

            for half, sampled in (
                (offsets <= 0, bottom_widths),
                (offsets >= 0, top_widths),
            ):
                weights = signal[half]
                distance = np.abs(offsets[half])
                if weights.size < 2 or weights.sum() <= 0:
                    continue
                sigma = np.sqrt(np.sum(weights * distance**2) / np.sum(weights))
                width = width_sigma * sigma
                if np.isfinite(width) and width > 0:
                    sampled.append(width)

        if len(bottom_widths) < 3 or len(top_widths) < 3:
            raise ValueError(
                "fewer than three valid samples for trace-width estimation"
            )
        return (
            float(np.clip(np.nanmedian(bottom_widths), 1.0, 11.0)),
            float(np.clip(np.nanmedian(top_widths), 1.0, 11.0)),
        )

    def _clamp_neighboring_apertures(self, chip, orderlet_gap_pixels=2.0):
        """Shrink the apertures of traces that would otherwise touch.

        Measured traces run ascending in row, so consecutive ones are the
        neighbors that must not overlap. Each keeps its profile width where
        there is room; where two would meet, both clamp to half the space
        between their centerlines less a guard band. That space is measured
        where the two centerlines run closest, so the apertures stay disjoint
        across the whole detector, not only at mid-dispersion.

        The clamped edges are written back into ``self._trace_tables[chip]``.
        """
        coefficient_fields = self._trace_fields(which="coeffs")
        table = self._trace_tables[chip]

        measured = table.index[table["Status"] != "missing"]
        test_columns = np.linspace(0, self.ccd["ncol"] - 1, 101)
        centers = [
            polynomial.polyval(
                test_columns,
                table.loc[trace_index, coefficient_fields].to_numpy(dtype=float),
            )
            for trace_index in measured
        ]
        for index, (below, above) in enumerate(
            zip(measured[:-1], measured[1:], strict=False)
        ):
            closest_approach = float(np.min(centers[index + 1] - centers[index]))
            half_space = (closest_approach - orderlet_gap_pixels) / 2.0
            if half_space <= 0:
                raise ValueError(
                    "neighboring fitted traces cross or leave no room for the "
                    "orderlet gap"
                )
            table.loc[below, "TopEdge"] = min(table.loc[below, "TopEdge"], half_space)
            table.loc[above, "BottomEdge"] = min(
                table.loc[above, "BottomEdge"], half_space
            )

    def _validate_trace_table(self, chip, full_coverage_fraction=0.9):
        """Validate output schema, geometry, labels, and detector coverage.

        Settles each measured trace's ``X1``-``X2`` as the columns over which
        its whole aperture lands on the detector, then its 'unknown' ``Status``
        from that span; no 'unknown' may survive.
        """
        table = self._trace_tables[chip]
        nrow, ncol = self.ccd["nrow"], self.ccd["ncol"]
        coefficient_fields = self._trace_fields(which="coeffs")
        if list(table.columns) != self._trace_fields():
            raise ValueError(f"{chip} output has incompatible fields")

        expected_traces = pd.MultiIndex.from_product(
            [range(self.norder[chip]), self.fibers], names=["Order", "Fiber"]
        )
        produced_traces = pd.MultiIndex.from_arrays([table["Order"], table["Fiber"]])
        if len(table) != len(expected_traces) or set(produced_traces) != set(
            expected_traces
        ):
            raise ValueError(
                f"{chip} output does not contain exactly one row per expected trace"
            )

        measured = table[table["Status"] != "missing"]
        geometry = measured[coefficient_fields + ["BottomEdge", "TopEdge"]].to_numpy(
            dtype=float
        )
        if not np.isfinite(geometry).all():
            raise ValueError(f"{chip} output contains non-finite measured geometry")
        if not ((measured["BottomEdge"] > 0) & (measured["TopEdge"] > 0)).all():
            raise ValueError(f"{chip} output contains non-positive widths")

        # Traces are written in ascending row, so evaluating every centerline
        # at the same columns must give rows that strictly increase down each
        # column; anything else means two traces cross or are mislabelled.
        detector_columns = np.arange(ncol, dtype=float)
        coeffs = measured[coefficient_fields].to_numpy(dtype=float)
        centers = np.array(
            [polynomial.polyval(detector_columns, coeff) for coeff in coeffs]
        )
        if np.any(np.diff(centers, axis=0) <= 0):
            raise ValueError(f"{chip} fitted traces cross or are out of detector order")

        # The apertures themselves, not just the centerlines, must stay disjoint:
        # every trace's upper edge below the next trace's lower edge everywhere.
        upper = centers + measured["TopEdge"].to_numpy(dtype=float)[:, None]
        lower = centers - measured["BottomEdge"].to_numpy(dtype=float)[:, None]
        if np.any(lower[1:] <= upper[:-1]):
            raise ValueError(f"{chip} fitted apertures overlap")

        # A trace is usable only where its whole aperture lands on the
        # detector; X1-X2 is the column span over which that holds.
        on_detector = (lower >= 0) & (upper <= nrow - 1)
        if not on_detector.any(axis=1).all():
            raise ValueError(f"{chip} output contains a wholly off-detector trace")
        first = on_detector.argmax(axis=1)
        last = ncol - 1 - on_detector[:, ::-1].argmax(axis=1)
        table.loc[measured.index, "X1"] = detector_columns[first]
        table.loc[measured.index, "X2"] = detector_columns[last]

        coverage = (last - first + 1) / ncol
        table.loc[measured.index, "Status"] = np.where(
            coverage >= full_coverage_fraction, "full", "partial"
        )
        if (table["Status"] == "unknown").any():
            raise ValueError(f"{chip} output contains an unclassified trace")

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def detect_traces(self, chip):
        """Detect and curate every trace cluster on one CCD.

        The CCD carries one trace per fiber of every order. Only the outermost
        order may lie partly off the detector, which leaves one trace missing or
        one orderlet of the next order in view, so the count is allowed to be
        off by one either way.

        The surviving clusters are cached as ``self._clusters[chip]``, which is
        where the identity and fitting steps read them from; they are returned
        as well so the step can be driven on its own. Everything already
        measured for the CCD is discarded with the clusters it was measured
        from, whose list positions this rebuild invalidates: the assigned
        identities, the fitted trace table, and the centers and accepted samples
        of each trace. The sampled profiles survive: they are a property of the
        image, not of the clusters.
        """
        illuminated = self._detect_illuminated_pixels(chip)
        clusters = self._detect_clusters(illuminated)
        logger.info("%s: %d illuminated clusters found", chip, len(clusters))

        clusters = self._reject_small_clusters(clusters)
        clusters = self._merge_fragmented_clusters(clusters)
        clusters = self._reject_malformed_clusters(clusters)
        logger.info("%s: %d clusters survive curation", chip, len(clusters))

        expected = len(self.fibers) * self.norder[chip]
        if abs(len(clusters) - expected) > 1:
            raise ValueError(
                f"{chip}: {len(clusters)} traces detected, expected {expected}"
            )
        if len(clusters) != expected:
            logger.warning(
                "%s: %d traces detected, expected %d; the fiber names of the "
                "order at the clipped edge rest on it being clipped",
                chip,
                len(clusters),
                expected,
            )

        self._clusters[chip] = clusters
        self._metadata.pop(chip, None)
        self._trace_tables.pop(chip, None)
        if chip in self._profiles:
            self._profiles[chip].update(centers=None, good=None)
        return clusters

    def assign_trace_identities(self, chip):
        """Label every cluster with its fiber and order index.

        The labelled traces are cached as ``self._metadata[chip]``, which is
        where the fitting step reads them from; they are returned as well so the
        step can be driven on its own.
        """
        metadata = self._track_cluster_metadata(chip)
        metadata = self._assign_fiber_identities(chip, metadata)
        self._metadata[chip] = self._assign_order_indexes(chip, metadata)
        return self._metadata[chip]

    def fit_trace_polynomials(self, chip, poly_degree=None):
        """Fit a centerline through every identified trace of one CCD.

        ``poly_degree`` overrides the configured fit degree for this tracer,
        which the output's ``Coeff`` field count follows.

        A trace no cluster was detected for is marked 'missing'; a trace that
        has a cluster must fit, and raises if it cannot. A fitted trace is left
        'unknown' for validation to classify from its dispersion span.

        The fitted traces become the CCD's ``self._trace_tables`` entry, with
        their aperture edges and column bounds left unmeasured for the aperture
        step to fill in. What each trace was measured at the sampled columns -- its row
        centers and the mask of the samples this fit accepted -- is recorded on
        ``self._profiles[chip]``, row for row with the CCD's table rows. Those
        rows are returned as well so the step can be driven on its own.
        """
        if poly_degree is not None:
            self.poly_degree = poly_degree

        coefficient_fields = self._trace_fields(which="coeffs")
        sampled = self._sample_profiles(chip)
        columns = sampled["columns"]

        traces = self._metadata[chip]
        sampled["centers"] = np.full((len(traces), columns.size), np.nan)
        sampled["good"] = np.zeros((len(traces), columns.size), dtype=bool)

        records = []
        for position, trace in enumerate(traces.itertuples(index=False)):
            order, fiber, cluster_index = trace.Order, trace.Fiber, trace.cluster
            record = dict.fromkeys(
                self._trace_fields(which="geometry") + ["PolyfitRMS"], np.nan
            )
            record.update(
                {"Chip": chip, "Fiber": fiber, "Order": order, "Status": "unknown"}
            )
            records.append(record)

            if pd.isna(cluster_index):
                logger.warning(
                    "%s %s order %d: no cluster detected", chip, fiber, order
                )
                record["Status"] = "missing"
                continue

            centers = self._trace_centers(chip, fiber, order)
            try:
                coeffs, good, rms = robust_polyfit(
                    columns.astype(float), centers, int(self.poly_degree), full=True
                )
            except ValueError as error:
                raise ValueError(f"{chip} {fiber} order {order}: {error}") from error
            sampled["good"][position] = good

            record.update(dict(zip(coefficient_fields, coeffs, strict=True)))
            record["PolyfitRMS"] = rms

        self._trace_tables[chip] = pd.DataFrame(records, columns=self._trace_fields())
        return self._trace_tables[chip]

    def estimate_trace_apertures(self, chip):
        """Measure every fitted trace's aperture and validate the CCD's table.

        The aperture is the band of rows an extraction takes as belonging to a
        trace: the flux widths below and above its centerline, clamped where a
        neighbor's aperture would otherwise be reached. Every fitted trace must
        yield one; a width that cannot be measured raises, as a centerline with
        no aperture cannot be extracted from.

        Each width is written into ``self._trace_tables[chip]`` as it is
        measured, completing the CCD's rows; the table is returned as well so
        the step can be driven on its own.
        """
        table = self._trace_tables[chip]

        for trace in table.itertuples():
            if trace.Status == "missing":
                continue
            try:
                widths = self._trace_width(chip, trace.Fiber, trace.Order)
            except ValueError as error:
                raise ValueError(
                    f"{chip} {trace.Fiber} order {trace.Order}: {error}"
                ) from error
            table.loc[trace.Index, ["BottomEdge", "TopEdge"]] = widths

        self._clamp_neighboring_apertures(chip)
        self._validate_trace_table(chip)
        return table

    def save_master(self, path, *, overwrite=False):
        """Write the assembled order-trace CSV to ``path``.

        Writes ``self._output_table``, assembled by ``make_master``, which must
        have run first. Parent directories are created as needed.
        """
        if self._output_table is None:
            raise RuntimeError("No traces available; run make_master() first")
        if not overwrite and os.path.exists(path):
            raise FileExistsError(
                f"{path} already exists; pass overwrite=True to replace it"
            )

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._output_table.to_csv(path, index=False, lineterminator="\n")
        self._output_path = path

    # ------------------------------------------------------------------
    # Private helpers - module execution
    # ------------------------------------------------------------------

    def _track_info(self, chips):
        """Build and cache the interactive execution summary."""
        lines = [
            "OrderTrace",
            f"  master flat: {self.master_flat_filename}",
            f"  output:      {self._output_path or '(not written)'}",
            "",
            f"  {'chip':<8s} {'full':>6s} {'partial':>8s} {'missing':>8s} "
            f"{'median RMS [pix]':>18s}",
            "  " + "-" * 60,
        ]
        for chip in chips:
            traced = self._trace_tables[chip]
            status = traced["Status"]
            median_rms = np.nanmedian(traced["PolyfitRMS"])
            lines.append(
                f"  {chip:<8s} {(status == 'full').sum():>6d} "
                f"{(status == 'partial').sum():>8d} "
                f"{(status == 'missing').sum():>8d} "
                f"{median_rms:>18.4f}"
            )
        self._info = "\n\n" + "\n".join(lines) + "\n\n"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def make_master(self, chips=None, *, poly_degree=None, output_dir=None):
        """
        Build order-trace calibrations from the requested master-flat CCDs.

        Every CCD's traces are gathered into one table, keyed by a leading
        ``Chip`` field, and returned; pass ``output_dir`` to also persist it as
        a single ``{obs_id}_master_order_trace.csv``, the obs id inherited from
        the master flat.

        Parameters
        ----------
        chips : list of str, optional
            CCDs to trace. Defaults to configured GREEN and RED.
        poly_degree : int, optional
            Degree of the fitted centerline, which the output's ``Coeff`` field
            count follows. Defaults to the configured degree.
        output_dir : str or pathlib.Path, optional
            If provided, write ``{obs_id}_master_order_trace.csv`` into this
            directory, replacing any existing file. When omitted, the table is
            built and returned but nothing is written to disk.

        Returns
        -------
        pandas.DataFrame
            Every requested CCD's table, assembled in the order the CCDs were
            measured. One row per expected trace, with fields ``Chip``,
            ``Fiber``, ``Order``, ``Coeff0..N``, ``BottomEdge``, ``TopEdge``,
            ``X1``, ``X2``, ``PolyfitRMS``, ``Status``.

        Raises
        ------
        FileNotFoundError
            If the master flat is absent.
        ValueError
            If no CAL orderlet can be identified, or if the assembled output
            table is invalid.

        Notes
        -----
        Pipeline steps, repeated per CCD:
        1. Threshold each detector column against its own inter-order flux,
           giving a mask of illuminated pixels set by orderlet contrast rather
           than by the absolute lamp level
        2. Collect mask pixels that touch into clusters, taking all eight pixels
           of the surrounding 3x3 neighborhood as adjacent (8-connectivity)
        3. Reject clusters too small to be part of any trace
        4. Rejoin clusters that are separated pieces of one trace
        5. Reject clusters that cannot be identified at the detector center
        6. Flag the CAL orderlet of each order, then count downward in row from
           each CAL to label every cluster with its fiber and zero-based order
           index
        7. Measure a subpixel row center per sample column -- a winsorized
           centroid for the peaked CAL profile, an edge midpoint for the
           flat-topped SKY and science profiles -- and fit each trace's row
           against column with sigma-clipping
        8. Estimate aperture edges above and below each centerline, shrinking
           them where neighboring apertures would otherwise touch
        9. Validate each CCD's table, then write every CCD's traces to one
           ``{obs_id}_master_order_trace.csv``

        Steps 1-6 are pure detection and identification: identity is fixed from
        the illuminated-pixel mask and cluster brightness before step 7 fits
        anything. The order *index* assigned in step 6 is a first-pass estimate;
        it is correct whenever the detected orders fill the detector, and is
        counted from the lowest complete fiber group otherwise.
        """
        if chips is None:
            chips = self.chips

        for chip in chips:
            self.detect_traces(chip)
            self.assign_trace_identities(chip)
            self.fit_trace_polynomials(chip, poly_degree)
            self.estimate_trace_apertures(chip)

        self._output_table = pd.concat(
            [self._trace_tables[chip] for chip in chips], ignore_index=True
        )

        if output_dir is not None:
            # The master flat is {obs_id}_master_flat_L1.fits (WMKO DRP-RUN-05),
            # so the master order-trace is {obs_id}_master_order_trace.csv.
            flat_name = os.path.basename(self.master_flat_filename)
            obs_id = flat_name.split("_master_flat")[0]
            self.save_master(
                os.path.join(output_dir, f"{obs_id}_master_order_trace.csv"),
                overwrite=True,
            )
        self._track_info(chips)
        logger.info("%s", self._info)
        return self._output_table

    def info(self):
        """Print a summary of the module configuration and tracing results."""
        if self._info is None:
            print(f"{type(self).__name__}: make_master() has not been called")
        else:
            print(self._info)
