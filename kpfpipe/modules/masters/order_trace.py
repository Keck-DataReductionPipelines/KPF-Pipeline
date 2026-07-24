"""
KPF Order Trace module.

Locates the spectral traces on one completed vNext master flat -- where the
illuminated pixels lie on each detector -- and writes a low-order polynomial
describing every trace centerline, plus its aperture edges, to one CSV per CCD.

Trace geometry is measured from the master flat alone. No pre-existing
order-trace table is consulted, so the module never inherits the geometry it
exists to determine. Trace identity is established before any polynomial is
fitted: the CAL orderlet is both narrower and several times brighter than the
SKY and science orderlets, which fixes the phase of the repeating
SKY-SCI1-SCI2-SCI3-CAL group, and each CAL closes one echelle order. Because
each order is anchored on its own CAL, a trace missing anywhere in the frame
shifts no other trace's label.

Only the within-order orderlet spacing is treated as regular. The spacing
*between* orders varies several-fold across a detector and is never assumed.

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

So every trace centerline is ``row = f(column)``, and the ``X1``/``X2`` output
fields are the first and last *pixel column* over which that fit was measured.
"Column" always means a pixel column; the fields of an output table are called
fields, never columns.

A bare ``i`` is a single row index and a bare ``j`` a single column index, as
in ``image[i, j]``. Plural ``rows``/``columns`` are arrays of such indices, and
names like ``sample_columns`` or ``middle_column`` are column positions the
polynomials are evaluated at. Reserving ``i``/``j`` for the scalars keeps a
loop counter from ever being mistaken for the array it walks.
"""

import logging
import os
import tempfile

import numpy as np
import pandas as pd
from astropy.stats import mad_std
from scipy.linalg import solve_banded
from scipy.ndimage import gaussian_filter1d, label

from kpfpipe import DEFAULTS
from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.utils.config import ConfigHandler

logger = logging.getLogger(__name__)

_TRACE_GEOMETRY_FIELDS = [
    "BottomEdge",
    "TopEdge",
    "X1",
    "X2",
]
_TRACE_ID_FIELDS = [
    "Fiber",
    "Order",
    "Status",
]

_DEFAULTS = {
    **DEFAULTS,
    "poly_degree": 3,
}


class OrderTrace:
    """
    Measure spectral traces from one vNext KPF master flat and write trace CSVs.

    Operations include:
      - reducing each detector column to a mask of illuminated pixels
      - collecting touching mask pixels into clusters (8-connectivity)
      - curating clusters (rejecting artifacts, rejoining split traces)
      - identifying each cluster's fiber and echelle order index
      - fitting a polynomial centerline, row against column, to each trace
      - estimating aperture edges and enforcing the gap between neighbors

    Output always contains one row per expected trace (``norder`` x five
    fibers), ordered by order index then by fiber. A trace that was not
    detected, or whose fit failed, is written with NaN geometry and a ``Status``
    of 'missing'; one measured over only part of the detector is written
    'partial'; one spanning essentially the whole detector is written 'full'.
    A trace that cannot be measured is therefore reported, never silently
    dropped, and the row count is fixed regardless of what was found.

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

        self._master_flat = None
        self._image = {}
        self._fit_rms = {}
        self._trace_tables = None
        self._output_paths = {}
        self._info = None

    # ------------------------------------------------------------------
    # Private helpers - input
    # ------------------------------------------------------------------

    def _coefficient_fields(self):
        """Return output coefficient fields for the configured fit degree."""
        degree = int(self.poly_degree)
        if degree < 0:
            raise ValueError(f"poly_degree must be non-negative, got {degree!r}")
        return [f"Coeff{i}" for i in range(degree + 1)]

    def _trace_fields(self):
        """Return the full output field order for one trace table."""
        return self._coefficient_fields() + _TRACE_GEOMETRY_FIELDS + _TRACE_ID_FIELDS

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

        self._image = {chip: master_flat.data[f"{chip}_IMG"] for chip in self.chips}
        return master_flat

    # ------------------------------------------------------------------
    # Private helpers - trace detection
    # ------------------------------------------------------------------

    @staticmethod
    def _single_column_inter_order_flux(pixels, smoothing_weight):
        """Return the flux lying between the orders of one pixel column.

        ``pixels`` are raw master-flat values, straight off the detector.

        A Whittaker smoother: the result minimizes
        ``sum (f - pixels)**2 + smoothing_weight * sum (df/drow)**2``. Those
        normal equations couple each pixel only to its two neighbors, so they
        are the tridiagonal system solved here. Interior rows carry
        ``1 + 2 * smoothing_weight``; the two ends carry one multiple less,
        having one neighbor rather than two.

        The default weight halves a 28-pixel period and passes a third of a
        20-pixel one, so the solution rides over the orderlets (5-11 pixels
        thick, spaced 16-20) while still following the far broader
        order-to-order flux envelope.

        ``solve_banded((1, 1), bands, pixels)`` solves ``A f = pixels`` for f,
        the ``(1, 1)`` declaring one superdiagonal and one subdiagonal. With w
        standing for ``smoothing_weight``, A is tridiagonal and symmetric:

            [ 1+w   -w                    ]
            [  -w  1+2w   -w              ]
            [        -w  1+2w   -w        ]
            [              -w  1+2w   -w  ]
            [                    -w   1+w ]

        Only those three diagonals are stored, in LAPACK's banded layout
        ``bands[1 + i - j, j] = A[i, j]`` -- row 0 the superdiagonal, row 1 the
        main diagonal, row 2 the subdiagonal. That indexing shifts row 0 one
        place right and row 2 one place left, which is why ``bands[0, 0]`` and
        ``bands[2, -1]`` fall outside A and are never read. LAPACK then runs a
        banded LU with partial pivoting, costing O(nrow) at this bandwidth
        rather than the O(nrow**3) a dense solve of the same system would.
        """
        nrow = pixels.size
        off_diagonal = np.full(nrow, -smoothing_weight)
        diagonal = np.full(nrow, 1.0 + 2.0 * smoothing_weight)
        diagonal[[0, -1]] = 1.0 + smoothing_weight

        # The matrix is symmetric, so one off-diagonal serves above and below.
        # Banded storage never reads the two corner entries zeroed here.
        bands = np.vstack([off_diagonal, diagonal, off_diagonal])
        bands[0, 0] = bands[2, -1] = 0.0
        return solve_banded((1, 1), bands, pixels)

    def _detect_illuminated_pixels(self, chip, smoothing_weight=20.0, trace_ratio=0.5):
        """Return a boolean mask of illuminated pixels, one column at a time.

        Each pixel column is reduced independently: no information crosses
        between columns, so the threshold is local along dispersion.
        """
        nrow, ncol = self.ccd["nrow"], self.ccd["ncol"]
        filled = np.nan_to_num(self._image[chip], nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.zeros((nrow, ncol), dtype=bool)
        for j in range(ncol):
            # Subtracting the inter-order flux leaves the orderlet peaks
            # standing alone, so the threshold below is set by orderlet
            # contrast rather than by the absolute lamp level, which varies by
            # an order of magnitude across the detector.
            column_pixels = filled[:, j]
            residual = column_pixels - self._single_column_inter_order_flux(
                column_pixels, smoothing_weight
            )
            positive_residual = np.clip(residual, 0.0, None)
            threshold = 0.5 * np.quantile(
                positive_residual, trace_ratio, method="lower"
            )
            mask[:, j] = residual > threshold + 1.0
        return mask

    @staticmethod
    def _cluster_record(rows, columns):
        """Bundle one cluster's pixels with the geometry used to curate it.

        ``rows`` and ``columns`` are the cross-dispersion and dispersion pixel
        indices of every pixel in the cluster. ``x1``/``x2`` are its first and
        last pixel column, matching the output fields of the same name, and
        ``row_at_x1``/``row_at_x2`` are where it sits across dispersion there.
        """
        first_j, last_j = int(columns.min()), int(columns.max())
        return {
            "rows": rows,
            "columns": columns,
            "npixel": int(rows.size),
            "x1": first_j,
            "x2": last_j,
            "row_at_x1": float(rows[columns == first_j].mean()),
            "row_at_x2": float(rows[columns == last_j].mean()),
        }

    def _label_clusters(self, mask):
        """Collect touching illuminated pixels into clusters.

        The 3x3 structure treats all eight surrounding pixels as adjacent
        (8-connectivity), so pixels meeting only at a corner still join one
        cluster.
        """
        labels, cluster_count = label(mask, structure=np.ones((3, 3), dtype=int))
        rows, columns = np.nonzero(labels)
        cluster_ids = labels[rows, columns]

        ordering = np.argsort(cluster_ids, kind="stable")
        rows, columns = rows[ordering], columns[ordering]
        # One contiguous run of pixels per cluster id, so a single searchsorted
        # gives every cluster's slice without revisiting the label image.
        bounds = np.searchsorted(cluster_ids[ordering], np.arange(1, cluster_count + 2))
        return [
            self._cluster_record(rows[start:stop], columns[start:stop])
            for start, stop in zip(bounds[:-1], bounds[1:], strict=True)
        ]

    def _center_column_band(self):
        """Return the pixel columns over which trace identity is measured.

        A narrow band at mid-dispersion, where every order is on the detector
        and the orderlets of an order are cleanly separated across dispersion.
        """
        ncol = self.ccd["ncol"]
        half_width = max(1, ncol // 200)
        center = ncol // 2
        return center - half_width, center + half_width + 1

    @staticmethod
    def _band_pixels(cluster, band_start, band_stop):
        """Return one cluster's pixel rows and columns inside a column band."""
        in_band = (cluster["columns"] >= band_start) & (cluster["columns"] < band_stop)
        return cluster["rows"][in_band], cluster["columns"][in_band]

    @staticmethod
    def _log_rejection(chip, cluster, reason):
        """Record one curated-away cluster and why it was discarded."""
        logger.debug(
            "%s: rejecting cluster at rows %d-%d, columns %d-%d (%s)",
            chip,
            cluster["rows"].min(),
            cluster["rows"].max(),
            cluster["x1"],
            cluster["x2"],
            reason,
        )

    def _reject_small_clusters(self, chip, clusters, min_cluster_pixels=500):
        """Drop clusters too small to be any part of a trace.

        Runs before fragments are rejoined, so it tests only pixel count: a
        genuine fragment is short and may sit anywhere on the detector.
        """
        kept = []
        for cluster in clusters:
            if cluster["npixel"] < min_cluster_pixels:
                self._log_rejection(chip, cluster, f"only {cluster['npixel']} pixels")
            else:
                kept.append(cluster)
        return kept

    def _reject_unidentifiable_clusters(
        self, chip, clusters, min_spanned_columns=200, min_thickness=3.0
    ):
        """Drop rejoined clusters that cannot be identified as traces."""
        band_start, band_stop = self._center_column_band()
        kept = []
        for cluster in clusters:
            spanned_columns = cluster["x2"] - cluster["x1"] + 1
            rows, columns = self._band_pixels(cluster, band_start, band_stop)
            # Cross-dispersion thickness: rows occupied per column, measured in
            # the same band that later fixes trace identity. A detector-edge
            # artifact running along a single row is therefore rejected even
            # when its full-frame pixel count looks trace-like.
            thickness = rows.size / np.unique(columns).size if rows.size else 0.0
            if spanned_columns < min_spanned_columns:
                reason = f"spans only {spanned_columns} pixel columns"
            elif rows.size == 0:
                reason = "no pixels in the central column band"
            elif thickness < min_thickness:
                reason = f"only {thickness:.1f} pixel rows thick at mid-detector"
            else:
                kept.append(cluster)
                continue
            self._log_rejection(chip, cluster, reason)
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

    def _bridges_gap(self, host, candidate, gap, max_residual):
        """Test whether a candidate cluster continues a host across a column gap.

        Compares one mean row per column rather than individual pixels, whose
        scatter is the trace's cross-dispersion thickness and would swamp the
        misalignment being measured. Only pixels within one gap width of either
        side are used and the fit is linear: the test asks whether the pieces
        line up locally, over which a trace's curvature is negligible.
        """
        host_near_gap = host["columns"] >= host["x2"] - gap
        candidate_near_gap = candidate["columns"] <= candidate["x1"] + gap
        if not host_near_gap.any() or not candidate_near_gap.any():
            return False

        host_columns, host_mean_rows = self._mean_row_per_column(
            host["rows"][host_near_gap], host["columns"][host_near_gap]
        )
        candidate_columns, candidate_mean_rows = self._mean_row_per_column(
            candidate["rows"][candidate_near_gap],
            candidate["columns"][candidate_near_gap],
        )
        if host_columns.size < 2:
            return False

        coeffs = np.polynomial.polynomial.polyfit(host_columns, host_mean_rows, 1)
        predicted = np.polynomial.polynomial.polyval(candidate_columns, coeffs)
        residual = np.median(np.abs(candidate_mean_rows - predicted))
        return residual <= max_residual

    def _find_mergeable_pair(
        self,
        clusters,
        max_gap_columns=512,
        max_row_offset=10.0,
        # Cluster edges are quantized to whole pixels, which puts a floor of
        # about one pixel on how well two fragments can be shown to line up.
        # Rival traces are a whole orderlet spacing apart, so two pixels of
        # slack stays far from ambiguous.
        max_residual_pixels=2.0,
    ):
        """Return the indices of the first two clusters lying on one curve."""
        for host_index, host in enumerate(clusters):
            for candidate_index, candidate in enumerate(clusters):
                gap = candidate["x1"] - host["x2"]
                if not 0 < gap <= max_gap_columns:
                    continue
                if abs(candidate["row_at_x1"] - host["row_at_x2"]) > max_row_offset:
                    continue
                if self._bridges_gap(host, candidate, gap, max_residual_pixels):
                    return host_index, candidate_index
        return None

    def _merge_fragmented_clusters(self, chip, clusters):
        """Rejoin clusters that are separated pieces of a single trace."""
        clusters = list(clusters)
        while True:
            pair = self._find_mergeable_pair(clusters)
            if pair is None:
                return clusters
            host, candidate = (clusters[index] for index in pair)
            logger.info(
                "%s: merging cluster fragments spanning columns %d-%d and %d-%d",
                chip,
                host["x1"],
                host["x2"],
                candidate["x1"],
                candidate["x2"],
            )
            combined = self._cluster_record(
                np.concatenate([host["rows"], candidate["rows"]]),
                np.concatenate([host["columns"], candidate["columns"]]),
            )
            clusters = [
                cluster for index, cluster in enumerate(clusters) if index not in pair
            ]
            clusters.append(combined)

    # ------------------------------------------------------------------
    # Private helpers - trace identity
    # ------------------------------------------------------------------

    def _cluster_center_metrics(self, chip, clusters):
        """Measure row, mask thickness, and flux for each cluster at mid-detector.

        The row is the cluster's cross-dispersion position there, and is what
        the clusters are subsequently ordered by: sorting on it walks the
        orderlets of the detector in the order they physically appear.
        """
        band_start, band_stop = self._center_column_band()
        records = []
        for index, cluster in enumerate(clusters):
            rows, columns = self._band_pixels(cluster, band_start, band_stop)
            records.append(
                {
                    "cluster": index,
                    "row": float(rows.mean()),
                    "thickness": rows.size / np.unique(columns).size,
                    "flux": float(np.median(self._image[chip][rows, columns])),
                }
            )
        metrics = pd.DataFrame(records)
        return metrics.sort_values("row").reset_index(drop=True)

    def _flag_cal_clusters(self, metrics, max_thickness=6.0, min_flux_ratio=1.8):
        """Flag CAL orderlets: thinner and brighter than the orderlets around them.

        Brightness is judged against the mean of the two clusters bracketing
        each candidate. Lamp flux varies by an order of magnitude across the
        detector, and bracketing cancels that gradient where a running median
        of the neighborhood does not.
        """
        flux = metrics["flux"].to_numpy()
        flux_below = np.concatenate([flux[1:2], flux[:-1]])
        flux_above = np.concatenate([flux[1:], flux[-2:-1]])
        return (metrics["thickness"].to_numpy() <= max_thickness) & (
            flux / ((flux_below + flux_above) / 2.0) > min_flux_ratio
        )

    def _orderlet_spacing(self, metrics, is_cal):
        """Return the median cross-dispersion step between orderlets of an order."""
        row_steps = np.diff(metrics["row"].to_numpy())
        if row_steps.size == 0:
            raise ValueError("a single cluster cannot fix the orderlet spacing")

        # A step that starts on a CAL crosses into the next order.
        within_order = row_steps[~is_cal[:-1]]
        if within_order.size:
            return float(np.median(within_order))
        return float(np.median(row_steps))

    def _assign_fiber_positions(self, chip, metrics, is_cal):
        """Give every cluster its position in the fiber pattern and its order group.

        Each CAL closes an order, so positions are counted downward from a CAL
        rather than propagated across the whole frame. Only the within-order
        orderlet spacing is used to decide whether the next cluster down is the
        adjacent orderlet or whether one was missed; the spacing *between*
        orders varies several-fold across the detector and is never assumed.
        """
        cluster_rows = metrics["row"].to_numpy()
        cal_position = len(self.fibers) - 1
        spacing = self._orderlet_spacing(metrics, is_cal)
        cal_indices = np.flatnonzero(is_cal)
        if cal_indices.size == 0:
            raise ValueError(
                f"{chip}: no CAL orderlet identified; cannot phase the fiber pattern"
            )

        positions = np.full(len(metrics), -1)
        groups = np.zeros(len(metrics), dtype=int)
        for group, cal_index in enumerate(cal_indices):
            positions[cal_index] = cal_position
            groups[cal_index] = group

            position, row = cal_position, cluster_rows[cal_index]
            previous_cal = cal_indices[group - 1] + 1 if group else 0
            for index in range(cal_index - 1, previous_cal - 1, -1):
                position -= max(1, round((row - cluster_rows[index]) / spacing))
                if position < 0:
                    break
                positions[index] = position
                groups[index] = group
                row = cluster_rows[index]

        # An order whose CAL lies off the detector keeps only its lower
        # orderlets, which are then in rank order from SKY upward.
        for position, index in enumerate(range(cal_indices[-1] + 1, len(metrics))):
            if position > cal_position:
                break
            positions[index] = position
            groups[index] = cal_indices.size

        for index in np.flatnonzero(positions < 0):
            logger.warning(
                "%s: discarding a cluster at row %.0f that fits no orderlet of the "
                "fiber pattern",
                chip,
                cluster_rows[index],
            )

        metrics = metrics.copy()
        metrics["Fiber"] = [self.fibers[max(p, 0)] for p in positions]
        metrics["group"] = groups
        return metrics[positions >= 0]

    def _drop_edge_groups(self, chip, metrics, norder):
        """Discard partial fiber groups beyond the expected number of orders.

        An order lying mostly off the detector still shows its innermost
        orderlets, which phase correctly but belong to no expected order. Such a
        group is short, and always at one edge or the other.
        """
        group_sizes = metrics["group"].value_counts().sort_index()
        while group_sizes.size > norder:
            edge_groups = [group_sizes.index[0], group_sizes.index[-1]]
            discarded = min(edge_groups, key=lambda group: group_sizes[group])
            logger.warning(
                "%s: %d fiber groups detected but %d orders expected; discarding a "
                "group of %d orderlets at row %.0f",
                chip,
                group_sizes.size,
                norder,
                group_sizes[discarded],
                metrics.loc[metrics["group"] == discarded, "row"].mean(),
            )
            metrics = metrics[metrics["group"] != discarded]
            group_sizes = group_sizes.drop(discarded)
        return metrics

    @staticmethod
    def _lowest_order_index(group_rows, norder):
        """Best-guess count of whole orders falling below the lowest group."""
        if group_rows.size >= norder:
            return 0
        group_spacing = np.median(np.diff(group_rows))
        orders_below = int(max(0.0, group_rows[0]) // group_spacing)
        return min(orders_below, norder - group_rows.size)

    def _assign_trace_identity(self, chip, clusters):
        """Label every cluster with its fiber and order index."""
        fibers = list(self.fibers)
        norder = self.norder[chip]

        metrics = self._cluster_center_metrics(chip, clusters)
        is_cal = self._flag_cal_clusters(metrics)
        metrics = self._assign_fiber_positions(chip, metrics, is_cal)
        metrics = self._drop_edge_groups(chip, metrics, norder)

        group_rows = metrics.groupby("group")["row"].mean().to_numpy()
        lowest_order = self._lowest_order_index(group_rows, norder)
        metrics["Order"] = metrics["group"] - metrics["group"].min() + lowest_order

        beyond_last_order = metrics["Order"] >= norder
        if beyond_last_order.any():
            logger.warning(
                "%s: discarding %d orderlets labelled beyond order %d",
                chip,
                int(beyond_last_order.sum()),
                norder - 1,
            )
            metrics = metrics[~beyond_last_order]

        cluster_by_trace = metrics.set_index(["Order", "Fiber"])["cluster"]
        if cluster_by_trace.index.has_duplicates:
            raise ValueError(f"{chip}: fiber phasing produced duplicate trace labels")

        expected_traces = pd.MultiIndex.from_product(
            [range(norder), fibers], names=["Order", "Fiber"]
        )
        return cluster_by_trace.reindex(expected_traces)

    # ------------------------------------------------------------------
    # Private helpers - trace measurement
    # ------------------------------------------------------------------

    def _sample_columns(self, sample_count=65):
        """Return the pixel columns each trace is measured at, both edges included.

        A centerline is fitted through one measurement per sample column, so
        these are the dispersion positions the polynomial is constrained at.
        """
        return np.unique(np.linspace(0, self.ccd["ncol"] - 1, sample_count, dtype=int))

    def _sample_column_profiles(self, chip, sample_columns, column_half_window=3):
        """Return the cross-dispersion profile at each sample column, keyed by column.

        Profiles rather than raw pixels: neighboring columns are median
        combined, which suppresses noise without smearing, because a trace
        moves only slightly across dispersion over so short a run along it.

        Every trace on the CCD is measured at these same few dozen columns, so
        all the profiles are built once here instead of being recomputed by
        each trace in turn.

        A window overhanging a detector edge is padded with NaN, which
        ``nanmedian`` then ignores, giving exactly the truncated median a
        clamped per-column slice would.
        """
        nrow, ncol = self.ccd["nrow"], self.ccd["ncol"]
        image = self._image[chip]
        offsets = np.arange(-column_half_window, column_half_window + 1)
        window_columns = sample_columns[:, None] + offsets[None, :]
        on_detector = (window_columns >= 0) & (window_columns < ncol)

        windows = np.full((nrow, *window_columns.shape), np.nan, dtype=image.dtype)
        windows[:, on_detector] = image[:, window_columns[on_detector]]
        profiles = np.nanmedian(windows, axis=2)

        # Fill non-finite rows with the profile's own median, as a wholly
        # unmeasurable column has no scale of its own to fall back on.
        finite = np.isfinite(profiles)
        fill = np.zeros(sample_columns.size)
        measured = finite.any(axis=0)
        fill[measured] = np.nanmedian(
            np.where(finite, profiles, np.nan)[:, measured], axis=0
        )
        profiles = np.where(finite, profiles, fill[None, :]).astype(float, copy=False)

        return dict(
            zip(sample_columns.tolist(), np.ascontiguousarray(profiles.T), strict=True)
        )

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

    @staticmethod
    def _threshold_crossing(rows, signal, core_index, direction, threshold):
        """Interpolate one threshold crossing away from an illuminated core."""
        current = int(core_index)
        adjacent = current + int(direction)
        while 0 <= adjacent < signal.size and signal[adjacent] >= threshold:
            current = adjacent
            adjacent = current + int(direction)
        if adjacent < 0 or adjacent >= signal.size:
            return np.nan

        value_inside = signal[current]
        value_outside = signal[adjacent]
        denominator = value_inside - value_outside
        if not np.isfinite(denominator) or denominator <= 0.0:
            return np.nan
        fraction = (value_inside - threshold) / denominator
        return float(rows[current] + fraction * (rows[adjacent] - rows[current]))

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
            illuminated = np.flatnonzero(signal >= threshold)
            if illuminated.size == 0:
                continue
            core_index = illuminated[np.argmin(np.abs(illuminated - row_guess_index))]
            bottom = self._threshold_crossing(rows, signal, core_index, -1, threshold)
            top = self._threshold_crossing(rows, signal, core_index, 1, threshold)
            width = top - bottom
            if (
                np.isfinite(bottom)
                and np.isfinite(top)
                and width >= edge_min_width_pixels
            ):
                centers.append((bottom + top) / 2.0)

        min_levels = max(2, int(np.ceil(len(edge_levels) / 2)))
        if len(centers) < min_levels:
            return np.nan
        centers = np.asarray(centers, dtype=float)
        if mad_std(centers, ignore_nan=True) > edge_max_center_spread:
            return np.nan
        return float(np.nanmedian(centers))

    def _trace_centers(self, column_profiles, cluster, sample_columns, fiber):
        """Measure a subpixel row center at every sample column of one cluster.

        The cluster's own mask pixels at a column give the starting row, so no
        prior trace geometry is needed. The CAL orderlet is peaked and centers
        well on its brightest pixels; SKY and the science orderlets are
        flat-topped, with no meaningful peak, and are centered from their two
        cross-dispersion edges instead.
        """
        measure_center = (
            self._local_peak_center if fiber == "CAL" else self._local_edge_center
        )
        centers = np.full(sample_columns.size, np.nan, dtype=float)
        for sample, j in enumerate(sample_columns):
            in_column = cluster["columns"] == j
            if not in_column.any():
                continue
            row_guess = float(cluster["rows"][in_column].mean())
            centers[sample] = measure_center(column_profiles[int(j)], row_guess)
        return centers

    def _robust_polynomial_fit(
        self,
        sample_columns,
        centers,
        min_valid_fraction=0.5,
        fit_max_iterations=8,
        fit_sigma=4.0,
    ):
        """Fit row center against pixel column, rejecting outliers by median/MAD."""
        degree = int(self.poly_degree)
        kept = np.isfinite(sample_columns) & np.isfinite(centers)
        min_kept = max(
            degree + 1, int(np.ceil(sample_columns.size * min_valid_fraction))
        )
        if kept.sum() < min_kept:
            raise ValueError(
                f"only {kept.sum()} of {sample_columns.size} trace centers are "
                f"valid; at least {min_kept} are required"
            )

        for _ in range(fit_max_iterations):
            coeffs = np.polynomial.polynomial.polyfit(
                sample_columns[kept], centers[kept], degree
            )
            residual = centers - np.polynomial.polynomial.polyval(
                sample_columns, coeffs
            )
            median_residual = np.nanmedian(residual[kept])
            residual_scatter = mad_std(residual[kept], ignore_nan=True)
            rejection_limit = max(0.25, fit_sigma * residual_scatter)
            still_valid = np.isfinite(centers) & (
                np.abs(residual - median_residual) <= rejection_limit
            )
            if still_valid.sum() < min_kept:
                break
            if np.array_equal(still_valid, kept):
                kept = still_valid
                break
            kept = still_valid

        coeffs = np.polynomial.polynomial.polyfit(
            sample_columns[kept], centers[kept], degree
        )
        residual = centers[kept] - np.polynomial.polynomial.polyval(
            sample_columns[kept], coeffs
        )
        rms = float(np.sqrt(np.mean(residual**2)))
        return coeffs, kept, rms

    def _width_at_column(
        self,
        column_profile,
        center,
        width_half_window=14,
        winsor_percentile=90.0,
        width_sigma=2.8,
    ):
        """Estimate the aperture's half-widths below and above one trace sample.

        Below and above are cross-dispersion, so the two widths become the
        ``BottomEdge`` and ``TopEdge`` output fields.
        """
        first_i = max(0, int(np.floor(center)) - width_half_window)
        last_i = min(column_profile.size, int(np.ceil(center)) + width_half_window + 1)
        if last_i - first_i < 5:
            return np.nan, np.nan

        profile = column_profile[first_i:last_i]
        rows = np.arange(first_i, last_i, dtype=float)
        signal = np.clip(profile - np.nanpercentile(profile, 20.0), 0.0, None)
        if not np.any(signal > 0):
            return np.nan, np.nan
        signal = np.minimum(signal, np.nanpercentile(signal, winsor_percentile))
        offsets = rows - center

        widths = []
        for half in (offsets <= 0, offsets >= 0):
            weights = signal[half]
            distance = np.abs(offsets[half])
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
        self,
        column_profiles,
        coeffs,
        sample_columns,
        kept,
        width_half_window=14,
        width_default=11.0,
    ):
        """Return robust bottom/top aperture widths from accepted samples."""
        bottom_widths = []
        top_widths = []
        kept_columns = sample_columns[kept]
        stride = max(1, kept_columns.size // 24)
        for j in kept_columns[::stride]:
            center = np.polynomial.polynomial.polyval(j, coeffs)
            bottom, top = self._width_at_column(column_profiles[int(j)], center)
            if np.isfinite(bottom) and bottom > 0:
                bottom_widths.append(bottom)
            if np.isfinite(top) and top > 0:
                top_widths.append(top)

        if len(bottom_widths) < 3 or len(top_widths) < 3:
            raise ValueError(
                "fewer than three valid samples for trace-width estimation"
            )
        width_ceiling = min(width_half_window, width_default)
        bottom = float(np.clip(np.nanmedian(bottom_widths), 1.0, width_ceiling))
        top = float(np.clip(np.nanmedian(top_widths), 1.0, width_ceiling))
        return bottom, top

    def _constrain_neighbor_widths(self, records, orderlet_gap_pixels=2.0):
        """Keep adjacent apertures separated by the required orderlet gap.

        Neighbors are adjacent across dispersion, which is why ``records`` must
        arrive ordered by order index then fiber -- that is ascending row.
        """
        coefficient_fields = self._coefficient_fields()
        middle_column = (self.ccd["ncol"] - 1) / 2.0
        for below, above in zip(records[:-1], records[1:], strict=False):
            below_center = np.polynomial.polynomial.polyval(
                middle_column, [below[field] for field in coefficient_fields]
            )
            above_center = np.polynomial.polynomial.polyval(
                middle_column, [above[field] for field in coefficient_fields]
            )
            available = (above_center - below_center) - orderlet_gap_pixels
            requested = below["TopEdge"] + above["BottomEdge"]
            if available <= 0:
                raise ValueError(
                    "neighboring fitted traces cross or have no aperture gap"
                )
            if requested > available:
                shrink = available / requested
                below["TopEdge"] *= shrink
                above["BottomEdge"] *= shrink

    def _validate_trace_table(self, chip, table, row_half_window=7):
        """Validate output schema, geometry, labels, and detector coverage."""
        nrow, ncol = self.ccd["nrow"], self.ccd["ncol"]
        coefficient_fields = self._coefficient_fields()
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
        if measured.empty:
            raise ValueError(f"{chip} produced no measured traces")

        geometry = measured[coefficient_fields + _TRACE_GEOMETRY_FIELDS].to_numpy(
            dtype=float
        )
        if not np.isfinite(geometry).all():
            raise ValueError(f"{chip} output contains non-finite measured geometry")
        if not ((measured["BottomEdge"] > 0) & (measured["TopEdge"] > 0)).all():
            raise ValueError(f"{chip} output contains non-positive widths")
        if not ((measured["X1"] >= 0) & (measured["X2"] < ncol)).all():
            raise ValueError(f"{chip} output contains out-of-range pixel columns")
        if not (measured["X1"] <= measured["X2"]).all():
            raise ValueError(f"{chip} output contains reversed pixel columns")

        # Traces are written in ascending row, so evaluating every centerline
        # at the same columns must give rows that strictly increase down each
        # column; anything else means two traces cross or are mislabelled.
        test_columns = np.linspace(0, ncol - 1, 17)
        coeffs = measured[coefficient_fields].to_numpy(dtype=float)
        centers = np.array(
            [np.polynomial.polynomial.polyval(test_columns, coeff) for coeff in coeffs]
        )
        if np.any(np.diff(centers, axis=0) <= 0):
            raise ValueError(f"{chip} fitted traces cross or are out of detector order")
        on_detector = (centers >= -row_half_window) & (
            centers <= nrow - 1 + row_half_window
        )
        if not on_detector.any(axis=1).all():
            raise ValueError(f"{chip} output contains a wholly off-detector trace")

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def _detect_traces(self, chip):
        """Detect, curate, and identify every trace on one CCD."""
        mask = self._detect_illuminated_pixels(chip)
        clusters = self._label_clusters(mask)
        logger.info("%s: %d illuminated clusters found", chip, len(clusters))

        clusters = self._reject_small_clusters(chip, clusters)
        clusters = self._merge_fragmented_clusters(chip, clusters)
        clusters = self._reject_unidentifiable_clusters(chip, clusters)
        logger.info("%s: %d clusters survive curation", chip, len(clusters))

        return clusters, self._assign_trace_identity(chip, clusters)

    def _measure_traces(self, chip, clusters, identities, full_coverage_fraction=0.9):
        """Fit every identified trace and assemble the output table for one CCD.

        Coverage is the fraction of the dispersion direction a trace was
        successfully measured over, and is what separates 'full' from 'partial'.
        """
        ncol = self.ccd["ncol"]
        coefficient_fields = self._coefficient_fields()
        sample_columns = self._sample_columns()
        column_profiles = self._sample_column_profiles(chip, sample_columns)

        records = []
        rms_values = []
        for (order, fiber), cluster_index in identities.items():
            record = dict.fromkeys(coefficient_fields + _TRACE_GEOMETRY_FIELDS, np.nan)
            record.update({"Fiber": fiber, "Order": order, "Status": "missing"})

            if pd.isna(cluster_index):
                logger.warning(
                    "%s %s order %d: no cluster detected", chip, fiber, order
                )
                records.append(record)
                continue

            cluster = clusters[int(cluster_index)]
            centers = self._trace_centers(
                column_profiles, cluster, sample_columns, fiber
            )
            try:
                coeffs, kept, rms = self._robust_polynomial_fit(
                    sample_columns.astype(float), centers
                )
                bottom, top = self._estimate_widths(
                    column_profiles, coeffs, sample_columns.astype(float), kept
                )
            except ValueError as error:
                logger.warning("%s %s order %d: %s", chip, fiber, order, error)
                records.append(record)
                continue

            kept_columns = sample_columns[kept]
            coverage = (kept_columns.max() - kept_columns.min() + 1) / ncol
            record.update(dict(zip(coefficient_fields, coeffs, strict=True)))
            record.update(
                {
                    "BottomEdge": bottom,
                    "TopEdge": top,
                    "X1": float(kept_columns.min()),
                    "X2": float(kept_columns.max()),
                    "Status": "full"
                    if coverage >= full_coverage_fraction
                    else "partial",
                }
            )
            records.append(record)
            rms_values.append(rms)

        measured = [record for record in records if record["Status"] != "missing"]
        self._constrain_neighbor_widths(measured)

        table = pd.DataFrame(records, columns=self._trace_fields())
        self._validate_trace_table(chip, table)
        self._fit_rms[chip] = np.asarray(rms_values, dtype=float)
        return table

    def _write_results(self, tables, output_dir, overwrite):
        """Stage and atomically install all requested CSV outputs."""
        os.makedirs(output_dir, exist_ok=True)
        final_paths = {
            chip: os.path.join(output_dir, f"order_trace_{chip.lower()}.csv")
            for chip in tables
        }
        existing = [path for path in final_paths.values() if os.path.exists(path)]
        if existing and not overwrite:
            raise FileExistsError(f"Order-trace output already exists: {existing}")

        staged_paths = {}
        try:
            for chip, table in tables.items():
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".csv.tmp",
                    prefix=f"order_trace_{chip.lower()}_",
                    dir=output_dir,
                    delete=False,
                ) as stream:
                    staged_paths[chip] = stream.name
                    table.to_csv(stream, lineterminator="\n")

            for chip, path in final_paths.items():
                os.replace(staged_paths[chip], path)
                staged_paths.pop(chip)
        finally:
            for path in staged_paths.values():
                if os.path.exists(path):
                    os.remove(path)

        self._output_paths = final_paths

    # ------------------------------------------------------------------
    # Private helpers - module execution
    # ------------------------------------------------------------------

    def _track_info(self, chips):
        """Build and cache the interactive execution summary."""
        lines = [
            "OrderTrace",
            f"  master flat: {self.master_flat_filename}",
            "",
            f"  {'chip':<8s} {'full':>6s} {'partial':>8s} {'missing':>8s} "
            f"{'median RMS [pix]':>18s}  output",
            "  " + "-" * 96,
        ]
        for chip in chips:
            status = self._trace_tables[chip]["Status"]
            median_rms = np.nanmedian(self._fit_rms[chip])
            lines.append(
                f"  {chip:<8s} {(status == 'full').sum():>6d} "
                f"{(status == 'partial').sum():>8d} "
                f"{(status == 'missing').sum():>8d} "
                f"{median_rms:>18.4f}  {self._output_paths[chip]}"
            )
        self._info = "\n\n" + "\n".join(lines) + "\n\n"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def make_master_order_trace(self, chips=None, *, output_dir, overwrite=False):
        """
        Build order-trace calibrations from the requested master-flat CCDs.

        Parameters
        ----------
        chips : list of str, optional
            CCDs to trace. Defaults to configured GREEN and RED.
        output_dir : str or pathlib.Path
            Directory receiving ``order_trace_<chip>.csv``. Required.
        overwrite : bool
            Replace existing output CSVs when True. Defaults to False.

        Returns
        -------
        dict
            Mapping of uppercase chip name to the DataFrame written for it.
            Each table holds one row per expected trace, ordered by order index
            then by fiber.

        Raises
        ------
        FileNotFoundError
            If the master flat is absent.
        ValueError
            If no CAL orderlet can be identified, or if the assembled output
            table is invalid.
        FileExistsError
            If a requested output exists and ``overwrite`` is False.

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
        9. Validate the assembled table and write ``order_trace_<chip>.csv``

        Steps 1-6 are pure detection and identification: identity is fixed from
        the illuminated-pixel mask and cluster brightness before step 7 fits
        anything. The order *index* assigned in step 6 is a first-pass estimate;
        it is correct whenever the detected orders fill the detector, and is
        counted from the lowest complete fiber group otherwise.
        """
        if chips is None:
            chips = self.chips

        self._master_flat = self._load_master_flat()

        tables = {}
        for chip in chips:
            clusters, identities = self._detect_traces(chip)
            tables[chip] = self._measure_traces(chip, clusters, identities)

        self._trace_tables = tables
        self._write_results(tables, output_dir, bool(overwrite))
        self._track_info(chips)
        logger.info("%s", self._info)
        return tables

    def info(self):
        """Print a summary of the module configuration and tracing results."""
        if self._info is None:
            print(
                f"{type(self).__name__}: make_master_order_trace() has not been called"
            )
        else:
            print(self._info)
