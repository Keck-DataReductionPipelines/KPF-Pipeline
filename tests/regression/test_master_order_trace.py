"""Tests for spectral order tracing from one vNext L1 master flat.

Numerical tests use synthetic master images and require no real data. The
synthetic flat reproduces the morphology the algorithm depends on: science and
sky orderlets are wide and flat-topped, and the CAL orderlet is narrow, which
is the difference identity rests on. The real master-flat test uses gitignored
``tests/testdata`` and is skipped when the vNext product is unavailable.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import label

import kpfpipe.modules.masters.order_trace as order_trace_module
from kpfpipe.modules.masters import OrderTrace
from kpfpipe.utils.config import ConfigHandler

_FIBERS = ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]
_TRACE_FIELDS = [
    "Chip",
    "Fiber",
    "Order",
    "Coeff0",
    "Coeff1",
    "Coeff2",
    "Coeff3",
    "BottomEdge",
    "TopEdge",
    "X1",
    "X2",
    "PolyfitRMS",
    "Status",
]

# Rows between orderlets inside one order, and between one order's CAL and the
# next order's SKY. The real inter-order gap is the wider of the two for all but
# a few orders at one end of a detector, where it narrows past the orderlet
# spacing; the narrow case is the one spacing cannot phase the fiber pattern
# from, so it is the one every synthetic order is built with.
_ORDERLET_SPACING = 19.0
_ORDER_GAP = 15.0


# ---------------------------------------------------------------------------
# Stubs and synthetic data
# ---------------------------------------------------------------------------


class StubMasterFlat:
    """Minimal vNext master flat used by the numerical tests."""

    def __init__(self, images, master_type="flat"):
        self.data = {}
        for chip, image in images.items():
            self.data[f"{chip}_IMG"] = np.asarray(image, dtype=np.float32)
        self.headers = {"PRIMARY": {"MASTYPE": master_type}}


def _stub_master_class(master_flat):
    """Return a KPFMasterL1 stand-in whose from_fits yields ``master_flat``."""

    class FakeKPFMasterL1:
        @classmethod
        def from_fits(cls, filename):
            return master_flat

    return FakeKPFMasterL1


def _orderlet_profile(offsets, fiber):
    """Return the cross-dispersion profile of one orderlet."""
    if fiber == "CAL":
        return 1500.0 * np.exp(-0.5 * (offsets / 1.8) ** 2)
    return 500.0 * np.exp(-0.5 * (offsets / 5.5) ** 6)


def _trace_labels(norder):
    """Return the (order, fiber) label of every trace, bottom to top."""
    return [(order, fiber) for order in range(norder) for fiber in _FIBERS]


def _labels(identities):
    """Return the (order index, fiber) label of every identified trace."""
    return list(zip(identities["Order"], identities["Fiber"], strict=True))


def _synthetic_flat(norder=3, ncol=400, drop=(), seed=0, converge=0.0):
    """Return a synthetic master-flat image and its exact trace centers.

    ``converge`` tilts each trace toward the one below it, closing every gap by
    that many pixels between the left and right edges. The traces then run
    closest at one end of the detector instead of everywhere, which is what
    separates a neighbor constraint measured at closest approach from one
    measured at mid-dispersion.
    """
    rng = np.random.default_rng(seed)
    order_height = 4 * _ORDERLET_SPACING + _ORDER_GAP
    nrow = int(24 + norder * order_height + 24)
    columns = np.arange(ncol, dtype=float)
    rows = np.arange(nrow, dtype=float)[:, None]
    image = np.full((nrow, ncol), 20.0)

    truth = {}
    first_row = 24.0
    tilt = 0.0
    for order, fiber in _trace_labels(norder):
        centers = (
            first_row
            - 0.004 * columns
            - 1e-6 * columns**2
            - tilt * columns / (ncol - 1)
        )
        truth[(order, fiber)] = centers
        first_row += _ORDER_GAP if fiber == "CAL" else _ORDERLET_SPACING
        tilt += converge
        if (order, fiber) in drop:
            continue
        image += _orderlet_profile(rows - centers[None, :], fiber)

    return image + rng.normal(0.0, 1.0, image.shape), truth


@pytest.fixture
def master_path(tmp_path, monkeypatch):
    """A master-flat path whose load yields a minimal two-chip stub."""
    path = tmp_path / "KP.20240405.00020.86_master_flat_L1.fits"
    path.touch()
    pixels = np.ones((8, 8), dtype=np.float32)
    monkeypatch.setattr(
        order_trace_module,
        "KPFMasterL1",
        _stub_master_class(StubMasterFlat({"GREEN": pixels, "RED": 2.0 * pixels})),
    )
    return path


def _tracer(tmp_path, image, monkeypatch, norder=3, **config):
    """Return an OrderTrace wired to a synthetic single-chip master flat."""
    master_path = tmp_path / "KP.20240405.00020.86_master_flat_L1.fits"
    master_path.touch()
    nrow, ncol = image.shape
    monkeypatch.setattr(
        order_trace_module,
        "KPFMasterL1",
        _stub_master_class(StubMasterFlat({"GREEN": image})),
    )
    return OrderTrace(
        master_path,
        {
            "chips": ["GREEN"],
            "norder": {"GREEN": norder},
            "ccd": {"nrow": nrow, "ncol": ncol},
            **config,
        },
    )


def _fiber_metadata(rows, cal_indices):
    """Return the cluster metadata that fiber phasing reads, bottom to top.

    Clusters named as CAL are given the thin, bright profile that identifies
    one; the rest are wide and dim.
    """
    is_cal = [index in cal_indices for index in range(len(rows))]
    return pd.DataFrame(
        {
            "cluster": range(len(rows)),
            "row": [float(row) for row in rows],
            "thickness": [2.0 if cal else 8.0 for cal in is_cal],
            "flux": [1000.0 if cal else 100.0 for cal in is_cal],
        }
    )


def _band_cluster(rows, columns):
    """Return a rectangular cluster covering `rows` at every one of `columns`."""
    grid_rows, grid_columns = np.meshgrid(rows, columns, indexing="ij")
    return {
        "row_indices": grid_rows.ravel(),
        "col_indices": grid_columns.ravel(),
        "npixel": int(grid_rows.size),
    }


def _curated_clusters(tracer):
    """Run detection and curation exactly as the module's own chain does.

    The clusters are cached where the later steps read them from, as
    detect_traces does, so identity can be exercised on cluster counts
    detection itself would refuse.
    """
    clusters = tracer._detect_clusters(tracer._detect_illuminated_pixels("GREEN"))
    clusters = tracer._reject_small_clusters(clusters)
    clusters = tracer._merge_fragmented_clusters(clusters)
    clusters = tracer._split_fused_clusters(clusters)
    clusters = tracer._reject_malformed_clusters(clusters)
    tracer._clusters["GREEN"] = tracer._reject_faint_clusters("GREEN", clusters)
    return tracer._clusters["GREEN"]


# ---------------------------------------------------------------------------
# Detection and cluster curation
# ---------------------------------------------------------------------------


class TestDetection:
    def test_labels_one_cluster_per_trace(self, tmp_path, monkeypatch):
        image, truth = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)

        curated = _curated_clusters(tracer)

        assert len(curated) == len(truth)
        for cluster in curated:
            assert cluster["col_indices"].min() == 0
            assert cluster["col_indices"].max() == image.shape[1] - 1

    def test_rejects_specks_and_single_row_artifacts(self, tmp_path, monkeypatch):
        image, truth = _synthetic_flat()
        image[300:303, 100:104] = 5000.0
        image[318, :] = 5000.0
        tracer = _tracer(tmp_path, image, monkeypatch)

        assert len(_curated_clusters(tracer)) == len(truth)

    def test_despeckles_specks_bridging_two_traces(self, tmp_path, monkeypatch):
        # A chain of noise specks across the gap corner-joins the traces either
        # side of it, and _detect_clusters would label the pair one cluster no
        # later step recovers whole. The mask has to come back already parted,
        # with both traces still whole.
        image = np.full((60, 400), 20.0, dtype=np.float32)
        image[20:23, :] = 1000.0
        image[30:33, :] = 1000.0
        specks = np.arange(23, 30)
        image[specks, 200 + specks - 23] = 1000.0
        tracer = _tracer(tmp_path, image, monkeypatch, norder=1)

        illuminated = tracer._detect_illuminated_pixels("GREEN")

        _, joined = label(illuminated, structure=np.ones((3, 3), dtype=int))
        assert joined == 2
        assert illuminated[20:23].all()
        assert illuminated[30:33].all()

    def test_merges_a_fragmented_trace(self, tmp_path, monkeypatch):
        image, truth = _synthetic_flat()
        # Blank a column band across the middle order's SCI1 trace only. The
        # far fragment is too short to be identified on its own, so the trace
        # is truncated at the gap unless the two pieces are rejoined.
        image[126:141, 300:340] = 20.0
        tracer = _tracer(tmp_path, image, monkeypatch)

        fragments = tracer._reject_small_clusters(
            tracer._detect_clusters(tracer._detect_illuminated_pixels("GREEN"))
        )
        merged = tracer._merge_fragmented_clusters(fragments)
        curated = _curated_clusters(tracer)

        assert len(merged) == len(fragments) - 1
        assert len(curated) == len(truth)
        for cluster in curated:
            assert cluster["col_indices"].min() == 0
            assert cluster["col_indices"].max() == image.shape[1] - 1

    def test_tolerates_one_trace_off_the_detector(self, tmp_path, monkeypatch, caplog):
        image, truth = _synthetic_flat(drop=[(2, "CAL")])
        tracer = _tracer(tmp_path, image, monkeypatch)

        with caplog.at_level("WARNING"):
            clusters = tracer.detect_traces("GREEN")

        assert len(clusters) == len(truth) - 1
        assert "14 traces detected, expected 15" in caplog.text

    def test_says_nothing_when_every_trace_is_detected(
        self, tmp_path, monkeypatch, caplog
    ):
        image, truth = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)

        with caplog.at_level("WARNING"):
            clusters = tracer.detect_traces("GREEN")

        assert len(clusters) == len(truth)
        assert caplog.text == ""

    def test_rejects_only_clusters_far_dimmer_than_the_traces(
        self, tmp_path, monkeypatch
    ):
        # Eleven trace-shaped clusters differing in nothing but flux: one at a
        # hundredth of the median, which is the junk, and one at a twentieth,
        # which is as dim as a real edge order gets.
        image = np.zeros((60, 400), dtype=np.float32)
        clusters = []
        for level, row in zip(
            [10.0, 50.0] + [1000.0] * 9, range(4, 59, 5), strict=True
        ):
            image[row : row + 3, :] = level
            rows, columns = np.mgrid[row : row + 3, 0:400]
            clusters.append(
                {
                    "row_indices": rows.ravel(),
                    "col_indices": columns.ravel(),
                    "npixel": rows.size,
                }
            )
        tracer = _tracer(tmp_path, image, monkeypatch)

        kept = tracer._reject_faint_clusters("GREEN", clusters)

        levels_kept = {
            float(np.median(image[c["row_indices"], c["col_indices"]])) for c in kept
        }
        assert len(kept) == len(clusters) - 1
        assert levels_kept == {50.0, 1000.0}

    def test_refuses_a_frame_whose_median_cluster_is_negative(
        self, tmp_path, monkeypatch
    ):
        image = np.zeros((60, 400), dtype=np.float32)
        clusters = []
        for level, row in zip([1000.0] + [-500.0] * 10, range(4, 59, 5), strict=True):
            image[row : row + 3, :] = level
            rows, columns = np.mgrid[row : row + 3, 0:400]
            clusters.append(
                {
                    "row_indices": rows.ravel(),
                    "col_indices": columns.ravel(),
                    "npixel": rows.size,
                }
            )
        tracer = _tracer(tmp_path, image, monkeypatch)

        with pytest.raises(ValueError, match="median cluster's flux is negative"):
            tracer._reject_faint_clusters("GREEN", clusters)

    def test_parts_a_cluster_holding_two_corner_joined_traces(
        self, tmp_path, monkeypatch
    ):
        image = np.zeros((60, 400), dtype=np.float32)
        tracer = _tracer(tmp_path, image, monkeypatch)
        columns = np.arange(400)
        ordinary = [
            _band_cluster(np.arange(row, row + 3), columns) for row in range(4, 34, 3)
        ]
        # Two ordinary traces meeting at one corner, as 8-connected labelling
        # hands them over: the lower reaches up one pixel at column 200 and the
        # upper is missing the pixel directly above it, so the two touch at a
        # corner and share no edge.
        lower = _band_cluster(np.arange(40, 43), columns)
        upper = _band_cluster(np.arange(44, 47), columns)
        above_corner = (upper["row_indices"] == 44) & (upper["col_indices"] == 200)
        upper = {
            "row_indices": upper["row_indices"][~above_corner],
            "col_indices": upper["col_indices"][~above_corner],
            "npixel": upper["npixel"] - 1,
        }
        fused = {
            "row_indices": np.concatenate(
                [lower["row_indices"], [43], upper["row_indices"]]
            ),
            "col_indices": np.concatenate(
                [lower["col_indices"], [200], upper["col_indices"]]
            ),
            "npixel": lower["npixel"] + 1 + upper["npixel"],
        }

        kept = tracer._split_fused_clusters([*ordinary, fused])

        assert len(kept) == len(ordinary) + 2
        assert sorted(c["npixel"] for c in kept[-2:]) == [
            upper["npixel"],
            lower["npixel"] + 1,
        ]

    def test_drops_an_overthick_cluster_that_will_not_part(self, tmp_path, monkeypatch):
        image = np.zeros((60, 400), dtype=np.float32)
        tracer = _tracer(tmp_path, image, monkeypatch)
        columns = np.arange(400)
        ordinary = [
            _band_cluster(np.arange(row, row + 3), columns) for row in range(4, 34, 3)
        ]
        solid = _band_cluster(np.arange(40, 48), columns)

        kept = tracer._split_fused_clusters([*ordinary, solid])

        assert len(kept) == len(ordinary)
        assert all(cluster["npixel"] == 3 * columns.size for cluster in kept)

    def test_reports_an_unexpected_trace_count(self, tmp_path, monkeypatch):
        image, _ = _synthetic_flat(norder=3)
        tracer = _tracer(tmp_path, image, monkeypatch, norder=2)

        with pytest.raises(ValueError, match="15 traces detected, expected 10"):
            tracer.detect_traces("GREEN")


class TestCalIdentification:
    def test_flags_every_cal_orderlet(self, tmp_path, monkeypatch):
        image, _ = _synthetic_flat(norder=4)
        tracer = _tracer(tmp_path, image, monkeypatch, norder=4)
        _curated_clusters(tracer)

        metadata = tracer._track_cluster_metadata("GREEN")
        flagged = np.flatnonzero(tracer._flag_cal_clusters(metadata)["is_cal"])

        assert flagged.size == 4
        assert np.all(np.diff(flagged) == len(_FIBERS))

    def test_cal_is_thinner_than_its_neighbours(self, tmp_path, monkeypatch):
        image, _ = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)
        _curated_clusters(tracer)

        metadata = tracer._track_cluster_metadata("GREEN")
        metadata = tracer._flag_cal_clusters(metadata)
        cal = metadata[metadata["is_cal"]]
        other = metadata[~metadata["is_cal"]]

        assert cal["thickness"].max() < other["thickness"].min()

    def test_cal_is_flagged_however_bright_it_is(self, tmp_path, monkeypatch):
        # Brightness is not tested, so a CAL no brighter than the orderlets it
        # closes -- which is what one instrument era looks like -- still flags.
        image, _ = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)
        _curated_clusters(tracer)
        metadata = tracer._track_cluster_metadata("GREEN")

        metadata["flux"] = 100.0

        assert tracer._flag_cal_clusters(metadata)["is_cal"].sum() == 3

    def test_reports_a_flat_with_no_identifiable_cal(self, tmp_path, monkeypatch):
        image, _ = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)
        _curated_clusters(tracer)
        metadata = tracer._track_cluster_metadata("GREEN")

        # A flat whose orderlets are all equally wide flags no CAL at all.
        metadata["thickness"] = 9.0
        with pytest.raises(ValueError, match="cannot phase the fiber pattern"):
            tracer._flag_cal_clusters(metadata)

    def test_reports_too_few_cal_orderlets_for_the_cluster_count(
        self, tmp_path, monkeypatch
    ):
        image, _ = _synthetic_flat(norder=4)
        tracer = _tracer(tmp_path, image, monkeypatch, norder=4)
        _curated_clusters(tracer)
        metadata = tracer._track_cluster_metadata("GREEN")

        # Two of the four CALs are widened to their neighbours' thickness, so
        # only half the orders end up anchored.
        cal_rows = metadata.index[metadata["thickness"] < 6.2][:2]
        metadata.loc[cal_rows, "thickness"] = 9.0
        with pytest.raises(ValueError, match="2 CAL orderlets identified among 20"):
            tracer._flag_cal_clusters(metadata)


# ---------------------------------------------------------------------------
# Trace identity
# ---------------------------------------------------------------------------


class TestTraceIdentity:
    def test_assigns_every_expected_fiber_and_order(self, tmp_path, monkeypatch):
        image, truth = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)

        clusters = tracer.detect_traces("GREEN")
        identities = tracer.assign_trace_identities("GREEN")

        assert _labels(identities) == _trace_labels(3)
        assert identities["cluster"].notna().all()
        for trace in identities.itertuples(index=False):
            cluster = clusters[int(trace.cluster)]
            middle = cluster["col_indices"] == image.shape[1] // 2
            assert cluster["row_indices"][middle].mean() == pytest.approx(
                truth[(trace.Order, trace.Fiber)][image.shape[1] // 2], abs=1.0
            )

    def test_labels_are_unshifted_when_edge_traces_are_absent(
        self, tmp_path, monkeypatch
    ):
        dropped = [(0, "SKY"), (2, "CAL")]
        image, truth = _synthetic_flat(drop=dropped)
        tracer = _tracer(tmp_path, image, monkeypatch)

        clusters = tracer.detect_traces("GREEN")
        identities = tracer.assign_trace_identities("GREEN")

        assert _labels(identities) == _trace_labels(3)
        assert _labels(identities[identities["cluster"].isna()]) == dropped
        for trace in identities.dropna().itertuples(index=False):
            cluster = clusters[int(trace.cluster)]
            middle = cluster["col_indices"] == image.shape[1] // 2
            assert cluster["row_indices"][middle].mean() == pytest.approx(
                truth[(trace.Order, trace.Fiber)][image.shape[1] // 2], abs=1.0
            )

    def test_phases_a_partial_order_at_the_bottom(self, master_path):
        # An order opening below the detector shows only its upper orderlets:
        # one trace fewer than expected, which detect_traces admits.
        rows = [119, 138, 157, 176, 191, 210, 229, 248, 267]
        metadata = _fiber_metadata(rows, cal_indices={3, 8})
        tracer = OrderTrace(master_path)

        phased = tracer._assign_fiber_identities("GREEN", metadata)

        assert list(phased["Fiber"]) == _FIBERS[1:] + _FIBERS

    def test_phases_a_partial_order_at_the_top(self, master_path):
        # Two complete orders and the SKY of a third whose other orderlets lie
        # off the top of the detector: one trace more than expected.
        rows = [100, 119, 138, 157, 176, 191, 210, 229, 248, 267, 282]
        metadata = _fiber_metadata(rows, cal_indices={4, 9})
        tracer = OrderTrace(master_path)

        phased = tracer._assign_fiber_identities("GREEN", metadata)

        assert list(phased["Fiber"]) == _FIBERS * 2 + ["SKY"]

    def test_names_the_ccd_when_no_cal_can_be_phased(self, master_path):
        # _flag_cal_clusters knows nothing of the CCD it was handed, so the
        # caller that does names it; two CCDs fail this way for different
        # reasons and the message has to say which one is being read.
        rows = [100, 119, 138, 157, 176, 191, 210, 229, 248, 267]
        metadata = _fiber_metadata(rows, cal_indices=set())
        tracer = OrderTrace(master_path)

        with pytest.raises(ValueError, match="^RED: 0 CAL orderlets identified"):
            tracer._assign_fiber_identities("RED", metadata)

    def test_reports_an_order_short_of_the_fiber_pattern(self, master_path):
        # An orderlet missing between two CALs leaves an order of four, which
        # cannot be told apart from an order whose fibers are mislabelled.
        rows = [100, 119, 138, 157, 176, 191, 210, 229]
        metadata = _fiber_metadata(rows, cal_indices={4, 7})
        tracer = OrderTrace(master_path)

        with pytest.raises(ValueError, match="2 orderlets below the CAL at row 229"):
            tracer._assign_fiber_identities("GREEN", metadata)

    def test_reports_an_order_beyond_the_fiber_pattern(self, master_path):
        # Seven clusters below the only CAL: more than a clipped edge explains.
        rows = [62, 81, 100, 119, 138, 157, 176]
        metadata = _fiber_metadata(rows, cal_indices={6})
        tracer = OrderTrace(master_path)

        with pytest.raises(ValueError, match="6 orderlets below the CAL at row 176"):
            tracer._assign_fiber_identities("GREEN", metadata)

    def test_discards_an_orderlet_clipped_by_the_bottom_edge(self, master_path, caplog):
        # The order below opens off the detector, leaving only its CAL in view,
        # too clipped to be recognized as one.
        rows = [8, 19, 40, 59, 79, 96]
        metadata = _fiber_metadata(rows, cal_indices={5})
        tracer = OrderTrace(master_path)

        with caplog.at_level("WARNING"):
            phased = tracer._assign_fiber_identities("GREEN", metadata)

        assert list(phased["Fiber"].fillna("clipped")) == ["clipped"] + _FIBERS
        assert "edge-clipped orderlet at detector row 8" in caplog.text

    def test_discards_an_orderlet_clipped_by_the_top_edge(self, master_path, caplog):
        # The mirror case: the order above closes off the detector, leaving its
        # CAL clipped at the top.
        rows = [100, 119, 138, 157, 176, 191, 210, 229, 248, 262]
        metadata = _fiber_metadata(rows, cal_indices={4})
        tracer = OrderTrace(master_path)

        with caplog.at_level("WARNING"):
            phased = tracer._assign_fiber_identities("GREEN", metadata)

        assert list(phased["Fiber"].fillna("clipped")) == (
            _FIBERS + _FIBERS[:-1] + ["clipped"]
        )
        assert "edge-clipped orderlet at detector row 262" in caplog.text

    def test_discards_a_lone_orderlet_beyond_the_expected_orders(
        self, master_path, caplog
    ):
        rows = [100, 119, 138, 157, 176, 191, 210, 229, 248, 267, 282]
        metadata = _fiber_metadata(rows, cal_indices={4, 9})
        tracer = OrderTrace(master_path, {"norder": {"GREEN": 2}})
        metadata = tracer._assign_fiber_identities("GREEN", metadata)

        with caplog.at_level("WARNING"):
            identities = tracer._assign_order_indexes("GREEN", metadata)

        assert _labels(identities) == _trace_labels(2)
        assert identities["cluster"].notna().all()
        assert "3 orders detected but 2 expected" in caplog.text

    def test_discards_a_lone_orderlet_at_each_edge(self, master_path, caplog):
        # Both edge orders open off the detector, leaving a lone CAL apiece. One
        # discard cannot settle this: the count has to be walked down twice.
        # Phasing is fed rather than run: two lone CALs put the CAL count two
        # clear of one fifth of the clusters, which _flag_cal_clusters refuses.
        rows = [85, 100, 119, 138, 157, 176, 191, 210, 229, 248, 267, 282]
        cal_indices = {0, 5, 10, 11}
        metadata = _fiber_metadata(rows, cal_indices=cal_indices)
        metadata["is_cal"] = [index in cal_indices for index in range(len(rows))]
        metadata["Fiber"] = ["CAL"] + _FIBERS * 2 + ["CAL"]
        tracer = OrderTrace(master_path, {"norder": {"GREEN": 2}})

        with caplog.at_level("WARNING"):
            identities = tracer._assign_order_indexes("GREEN", metadata)

        assert _labels(identities) == _trace_labels(2)
        assert identities["cluster"].notna().all()
        assert "4 orders detected but 2 expected" in caplog.text
        assert "3 orders detected but 2 expected" in caplog.text


# ---------------------------------------------------------------------------
# End-to-end tracing
# ---------------------------------------------------------------------------


class TestMakeMasterOrderTrace:
    def test_traces_a_synthetic_flat(self, tmp_path, monkeypatch):
        image, truth = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)
        output_dir = tmp_path / "out"

        table = tracer.make_master(output_dir=output_dir)

        assert list(table.columns) == _TRACE_FIELDS
        assert (table["Chip"] == "GREEN").all()
        assert len(table) == len(truth)
        labels = list(zip(table["Order"], table["Fiber"], strict=True))
        assert labels == _trace_labels(3)
        assert (table["Status"] == "full").all()
        assert ((table["BottomEdge"] > 0) & (table["TopEdge"] > 0)).all()
        _assert_apertures_disjoint(table, image.shape[1])

        columns = np.linspace(0, image.shape[1] - 1, 9)
        for row in table.itertuples(index=False):
            coeffs = [getattr(row, f"Coeff{i}") for i in range(4)]
            fitted = np.polynomial.polynomial.polyval(columns, coeffs)
            expected = truth[(row.Order, row.Fiber)][columns.astype(int)]
            assert np.max(np.abs(fitted - expected)) < 0.5

        written = pd.read_csv(
            output_dir / "KP.20240405.00020.86_master_order_trace.csv"
        )
        pd.testing.assert_frame_equal(written, table)

    def test_omitting_output_dir_builds_without_writing(self, tmp_path, monkeypatch):
        image, _ = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)

        table = tracer.make_master()

        assert set(table["Chip"]) == {"GREEN"}
        assert not list(tmp_path.glob("*master_order_trace.csv"))
        assert "(not written)" in tracer._info

    def test_missing_trace_keeps_its_row(self, tmp_path, monkeypatch):
        image, truth = _synthetic_flat(drop=[(0, "SKY")])
        tracer = _tracer(tmp_path, image, monkeypatch)

        table = tracer.make_master(output_dir=tmp_path / "out")
        missing = table[table["Status"] == "missing"]

        assert len(table) == len(truth)
        labels = list(zip(missing["Order"], missing["Fiber"], strict=True))
        assert labels == [(0, "SKY")]
        assert missing[["Coeff0", "BottomEdge", "X1"]].isna().all(axis=None)
        assert (table[table["Status"] != "missing"]["Status"] == "full").all()

    def test_poly_degree_sets_the_coefficient_fields(self, tmp_path, monkeypatch):
        image, _ = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch, poly_degree=5)

        table = tracer.make_master(output_dir=tmp_path / "out")

        coeff_fields = [field for field in table.columns if field.startswith("Coeff")]
        assert coeff_fields == [f"Coeff{i}" for i in range(6)]

    def test_poly_degree_argument_overrides_the_configured_degree(
        self, tmp_path, monkeypatch
    ):
        image, _ = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch, poly_degree=5)

        table = tracer.make_master(poly_degree=2, output_dir=tmp_path / "out")

        coeff_fields = [field for field in table.columns if field.startswith("Coeff")]
        assert coeff_fields == ["Coeff0", "Coeff1", "Coeff2"]

    def test_reports_progress_only_after_running(self, tmp_path, monkeypatch, capsys):
        image, _ = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)

        tracer.info()
        assert "has not been called" in capsys.readouterr().out

        tracer.make_master(output_dir=tmp_path / "out")
        tracer.info()
        assert "OrderTrace" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Configuration, input validation, and output
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_only_polynomial_degree_is_module_configurable(self, master_path):
        tracer = OrderTrace(
            master_path,
            {"poly_degree": 2, "cal_flux_ratio": 2.5, "sample_count": 3},
        )

        assert tracer.poly_degree == 2
        assert not hasattr(tracer, "cal_flux_ratio")
        assert not hasattr(tracer, "sample_count")

    def test_config_does_not_require_data_directory_paths(self, tmp_path, master_path):
        config_path = tmp_path / "order_trace.toml"
        config_path.write_text(
            '[TRACES]\nchips = ["GREEN", "RED"]\n[ORDER_TRACE]\npoly_degree = 2\n'
        )

        tracer = OrderTrace(master_path, ConfigHandler(config_path))

        assert tracer.chips == ["GREEN", "RED"]
        assert tracer.poly_degree == 2

    def test_rejects_an_unusable_config(self, master_path):
        with pytest.raises(TypeError, match="config must be"):
            OrderTrace(master_path, config=1)

    def test_rejects_a_negative_polynomial_degree(self, master_path):
        tracer = OrderTrace(master_path, {"poly_degree": -1})
        with pytest.raises(ValueError, match="poly_degree must be non-negative"):
            tracer._trace_fields()

    def test_rejects_an_unknown_field_selection(self, master_path):
        tracer = OrderTrace(master_path)
        with pytest.raises(ValueError, match="which must be one of"):
            tracer._trace_fields(which="labels")


class TestMasterFlatLoading:
    def test_loads_vnext_master_flat(self, tmp_path, monkeypatch):
        master_path = tmp_path / "KP.20240405.00020.86_master_flat_L1.fits"
        master_path.touch()
        pixels = np.ones((8, 8), dtype=np.float32)
        master_flat = StubMasterFlat({"GREEN": pixels, "RED": 2.0 * pixels})
        monkeypatch.setattr(
            order_trace_module, "KPFMasterL1", _stub_master_class(master_flat)
        )

        tracer = OrderTrace(master_path)

        assert tracer._master_flat is master_flat
        assert set(tracer._image) == {"GREEN", "RED"}
        assert tracer._image["RED"] is master_flat.data["RED_IMG"]

    def test_rejects_non_flat_master(self, tmp_path, monkeypatch):
        master_path = tmp_path / "KP.20240405.00020.86_master_dark_L1.fits"
        master_path.touch()
        dark_master = StubMasterFlat(
            {"GREEN": np.ones((8, 8), dtype=np.float32)}, master_type="dark"
        )

        monkeypatch.setattr(
            order_trace_module, "KPFMasterL1", _stub_master_class(dark_master)
        )
        with pytest.raises(ValueError, match="not a vNext flat master"):
            OrderTrace(master_path)

    def test_reports_a_missing_master_flat(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Master flat not found"):
            OrderTrace(tmp_path / "absent_master_flat_L1.fits")


class TestCSVWriting:
    def test_refuses_overwrite(self, tmp_path, master_path):
        tracer = OrderTrace(master_path)
        table = pd.DataFrame(
            [
                {
                    "Chip": "GREEN",
                    "Coeff0": 10.0,
                    "Coeff1": 0.0,
                    "Coeff2": 0.0,
                    "Coeff3": 0.0,
                    "BottomEdge": 2.0,
                    "TopEdge": 2.0,
                    "X1": 0.0,
                    "X2": 10.0,
                    "Fiber": "SKY",
                    "Order": 0,
                    "Status": "full",
                }
            ],
            columns=_TRACE_FIELDS,
        )
        tracer.output_table = table
        output = tmp_path / "traces.csv"

        tracer.save_master(output, overwrite=False)
        assert output.is_file()
        with pytest.raises(FileExistsError):
            tracer.save_master(output, overwrite=False)
        tracer.save_master(output, overwrite=True)

    def test_refuses_to_save_before_make_master(self, tmp_path, master_path):
        tracer = OrderTrace(master_path)
        with pytest.raises(RuntimeError, match="run make_master"):
            tracer.save_master(tmp_path / "traces.csv")


# ---------------------------------------------------------------------------
# Aperture non-overlap constraint
# ---------------------------------------------------------------------------


def _straight_trace(center, slope, bottom_edge, top_edge):
    """A degree-3 record with a straight centerline and the given edges."""
    return {
        "Coeff0": center,
        "Coeff1": slope,
        "Coeff2": 0.0,
        "Coeff3": 0.0,
        "BottomEdge": bottom_edge,
        "TopEdge": top_edge,
    }


def _assert_apertures_disjoint(table, ncol):
    """No adjacent measured apertures overlap at any column."""
    measured = table[table["Status"] != "missing"]
    coeff_fields = [field for field in table.columns if field.startswith("Coeff")]
    columns = np.linspace(0, ncol - 1, 201)
    centers = np.array(
        [
            np.polynomial.polynomial.polyval(columns, coeffs)
            for coeffs in measured[coeff_fields].to_numpy(dtype=float)
        ]
    )
    upper = centers + measured["TopEdge"].to_numpy(dtype=float)[:, None]
    lower = centers - measured["BottomEdge"].to_numpy(dtype=float)[:, None]
    assert np.all(lower[1:] > upper[:-1])


def _fitted_tracer(tmp_path, monkeypatch, **kwargs):
    """Return a tracer over a one-order synthetic flat, fitted but unapertured."""
    image, _ = _synthetic_flat(norder=1)
    tracer = _tracer(tmp_path, image, monkeypatch, norder=1, **kwargs)
    tracer.detect_traces("GREEN")
    tracer.assign_trace_identities("GREEN")
    tracer.fit_trace_polynomials("GREEN")
    return tracer


class TestApertureConstraint:
    def _make_tracer(self, master_path, norder=1, ncol=400):
        return OrderTrace(
            master_path,
            {"ccd": {"nrow": 400, "ncol": ncol}, "norder": {"GREEN": norder}},
        )

    def test_clamps_contended_neighbours_and_keeps_roomy_ones(
        self, tmp_path, monkeypatch
    ):
        roomy = _fitted_tracer(tmp_path, monkeypatch).estimate_trace_apertures("GREEN")
        # A guard band this wide leaves the neighbours 4 px of their
        # _ORDERLET_SPACING to share, so every interior edge is contended.
        # Re-clamping only ever narrows, so the wider band decides every edge.
        tracer = _fitted_tracer(tmp_path, monkeypatch)
        clamped = tracer.estimate_trace_apertures("GREEN")
        tracer._clamp_neighboring_apertures(
            "GREEN", orderlet_gap_pixels=_ORDERLET_SPACING - 4.0
        )

        below, above = clamped["TopEdge"][:-1], clamped["BottomEdge"][1:]
        assert np.allclose(below.to_numpy(), 2.0, atol=0.1)
        assert np.allclose(below.to_numpy(), above.to_numpy())
        assert (below.to_numpy() < roomy["TopEdge"][:-1].to_numpy()).all()
        # The outermost edges are uncontended and keep their profile width.
        assert clamped["BottomEdge"].iloc[0] == roomy["BottomEdge"].iloc[0]
        assert clamped["TopEdge"].iloc[-1] == roomy["TopEdge"].iloc[-1]

    def test_clamps_convergent_neighbours_at_their_closest_approach(
        self, tmp_path, monkeypatch
    ):
        image, truth = _synthetic_flat(norder=1, converge=4.0)
        tracer = _tracer(tmp_path, image, monkeypatch, norder=1)
        table = tracer.make_master()
        roomy = _fitted_tracer(tmp_path, monkeypatch).estimate_trace_apertures("GREEN")

        centers = np.array([truth[label] for label in _trace_labels(1)])
        separation = np.diff(centers, axis=0)
        mid_dispersion = separation[:, image.shape[1] // 2]
        assert (separation.min(axis=1) < mid_dispersion - 1.0).all()

        # Each contended edge is half the closest approach less the default 2 px
        # guard band; the rest keep the width their profile was measured at.
        half_space = (separation.min(axis=1) - 2.0) / 2.0
        below, above = table["TopEdge"][:-1], table["BottomEdge"][1:]
        assert np.allclose(
            below.to_numpy(), np.minimum(roomy["TopEdge"][:-1], half_space), atol=0.1
        )
        assert np.allclose(
            above.to_numpy(), np.minimum(roomy["BottomEdge"][1:], half_space), atol=0.1
        )
        # Measuring at mid-dispersion instead would have left them wider.
        assert (below.to_numpy() < (mid_dispersion - 2.0) / 2.0).all()
        _assert_apertures_disjoint(table, image.shape[1])

    def test_rejects_neighbours_whose_centerlines_cross(self, tmp_path, monkeypatch):
        tracer = _fitted_tracer(tmp_path, monkeypatch)
        # Tilt SCI1 down through SKY so the two touch and then cross. Detection
        # cannot deliver such a pair -- they would be one cluster -- but a fit
        # running away over part of the detector can.
        fitted = tracer._trace_tables["GREEN"]
        crossing = fitted.index[fitted["Fiber"] == "SCI1"][0]
        fitted.loc[crossing, "Coeff1"] -= (
            2 * _ORDERLET_SPACING / (tracer.ccd["ncol"] - 1)
        )

        with pytest.raises(ValueError, match="orderlet gap"):
            tracer.estimate_trace_apertures("GREEN")

    def test_rejects_traces_too_close_for_the_gap(self, tmp_path, monkeypatch):
        tracer = _fitted_tracer(tmp_path, monkeypatch)
        # The clamp reads the widths to see where an aperture is on the
        # detector, which the aperture step would have measured by now.
        tracer._trace_tables["GREEN"][["BottomEdge", "TopEdge"]] = 5.0
        with pytest.raises(ValueError, match="^GREEN: neighboring fitted traces"):
            tracer._clamp_neighboring_apertures(
                "GREEN", orderlet_gap_pixels=_ORDERLET_SPACING + 1.0
            )

    def test_fits_only_the_columns_the_orderlet_can_be_centered_at(
        self, tmp_path, monkeypatch
    ):
        # A column where the cluster reaches a detector edge cannot yield a
        # center, so an order running off the detector is judged on its own
        # span rather than on the whole dispersion.
        image, _ = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)
        tracer.detect_traces("GREEN")
        tracer.assign_trace_identities("GREEN")
        columns = tracer._sample_profiles("GREEN")["columns"]

        cluster = tracer._clusters["GREEN"][
            int(tracer._metadata["GREEN"]["cluster"].iloc[0])
        ]
        half = image.shape[1] // 2
        reaching = cluster["col_indices"][cluster["col_indices"] >= half]
        cluster["row_indices"] = np.concatenate(
            [cluster["row_indices"], np.zeros(reaching.size, dtype=int)]
        )
        cluster["col_indices"] = np.concatenate([cluster["col_indices"], reaching])

        offered = []
        fit_polynomial = order_trace_module.robust_polyfit

        def spy(x, y, deg, **kwargs):
            offered.append(np.asarray(x))
            return fit_polynomial(x, y, deg, **kwargs)

        monkeypatch.setattr(order_trace_module, "robust_polyfit", spy)
        tracer.fit_trace_polynomials("GREEN")

        assert list(offered[0]) == list(columns[columns < half])
        assert all(x.size == columns.size for x in offered[1:])

    def test_a_partial_trace_is_bounded_by_the_columns_it_was_fitted_over(
        self, tmp_path, monkeypatch
    ):
        # A cubic extrapolated beyond its fitted span can curl back over the
        # detector and read as carried, so X1-X2 must stay inside that span.
        image, _ = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)
        tracer.detect_traces("GREEN")
        tracer.assign_trace_identities("GREEN")

        cluster = tracer._clusters["GREEN"][
            int(tracer._metadata["GREEN"]["cluster"].iloc[0])
        ]
        half = image.shape[1] // 2
        reaching = cluster["col_indices"][cluster["col_indices"] >= half]
        cluster["row_indices"] = np.concatenate(
            [cluster["row_indices"], np.zeros(reaching.size, dtype=int)]
        )
        cluster["col_indices"] = np.concatenate([cluster["col_indices"], reaching])

        fitted = tracer.fit_trace_polynomials("GREEN")
        assert fitted[["X1", "X2"]].iloc[0].tolist() == [0.0, half - 1.0]
        assert (fitted["X2"].iloc[1:] == image.shape[1] - 1).all()

        table = tracer.estimate_trace_apertures("GREEN")
        assert table["X1"].iloc[0] == 0.0 and table["X2"].iloc[0] <= half - 1
        assert table["Status"].iloc[0] == "partial"
        assert (table["Status"].iloc[1:] == "full").all()

    def test_refits_a_partial_trace_whose_fit_turns_back_within_its_span(
        self, tmp_path, monkeypatch, caplog
    ):
        # Over the columns a partial trace was not fitted at, a cubic is free
        # to turn back and carry the aperture off the detector and onto it
        # again, so such a fit is redone as a parabola.
        image, _ = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)
        tracer.detect_traces("GREEN")
        tracer.assign_trace_identities("GREEN")

        cluster = tracer._clusters["GREEN"][
            int(tracer._metadata["GREEN"]["cluster"].iloc[0])
        ]
        half = image.shape[1] // 2
        reaching = cluster["col_indices"][cluster["col_indices"] >= half]
        cluster["row_indices"] = np.concatenate(
            [cluster["row_indices"], np.zeros(reaching.size, dtype=int)]
        )
        cluster["col_indices"] = np.concatenate([cluster["col_indices"], reaching])

        degrees = []
        fit_polynomial = order_trace_module.robust_polyfit
        turn = half / 2.0

        def spy(x, y, deg, **kwargs):
            degrees.append(deg)
            coeffs, good, rms = fit_polynomial(x, y, deg, **kwargs)
            if len(degrees) == 1:
                # (x - turn)**3, whose only turning point is inside the span.
                coeffs = np.array([-(turn**3), 3 * turn**2, -3 * turn, 1.0])
            return coeffs, good, rms

        monkeypatch.setattr(order_trace_module, "robust_polyfit", spy)
        with caplog.at_level("WARNING"):
            fitted = tracer.fit_trace_polynomials("GREEN")

        assert degrees[:2] == [3, 2]
        assert set(degrees[2:]) == {3}
        assert fitted["Coeff3"].iloc[0] == 0.0
        assert (fitted["Coeff3"].iloc[1:] != 0.0).all()
        assert "turns back inside the span" in caplog.text

    def test_fitting_leaves_the_apertures_to_the_aperture_step(
        self, tmp_path, monkeypatch
    ):
        tracer = _fitted_tracer(tmp_path, monkeypatch)

        fitted = tracer._trace_tables["GREEN"]
        assert list(fitted.columns) == _TRACE_FIELDS
        assert fitted[["BottomEdge", "TopEdge"]].isna().all(axis=None)
        assert fitted[["Coeff0", "PolyfitRMS"]].notna().all(axis=None)
        assert (fitted["Status"] == "unknown").all()
        # X1-X2 arrives as the span the fit was constrained over.
        assert (fitted["X1"] == 0).all()
        assert (fitted["X2"] == tracer.ccd["ncol"] - 1).all()

        # What the fit accepted at the sampled columns stays with them.
        sampled = tracer._profiles["GREEN"]
        assert sampled["good"].shape == (len(fitted), sampled["columns"].size)
        assert sampled["good"].any(axis=1).all()

        table = tracer.estimate_trace_apertures("GREEN")
        assert list(table.columns) == _TRACE_FIELDS
        assert (table[["BottomEdge", "TopEdge"]] > 0).all(axis=None)
        assert (table["X1"] <= table["X2"]).all()

    def test_reports_a_trace_whose_widths_cannot_be_measured(
        self, tmp_path, monkeypatch
    ):
        tracer = _fitted_tracer(tmp_path, monkeypatch)

        def fail(*args, **kwargs):
            raise ValueError("fewer than three valid samples")

        monkeypatch.setattr(tracer, "_trace_width", fail)

        with pytest.raises(
            ValueError, match="GREEN SKY order 0: fewer than three valid samples"
        ):
            tracer.estimate_trace_apertures("GREEN")

        # The fit stands as it was left; nothing is blanked on the way out.
        fitted = tracer._trace_tables["GREEN"]
        assert (fitted["Status"] == "unknown").all()
        assert fitted["PolyfitRMS"].notna().all()

    def test_reports_a_trace_that_cannot_be_fitted(self, tmp_path, monkeypatch):
        image, _ = _synthetic_flat(norder=1)
        tracer = _tracer(tmp_path, image, monkeypatch, norder=1)
        tracer.detect_traces("GREEN")
        tracer.assign_trace_identities("GREEN")

        def fail(*args, **kwargs):
            raise ValueError("only 2 of 65 samples are valid")

        monkeypatch.setattr(order_trace_module, "robust_polyfit", fail)

        with pytest.raises(
            ValueError, match="GREEN SKY order 0: only 2 of 65 samples are valid"
        ):
            tracer.fit_trace_polynomials("GREEN")

    def test_redetection_discards_what_was_measured_per_trace(
        self, tmp_path, monkeypatch
    ):
        tracer = _fitted_tracer(tmp_path, monkeypatch)
        profiles = tracer._profiles["GREEN"]["profiles"]
        assert "GREEN" in tracer._trace_tables

        tracer.detect_traces("GREEN")

        sampled = tracer._profiles["GREEN"]
        assert sampled["good"] is None
        # The fitted rows named clusters this rebuild has renumbered.
        assert "GREEN" not in tracer._trace_tables
        assert "GREEN" not in tracer._metadata
        # The profiles are a property of the image, so they survive.
        assert sampled["profiles"] is profiles

    def test_validation_rejects_overlapping_apertures(self, master_path):
        tracer = self._make_tracer(master_path, norder=1)
        centers = [100.0, 120.0, 140.0, 160.0, 180.0]
        edges = [(6.0, 6.0), (6.0, 6.0), (6.0, 25.0), (25.0, 6.0), (6.0, 6.0)]
        rows = [
            {
                **_straight_trace(center, 0.0, bottom, top),
                "Chip": "GREEN",
                "X1": 0.0,
                "X2": 399.0,
                "Fiber": fiber,
                "Order": 0,
                "Status": "full",
            }
            for fiber, center, (bottom, top) in zip(
                _FIBERS, centers, edges, strict=True
            )
        ]
        tracer._trace_tables["GREEN"] = pd.DataFrame(rows, columns=_TRACE_FIELDS)

        # Centerlines stay ordered, but the SCI2 and SCI3 apertures overlap.
        with pytest.raises(ValueError, match="apertures overlap"):
            tracer._validate_trace_table("GREEN")

    def test_validation_rejects_a_trace_that_leaves_and_returns(self, master_path):
        tracer = self._make_tracer(master_path, norder=1)
        rows = [
            {
                **_straight_trace(100.0 + 20.0 * position, 0.0, 6.0, 6.0),
                "Chip": "GREEN",
                "X1": 0.0,
                "X2": 399.0,
                "Fiber": fiber,
                "Order": 0,
                "Status": "full",
            }
            for position, fiber in enumerate(_FIBERS)
        ]
        # Sink SKY below the detector at mid-dispersion and bring it back;
        # X1-X2 is one span, so both ends cannot be carried.
        rows[0].update(Coeff0=100.0, Coeff1=-2.0, Coeff2=0.005)
        tracer._trace_tables["GREEN"] = pd.DataFrame(rows, columns=_TRACE_FIELDS)

        with pytest.raises(ValueError, match="SKY order 0 leaves the detector"):
            tracer._validate_trace_table("GREEN")

    def test_neighbours_sharing_no_fitted_column_are_left_unclamped(
        self, master_path, caplog
    ):
        tracer = self._make_tracer(master_path, norder=1)
        rows = [
            {
                **_straight_trace(100.0 + 20.0 * position, 0.0, 6.0, 6.0),
                "Chip": "GREEN",
                "X1": 0.0,
                "X2": 399.0,
                "Fiber": fiber,
                "Order": 0,
                "Status": "full",
            }
            for position, fiber in enumerate(_FIBERS)
        ]
        # SKY was fitted over one end of the dispersion and SCI1 the other, so
        # their spacing could be measured nowhere.
        rows[0]["X2"], rows[1]["X1"] = 150.0, 250.0
        tracer._trace_tables["GREEN"] = pd.DataFrame(rows, columns=_TRACE_FIELDS)

        with caplog.at_level("WARNING"):
            tracer._clamp_neighboring_apertures("GREEN")

        assert "fitted over no common column" in caplog.text
        # Every other pair is still clamped against its neighbour.
        assert (tracer._trace_tables["GREEN"]["TopEdge"] == 6.0).all()


# ---------------------------------------------------------------------------
# Real-data test
# ---------------------------------------------------------------------------


class TestRealData:
    @pytest.mark.slow
    @pytest.mark.requires_testdata
    def test_real_20240405_master_flat(self, tmp_path):
        """Trace a real vNext master flat and check it reproduces the reference.

        The comparison is against the vetted reference the pipeline ships for
        this era, which was measured from this flat -- so it pins the shipped
        artifact to what the module produces today, rather than standing as
        independent truth."""
        testdata = Path(__file__).parent.parent / "testdata"
        masters = sorted(testdata.glob("**/KP.20240405.*_master_flat_L1.fits"))
        if not masters:
            pytest.skip("a 20240405 vNext master flat is not installed")

        vetted = pd.read_csv(
            Path(__file__).parents[2]
            / "reference"
            / "order_traces"
            / "order_trace_20240405.csv"
        )

        tracer = OrderTrace(masters[0])
        combined = tracer.make_master(output_dir=tmp_path)
        tables = {chip: combined[combined["Chip"] == chip] for chip in ("GREEN", "RED")}

        assert len(tables["GREEN"]) == 175
        assert len(tables["RED"]) == 160
        assert (tables["GREEN"]["Status"] != "missing").sum() == 175
        assert (tables["RED"]["Status"] != "missing").sum() >= 158
        obs_id = masters[0].name.split("_master_flat")[0]
        assert (tmp_path / f"{obs_id}_master_order_trace.csv").is_file()

        for chip in ("GREEN", "RED"):
            assert np.nanmedian(tables[chip]["PolyfitRMS"]) < 1.0
            _assert_apertures_disjoint(tables[chip], tracer.ccd["ncol"])
            assert np.nanmax(tables[chip]["PolyfitRMS"]) < 2.0

            reference = vetted[
                (vetted["Chip"] == chip) & (vetted["Status"] != "missing")
            ]
            merged = tables[chip].merge(
                reference, on=["Order", "Fiber"], suffixes=("", "_ref")
            )
            assert len(merged) >= len(reference) - 1

            middle = 2040
            measured = np.polynomial.polynomial.polyval(
                middle, merged[[f"Coeff{i}" for i in range(4)]].to_numpy().T
            )
            expected = np.polynomial.polynomial.polyval(
                middle, merged[[f"Coeff{i}_ref" for i in range(4)]].to_numpy().T
            )
            assert np.nanmedian(np.abs(measured - expected)) < 0.5
            assert np.nanmax(np.abs(measured - expected)) < 1.0
