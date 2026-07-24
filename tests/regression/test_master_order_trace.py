"""Tests for spectral order tracing from one vNext L1 master flat.

Numerical tests use synthetic master images and require no real data. The
synthetic flat reproduces the morphology the algorithm depends on: science and
sky orderlets are wide and flat-topped, and the CAL orderlet is narrow and
several times brighter. The real master-flat test uses gitignored
``tests/testdata`` and is skipped when the vNext product is unavailable.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import kpfpipe.modules.masters.order_trace as order_trace_module
from kpfpipe.modules.masters import OrderTrace
from kpfpipe.utils.config import ConfigHandler

_FIBERS = ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]
_TRACE_FIELDS = [
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
    "Status",
]

# Rows between orderlets inside one order, and between one order's CAL and the
# next order's SKY. As on the real detector the inter-order gap is the smaller
# of the two, so spacing alone cannot phase the fiber pattern.
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


def _orderlet_profile(offsets, fiber):
    """Return the cross-dispersion profile of one orderlet."""
    if fiber == "CAL":
        return 1500.0 * np.exp(-0.5 * (offsets / 1.8) ** 2)
    return 500.0 * np.exp(-0.5 * (offsets / 5.5) ** 6)


def _trace_labels(norder):
    """Return the (order, fiber) label of every trace, bottom to top."""
    return [(order, fiber) for order in range(norder) for fiber in _FIBERS]


def _synthetic_flat(norder=3, ncol=400, drop=(), seed=0):
    """Return a synthetic master-flat image and its exact trace centers."""
    rng = np.random.default_rng(seed)
    order_height = 4 * _ORDERLET_SPACING + _ORDER_GAP
    nrow = int(24 + norder * order_height + 24)
    columns = np.arange(ncol, dtype=float)
    rows = np.arange(nrow, dtype=float)[:, None]
    image = np.full((nrow, ncol), 20.0)

    truth = {}
    first_row = 24.0
    for order, fiber in _trace_labels(norder):
        centers = first_row - 0.004 * columns - 1e-6 * columns**2
        truth[(order, fiber)] = centers
        first_row += _ORDER_GAP if fiber == "CAL" else _ORDERLET_SPACING
        if (order, fiber) in drop:
            continue
        image += _orderlet_profile(rows - centers[None, :], fiber)

    return image + rng.normal(0.0, 1.0, image.shape), truth


def _tracer(tmp_path, image, monkeypatch, norder=3, **config):
    """Return an OrderTrace wired to a synthetic single-chip master flat."""
    master_path = tmp_path / "KP.20240405.00020.86_master_flat_L1.fits"
    master_path.touch()
    tracer = OrderTrace(
        master_path, {"chips": ["GREEN"], "norder": {"GREEN": norder}, **config}
    )
    monkeypatch.setattr(
        tracer, "_load_master_flat", lambda: StubMasterFlat({"GREEN": image})
    )
    return tracer


def _curated_clusters(tracer, image):
    """Run detection and curation exactly as the module's own chain does."""
    clusters = tracer._label_clusters(tracer._detect_illuminated_pixels(image))
    clusters = tracer._reject_small_clusters("GREEN", clusters)
    clusters = tracer._merge_fragmented_clusters("GREEN", clusters)
    return tracer._reject_unidentifiable_clusters("GREEN", clusters, image.shape[1])


# ---------------------------------------------------------------------------
# Detection and cluster curation
# ---------------------------------------------------------------------------


class TestDetection:
    def test_labels_one_cluster_per_trace(self, tmp_path, monkeypatch):
        image, truth = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)

        curated = _curated_clusters(tracer, image)

        assert len(curated) == len(truth)
        for cluster in curated:
            assert cluster["x1"] == 0
            assert cluster["x2"] == image.shape[1] - 1

    def test_rejects_specks_and_single_row_artifacts(self, tmp_path, monkeypatch):
        image, truth = _synthetic_flat()
        image[300:303, 100:104] = 5000.0
        image[318, :] = 5000.0
        tracer = _tracer(tmp_path, image, monkeypatch)

        assert len(_curated_clusters(tracer, image)) == len(truth)

    def test_merges_a_fragmented_trace(self, tmp_path, monkeypatch):
        image, truth = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)

        # Blank a column band across the middle order's SCI1 trace only. The
        # far fragment is too short to be identified on its own, so the trace
        # is truncated at the gap unless the two pieces are rejoined.
        image[126:141, 300:340] = 20.0
        fragments = tracer._reject_small_clusters(
            "GREEN", tracer._label_clusters(tracer._detect_illuminated_pixels(image))
        )
        merged = tracer._merge_fragmented_clusters("GREEN", fragments)
        curated = _curated_clusters(tracer, image)

        assert len(merged) == len(fragments) - 1
        assert len(curated) == len(truth)
        for cluster in curated:
            assert cluster["x1"] == 0
            assert cluster["x2"] == image.shape[1] - 1


class TestCalIdentification:
    def test_flags_every_cal_orderlet(self, tmp_path, monkeypatch):
        image, _ = _synthetic_flat(norder=4)
        tracer = _tracer(tmp_path, image, monkeypatch, norder=4)
        clusters = _curated_clusters(tracer, image)

        metrics = tracer._cluster_center_metrics(clusters, image)
        flagged = np.flatnonzero(tracer._flag_cal_clusters(metrics))

        assert flagged.size == 4
        assert np.all(np.diff(flagged) == len(_FIBERS))

    def test_cal_is_thinner_and_brighter_than_its_neighbours(
        self, tmp_path, monkeypatch
    ):
        image, _ = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)
        clusters = _curated_clusters(tracer, image)

        metrics = tracer._cluster_center_metrics(clusters, image)
        metrics["is_cal"] = tracer._flag_cal_clusters(metrics)
        cal = metrics[metrics["is_cal"]]
        other = metrics[~metrics["is_cal"]]

        assert cal["thickness"].max() < other["thickness"].min()
        assert cal["flux"].min() > other["flux"].max()

    def test_reports_a_flat_with_no_identifiable_cal(self, tmp_path, monkeypatch):
        image, _ = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)
        metrics = tracer._cluster_center_metrics(
            _curated_clusters(tracer, image), image
        )

        with pytest.raises(ValueError, match="no CAL orderlet identified"):
            tracer._assign_fiber_positions(
                "GREEN", metrics, np.zeros(len(metrics), dtype=bool)
            )


# ---------------------------------------------------------------------------
# Trace identity
# ---------------------------------------------------------------------------


class TestTraceIdentity:
    def test_assigns_every_expected_fiber_and_order(self, tmp_path, monkeypatch):
        image, truth = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)

        clusters, identities = tracer._detect_traces("GREEN", image)

        assert list(identities.index) == _trace_labels(3)
        assert identities.notna().all()
        for (order, fiber), index in identities.items():
            cluster = clusters[int(index)]
            middle = cluster["columns"] == image.shape[1] // 2
            assert cluster["rows"][middle].mean() == pytest.approx(
                truth[(order, fiber)][image.shape[1] // 2], abs=1.0
            )

    def test_labels_are_unshifted_when_edge_traces_are_absent(
        self, tmp_path, monkeypatch
    ):
        dropped = [(0, "SKY"), (2, "CAL")]
        image, truth = _synthetic_flat(drop=dropped)
        tracer = _tracer(tmp_path, image, monkeypatch)

        clusters, identities = tracer._detect_traces("GREEN", image)

        assert list(identities.index) == _trace_labels(3)
        assert set(identities[identities.isna()].index) == set(dropped)
        for (order, fiber), index in identities[identities.notna()].items():
            cluster = clusters[int(index)]
            middle = cluster["columns"] == image.shape[1] // 2
            assert cluster["rows"][middle].mean() == pytest.approx(
                truth[(order, fiber)][image.shape[1] // 2], abs=1.0
            )

    def test_discards_a_group_beyond_the_expected_order_count(
        self, tmp_path, monkeypatch, caplog
    ):
        image, _ = _synthetic_flat(norder=3)
        tracer = _tracer(tmp_path, image, monkeypatch, norder=2)

        with caplog.at_level("WARNING"):
            _, identities = tracer._detect_traces("GREEN", image)

        assert list(identities.index) == _trace_labels(2)
        assert identities.notna().all()
        assert "3 fiber groups detected but 2 orders expected" in caplog.text


# ---------------------------------------------------------------------------
# End-to-end tracing
# ---------------------------------------------------------------------------


class TestMakeMasterOrderTrace:
    def test_traces_a_synthetic_flat(self, tmp_path, monkeypatch):
        image, truth = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)
        output_dir = tmp_path / "out"

        tables = tracer.make_master_order_trace(output_dir=output_dir)
        table = tables["GREEN"]

        assert list(table.columns) == _TRACE_FIELDS
        assert len(table) == len(truth)
        labels = list(zip(table["Order"], table["Fiber"], strict=True))
        assert labels == _trace_labels(3)
        assert (table["Status"] == "full").all()
        assert ((table["BottomEdge"] > 0) & (table["TopEdge"] > 0)).all()

        columns = np.linspace(0, image.shape[1] - 1, 9)
        for row in table.itertuples(index=False):
            coeffs = [getattr(row, f"Coeff{i}") for i in range(4)]
            fitted = np.polynomial.polynomial.polyval(columns, coeffs)
            expected = truth[(row.Order, row.Fiber)][columns.astype(int)]
            assert np.max(np.abs(fitted - expected)) < 0.5

        written = pd.read_csv(output_dir / "order_trace_green.csv", index_col=0)
        pd.testing.assert_frame_equal(written, table)

    def test_missing_trace_keeps_its_row(self, tmp_path, monkeypatch):
        image, truth = _synthetic_flat(drop=[(1, "SCI2")])
        tracer = _tracer(tmp_path, image, monkeypatch)

        table = tracer.make_master_order_trace(output_dir=tmp_path / "out")["GREEN"]
        missing = table[table["Status"] == "missing"]

        assert len(table) == len(truth)
        labels = list(zip(missing["Order"], missing["Fiber"], strict=True))
        assert labels == [(1, "SCI2")]
        assert missing[["Coeff0", "BottomEdge", "X1"]].isna().all(axis=None)
        assert (table[table["Status"] != "missing"]["Status"] == "full").all()

    def test_poly_degree_sets_the_coefficient_fields(self, tmp_path, monkeypatch):
        image, _ = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch, poly_degree=5)

        table = tracer.make_master_order_trace(output_dir=tmp_path / "out")["GREEN"]

        assert list(table.columns)[:6] == [f"Coeff{i}" for i in range(6)]

    def test_reports_progress_only_after_running(self, tmp_path, monkeypatch, capsys):
        image, _ = _synthetic_flat()
        tracer = _tracer(tmp_path, image, monkeypatch)

        tracer.info()
        assert "has not been called" in capsys.readouterr().out

        tracer.make_master_order_trace(output_dir=tmp_path / "out")
        tracer.info()
        assert "OrderTrace" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Configuration, input validation, and output
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_only_polynomial_degree_is_module_configurable(self):
        tracer = OrderTrace(
            "KP.20240405.00020.86_master_flat_L1.fits",
            {"poly_degree": 2, "cal_flux_ratio": 2.5, "sample_count": 3},
        )

        assert tracer.poly_degree == 2
        assert not hasattr(tracer, "cal_flux_ratio")
        assert not hasattr(tracer, "sample_count")

    def test_config_does_not_require_data_directory_paths(self, tmp_path):
        config_path = tmp_path / "order_trace.toml"
        config_path.write_text(
            '[TRACES]\nchips = ["GREEN", "RED"]\n[ORDER_TRACE]\npoly_degree = 2\n'
        )

        tracer = OrderTrace(
            "KP.20240405.00020.86_master_flat_L1.fits", ConfigHandler(config_path)
        )

        assert tracer.chips == ["GREEN", "RED"]
        assert tracer.poly_degree == 2

    def test_rejects_an_unusable_config(self):
        with pytest.raises(TypeError, match="config must be"):
            OrderTrace("KP.20240405.00020.86_master_flat_L1.fits", config=1)

    def test_rejects_a_negative_polynomial_degree(self):
        tracer = OrderTrace(
            "KP.20240405.00020.86_master_flat_L1.fits", {"poly_degree": -1}
        )
        with pytest.raises(ValueError, match="poly_degree must be non-negative"):
            tracer._coefficient_fields()


class TestMasterFlatLoading:
    def test_loads_vnext_master_flat(self, tmp_path, monkeypatch):
        master_path = tmp_path / "KP.20240405.00020.86_master_flat_L1.fits"
        master_path.touch()
        master_flat = StubMasterFlat({"GREEN": np.ones((8, 8), dtype=np.float32)})
        calls = {}

        class FakeKPFMasterL1:
            @classmethod
            def from_fits(cls, filename):
                calls["filename"] = filename
                return master_flat

        monkeypatch.setattr(order_trace_module, "KPFMasterL1", FakeKPFMasterL1)

        tracer = OrderTrace(master_path)

        assert tracer._load_master_flat() is master_flat
        assert calls["filename"] == str(master_path)

    def test_rejects_non_flat_master(self, tmp_path, monkeypatch):
        master_path = tmp_path / "KP.20240405.00020.86_master_dark_L1.fits"
        master_path.touch()
        dark_master = StubMasterFlat(
            {"GREEN": np.ones((8, 8), dtype=np.float32)}, master_type="dark"
        )

        class FakeKPFMasterL1:
            @classmethod
            def from_fits(cls, filename):
                return dark_master

        monkeypatch.setattr(order_trace_module, "KPFMasterL1", FakeKPFMasterL1)
        with pytest.raises(ValueError, match="not a vNext flat master"):
            OrderTrace(master_path)._load_master_flat()

    def test_reports_a_missing_master_flat(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Master flat not found"):
            OrderTrace(tmp_path / "absent_master_flat_L1.fits")._load_master_flat()


class TestCSVWriting:
    def test_refuses_overwrite_and_supports_chip_subset(self, tmp_path):
        tracer = OrderTrace("KP.20240405.00020.86_master_flat_L1.fits")
        table = pd.DataFrame(
            [
                {
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

        tracer._write_results({"GREEN": table}, tmp_path, overwrite=False)
        assert (tmp_path / "order_trace_green.csv").is_file()
        assert not (tmp_path / "order_trace_red.csv").exists()
        with pytest.raises(FileExistsError):
            tracer._write_results({"GREEN": table}, tmp_path, overwrite=False)
        tracer._write_results({"GREEN": table}, tmp_path, overwrite=True)


# ---------------------------------------------------------------------------
# Real-data test
# ---------------------------------------------------------------------------


class TestRealData:
    @pytest.mark.slow
    @pytest.mark.requires_testdata
    def test_real_20240405_master_flat(self, tmp_path):
        """Trace a real vNext master flat and check it against the references."""
        testdata = Path(__file__).parent.parent / "testdata"
        masters = sorted(testdata.glob("**/KP.20240405.*_master_flat_L1.fits"))
        if not masters:
            pytest.skip("a 20240405 vNext master flat is not installed")

        repo_root = Path(__file__).parent.parent.parent
        tracer = OrderTrace(masters[0])
        tables = tracer.make_master_order_trace(output_dir=tmp_path)

        assert len(tables["GREEN"]) == 175
        assert len(tables["RED"]) == 160
        assert (tables["GREEN"]["Status"] != "missing").sum() == 175
        assert (tables["RED"]["Status"] != "missing").sum() >= 158

        for chip in ("GREEN", "RED"):
            assert np.nanmedian(tracer._fit_rms[chip]) < 1.0
            assert np.nanmax(tracer._fit_rms[chip]) < 2.0
            assert (tmp_path / f"order_trace_{chip.lower()}.csv").is_file()

            # The reference tables are 1-based in Order and predate this module;
            # they are used only as independent truth for position and labelling.
            reference = pd.read_csv(
                repo_root / "reference" / f"order_trace_{chip.lower()}.csv",
                index_col=0,
            )
            reference["Order"] -= 1
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
