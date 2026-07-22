"""Tests for spectral order tracing from one vNext L1 master flat.

Numerical tests use synthetic master images and require no real data. The real
master-flat smoke test uses gitignored ``tests/testdata`` and is skipped when
the required vNext product is unavailable.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import kpfpipe.modules.masters.order_trace as order_trace_module
from kpfpipe.modules.masters import OrderTrace
from kpfpipe.modules.spectral_extraction import SpectralExtraction
from kpfpipe.utils.config import ConfigHandler

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
_FIBERS = ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]


class StubMasterFlat:
    """Minimal vNext master flat used by the numerical tests."""

    def __init__(self, images, master_type="flat"):
        self.data = {}
        for chip, image in images.items():
            self.data[f"{chip}_IMG"] = np.asarray(image, dtype=np.float32)
        self.headers = {"PRIMARY": {"MASTYPE": master_type}}


class StubScienceL1:
    """Minimal science L1 used for the extraction-compatibility assertion."""

    def __init__(self, images):
        self.data = {}
        for chip, image in images.items():
            self.data[f"{chip}_CCD"] = np.asarray(image, dtype=np.float32)
            self.data[f"{chip}_VAR"] = np.ones_like(image, dtype=np.float32)


def _synthetic_traces(norder=3, nrow=230, ncol=160, shift=0.0):
    """Return a curved master-flat image and its exact trace coefficients."""
    rng = np.random.default_rng(1398)
    image = rng.normal(2.0, 0.15, size=(nrow, ncol))
    x = np.arange(ncol, dtype=float)
    rows = []

    for order in range(1, norder + 1):
        for fiber_index, fiber in enumerate(_FIBERS):
            trace_index = (order - 1) * len(_FIBERS) + fiber_index
            coeffs = np.array(
                [18.0 + 13.0 * trace_index + shift, 0.018, 7.0e-5, -1.2e-7]
            )
            center = np.polynomial.polynomial.polyval(x, coeffs)
            y = np.arange(nrow, dtype=float)
            for column, row_center in enumerate(center):
                image[:, column] += 500.0 * np.exp(-0.5 * ((y - row_center) / 1.7) ** 2)
            rows.append((fiber, order, coeffs))

    return image.astype(np.float32), rows


def _write_seed_table(path, trace_rows, seed_offset=0.0):
    rows = []
    for fiber, order, coeffs in trace_rows:
        seed = coeffs.copy()
        seed[0] += seed_offset
        rows.append(
            {
                "Coeff0": seed[0],
                "Coeff1": seed[1],
                "Coeff2": seed[2],
                "Coeff3": seed[3],
                "BottomEdge": 5.0,
                "TopEdge": 5.0,
                "X1": 0,
                "X2": 159,
                "Fiber": fiber,
                "Order": order,
            }
        )
    pd.DataFrame(rows, columns=_TRACE_COLUMNS).to_csv(path)


def _write_era_table(path, start="2024-02-23 12:00:01", end="2024-11-01 00:00:00"):
    pd.DataFrame(
        [
            {
                "INSTERA": 2.0,
                "UT_start_date": start,
                "UT_end_date": end,
                "Comments": "synthetic test era",
            }
        ]
    ).to_csv(path, index=False)


@pytest.fixture
def synthetic_setup(tmp_path, monkeypatch):
    image, trace_rows = _synthetic_traces()
    seed_path = tmp_path / "seed.csv"
    era_path = tmp_path / "eras.csv"
    _write_seed_table(seed_path, trace_rows, seed_offset=1.5)
    _write_era_table(era_path)
    monkeypatch.setattr(order_trace_module, "_ERA_DEFINITIONS_PATH", era_path)
    monkeypatch.setattr(
        order_trace_module,
        "_ORDER_TRACE_PATHS",
        {"GREEN": seed_path, "RED": seed_path},
    )
    config = {"poly_degree": 3}
    return StubMasterFlat({"GREEN": image}), trace_rows, config


class TestTraceRecovery:
    def test_edge_center_recovers_flat_topped_profile(self):
        rng = np.random.default_rng(1398)
        rows = np.arange(90, dtype=float)
        expected_center = 41.35
        half_width = 4.4
        profile = 0.5 * (
            np.tanh((rows - (expected_center - half_width)) / 0.45)
            - np.tanh((rows - (expected_center + half_width)) / 0.45)
        )
        image = np.repeat((10.0 + 1000.0 * profile)[:, None], 31, axis=1)
        image += rng.normal(0.0, 2.0, image.shape)

        tracer = OrderTrace("KP.20240923.00139.44_master_flat_L1.fits")
        measured = tracer._local_edge_center(
            image,
            column=15,
            guess=expected_center + 1.5,
            row_half_window=9,
            profile_smoothing_sigma=0.7,
        )

        assert measured == pytest.approx(expected_center, abs=0.15)

    def test_recovers_cubic_centers_widths_and_schema(
        self, tmp_path, monkeypatch, synthetic_setup
    ):
        master_flat, expected, config = synthetic_setup
        master_path = tmp_path / "KP.20240923.00139.44_master_flat_L1.fits"
        tracer = OrderTrace(master_path, config)
        monkeypatch.setattr(tracer, "_load_master_flat", lambda: master_flat)

        result = tracer.make_master_order_trace(
            chips=["GREEN"], output_dir=tmp_path / "out"
        )
        table = result["GREEN"]

        assert list(table.columns) == _TRACE_COLUMNS
        assert len(table) == len(expected)
        assert not table.duplicated(["Fiber", "Order"]).any()
        assert (table["BottomEdge"] > 0).all()
        assert (table["TopEdge"] > 0).all()
        assert (table["X1"] >= 0).all()
        assert (table["X2"] <= master_flat.data["GREEN_IMG"].shape[1] - 1).all()

        x = np.arange(master_flat.data["GREEN_IMG"].shape[1], dtype=float)
        for (_, _, expected_coeffs), measured in zip(
            expected, table.itertuples(index=False), strict=True
        ):
            measured_coeffs = np.array(
                [getattr(measured, f"Coeff{i}") for i in range(4)]
            )
            expected_y = np.polynomial.polynomial.polyval(x, expected_coeffs)
            measured_y = np.polynomial.polynomial.polyval(x, measured_coeffs)
            assert np.median(np.abs(measured_y - expected_y)) < 1.0

        written = pd.read_csv(tmp_path / "out/order_trace_green.csv", index_col=0)
        pd.testing.assert_frame_equal(written, table, check_dtype=False)

        science_l1 = StubScienceL1({"GREEN": master_flat.data["GREEN_IMG"]})
        extractor = SpectralExtraction(science_l1)
        extractor.order_trace = {
            "GREEN": table.set_index(["Fiber", "Order"]).sort_index()
        }
        extractor.order_trace_path = {"GREEN": str(tmp_path / "out")}
        data, variance, weight = extractor._get_orderlet_pixels("GREEN", "SCI2", 2)
        assert data.shape == variance.shape == weight.shape
        assert data.shape[1] == master_flat.data["GREEN_IMG"].shape[1]
        assert np.nanmax(weight) == 1.0

    def test_higher_degree_extends_coefficient_schema(
        self, tmp_path, monkeypatch, synthetic_setup
    ):
        master_flat, expected, _ = synthetic_setup
        master_path = tmp_path / "KP.20240923.00139.44_master_flat_L1.fits"
        tracer = OrderTrace(master_path, {"poly_degree": 5})
        monkeypatch.setattr(tracer, "_load_master_flat", lambda: master_flat)

        table = tracer.make_master_order_trace(
            chips=["GREEN"], output_dir=tmp_path / "degree-five"
        )["GREEN"]

        expected_columns = [f"Coeff{i}" for i in range(6)] + _TRACE_COLUMNS[4:]
        assert list(table.columns) == expected_columns
        assert len(table) == len(expected)
        assert np.isfinite(table[[f"Coeff{i}" for i in range(6)]]).all().all()

    def test_manual_new_era_anchor_translates_seed(
        self, tmp_path, monkeypatch, synthetic_setup
    ):
        _, _, config = synthetic_setup
        shifted_image, shifted_rows = _synthetic_traces(shift=9.0)
        unknown_master = StubMasterFlat({"GREEN": shifted_image})
        master_path = tmp_path / "KP.20410101.00000.00_master_flat_L1.fits"
        tracer = OrderTrace(master_path, config)
        monkeypatch.setattr(tracer, "_load_master_flat", lambda: unknown_master)

        with pytest.raises(ValueError, match="outside the defined instrument eras"):
            tracer.make_master_order_trace(
                chips=["GREEN"], output_dir=tmp_path / "missing-anchor"
            )

        cal_order3 = next(
            coeffs[0]
            for fiber, order, coeffs in shifted_rows
            if fiber == "CAL" and order == 3
        )
        result = tracer.make_master_order_trace(
            chips=["GREEN"],
            output_dir=tmp_path / "manual",
            cal_order3_y={"GREEN": cal_order3},
        )

        assert tracer._instrument_era is None
        assert tracer._anchor_shifts["GREEN"] == pytest.approx(7.5)
        measured_anchor = (
            result["GREEN"]
            .loc[
                (result["GREEN"]["Fiber"] == "CAL") & (result["GREEN"]["Order"] == 3),
                "Coeff0",
            ]
            .item()
        )
        assert measured_anchor == pytest.approx(cal_order3, abs=1.0)


class TestEraAndInputValidation:
    def test_only_polynomial_degree_is_module_configurable(self):
        tracer = OrderTrace(
            "KP.20240923.00139.44_master_flat_L1.fits",
            {
                "poly_degree": 2,
                "sample_count": 3,
                "era_definitions_path": "/tmp/not-used.csv",
            },
        )

        assert tracer.poly_degree == 2
        assert not hasattr(tracer, "sample_count")
        assert not hasattr(tracer, "era_definitions_path")

    def test_config_does_not_require_data_directory_paths(self, tmp_path):
        config_path = tmp_path / "order_trace.toml"
        config_path.write_text(
            '[TRACES]\nchips = ["GREEN", "RED"]\n[ORDER_TRACE]\npoly_degree = 2\n'
        )

        tracer = OrderTrace(
            "KP.20240923.00139.44_master_flat_L1.fits",
            ConfigHandler(config_path),
        )

        assert tracer.chips == ["GREEN", "RED"]
        assert tracer.poly_degree == 2

    def test_manual_anchor_requires_every_requested_chip(self):
        tracer = OrderTrace("KP.20240923.00139.44_master_flat_L1.fits")
        with pytest.raises(ValueError, match="missing requested chips"):
            tracer._validate_manual_anchors(["GREEN", "RED"], {"GREEN": 272.0})


class TestMasterFlatLoading:
    def test_loads_vnext_master_flat(self, tmp_path, monkeypatch):
        master_path = tmp_path / "KP.20240923.00139.44_master_flat_L1.fits"
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
        result = tracer._load_master_flat()

        assert result is master_flat
        assert calls["filename"] == str(master_path)
        tracer._master_flat = result
        assert tracer._chip_image("GREEN").shape == (8, 8)

    def test_rejects_non_flat_master(self, tmp_path, monkeypatch):
        master_path = tmp_path / "KP.20240923.00139.44_master_dark_L1.fits"
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


class TestCSVWriting:
    def test_refuses_overwrite_and_supports_chip_subset(self, tmp_path):
        tracer = OrderTrace("KP.20240923.00139.44_master_flat_L1.fits")
        table = pd.DataFrame(
            [
                {
                    "Coeff0": 10.0,
                    "Coeff1": 0.0,
                    "Coeff2": 0.0,
                    "Coeff3": 0.0,
                    "BottomEdge": 2.0,
                    "TopEdge": 2.0,
                    "X1": 0,
                    "X2": 10,
                    "Fiber": "SKY",
                    "Order": 1,
                }
            ],
            columns=_TRACE_COLUMNS,
        )

        tracer._write_results({"GREEN": table}, tmp_path, overwrite=False)
        assert (tmp_path / "order_trace_green.csv").is_file()
        assert not (tmp_path / "order_trace_red.csv").exists()
        with pytest.raises(FileExistsError):
            tracer._write_results({"GREEN": table}, tmp_path, overwrite=False)
        tracer._write_results({"GREEN": table}, tmp_path, overwrite=True)


@pytest.mark.slow
@pytest.mark.requires_testdata
def test_real_20240923_master_flat(tmp_path):
    """Smoke-test a vNext 2024 master flat when local testdata are installed."""
    testdata = Path(__file__).parent.parent / "testdata"
    masters = sorted(testdata.glob("**/KP.20240923.*_master_flat_L1.fits"))
    if not masters:
        pytest.skip("a 20240923 vNext master flat is not installed")

    tracer = OrderTrace(masters[0])
    tables = tracer.make_master_order_trace(output_dir=tmp_path)

    assert len(tables["GREEN"]) == 175
    assert len(tables["RED"]) == 159
    for chip in ("GREEN", "RED"):
        assert np.nanmedian(tracer._fit_rms[chip]) < 1.0
        assert np.nanmax(tracer._fit_rms[chip]) < 2.0
    assert (tmp_path / "order_trace_green.csv").is_file()
    assert (tmp_path / "order_trace_red.csv").is_file()
