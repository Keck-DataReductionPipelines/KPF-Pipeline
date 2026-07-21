"""Tests for single-wideflat spectral order tracing.

Numerical tests use synthetic assembled images and require no real data. The
real-wideflat smoke test uses gitignored ``tests/testdata`` and is skipped when
the required raw exposure or master bias is unavailable.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import kpfpipe.modules.order_trace as order_trace_module
from kpfpipe.modules.order_trace import OrderTrace
from kpfpipe.modules.spectral_extraction import SpectralExtraction

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


class StubL1:
    """Minimal assembled frame used by the numerical tests."""

    def __init__(self, images, date_obs="2024-09-23T00:02:19"):
        self.data = {}
        for chip, image in images.items():
            self.data[f"{chip}_CCD"] = np.asarray(image, dtype=np.float32)
            self.data[f"{chip}_VAR"] = np.ones_like(image, dtype=np.float32)
        self.headers = {
            "PRIMARY": {"DATE-OBS": date_obs},
            "RECEIPT": {"BIASFILE": "/tmp/master_bias.fits"},
        }
        self.receipt = []

    def receipt_add_entry(self, module, args, status):
        self.receipt.append((module, args, status))


def _synthetic_traces(norder=3, nrow=230, ncol=160, shift=0.0):
    """Return a curved wideflat image and its exact trace coefficients."""
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
def synthetic_setup(tmp_path):
    image, trace_rows = _synthetic_traces()
    seed_path = tmp_path / "seed.csv"
    era_path = tmp_path / "eras.csv"
    _write_seed_table(seed_path, trace_rows, seed_offset=1.5)
    _write_era_table(era_path)
    config = {
        "chips": ["GREEN"],
        "sample_count": 33,
        "col_half_window": 1,
        "background_smoothing_sigma": 8.0,
        "candidate_prominence_sigma": 2.0,
        "candidate_distance_pixels": 6,
        "width_half_window": 8,
        "era_definitions_path": str(era_path),
        "order_trace_green_path": str(seed_path),
        "order_trace_red_path": str(seed_path),
    }
    return StubL1({"GREEN": image}), trace_rows, config


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

        tracer = OrderTrace(
            "wideflat.fits",
            {
                "row_half_window": 9,
                "col_half_window": 2,
                "profile_smoothing_sigma": 0.7,
            },
        )
        measured = tracer._local_edge_center(
            image, column=15, guess=expected_center + 1.5
        )

        assert measured == pytest.approx(expected_center, abs=0.15)

    def test_recovers_cubic_centers_widths_and_schema(
        self, tmp_path, monkeypatch, synthetic_setup
    ):
        l1_obj, expected, config = synthetic_setup
        tracer = OrderTrace(tmp_path / "wideflat.fits", config)
        monkeypatch.setattr(tracer, "_preprocess", lambda chips: l1_obj)

        result = tracer.perform(output_dir=tmp_path / "out")
        table = result["GREEN"]

        assert list(table.columns) == _TRACE_COLUMNS
        assert len(table) == len(expected)
        assert not table.duplicated(["Fiber", "Order"]).any()
        assert (table["BottomEdge"] > 0).all()
        assert (table["TopEdge"] > 0).all()
        assert (table["X1"] >= 0).all()
        assert (table["X2"] <= l1_obj.data["GREEN_CCD"].shape[1] - 1).all()

        x = np.arange(l1_obj.data["GREEN_CCD"].shape[1], dtype=float)
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
        assert ("order_trace", "", "PASS") in l1_obj.receipt

        extractor = SpectralExtraction(l1_obj)
        extractor.order_trace = {
            "GREEN": table.set_index(["Fiber", "Order"]).sort_index()
        }
        extractor.order_trace_path = {"GREEN": str(tmp_path / "out")}
        data, variance, weight = extractor._get_orderlet_pixels("GREEN", "SCI2", 2)
        assert data.shape == variance.shape == weight.shape
        assert data.shape[1] == l1_obj.data["GREEN_CCD"].shape[1]
        assert np.nanmax(weight) == 1.0

    def test_manual_new_era_anchor_translates_seed(
        self, tmp_path, monkeypatch, synthetic_setup
    ):
        _, _, config = synthetic_setup
        shifted_image, shifted_rows = _synthetic_traces(shift=9.0)
        unknown_l1 = StubL1({"GREEN": shifted_image}, date_obs="2041-01-01T00:00:00")
        tracer = OrderTrace(tmp_path / "wideflat.fits", config)
        monkeypatch.setattr(tracer, "_preprocess", lambda chips: unknown_l1)

        with pytest.raises(ValueError, match="outside the defined instrument eras"):
            tracer.perform(output_dir=tmp_path / "missing-anchor")

        cal_order3 = next(
            coeffs[0]
            for fiber, order, coeffs in shifted_rows
            if fiber == "CAL" and order == 3
        )
        result = tracer.perform(
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
    def test_era_boundaries_and_gap(self, tmp_path):
        tracer = OrderTrace("wideflat.fits")
        tracer.era_definitions_path = str(tmp_path / "eras.csv")
        _write_era_table(tracer.era_definitions_path)

        assert tracer._resolve_instrument_era("2024-02-23T12:00:01") == 2.0
        assert tracer._resolve_instrument_era("2024-11-01T00:00:00") == 2.0
        assert tracer._resolve_instrument_era("2024-11-01T06:00:00") is None

    def test_manual_anchor_requires_every_requested_chip(self):
        tracer = OrderTrace("wideflat.fits")
        with pytest.raises(ValueError, match="missing requested chips"):
            tracer._validate_manual_anchors(["GREEN", "RED"], {"GREEN": 272.0})

    @pytest.mark.parametrize("chips", [[], ["BLUE"], ["GREEN", "GREEN"]])
    def test_invalid_chip_selection_fails(self, chips):
        with pytest.raises(ValueError):
            OrderTrace._normalise_chips(chips)


class TestPreprocessing:
    def test_pipeline_modules_are_called_with_bias_only(self, tmp_path, monkeypatch):
        wideflat = tmp_path / "wideflat.fits"
        wideflat.touch()
        l1_obj = StubL1({"GREEN": np.ones((8, 8), dtype=np.float32)})
        calls = {}

        class FakeKPF0:
            @classmethod
            def from_fits(cls, filename):
                calls["filename"] = filename
                return "l0"

        class FakeAssembly:
            def __init__(self, l0_obj, config):
                calls["assembly_init"] = (l0_obj, config)

            def perform(self, chips):
                calls["assembly_chips"] = chips
                return l1_obj

        class FakeAssociation:
            def __init__(self, assembled, config):
                calls["association_init"] = (assembled, config)

            def perform(self, cal_types):
                calls["cal_types"] = cal_types
                return l1_obj

        class FakeProcessing:
            def __init__(self, assembled, config):
                calls["processing_init"] = (assembled, config)

            def perform(self, chips, *, bias, dark, flat):
                calls["processing"] = (chips, bias, dark, flat)
                return l1_obj

        monkeypatch.setattr(order_trace_module, "KPF0", FakeKPF0)
        monkeypatch.setattr(order_trace_module, "ImageAssembly", FakeAssembly)
        monkeypatch.setattr(
            order_trace_module, "CalibrationAssociation", FakeAssociation
        )
        monkeypatch.setattr(order_trace_module, "ImageProcessing", FakeProcessing)

        tracer = OrderTrace(wideflat, config={"KPF_MASTERS_OUTPUT": str(tmp_path)})
        result = tracer._preprocess(["GREEN"])

        assert result is l1_obj
        assert calls["filename"] == str(wideflat)
        assert calls["assembly_chips"] == ["GREEN"]
        assert calls["cal_types"] == ["bias"]
        assert calls["processing"] == (["GREEN"], True, False, False)
        assert tracer._bias_path == "/tmp/master_bias.fits"


class TestCSVWriting:
    def test_refuses_overwrite_and_supports_chip_subset(self, tmp_path):
        tracer = OrderTrace("wideflat.fits")
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
def test_real_20240923_wideflat(tmp_path):
    """Smoke-test the issue-1398 frame when its local truth data are installed."""
    testdata = Path(__file__).parent.parent / "testdata"
    wideflat = testdata / "L0/20240923/KP.20240923.00139.44.fits"
    masters = list(testdata.glob("**/*master_bias*L1.fits"))
    if not wideflat.is_file() or not masters:
        pytest.skip("20240923 wideflat and master bias are not installed")

    tracer = OrderTrace(wideflat, config={"KPF_MASTERS_OUTPUT": str(testdata)})
    tables = tracer.perform(output_dir=tmp_path)

    assert len(tables["GREEN"]) == 175
    assert len(tables["RED"]) == 159
    for chip in ("GREEN", "RED"):
        assert np.nanmedian(tracer._fit_rms[chip]) < 1.0
        assert np.nanmax(tracer._fit_rms[chip]) < 2.0
    assert (tmp_path / "order_trace_green.csv").is_file()
    assert (tmp_path / "order_trace_red.csv").is_file()
