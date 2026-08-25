"""Tests for the SpectralExtraction module (L1 -> L2).

Extraction-algorithm and perform() tests run on synthetic arrays, the latter with
extract_ffi monkeypatched; the real-L0 regression class is marked slow.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

import kpfpipe.modules.spectral_extraction as se_module
from kpfpipe import DETECTOR
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.modules.image_assembly import ImageAssembly
from kpfpipe.modules.spectral_extraction import SpectralExtraction

from ._dtype_policy import FLUX, assert_dtype

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
NCOL = DETECTOR["ccd"]["ncol"]

TESTDATA_L0_DIR = Path(__file__).parent.parent / "testdata" / "L0" / "20240405"
L0_FILE = str(TESTDATA_L0_DIR / "KP.20240405.00020.86.fits")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_l1(tmp_path):
    """Minimal KPF1 sufficient for to_kpf2() and SpectralExtraction init."""
    fn = str(tmp_path / "kpf_L1_20240101T000000.fits")
    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-01-01T00:00:00"
    primary.header["JD_UTC"] = 2460310.5
    primary.header["INSTERA"] = "1.0"
    green_ccd = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="GREEN_CCD")
    green_var = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="GREEN_VAR")
    red_ccd = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="RED_CCD")
    red_var = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="RED_VAR")
    fits.HDUList([primary, green_ccd, green_var, red_ccd, red_var]).writeto(
        fn, overwrite=True
    )
    return KPF1.from_fits(fn)


# ---------------------------------------------------------------------------
# Box extraction (unit tests -- no data or fixtures required)
# ---------------------------------------------------------------------------


class TestBoxExtraction:
    def test_basic_extraction(self):
        D = np.ones((5, 10), dtype=np.float32)
        V = np.ones((5, 10), dtype=np.float32)
        flux, var = SpectralExtraction._box_extraction(D, V)
        assert flux.shape == (10,)
        assert var.shape == (10,)
        np.testing.assert_allclose(flux, 5.0)

    def test_with_weights(self):
        D = np.ones((5, 10), dtype=np.float32)
        V = np.ones((5, 10), dtype=np.float32)
        W = np.ones((5, 10), dtype=np.float32)
        W[0, :] = 0.5
        W[-1, :] = 0.5
        flux_full, _ = SpectralExtraction._box_extraction(D, V)
        flux_w, _ = SpectralExtraction._box_extraction(D, V, W=W)
        assert np.all(flux_w < flux_full)

    def test_with_sky_subtraction(self):
        D = np.full((5, 10), 10.0, dtype=np.float32)
        V = np.ones((5, 10), dtype=np.float32)
        S = np.full((5, 10), 3.0, dtype=np.float32)
        flux, _ = SpectralExtraction._box_extraction(D, V, S=S)
        np.testing.assert_allclose(flux, 5 * (10.0 - 3.0))

    def test_with_mask(self):
        D = np.ones((5, 10), dtype=np.float32)
        V = np.ones((5, 10), dtype=np.float32)
        M = np.ones((5, 10), dtype=np.float32)
        M[2, :] = 0
        flux, _ = SpectralExtraction._box_extraction(D, V, M=M)
        # M is renormalised to nrow, so masking a row redistributes its weight.
        np.testing.assert_allclose(flux, 5.0)

    def test_fully_masked_column_raises(self):
        D = np.ones((5, 10), dtype=np.float32)
        V = np.ones((5, 10), dtype=np.float32)
        M = np.ones((5, 10), dtype=np.float32)
        M[:, 3] = 0
        with pytest.raises(ValueError, match="Fully masked"):
            SpectralExtraction._box_extraction(D, V, M=M)

    def test_variance_propagation(self):
        D = np.ones((5, 10), dtype=np.float32)
        V = np.full((5, 10), 4.0, dtype=np.float32)
        _, var = SpectralExtraction._box_extraction(D, V)
        np.testing.assert_allclose(var, 5 * 4.0)


# ---------------------------------------------------------------------------
# Dtype provenance
# ---------------------------------------------------------------------------


class TestDtypeProvenance:
    """L2 FLUX/VAR are float32; the box-extraction kernel preserves float32."""

    def test_box_extraction_preserves_float32(self):
        D = np.ones((5, 32), dtype=np.float32)
        V = np.ones((5, 32), dtype=np.float32)
        flux_1d, var_1d = SpectralExtraction._box_extraction(D, V)
        assert_dtype(flux_1d, FLUX, "box flux_1d")
        assert_dtype(var_1d, FLUX, "box var_1d")

    def test_l2_flux_var_float32(self, minimal_l1, monkeypatch):
        norder = {"GREEN": NORDER_GREEN, "RED": NORDER_RED}
        arrays = {
            f"{chip}_{fiber}_{q}": np.ones((norder[chip], NCOL), dtype=np.float32)
            for chip in ("GREEN", "RED")
            for fiber in ("CAL", "SCI1", "SCI2", "SCI3", "SKY")
            for q in ("FLUX", "VAR")
        }
        monkeypatch.setattr(
            SpectralExtraction,
            "extract_ffi",
            lambda self, chip, fibers, extraction_method, **kw: {
                k: v for k, v in arrays.items() if k.startswith(chip)
            },
        )
        l2 = SpectralExtraction(minimal_l1).perform()
        assert_dtype(l2.data["SCI2_FLUX"], FLUX, "L2 SCI2_FLUX")
        assert_dtype(l2.data["SCI2_VAR"], FLUX, "L2 SCI2_VAR")


# ---------------------------------------------------------------------------
# Unimplemented extraction stubs
# ---------------------------------------------------------------------------


class TestUnimplementedExtraction:
    def test_optimal_raises(self):
        D = V = np.ones((5, 10))
        with pytest.raises(NotImplementedError, match="[Oo]ptimal"):
            SpectralExtraction._optimal_extraction(D, V)

    def test_flat_relative_raises(self):
        D = V = np.ones((5, 10))
        with pytest.raises(NotImplementedError, match="[Ff]lat"):
            SpectralExtraction._flat_relative_extraction(D, V)


# ---------------------------------------------------------------------------
# perform() shape tests (monkeypatched -- no real data or trace files needed)
# ---------------------------------------------------------------------------


class TestPerformShapes:
    """perform() assembles GREEN and RED arrays into correctly shaped KPF2 traces."""

    @pytest.fixture
    def mock_ffi_arrays(self):
        """Pre-built (chip, fiber) arrays matching real detector dimensions."""
        chips = ["GREEN", "RED"]
        fibers = ["CAL", "SCI1", "SCI2", "SCI3", "SKY"]
        norder = {"GREEN": NORDER_GREEN, "RED": NORDER_RED}
        arrays = {}
        for chip in chips:
            for fiber in fibers:
                n = norder[chip]
                arrays[f"{chip}_{fiber}_FLUX"] = np.ones((n, NCOL), dtype=np.float32)
                arrays[f"{chip}_{fiber}_VAR"] = np.ones((n, NCOL), dtype=np.float32)
        return arrays

    def test_whole_trace_concatenates_green_then_red(
        self, minimal_l1, mock_ffi_arrays, monkeypatch
    ):
        # Per-chip shapes would only restate the arrays the mock handed in; the
        # concatenation is the part perform() actually computes.
        monkeypatch.setattr(
            SpectralExtraction,
            "extract_ffi",
            lambda self, chip, fibers, extraction_method, **kwargs: {
                k: v for k, v in mock_ffi_arrays.items() if k.startswith(chip)
            },
        )
        l2 = SpectralExtraction(minimal_l1).perform()
        assert l2.level == 2
        assert l2.data["SCI2_FLUX"].shape == (NORDER_GREEN + NORDER_RED, NCOL)

    def test_every_fiber_lands_on_its_own_trace(self, minimal_l1, monkeypatch):
        # One fill value per fiber, so this sees a fiber written over another's
        # trace -- which a shape assertion cannot.
        fills = {"CAL": 1.0, "SCI1": 2.0, "SCI2": 3.0, "SCI3": 4.0, "SKY": 5.0}
        norder = {"GREEN": NORDER_GREEN, "RED": NORDER_RED}

        def mock_extract(self, chip, fibers, extraction_method, **kwargs):
            return {
                f"{chip}_{fiber}_{quantity}": np.full(
                    (norder[chip], NCOL), fill, dtype=np.float32
                )
                for fiber, fill in fills.items()
                for quantity in ("FLUX", "VAR")
            }

        monkeypatch.setattr(SpectralExtraction, "extract_ffi", mock_extract)
        l2 = SpectralExtraction(minimal_l1).perform()
        for fiber, fill in fills.items():
            for quantity in ("FLUX", "VAR"):
                array = l2.data[f"{fiber}_{quantity}"]
                assert array.shape == (NORDER_GREEN + NORDER_RED, NCOL)
                np.testing.assert_array_equal(array, fill)

    def test_green_red_slices_independent(self, minimal_l1, monkeypatch):
        def mock_extract(self, chip, fibers, extraction_method, **kwargs):
            fill = 1.0 if chip == "GREEN" else 2.0
            n = NORDER_GREEN if chip == "GREEN" else NORDER_RED
            return {
                f"{chip}_SCI2_FLUX": np.full((n, NCOL), fill, dtype=np.float32),
                f"{chip}_SCI2_VAR": np.full((n, NCOL), fill, dtype=np.float32),
            }

        monkeypatch.setattr(SpectralExtraction, "extract_ffi", mock_extract)

        se = SpectralExtraction(minimal_l1, config={"fibers": ["SCI2"]})
        l2 = se.perform(fibers=["SCI2"])

        np.testing.assert_array_equal(l2.data["GREEN_SCI2_FLUX"], 1.0)
        np.testing.assert_array_equal(l2.data["RED_SCI2_FLUX"], 2.0)

    def test_each_order_lands_on_its_own_row(self, monkeypatch):
        # The reference numbers orders from zero, so order n must land on row n.
        # Shape alone cannot see a rebase: every row stays populated either way.

        def mock_orderlet(self, chip, fiber, order, extraction_method=None):
            spectrum = np.full(100, order, dtype=np.float32)
            return spectrum, spectrum

        monkeypatch.setattr(SpectralExtraction, "extract_orderlet", mock_orderlet)

        se = SpectralExtraction(_StubL1("2024-06-01T11:08:33", "2.0"))
        l2_arrays = se.extract_ffi("GREEN", ["SCI1"])

        for order in range(NORDER_GREEN):
            np.testing.assert_array_equal(
                l2_arrays["GREEN_SCI1_FLUX"][order], float(order)
            )

    def test_receipt_chain(self, minimal_l1, mock_ffi_arrays, monkeypatch):
        monkeypatch.setattr(
            SpectralExtraction,
            "extract_ffi",
            lambda self, chip, fibers, extraction_method, **kwargs: {
                k: v for k, v in mock_ffi_arrays.items() if k.startswith(chip)
            },
        )
        se = SpectralExtraction(minimal_l1)
        l2 = se.perform()
        modules = l2.receipt["FUNCTION"].values
        assert "to_kpf2" in modules
        assert "spectral_extraction" in modules


# ---------------------------------------------------------------------------
# Regression tests (real L0 data -> assemble L1 -> extract)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.requires_testdata
class TestSpectralExtractionRealData:
    @pytest.fixture(scope="class")
    def l2_from_flat(self):
        l0 = KPF0.from_fits(L0_FILE)
        ia = ImageAssembly(l0)
        l1 = ia.perform()
        se = SpectralExtraction(l1)
        return se.perform(), se

    def test_returns_kpf2(self, l2_from_flat):
        l2, _ = l2_from_flat
        assert isinstance(l2, KPF2)

    @pytest.mark.parametrize(
        "key, expected_rows",
        [
            ("GREEN_SCI2_FLUX", NORDER_GREEN),
            ("RED_SCI2_FLUX", NORDER_RED),
            ("SCI2_FLUX", NORDER_GREEN + NORDER_RED),
        ],
    )
    def test_sci2_flux_shape(self, l2_from_flat, key, expected_rows):
        l2, _ = l2_from_flat
        assert l2.data[key].shape == (expected_rows, NCOL)

    def test_flux_positive(self, l2_from_flat):
        l2, _ = l2_from_flat
        assert np.nanmedian(l2.data["GREEN_SCI2_FLUX"]) > 0
        assert np.nanmedian(l2.data["RED_SCI2_FLUX"]) > 0

    def test_variance_positive(self, l2_from_flat):
        # A column outside its trace's valid span is NaN by design, not negative.
        l2, _ = l2_from_flat
        for key in ("GREEN_SCI2_VAR", "RED_SCI2_VAR"):
            var = l2.data[key]
            measured = var[np.isfinite(var)]
            assert measured.size > 0
            assert np.all(measured >= 0)

    def test_receipt_chain(self, l2_from_flat):
        l2, _ = l2_from_flat
        modules = l2.receipt["FUNCTION"].values
        assert "image_assembly" in modules
        assert "spectral_extraction" in modules


# ---------------------------------------------------------------------------
# TestPolynomialOrderTrace
# ---------------------------------------------------------------------------


class TestPolynomialOrderTrace:
    def test_uses_all_available_polynomial_coefficients(self):
        class StubL1:
            data = {
                "GREEN_CCD": np.zeros((100, 100), dtype=np.float32),
                "GREEN_VAR": np.ones((100, 100), dtype=np.float32),
            }

        trace = pd.DataFrame(
            [
                {
                    "Fiber": "SCI1",
                    "Order": 1,
                    "TopEdge": 5.0,
                    "BottomEdge": 5.0,
                    "X1": 0.0,
                    "X2": 99.0,
                    "Coeff0": 20.0,
                    "Coeff1": 0.0,
                    "Coeff2": 0.0,
                    "Coeff3": 0.0,
                    "Coeff4": 1.0e-7,
                }
            ]
        ).set_index(["Fiber", "Order"])
        extraction = SpectralExtraction(StubL1())
        extraction._order_trace = {"GREEN": trace}
        extraction._order_trace_path = "<stub>"

        _, _, _, row_min, row_max = extraction._get_orderlet_pixels(
            "GREEN", "SCI1", 1, return_coords=True
        )

        x = np.arange(100, dtype=np.float32)
        center = np.polynomial.polynomial.polyval(x, [20.0, 0.0, 0.0, 0.0, 1.0e-7])
        assert row_min == int(np.floor((center - 5.0).min()))
        assert row_max == int(np.ceil((center + 5.0).max()))


# ---------------------------------------------------------------------------
# TestOrderTraceErrors
# ---------------------------------------------------------------------------


class TestOrderTraceErrors:
    """Errors raised from _get_orderlet_pixels via extract_orderlet."""

    def _make_se(self, rows):
        """Build a SpectralExtraction with a pre-populated order_trace cache."""

        class StubL1:
            data = {
                "GREEN_CCD": np.zeros((100, 100), dtype=np.float32),
                "GREEN_VAR": np.ones((100, 100), dtype=np.float32),
            }
            headers = {"PRIMARY": {}}

        rows = [{"X1": 0.0, "X2": 99.0, **row} for row in rows]
        df = pd.DataFrame(rows).set_index(["Fiber", "Order"]).sort_index()
        se = SpectralExtraction(StubL1())
        se._order_trace = {"GREEN": df}
        se._order_trace_path = "<stub>"
        return se

    def test_missing_trace_raises_lookup_error(self):
        # Table has SCI1 order 1 only; asking for order 2 misses.
        rows = [
            {
                "Fiber": "SCI1",
                "Order": 1,
                "TopEdge": 5.0,
                "BottomEdge": 5.0,
                "Coeff0": 50.0,
                "Coeff1": 0.0,
                "Coeff2": 0.0,
                "Coeff3": 0.0,
            },
        ]
        se = self._make_se(rows)
        with pytest.raises(LookupError, match="No trace found"):
            se.extract_orderlet("GREEN", "SCI1", 2)

    def test_duplicate_trace_raises_value_error(self):
        # Two rows for the same (Fiber, Order) -- a corrupt reference file.
        rows = [
            {
                "Fiber": "SCI1",
                "Order": 1,
                "TopEdge": 5.0,
                "BottomEdge": 5.0,
                "Coeff0": 50.0,
                "Coeff1": 0.0,
                "Coeff2": 0.0,
                "Coeff3": 0.0,
            },
            {
                "Fiber": "SCI1",
                "Order": 1,
                "TopEdge": 5.0,
                "BottomEdge": 5.0,
                "Coeff0": 50.0,
                "Coeff1": 0.0,
                "Coeff2": 0.0,
                "Coeff3": 0.0,
            },
        ]
        se = self._make_se(rows)
        with pytest.raises(ValueError, match="Expected exactly one row"):
            se.extract_orderlet("GREEN", "SCI1", 1)


# ---------------------------------------------------------------------------
# TestValidColumnSpan
# ---------------------------------------------------------------------------


class TestValidColumnSpan:
    """Only X1..X2 carries flux, whichever detector edge the trace ran off."""

    def _make_se(self, center, x1, x2, slope=0.0):
        class StubL1:
            data = {
                "GREEN_CCD": np.full((100, 100), 1234.0, dtype=np.float32),
                "GREEN_VAR": np.ones((100, 100), dtype=np.float32),
            }
            headers = {"PRIMARY": {}}

        trace = pd.DataFrame(
            [
                {
                    "Fiber": "SCI1",
                    "Order": 0,
                    "TopEdge": 5.0,
                    "BottomEdge": 5.0,
                    "X1": x1,
                    "X2": x2,
                    "Coeff0": center,
                    "Coeff1": slope,
                    "Coeff2": 0.0,
                    "Coeff3": 0.0,
                }
            ]
        ).set_index(["Fiber", "Order"])
        se = SpectralExtraction(StubL1())
        se._order_trace = {"GREEN": trace}
        se._order_trace_path = "<stub>"
        return se

    @pytest.mark.parametrize("center", [3.0, 96.0])
    def test_columns_beyond_the_span_are_nan(self, center):
        # The bottom edge is the one that used to leak: a clamped aperture still
        # lands inside the box and would carry detector row 0 off as flux.
        se = self._make_se(center, x1=0.0, x2=59.0)

        flux_1d, var_1d = se.extract_orderlet("GREEN", "SCI1", 0)

        assert np.all(np.isfinite(flux_1d[:60]))
        assert np.all(np.isnan(flux_1d[60:]))
        assert np.all(np.isnan(var_1d[60:]))

    def test_a_span_covering_the_detector_nans_nothing(self):
        se = self._make_se(50.0, x1=0.0, x2=99.0)

        flux_1d, _ = se.extract_orderlet("GREEN", "SCI1", 0)

        assert np.all(np.isfinite(flux_1d))

    @pytest.mark.parametrize("center", [50.0, 50.5])
    def test_aperture_weights_sum_to_the_aperture_height(self, center):
        # TopEdge = BottomEdge = 5.0 is a 10-pixel aperture, so on a uniform
        # detector every carried column must weigh exactly 10 -- whole pixels
        # inside plus the two fractional edge rows. Existing tests assert only
        # WHERE W is non-zero; an off-by-one in edge_pixel_top/bottom or an
        # inverted frac_bot is a ~10% photometric error they all tolerate.
        se = self._make_se(center, x1=0.0, x2=99.0)

        _, _, W = se._get_orderlet_pixels("GREEN", "SCI1", 0)
        np.testing.assert_allclose(W.sum(axis=0), 10.0, atol=1e-5)

        flux_1d, _ = se.extract_orderlet("GREEN", "SCI1", 0)
        np.testing.assert_allclose(flux_1d, 10.0 * 1234.0, rtol=1e-5)

    def test_the_box_is_sized_from_the_carried_columns_alone(self):
        # This trace climbs a row per column, so past X2 its extrapolation runs
        # off the top of the detector and would size a box of rows the orderlet
        # never carries flux in.
        se = self._make_se(10.0, x1=0.0, x2=39.0, slope=1.0)

        _, _, W, row_min, row_max = se._get_orderlet_pixels(
            "GREEN", "SCI1", 0, return_coords=True
        )

        # Rows 10-49 are traced over columns 0-39, plus the 5 px aperture.
        assert (row_min, row_max) == (5, 54)
        assert W[:, :40].any(axis=0).all()
        assert not W[:, 40:].any()


# ---------------------------------------------------------------------------
# extract_ffi's missing-trace tolerance, and the extraction-method guard
# ---------------------------------------------------------------------------


class TestExtractFfiFailureTolerance:
    """One missing trace degrades to a NaN row; two or more fail loudly.

    The single-failure tolerance exists because in some KPF eras one trace does
    not fall on the detector. Widening it would let a systematically broken
    order trace produce an all-NaN L2 that only surfaces far downstream, as a
    zero CCF.
    """

    def _make_se(self, n_traced):
        """SpectralExtraction over 2 GREEN orders with only ``n_traced`` traced."""

        class StubL1:
            data = {
                "GREEN_CCD": np.full((100, 100), 1.0, dtype=np.float32),
                "GREEN_VAR": np.ones((100, 100), dtype=np.float32),
            }
            headers = {"PRIMARY": {}}

        rows = [
            {
                "Fiber": "SCI1",
                "Order": order,
                "TopEdge": 5.0,
                "BottomEdge": 5.0,
                "X1": 0.0,
                "X2": 99.0,
                "Coeff0": 20.0 + 20.0 * order,
                "Coeff1": 0.0,
                "Coeff2": 0.0,
                "Coeff3": 0.0,
            }
            for order in range(n_traced)
        ]
        trace = pd.DataFrame(
            rows,
            columns=[
                "Fiber",
                "Order",
                "TopEdge",
                "BottomEdge",
                "X1",
                "X2",
                "Coeff0",
                "Coeff1",
                "Coeff2",
                "Coeff3",
            ],
        ).set_index(["Fiber", "Order"])
        se = SpectralExtraction(StubL1())
        se._order_trace = {"GREEN": trace}
        se._order_trace_path = "<stub>"
        # Rebind rather than mutate: self.norder is the shared DETECTOR dict, so
        # an item assignment here would resize every later test's detector.
        se.norder = dict(se.norder, GREEN=2)
        return se

    def test_one_missing_trace_leaves_a_nan_row(self, caplog):
        se = self._make_se(n_traced=1)
        with caplog.at_level(logging.WARNING):
            arrays = se.extract_ffi("GREEN", ["SCI1"])
        flux = arrays["GREEN_SCI1_FLUX"]
        assert np.all(np.isfinite(flux[0]))
        assert np.all(np.isnan(flux[1]))
        assert "1 orderlet failed" in caplog.text

    def test_two_missing_traces_raise(self):
        se = self._make_se(n_traced=0)
        with pytest.raises(LookupError, match="Failed to extract 2"):
            se.extract_ffi("GREEN", ["SCI1"])


class TestExtractionMethodGuard:
    def test_unsupported_method_names_itself(self):
        # Dispatch is getattr(self, f"_{extraction_method}_extraction"), so a
        # config typo must not surface as a bare AttributeError mid-loop.
        se = TestValidColumnSpan()._make_se(50.0, x1=0.0, x2=99.0)
        with pytest.raises(AttributeError, match="Unsupported extraction"):
            se.extract_orderlet("GREEN", "SCI1", 0, extraction_method="boxx")


# ---------------------------------------------------------------------------
# TestOrderTraceSelection
# ---------------------------------------------------------------------------


_STUB_ERAS = """INSTERA,UT_start_date,UT_end_date,Comments
1.0,2022-11-09 00:00:01,2024-02-03 00:00:00,First science era
2.0,2024-02-23 12:00:01,2024-11-01 00:00:00,Second science era
2.5,2024-11-01 12:00:01,2025-01-01 00:00:00,Service Mission #2 (engineering)
"""

_STUB_TRACE = (
    "Chip,Fiber,Order,Coeff0,Coeff1,Coeff2,Coeff3,BottomEdge,TopEdge,X1,X2,"
    "PolyfitRMS,Status\n"
    "GREEN,SCI1,0,20.0,0.0,0.0,0.0,5.0,5.0,0,99,0.1,full\n"
    "GREEN,SCI1,1,,,,,,,,,,missing\n"
    "RED,SCI1,0,30.0,0.0,0.0,0.0,5.0,5.0,0,99,0.1,full\n"
)


def _stub_reference_tree(tmp_path, monkeypatch):
    """Stub the repo reference tree: three instrument eras, three order traces."""
    traces = tmp_path / "reference" / "order_traces"
    traces.mkdir(parents=True)
    (tmp_path / "reference" / "instrument_eras.csv").write_text(_STUB_ERAS)
    for datecode in ("20231101", "20240301", "20240501"):
        (traces / f"order_trace_{datecode}.csv").write_text(_STUB_TRACE)
    monkeypatch.setattr(se_module, "REPO_ROOT", str(tmp_path))
    return traces


class _StubL1:
    def __init__(self, date_obs, instera):
        self.obs_id = "KP.20240601.40113.00"
        self.data = {
            "GREEN_CCD": np.zeros((100, 100), dtype=np.float32),
            "GREEN_VAR": np.ones((100, 100), dtype=np.float32),
        }
        self.headers = {
            "PRIMARY": {
                # date_obs None stands for a frame the L0 could not date.
                "JD_UTC": pd.Timestamp(date_obs).to_julian_date() if date_obs else None,
                "INSTERA": instera,
            }
        }

    def set_keyword(self, key, value, ext=None):
        self.headers["PRIMARY"][key] = value


class TestOrderTraceSelection:
    def test_selects_the_most_recent_trace_of_the_frames_era(
        self, tmp_path, monkeypatch
    ):
        traces = _stub_reference_tree(tmp_path, monkeypatch)
        se = SpectralExtraction(_StubL1("2024-06-01T11:08:33", "2.0"))

        se._read_order_trace_reference()

        assert se._order_trace_path == str(traces / "order_trace_20240501.csv")

    def test_ignores_a_trace_measured_after_the_frame(self, tmp_path, monkeypatch):
        traces = _stub_reference_tree(tmp_path, monkeypatch)
        se = SpectralExtraction(_StubL1("2024-04-01T11:08:33", "2.0"))

        se._read_order_trace_reference()

        assert se._order_trace_path == str(traces / "order_trace_20240301.csv")

    def test_does_not_reach_into_a_neighbouring_era(self, tmp_path, monkeypatch):
        # The frame is in era 2.5, whose traces are all in eras 1.0 and 2.0.
        _stub_reference_tree(tmp_path, monkeypatch)
        se = SpectralExtraction(_StubL1("2024-12-01T11:08:33", "2.5"))

        with pytest.raises(FileNotFoundError, match="instrument era 2.5"):
            se._read_order_trace_reference()

    def test_a_frame_between_eras_fails_loudly(self, tmp_path, monkeypatch):
        # 2024-02-03 -> 2024-02-23 is uncovered by the era table.
        _stub_reference_tree(tmp_path, monkeypatch)
        se = SpectralExtraction(_StubL1("2024-02-10T11:08:33", "1.0"))

        with pytest.raises(ValueError, match="No KPF instrument era covers"):
            se._read_order_trace_reference()

    def test_an_undated_frame_fails_loudly(self, tmp_path, monkeypatch):
        # MJD-OBS absent at L0 leaves JD_UTC seeded but unset.
        _stub_reference_tree(tmp_path, monkeypatch)
        se = SpectralExtraction(_StubL1(None, "2.0"))

        with pytest.raises(ValueError, match="JD_UTC is None"):
            se._read_order_trace_reference()

    def test_an_undatable_frame_is_not_swallowed_as_a_missing_orderlet(
        self, tmp_path, monkeypatch
    ):
        # extract_ffi catches LookupError to NaN-fill an absent orderlet. An era
        # that cannot be inferred must not arrive disguised as one -- it must
        # abort on the first orderlet rather than be retried for every order.
        _stub_reference_tree(tmp_path, monkeypatch)
        se = SpectralExtraction(_StubL1(None, "2.0"))

        with pytest.raises(ValueError):
            se.extract_ffi("GREEN", ["SCI1"])

    def test_restamps_an_instera_that_disagrees_with_jd_utc(
        self, tmp_path, monkeypatch, caplog
    ):
        traces = _stub_reference_tree(tmp_path, monkeypatch)
        l1 = _StubL1("2024-06-01T11:08:33", "1.0")
        se = SpectralExtraction(l1)

        with caplog.at_level("WARNING"):
            se._read_order_trace_reference()

        assert "disagrees with instrument era 2.0" in caplog.text
        assert l1.headers["PRIMARY"]["INSTERA"] == "2.0"
        # The era from JD_UTC, not the stamped one, picks the trace.
        assert se._order_trace_path == str(traces / "order_trace_20240501.csv")

    def test_restamps_an_unset_instera_without_failing(self, tmp_path, monkeypatch):
        _stub_reference_tree(tmp_path, monkeypatch)
        l1 = _StubL1("2024-06-01T11:08:33", "UNKNOWN")
        se = SpectralExtraction(l1)

        se._read_order_trace_reference()

        assert l1.headers["PRIMARY"]["INSTERA"] == "2.0"

    def test_drops_missing_traces_and_keys_them_by_fiber_and_order(
        self, tmp_path, monkeypatch
    ):
        _stub_reference_tree(tmp_path, monkeypatch)
        se = SpectralExtraction(_StubL1("2024-06-01T11:08:33", "2.0"))

        se._read_order_trace_reference()

        assert set(se._order_trace) == {"GREEN", "RED"}
        assert se._order_trace["GREEN"].index.tolist() == [("SCI1", 0)]

    def test_writes_the_trace_path_to_the_l2_receipt(
        self, minimal_l1, tmp_path, monkeypatch
    ):
        _stub_reference_tree(tmp_path, monkeypatch)

        def extract(self, chip, fibers, extraction_method, **kwargs):
            self._read_order_trace_reference()
            return {}

        monkeypatch.setattr(SpectralExtraction, "extract_ffi", extract)
        l2 = SpectralExtraction(minimal_l1).perform(fibers=[])

        assert l2.headers["RECEIPT"]["TRACEREF"] == "order_trace_20231101.csv"

    def test_writes_the_corrected_instera_to_the_l2(
        self, minimal_l1, tmp_path, monkeypatch
    ):
        # to_kpf2() copies the L1 PRIMARY before the era is inferred, so the L2
        # only carries the correction because _set_headers restamps it.
        _stub_reference_tree(tmp_path, monkeypatch)
        minimal_l1.headers["PRIMARY"]["INSTERA"] = "2.0"

        def extract(self, chip, fibers, extraction_method, **kwargs):
            self._read_order_trace_reference()
            return {}

        monkeypatch.setattr(SpectralExtraction, "extract_ffi", extract)
        l2 = SpectralExtraction(minimal_l1).perform(fibers=[])

        assert l2.headers["PRIMARY"]["INSTERA"] == "1.0"
