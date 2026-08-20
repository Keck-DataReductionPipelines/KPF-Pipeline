"""Regression tests for image assembly (L0 -> L1).

The 2-amp regression and round-trip tests are marked ``slow`` and read real L0
FITS frames from the gitignored ``tests/testdata/L0/20240405`` tree. Every other
test runs on synthetic data and needs no external frames.
"""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.modules.image_assembly import ImageAssembly

from ._dtype_policy import (
    L1_IMAGE,
    assert_dtype,
    assert_not_float64,
    assert_roundtrip_dtype,
)

TESTDATA_L0_DIR = Path(__file__).parent.parent / "testdata" / "L0" / "20240405"
L0_BIAS = str(TESTDATA_L0_DIR / "KP.20240405.03637.74.fits")
L0_FLAT = str(TESTDATA_L0_DIR / "KP.20240405.00020.86.fits")


# ---------------------------------------------------------------------------
# Synthetic 4-amp L0 fixture (no real data needed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_4amp_l0(tmp_path_factory):
    """Synthetic L0 with 4-amp readout on both CCDs.

    Module-scoped and read-only, so the ~140 MB write happens once, not per test.
    """
    fn = str(tmp_path_factory.mktemp("l0_4amp") / "KP.20240101.00001.00.fits")
    rng = np.random.default_rng(42)

    # 4-amp dimensions: 2040 imaging rows + 30 parallel overscan,
    # 4 prescan + 2040 imaging cols + 50 serial overscan
    nrow, ncol = 2070, 2094
    bias_level = 1000.0

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["OBJECT"] = "synthetic-4amp"
    primary.header["IMTYPE"] = "Bias"
    primary.header["DATE-OBS"] = "2024-01-01T00:00:01"
    primary.header["OFNAME"] = os.path.basename(fn)
    primary.header["PROGNAME"] = "K123"

    hdus = [primary]
    for chip in ["GREEN", "RED"]:
        for amp in range(1, 5):
            data = (bias_level + rng.normal(0, 3.0, (nrow, ncol))).astype(np.float32)
            hdu = fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}")
            hdus.append(hdu)

    hdul = fits.HDUList(hdus)
    hdul.writeto(fn, overwrite=True)
    hdul.close()
    return fn


# ---------------------------------------------------------------------------
# 2-amp regression tests (real data)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestImageAssemblyBias:
    """Regression tests using a bias frame (no signal, 2-amp mode)."""

    @pytest.fixture(scope="class")
    def l1_bias(self):
        l0 = KPF0.from_fits(L0_BIAS)
        ia = ImageAssembly(l0)
        return ia.perform(), ia

    def test_returns_kpf1(self, l1_bias):
        l1, _ = l1_bias
        assert isinstance(l1, KPF1)
        assert l1.level == 1

    @pytest.mark.parametrize("chip", ["GREEN", "RED"])
    def test_ccd_shape(self, l1_bias, chip):
        l1, _ = l1_bias
        assert l1.data[f"{chip}_CCD"].shape == (4080, 4080)
        assert l1.data[f"{chip}_CCD"].dtype == np.float32

    def test_variance_frames_exist(self, l1_bias):
        l1, _ = l1_bias
        assert l1.data["GREEN_VAR"].shape == (4080, 4080)
        assert l1.data["RED_VAR"].shape == (4080, 4080)

    def test_variance_positive(self, l1_bias):
        l1, _ = l1_bias
        assert np.all(l1.data["GREEN_VAR"] >= 0)
        assert np.all(l1.data["RED_VAR"] >= 0)

    def test_bias_near_zero(self, l1_bias):
        l1, _ = l1_bias
        assert abs(np.nanmedian(l1.data["GREEN_CCD"])) < 5.0
        assert abs(np.nanmedian(l1.data["RED_CCD"])) < 5.0

    def test_primary_header_carried_forward(self, l1_bias):
        l1, _ = l1_bias
        assert l1.headers["PRIMARY"]["INSTRUME"] == "KPF"

    def test_obs_id_carried_forward(self, l1_bias):
        l1, _ = l1_bias
        assert l1.obs_id == "KP.20240405.03637.74"

    def test_datalvl_set(self, l1_bias):
        l1, _ = l1_bias
        val = l1.headers["PRIMARY"].get("DATALVL")
        assert val == "L1"

    def test_read_noise_in_header(self, l1_bias):
        l1, _ = l1_bias
        # 2-amp mode names the amps RNGREEN1/2 and RNRED1/2.
        assert "RNGREEN1" in l1.headers["QUALITY_CONTROL"]
        assert "RNRED1" in l1.headers["QUALITY_CONTROL"]

    def test_read_noise_reasonable(self, l1_bias):
        # 1-20 e- brackets any physically sane KPF read noise.
        _, ia = l1_bias
        for channel_ext, rn in ia.readnoise.items():
            assert 1.0 < rn < 20.0, f"Read noise for {channel_ext} = {rn} e-"

    def test_rnng_in_header(self, l1_bias):
        l1, _ = l1_bias
        # 2-amp mode names the amps RNNGGR1/2 and RNNGRD1/2.
        assert "RNNGGR1" in l1.headers["QUALITY_CONTROL"]
        assert "RNNGRD1" in l1.headers["QUALITY_CONTROL"]

    def test_overscan_applied_in_header(self, l1_bias):
        l1, _ = l1_bias
        val = l1.headers["RECEIPT"].get("OSCANSUB")
        assert val == 1

    def test_receipt_chain(self, l1_bias):
        l1, _ = l1_bias
        modules = l1.receipt["FUNCTION"].values
        assert "from_fits" in modules
        assert "to_kpf1" in modules
        assert "image_assembly" in modules

    def test_passthrough_telemetry(self, l1_bias):
        l1, _ = l1_bias
        assert "TELEMETRY" in l1.extensions

    def test_no_nans_in_ccd(self, l1_bias):
        l1, _ = l1_bias
        assert not np.any(np.isnan(l1.data["GREEN_CCD"]))
        assert not np.any(np.isnan(l1.data["RED_CCD"]))


@pytest.mark.slow
class TestImageAssemblyFlat:
    """Regression tests using a flat lamp frame (has signal)."""

    @pytest.fixture(scope="class")
    def l1_flat(self):
        l0 = KPF0.from_fits(L0_FLAT)
        ia = ImageAssembly(l0)
        return ia.perform()

    def test_flat_has_signal(self, l1_flat):
        assert np.nanmedian(l1_flat.data["GREEN_CCD"]) > 100.0
        assert np.nanmedian(l1_flat.data["RED_CCD"]) > 100.0

    def test_flat_variance_exceeds_readnoise(self, l1_flat):
        # Photon noise dominates read noise in a flat, so VAR is well above it.
        assert np.nanmedian(l1_flat.data["GREEN_VAR"]) > 10.0
        assert np.nanmedian(l1_flat.data["RED_VAR"]) > 10.0


# ---------------------------------------------------------------------------
# 4-amp mode tests (synthetic data)
# ---------------------------------------------------------------------------


class TestImageAssembly4Amp:
    @pytest.fixture(scope="class")
    def l1_4amp(self, synthetic_4amp_l0):
        """Assemble the synthetic 4-amp L0 once, shared read-only across the class.

        perform() counts the amplifiers, so ia.namp/ia.dims come back populated.
        """
        l0 = KPF0.from_fits(synthetic_4amp_l0)
        ia = ImageAssembly(l0)
        return ia.perform(), ia

    def test_4amp_produces_valid_l1(self, l1_4amp):
        l1, _ = l1_4amp
        assert isinstance(l1, KPF1)
        assert l1.data["GREEN_CCD"].shape == (4080, 4080)
        assert l1.data["RED_CCD"].shape == (4080, 4080)

    def test_4amp_detects_four_amplifiers(self, l1_4amp):
        _, ia = l1_4amp
        assert ia.namp["GREEN"] == 4
        assert ia.namp["RED"] == 4
        assert ia.dims["GREEN"] == (2040, 2040)

    def test_4amp_read_noise_all_amps(self, l1_4amp):
        l1, ia = l1_4amp

        assert len(ia.readnoise) == 8
        for channel_ext in [
            "GREEN_AMP1",
            "GREEN_AMP2",
            "GREEN_AMP3",
            "GREEN_AMP4",
            "RED_AMP1",
            "RED_AMP2",
            "RED_AMP3",
            "RED_AMP4",
        ]:
            assert channel_ext in ia.readnoise

        for key in [
            "RNGREEN1",
            "RNGREEN2",
            "RNGREEN3",
            "RNGREEN4",
            "RNRED1",
            "RNRED2",
            "RNRED3",
            "RNRED4",
        ]:
            assert key in l1.headers["QUALITY_CONTROL"]

    def test_4amp_bias_near_zero(self, l1_4amp):
        l1, _ = l1_4amp
        assert abs(np.nanmedian(l1.data["GREEN_CCD"])) < 10.0
        assert abs(np.nanmedian(l1.data["RED_CCD"])) < 10.0

    def test_4amp_no_nans(self, l1_4amp):
        l1, _ = l1_4amp
        assert not np.any(np.isnan(l1.data["GREEN_CCD"]))
        assert not np.any(np.isnan(l1.data["RED_CCD"]))


# ---------------------------------------------------------------------------
# Dtype provenance (L1 CCD/VAR float32; L0 amps never upscale to float64)
# ---------------------------------------------------------------------------


class TestDtypeProvenance:
    def test_l1_ccd_var_float32_and_roundtrip(self, synthetic_4amp_l0, tmp_path):
        l0 = KPF0.from_fits(synthetic_4amp_l0)
        l1 = ImageAssembly(l0).perform()
        for ext in ("GREEN_CCD", "GREEN_VAR", "RED_CCD", "RED_VAR"):
            assert_dtype(l1.data[ext], L1_IMAGE, ext)
        assert_roundtrip_dtype(
            KPF1,
            l1,
            "GREEN_CCD",
            L1_IMAGE,
            tmp_path,
            name="kpf_L1_20240113T102656.fits",
        )

    def test_l0_amps_not_float64(self, synthetic_4amp_l0):
        l0 = KPF0.from_fits(synthetic_4amp_l0)
        for ext in ("GREEN_AMP1", "RED_AMP1"):
            assert_not_float64(l0.data[ext], ext)


class TestOrientFFI:
    """Unit tests for the static FFI orientation helper.

    orient_ffi is the single source of truth for FFI orientation, shared by
    stitch_ffi and the L0 quicklook. Flux and variance frames all run through it,
    so co-orientation hinges on it applying the same deterministic flip.
    """

    # A small asymmetric array makes every flip unambiguous.
    BASE = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    def test_red_flips_columns_only(self):
        # RED disperses blue->red across columns, so only a left-right flip.
        out = ImageAssembly.orient_ffi(self.BASE, "RED", 2)
        expected = np.array([[3, 2, 1], [6, 5, 4]], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_green_flips_both_axes(self):
        # GREEN's raw image is inverted relative to RED, so rows flip as well.
        out = ImageAssembly.orient_ffi(self.BASE, "GREEN", 2)
        expected = np.array([[6, 5, 4], [3, 2, 1]], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_green_read_out_on_four_amps_flips_columns_only(self):
        # 4-amp GREEN arrives with its rows already running bottom-up.
        out = ImageAssembly.orient_ffi(self.BASE, "GREEN", 4)
        np.testing.assert_array_equal(out, np.flip(self.BASE, axis=1))

    def test_green_is_red_plus_row_flip(self):
        red = ImageAssembly.orient_ffi(self.BASE, "RED", 2)
        green = ImageAssembly.orient_ffi(self.BASE, "GREEN", 2)
        np.testing.assert_array_equal(green, np.flip(red, axis=0))

    def test_chip_name_is_case_insensitive(self):
        for name in ("green", "Green", "GREEN"):
            np.testing.assert_array_equal(
                ImageAssembly.orient_ffi(self.BASE, name, 2),
                ImageAssembly.orient_ffi(self.BASE, "GREEN", 2),
            )
        for name in ("red", "Red", "RED"):
            np.testing.assert_array_equal(
                ImageAssembly.orient_ffi(self.BASE, name, 2),
                ImageAssembly.orient_ffi(self.BASE, "RED", 2),
            )

    def test_does_not_mutate_input(self):
        original = self.BASE.copy()
        ImageAssembly.orient_ffi(self.BASE, "GREEN", 2)
        np.testing.assert_array_equal(self.BASE, original)

    def test_double_application_is_identity(self):
        for chip in ("GREEN", "RED"):
            once = ImageAssembly.orient_ffi(self.BASE, chip, 2)
            twice = ImageAssembly.orient_ffi(once, chip, 2)
            np.testing.assert_array_equal(twice, self.BASE)

    def test_flux_and_wave_are_co_oriented(self):
        # The load-bearing property: frames of the same shape get the identical
        # index remapping, so flux and its wave/var counterpart stay aligned.
        flux = np.arange(12, dtype=np.float32).reshape(3, 4)
        # Index markers track where each (row, col) lands under the transform.
        rows = np.broadcast_to(np.arange(3)[:, None], (3, 4)).astype(np.float32)
        cols = np.broadcast_to(np.arange(4)[None, :], (3, 4)).astype(np.float32)

        for chip in ("GREEN", "RED"):
            f = ImageAssembly.orient_ffi(flux, chip, 2)
            r = ImageAssembly.orient_ffi(rows, chip, 2)
            c = ImageAssembly.orient_ffi(cols, chip, 2)
            for i in range(3):
                for j in range(4):
                    src_row, src_col = int(r[i, j]), int(c[i, j])
                    assert f[i, j] == flux[src_row, src_col]

    def test_unknown_chip_treated_as_non_green(self):
        out = ImageAssembly.orient_ffi(self.BASE, "BLUE", 2)
        expected = np.flip(self.BASE, axis=1)
        np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# Expmeter wavelength unit conversion (nm -> Angstrom at L0 -> L1)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_4amp_l0_with_expmeter(tmp_path):
    """Synthetic 4-amp L0 with EXPMETER_SCI/SKY tables labeled in nm.

    The column labels mirror real KPF expmeter native units (e.g. '498.12' nm).
    """
    fn = str(tmp_path / "KP.20240101.00002.00.fits")
    rng = np.random.default_rng(7)

    nrow, ncol = 2070, 2094
    bias_level = 1000.0

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["OBJECT"] = "synthetic-expmeter"
    primary.header["IMTYPE"] = "Bias"
    primary.header["DATE-OBS"] = "2024-01-01T00:00:01"
    primary.header["OFNAME"] = os.path.basename(fn)
    primary.header["PROGNAME"] = "K123"

    hdus = [primary]
    for chip in ["GREEN", "RED"]:
        for amp in range(1, 5):
            data = (bias_level + rng.normal(0, 3.0, (nrow, ncol))).astype(np.float32)
            hdus.append(fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}"))

    wave_nm_labels = ["498.12", "604.38", "710.62", "816.88"]
    nrows = 3
    for ext_name in ["EXPMETER_SCI", "EXPMETER_SKY"]:
        cols = [
            fits.Column(
                name="Date-Beg", format="25A", array=["2024-01-01T00:00:00.000"] * nrows
            ),
            fits.Column(
                name="Date-End", format="25A", array=["2024-01-01T00:00:01.000"] * nrows
            ),
        ]
        for w in wave_nm_labels:
            cols.append(
                fits.Column(
                    name=w, format="E", array=np.full(nrows, 100.0, dtype=np.float32)
                )
            )
        hdus.append(fits.BinTableHDU.from_columns(cols, name=ext_name))

    fits.HDUList(hdus).writeto(fn, overwrite=True)
    return fn


def _expmeter_table():
    """Synthetic EXPMETER table: nm-labeled wavelength columns plus non-numeric ones."""
    return Table(
        {
            "498.12": np.full(3, 100.0, dtype=np.float32),
            "604.38": np.full(3, 200.0, dtype=np.float32),
            "710.62": np.full(3, 300.0, dtype=np.float32),
            "816.88": np.full(3, 400.0, dtype=np.float32),
            "Date-Beg": ["a", "b", "c"],
            "Date-End": ["x", "y", "z"],
        }
    )


class TestExpmeterWavelengthConversion:
    """L0 -> L1 relabels EXPMETER_SCI/SKY wavelength columns from nm to Angstroms.

    The conversion is a discrete static step, so it is unit-tested on a synthetic
    table rather than through a full perform() (~1s) for a column rename.
    ``test_conversion_applied_by_perform`` guards that perform() still wires it in.
    """

    def _convert(self, **exts):
        """Run the converter on a minimal l1-like object; return it."""
        l1 = SimpleNamespace(data=dict(exts))
        ImageAssembly._convert_expmeter_wavelengths_to_angstroms(l1)
        return l1

    def test_sci_columns_converted_to_angstroms(self):
        cols = (
            self._convert(EXPMETER_SCI=_expmeter_table()).data["EXPMETER_SCI"].colnames
        )
        for expected in ("4981.2", "6043.8", "7106.2", "8168.8"):
            assert expected in cols, f"missing Å column {expected!r}; got {cols}"

    def test_sky_columns_converted_to_angstroms(self):
        cols = (
            self._convert(EXPMETER_SKY=_expmeter_table()).data["EXPMETER_SKY"].colnames
        )
        for expected in ("4981.2", "6043.8", "7106.2", "8168.8"):
            assert expected in cols

    def test_nm_labels_removed(self):
        cols = (
            self._convert(EXPMETER_SCI=_expmeter_table()).data["EXPMETER_SCI"].colnames
        )
        for nm_label in ("498.12", "604.38", "710.62", "816.88"):
            assert nm_label not in cols, f"nm label {nm_label!r} should be gone"

    def test_non_numeric_columns_preserved(self):
        cols = (
            self._convert(EXPMETER_SCI=_expmeter_table()).data["EXPMETER_SCI"].colnames
        )
        assert "Date-Beg" in cols
        assert "Date-End" in cols

    def test_values_preserved(self):
        l1 = self._convert(EXPMETER_SCI=_expmeter_table())
        np.testing.assert_array_equal(
            np.asarray(l1.data["EXPMETER_SCI"]["4981.2"]),
            np.full(3, 100.0, dtype=np.float32),
        )

    def test_no_error_when_expmeter_absent(self):
        # Biases carry no expmeter: a missing key and an explicit None are no-ops.
        self._convert()
        self._convert(EXPMETER_SCI=None)

    def test_conversion_applied_by_perform(self, synthetic_4amp_l0_with_expmeter):
        l0 = KPF0.from_fits(synthetic_4amp_l0_with_expmeter)
        l1 = ImageAssembly(l0).perform()
        assert "4981.2" in l1.data["EXPMETER_SCI"].colnames


# ---------------------------------------------------------------------------
# FITS round-trip tests (real data)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestImageAssemblyRoundTrip:
    def test_write_and_read_back(self):
        l0 = KPF0.from_fits(L0_BIAS)
        ia = ImageAssembly(l0)
        l1 = ia.perform()

        with tempfile.TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, "kpf_L1_20240113T102656.fits")
            l1.to_fits(fn)

            l1_read = KPF1.from_fits(fn)

            assert l1_read.data["GREEN_CCD"].shape == (4080, 4080)
            assert l1_read.data["RED_CCD"].shape == (4080, 4080)
            np.testing.assert_array_almost_equal(
                l1_read.data["GREEN_CCD"], l1.data["GREEN_CCD"], decimal=4
            )
            np.testing.assert_array_almost_equal(
                l1_read.data["RED_CCD"], l1.data["RED_CCD"], decimal=4
            )

    def test_roundtrip_preserves_header(self):
        l0 = KPF0.from_fits(L0_BIAS)
        ia = ImageAssembly(l0)
        l1 = ia.perform()

        with tempfile.TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, "kpf_L1_20240113T102656.fits")
            l1.to_fits(fn)

            l1_read = KPF1.from_fits(fn)
            assert l1_read.headers["PRIMARY"]["INSTRUME"] == "KPF"
