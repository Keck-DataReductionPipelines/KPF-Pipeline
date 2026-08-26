"""Regression tests for image assembly (L0 -> L1).

The 2-amp regression and round-trip tests are marked ``slow`` and read real L0
FITS frames from the gitignored ``tests/testdata/L0/20240405`` tree. Every other
test runs on synthetic data and needs no external frames.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.modules.image_assembly import ImageAssembly

from ._data_models import write_amp_l0
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
    # 4-amp dimensions: 2040 imaging rows + 30 parallel overscan,
    # 4 prescan + 2040 imaging cols + 50 serial overscan
    return write_amp_l0(
        tmp_path_factory.mktemp("l0_4amp") / "KP.20240101.00001.00.fits",
        shape=(2070, 2094),
        bias_level=1000.0,
        primary_cards={"OBJECT": "synthetic-4amp"},
    )


# ---------------------------------------------------------------------------
# 2-amp regression tests (real data)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.requires_testdata
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
        assert_dtype(l1.data[f"{chip}_CCD"], L1_IMAGE, f"{chip}_CCD")

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
        # INSTRUME is stamped by from_fits on any KPF file, so asserting it
        # cannot catch an assembly bug. DATE-OBS comes from the source frame,
        # so compare the L1 card against the L0 it was assembled from.
        l1, ia = l1_bias
        assert (
            l1.headers["PRIMARY"]["DATE-OBS"]
            == ia.l0_obj.headers["PRIMARY"]["DATE-OBS"]
        )

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

    def test_read_mode_in_header(self, l1_bias):
        l1, _ = l1_bias
        # The frame was read out with regular-read-{green,red}.acf.
        assert l1.headers["PRIMARY"]["READMODE"] == "regular"

    def test_read_time_in_header(self, l1_bias):
        l1, _ = l1_bias
        # A regular readout takes ~48 s.
        assert 40 < l1.headers["QUALITY_CONTROL"]["TRTGREEN"] < 60
        assert 40 < l1.headers["QUALITY_CONTROL"]["TRTRED"] < 60

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
@pytest.mark.requires_testdata
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
        # BITPIX and the re-read dtype are shape-independent, so the round-trip
        # runs on a corner of each array rather than 270 MB of full detector.
        for ext in ("GREEN_CCD", "GREEN_VAR", "RED_CCD", "RED_VAR"):
            l1.data[ext] = l1.data[ext][:16, :16]
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
    wave_nm_labels = ["498.12", "604.38", "710.62", "816.88"]
    nrows = 3
    expmeter_hdus = []
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
        expmeter_hdus.append(fits.BinTableHDU.from_columns(cols, name=ext_name))

    return write_amp_l0(
        tmp_path / "KP.20240101.00002.00.fits",
        shape=(2070, 2094),
        bias_level=1000.0,
        seed=7,
        primary_cards={"OBJECT": "synthetic-expmeter"},
        extra_hdus=expmeter_hdus,
    )


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

    @pytest.mark.parametrize("ext", ["EXPMETER_SCI", "EXPMETER_SKY"])
    def test_columns_converted_to_angstroms(self, ext):
        cols = self._convert(**{ext: _expmeter_table()}).data[ext].colnames
        for expected in ("4981.2", "6043.8", "7106.2", "8168.8"):
            assert expected in cols, f"missing Å column {expected!r}; got {cols}"

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
# Overscan methods and the config-typo guards (synthetic, no perform())
# ---------------------------------------------------------------------------


class TestOverscanMethods:
    """The three ``_oscan_*`` kernels and the OSCANSUB provenance flag.

    ``rowmedian`` is the configured default, so it is the only method the
    real-frame tests above ever run; ``median`` and ``zero`` are equally
    selectable in production and are exercised here on a hand-built amp.
    """

    @pytest.fixture
    def ia(self, tmp_path):
        """ImageAssembly on a tiny 4-amp frame, with the geometry set by hand.

        ``dims``/``prescan`` normally come from count_amplifiers(); setting
        them directly keeps the overscan slicing arithmetic in view.
        """
        path = write_amp_l0(tmp_path / "KP.20240101.00001.00.fits", shape=(12, 12))
        module = ImageAssembly(KPF0.from_fits(path))
        module.namp["GREEN"] = 4
        module.dims["GREEN"] = (8, 8)
        module.prescan = 2
        return module

    def _set_serial_overscan(self, ia, values):
        """Write ``values`` into GREEN_AMP1's serial overscan strip (cols 10:12)."""
        amp = np.array(ia.l0_obj.data["GREEN_AMP1"], dtype=np.float32)
        amp[:8, 10:] = values
        ia.l0_obj.data["GREEN_AMP1"] = amp

    def test_zero_returns_scalar_zero(self, ia):
        assert ia._oscan_zero("GREEN", 1) == 0.0

    def test_median_is_the_strip_median(self, ia):
        self._set_serial_overscan(ia, 7.0)
        assert ia._oscan_median("GREEN", 1) == pytest.approx(7.0)

    def test_rowmedian_is_per_row_and_column_shaped(self, ia):
        # A row ramp: row i of the overscan strip holds the value i.
        self._set_serial_overscan(ia, np.arange(8, dtype=np.float32)[:, None])
        bias = ia._oscan_rowmedian("GREEN", 1)
        assert bias.shape == (8, 1)
        np.testing.assert_array_equal(bias[:, 0], np.arange(8))

    def test_unsupported_method_raises(self, ia):
        # The dispatch is getattr(self, f"_oscan_{method}"), so a config typo
        # must name itself rather than surface as a bare AttributeError.
        with pytest.raises(AttributeError, match="Unsupported overscan"):
            ia.subtract_overscan("GREEN", method="rowmedain")

    @pytest.mark.parametrize(
        "method, expected", [("rowmedian", 1), ("median", 1), ("zero", 0)]
    )
    def test_oscansub_flag_tracks_method(self, ia, method, expected):
        # OSCANSUB is a provenance card: 'zero' subtracts nothing and must say so.
        ia.overscan_method = method
        l1 = KPF1()
        ia._set_headers(l1)
        assert l1.headers["RECEIPT"].get("OSCANSUB") == expected


class TestReadMode:
    """READMODE classification: the ACF filename first, readout duration after."""

    def _infer(self, tmp_path, cards):
        path = write_amp_l0(
            tmp_path / "KP.20240101.00001.00.fits", shape=(12, 12), primary_cards=cards
        )
        return ImageAssembly(KPF0.from_fits(path)).infer_read_mode()

    @pytest.mark.parametrize(
        "green_acf, red_acf, expected",
        [
            ("regular-read-green.acf", "regular-read-red.acf", "regular"),
            ("fast-read-green.acf", "fast-read-red.acf", "fast"),
            ("fast-read-green.acf", "regular-read-red.acf", "fast"),
        ],
    )
    def test_acf_filename_names_the_mode(self, tmp_path, green_acf, red_acf, expected):
        cards = {"GRACFFLN": green_acf, "RDACFFLN": red_acf}
        assert self._infer(tmp_path, cards) == expected

    @pytest.mark.parametrize("read_time, expected", [(12.0, "fast"), (48.0, "regular")])
    def test_falls_back_to_readout_duration(self, tmp_path, read_time, expected):
        # An ACF named after neither mode leaves the shutter-close to file-write
        # interval as the only discriminator.
        shutter_close = "2024-01-01T00:00:00"
        file_write = f"2024-01-01T00:00:{read_time:04.1f}"
        cards = {
            "GRACFFLN": "unknown.acf",
            "RDACFFLN": "unknown.acf",
            "GRDATE": file_write,
            "GRDATE-E": shutter_close,
            "RDDATE": file_write,
            "RDDATE-E": shutter_close,
        }
        assert self._infer(tmp_path, cards) == expected

    def test_read_time_measured_for_both_chips(self, tmp_path):
        path = write_amp_l0(
            tmp_path / "KP.20240101.00001.00.fits",
            shape=(12, 12),
            primary_cards={
                "GRDATE-E": "2024-01-01T00:00:00",
                "GRDATE": "2024-01-01T00:00:47.5",
                "RDDATE-E": "2024-01-01T00:00:00",
                "RDDATE": "2024-01-01T00:00:12.0",
            },
        )
        assembly = ImageAssembly(KPF0.from_fits(path))
        assembly.infer_read_mode()
        assert assembly.read_time == {"GREEN": 47.5, "RED": 12.0}

    def test_read_time_only_for_processed_chips(self, tmp_path):
        path = write_amp_l0(tmp_path / "KP.20240101.00002.00.fits", shape=(12, 12))
        assembly = ImageAssembly(KPF0.from_fits(path), config={"chips": ["GREEN"]})
        assembly.infer_read_mode()
        assert set(assembly.read_time) == {"GREEN"}


class TestAmplifierGuards:
    """The two fail-loud guards on an unexpected readout geometry."""

    def test_one_amp_mode_raises(self, tmp_path):
        path = write_amp_l0(
            tmp_path / "KP.20240101.00001.00.fits", namps=1, shape=(10, 10)
        )
        module = ImageAssembly(KPF0.from_fits(path))
        with pytest.raises(ValueError, match="Only 2-amp and 4-amp"):
            module.count_amplifiers("GREEN")

    def test_unexpected_flip_entry_raises(self, tmp_path):
        path = write_amp_l0(tmp_path / "KP.20240101.00001.00.fits", shape=(10, 10))
        module = ImageAssembly(KPF0.from_fits(path))
        module.count_amplifiers("GREEN")
        module.orientation["GREEN_AMP1"] = "sideways"
        with pytest.raises(ValueError, match="unexpected 'flip' entry"):
            module.orient_channels("GREEN")


# ---------------------------------------------------------------------------
# FITS round-trip tests (real data)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.requires_testdata
class TestImageAssemblyRoundTrip:
    @pytest.fixture(scope="class")
    def assembled(self):
        return ImageAssembly(KPF0.from_fits(L0_BIAS)).perform()

    def test_write_and_read_back(self, assembled, tmp_path):
        fn = str(tmp_path / "kpf_L1_20240113T102656.fits")
        assembled.to_fits(fn)
        l1_read = KPF1.from_fits(fn)

        for chip in ("GREEN", "RED"):
            ext = f"{chip}_CCD"
            assert l1_read.data[ext].shape == (4080, 4080)
            np.testing.assert_array_almost_equal(
                l1_read.data[ext], assembled.data[ext], decimal=4
            )

        # INSTRUME is guaranteed by from_fits, so it cannot catch an assembly
        # bug; obs_id is carried by the write path this test exercises.
        assert l1_read.obs_id == assembled.obs_id
