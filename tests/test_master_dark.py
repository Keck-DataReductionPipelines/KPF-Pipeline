"""
Unit tests for the master dark module.

Uses mocked stack_frames for unit tests (no real data needed). A real-data
regression suite (as in test_master_bias.py) is not viable here: the bundled
dark testdata forms two undersized clusters, so a full dark stack cannot be
built from it. Bias subtraction (via the shared `_process_frame` hook) is
covered separately with mocked CalibrationAssociation/ImageProcessing.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.modules.masters.dark import Dark

CHIPS = ["GREEN", "RED"]
NROW, NCOL = 10, 10  # small arrays for unit tests


def make_l1_arrays(rng=None):
    """Return a synthetic stack_frames output dict."""
    if rng is None:
        rng = np.random.default_rng(42)
    arrays = {}
    for chip in CHIPS:
        arrays[f"{chip}_IMG"] = rng.normal(0.0, 5.0, (NROW, NCOL)).astype(np.float32)
        arrays[f"{chip}_SNR"] = np.abs(rng.normal(10.0, 1.0, (NROW, NCOL))).astype(
            np.float32
        )
        arrays[f"{chip}_MASK"] = np.ones((NROW, NCOL), dtype=bool)
    return arrays


FILE_LIST = [f"KP.20240101.{i:05d}.00.fits" for i in range(8)]


def _header_value(value):
    """Return a header value, unwrapping a (value, comment) tuple if present."""
    return value[0] if isinstance(value, tuple) else value


# ---------------------------------------------------------------------------
# Unit tests (mocked stack_frames)
# ---------------------------------------------------------------------------


class TestMasterDarkUnit:
    """Unit tests using a mocked stack_frames — no real data needed."""

    @pytest.fixture(scope="class")
    def master_dark(self):
        synthetic = make_l1_arrays()
        dark = Dark(FILE_LIST)
        with patch.object(dark, "stack_frames", return_value=synthetic):
            return dark.make_master_l1()

    def test_returns_kpf_master_l1(self, master_dark):
        assert isinstance(master_dark, KPFMasterL1)

    def test_green_img_shape(self, master_dark):
        assert master_dark.data["GREEN_IMG"].shape == (NROW, NCOL)

    def test_red_img_shape(self, master_dark):
        assert master_dark.data["RED_IMG"].shape == (NROW, NCOL)

    def test_green_snr_shape(self, master_dark):
        assert master_dark.data["GREEN_SNR"].shape == (NROW, NCOL)

    def test_red_snr_shape(self, master_dark):
        assert master_dark.data["RED_SNR"].shape == (NROW, NCOL)

    def test_mask_is_boolean(self, master_dark):
        assert master_dark.data["GREEN_MASK"].dtype == bool
        assert master_dark.data["RED_MASK"].dtype == bool

    def test_snr_non_negative(self, master_dark):
        assert np.all(master_dark.data["GREEN_SNR"] >= 0)
        assert np.all(master_dark.data["RED_SNR"] >= 0)

    def test_receipt_entry(self, master_dark):
        assert "master_dark" in master_dark.receipt["Module_Name"].values

    def test_bunit_is_rate(self, master_dark):
        for chip in CHIPS:
            bunit = _header_value(master_dark.headers[f"{chip}_IMG"]["BUNIT"])
            assert bunit == "electrons/sec"

    def test_datalvl_class_attribute(self, master_dark):
        assert master_dark._DATALVL == "ML1"


# ---------------------------------------------------------------------------
# info() smoke tests
# ---------------------------------------------------------------------------


class TestMasterDarkInfo:
    """Smoke tests for Dark.info() in both pre- and post-perform states."""

    def test_info_before_make_master_l1(self, capsys):
        dark = Dark(FILE_LIST)
        dark.info()
        out = capsys.readouterr().out
        assert "Dark" in out
        assert "make_master_l1() has not been called" in out

    def test_info_after_make_master_l1(self, capsys):
        synthetic = make_l1_arrays()
        dark = Dark(FILE_LIST)
        with patch.object(dark, "stack_frames", return_value=synthetic):
            dark.make_master_l1()
        dark.info()
        out = capsys.readouterr().out
        assert "Dark" in out
        assert "make_master_l1() has not been called" not in out
        for chip in CHIPS:
            assert chip in out


# ---------------------------------------------------------------------------
# FITS round-trip (mocked stack_frames)
# ---------------------------------------------------------------------------


class TestMasterDarkRoundTrip:
    """Test that master dark output survives a FITS write/read cycle."""

    def test_roundtrip_arrays(self):
        synthetic = make_l1_arrays()
        dark = Dark(FILE_LIST)
        with patch.object(dark, "stack_frames", return_value=synthetic):
            ml1 = dark.make_master_l1()

        with tempfile.TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, "master_dark.fits")
            ml1.to_fits(fn)
            ml1_read = KPFMasterL1.from_fits(fn)

        np.testing.assert_array_almost_equal(
            ml1_read.data["GREEN_IMG"], ml1.data["GREEN_IMG"], decimal=4
        )
        np.testing.assert_array_almost_equal(
            ml1_read.data["RED_IMG"], ml1.data["RED_IMG"], decimal=4
        )

    def test_roundtrip_datalvl(self):
        synthetic = make_l1_arrays()
        dark = Dark(FILE_LIST)
        with patch.object(dark, "stack_frames", return_value=synthetic):
            ml1 = dark.make_master_l1()

        with tempfile.TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, "master_dark.fits")
            ml1.to_fits(fn)
            ml1_read = KPFMasterL1.from_fits(fn)

        assert _header_value(ml1_read.headers["PRIMARY"]["DATALVL"]) == "ML1"


# ---------------------------------------------------------------------------
# filepath integration with save_master
# ---------------------------------------------------------------------------


class TestMasterDarkSaveMaster:
    """make_master_l1(filepath=...) should write the FITS via save_master."""

    def test_filepath_writes_fits(self, tmp_path):
        synthetic = make_l1_arrays()
        dark = Dark(FILE_LIST)
        master_path = tmp_path / "master_dark.fits"
        with patch.object(dark, "stack_frames", return_value=synthetic):
            dark.make_master_l1(filepath=str(master_path))
        assert master_path.exists()

    def test_filepath_creates_parent_dir(self, tmp_path):
        synthetic = make_l1_arrays()
        dark = Dark(FILE_LIST)
        master_path = tmp_path / "nested" / "subdir" / "master_dark.fits"
        with patch.object(dark, "stack_frames", return_value=synthetic):
            dark.make_master_l1(filepath=str(master_path))
        assert master_path.exists()

    def test_save_master_before_make_raises(self):
        dark = Dark(FILE_LIST)
        with pytest.raises(RuntimeError, match="run make_master_l1"):
            dark.save_master("L1", "/tmp/should_not_be_created.fits")


# ---------------------------------------------------------------------------
# _process_frame: bias subtraction via CalibrationAssociation + ImageProcessing
# ---------------------------------------------------------------------------


class TestProcessFrame:
    """Dark subtracts the master bias by reusing the standard science modules
    through the shared `_process_frame` hook, not by hand-rolling the math."""

    def test_process_frame_runs_ca_then_ip(self):
        dark = Dark(FILE_LIST, config={"KPF_MASTERS_OUTPUT": "/masters"})
        frame_in = MagicMock(name="l1_in")
        associated = MagicMock(name="l1_associated")
        processed = MagicMock(name="l1_processed")

        with (
            patch("kpfpipe.modules.masters.base.CalibrationAssociation") as mock_ca,
            patch("kpfpipe.modules.masters.base.ImageProcessing") as mock_ip,
        ):
            mock_ca.return_value.perform.return_value = associated
            mock_ip.return_value.perform.return_value = processed

            result = dark._process_frame(frame_in)

        # Associate only the bias master, reading it from the masters root.
        mock_ca.assert_called_once_with(frame_in, {"KPF_MASTERS_OUTPUT": "/masters"})
        mock_ca.return_value.perform.assert_called_once_with(["bias"])

        # Subtract the bias (and only the bias) from the associated frame.
        mock_ip.assert_called_once_with(associated)
        mock_ip.return_value.perform.assert_called_once_with(
            bias=True, dark=False, flat=False
        )

        assert result is processed

    def test_explicit_override_skips_association(self):
        # A bias given as a master object (not True) needs no association.
        dark = Dark(FILE_LIST, config={"KPF_MASTERS_OUTPUT": "/masters"})
        master_obj = MagicMock(name="master_bias")
        dark._active_corrections = {"bias": master_obj, "dark": False, "flat": False}
        frame_in = MagicMock(name="l1_in")
        processed = MagicMock(name="l1_processed")

        with (
            patch("kpfpipe.modules.masters.base.CalibrationAssociation") as mock_ca,
            patch("kpfpipe.modules.masters.base.ImageProcessing") as mock_ip,
        ):
            mock_ip.return_value.perform.return_value = processed
            result = dark._process_frame(frame_in)

        mock_ca.assert_not_called()
        mock_ip.assert_called_once_with(frame_in)
        mock_ip.return_value.perform.assert_called_once_with(
            bias=master_obj, dark=False, flat=False
        )
        assert result is processed

    def test_mixed_true_and_explicit_associates_only_true(self):
        # True corrections are associated; explicit paths pass straight through.
        dark = Dark(FILE_LIST, config={"KPF_MASTERS_OUTPUT": "/masters"})
        dark._active_corrections = {
            "bias": True,
            "dark": "/p/master_dark.fits",
            "flat": False,
        }
        frame_in = MagicMock(name="l1_in")
        associated = MagicMock(name="l1_associated")
        processed = MagicMock(name="l1_processed")

        with (
            patch("kpfpipe.modules.masters.base.CalibrationAssociation") as mock_ca,
            patch("kpfpipe.modules.masters.base.ImageProcessing") as mock_ip,
        ):
            mock_ca.return_value.perform.return_value = associated
            mock_ip.return_value.perform.return_value = processed
            result = dark._process_frame(frame_in)

        mock_ca.return_value.perform.assert_called_once_with(["bias"])
        mock_ip.assert_called_once_with(associated)
        mock_ip.return_value.perform.assert_called_once_with(
            bias=True, dark="/p/master_dark.fits", flat=False
        )
        assert result is processed


# ---------------------------------------------------------------------------
# _resolve_corrections: standard ∩ resolved(bias/dark/flat) clamp
# ---------------------------------------------------------------------------


class TestResolveCorrections:
    """Effective corrections = the per-master standard intersected with the
    resolved flags; flags can only turn standard corrections off."""

    def test_default_is_bias_only(self):
        # Dark's standard is ("bias",); the enabled defaults are (bias, dark).
        dark = Dark(FILE_LIST)
        assert dark._resolve_corrections() == {
            "bias": True,
            "dark": False,
            "flat": False,
        }

    def test_kwarg_can_disable_standard_correction(self):
        dark = Dark(FILE_LIST)
        assert dark._resolve_corrections(bias=False)["bias"] is False

    def test_illogical_enable_is_clamped(self):
        # dark/flat are outside a dark master's standard -> stay off.
        dark = Dark(FILE_LIST)
        assert dark._resolve_corrections(dark=True, flat=True) == {
            "bias": True,
            "dark": False,
            "flat": False,
        }

    def test_config_can_disable_standard_bias(self):
        dark = Dark(FILE_LIST, config={"bias": False})
        assert dark._resolve_corrections()["bias"] is False

    def test_str_override_passes_through_for_standard_correction(self):
        dark = Dark(FILE_LIST)
        assert dark._resolve_corrections(bias="/p/master_bias.fits") == {
            "bias": "/p/master_bias.fits",
            "dark": False,
            "flat": False,
        }

    def test_object_override_passes_through_for_standard_correction(self):
        dark = Dark(FILE_LIST)
        sentinel = object()  # stands in for a KPFMasterL1 instance
        assert dark._resolve_corrections(bias=sentinel)["bias"] is sentinel

    def test_str_override_outside_standard_is_clamped(self):
        # flat is not in a dark master's standard -> ignored even as a path.
        dark = Dark(FILE_LIST)
        assert dark._resolve_corrections(flat="/p/master_flat.fits")["flat"] is False
