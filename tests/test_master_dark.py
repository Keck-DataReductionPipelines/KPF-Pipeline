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
        loaded_bias = MagicMock(name="loaded_bias")

        with (
            patch("kpfpipe.modules.masters.base.CalibrationAssociation") as mock_ca,
            patch("kpfpipe.modules.masters.base.ImageProcessing") as mock_ip,
            patch.object(
                dark,
                "_load_calibration",
                side_effect=lambda l1, cal: loaded_bias if cal == "bias" else False,
            ) as mock_load,
        ):
            mock_ca.return_value.perform.return_value = associated
            mock_ip.return_value.perform.return_value = processed
            mock_ip.calibration_applied.return_value = False  # frame not yet calibrated

            result = dark._process_frame(frame_in)

        # Associate only the bias master, reading it from the masters root.
        mock_ca.assert_called_once_with(frame_in, {"KPF_MASTERS_OUTPUT": "/masters"})
        mock_ca.return_value.perform.assert_called_once_with(["bias"])

        # The bias master is loaded (and cached) from the associated frame...
        mock_load.assert_any_call(associated, "bias")

        # ...then the loaded bias (and only the bias) is subtracted.
        mock_ip.assert_called_once_with(associated)
        mock_ip.return_value.perform.assert_called_once_with(
            bias=loaded_bias, dark=False, flat=False
        )

        assert result is processed

    def test_explicit_override_skips_association(self):
        # A bias given as a master object (not True) needs no association.
        dark = Dark(FILE_LIST, config={"KPF_MASTERS_OUTPUT": "/masters"})
        master_obj = MagicMock(name="master_bias")
        dark._active_calibrations = {"bias": master_obj, "dark": False, "flat": False}
        frame_in = MagicMock(name="l1_in")
        processed = MagicMock(name="l1_processed")

        with (
            patch("kpfpipe.modules.masters.base.CalibrationAssociation") as mock_ca,
            patch("kpfpipe.modules.masters.base.ImageProcessing") as mock_ip,
        ):
            mock_ip.return_value.perform.return_value = processed
            mock_ip.calibration_applied.return_value = False  # frame not yet calibrated
            result = dark._process_frame(frame_in)

        mock_ca.assert_not_called()
        mock_ip.assert_called_once_with(frame_in)
        mock_ip.return_value.perform.assert_called_once_with(
            bias=master_obj, dark=False, flat=False
        )
        assert result is processed

    def test_mixed_true_and_explicit_associates_only_true(self):
        # True calibrations are associated; explicit paths skip association but
        # are still loaded by the module before ImageProcessing runs.
        dark = Dark(FILE_LIST, config={"KPF_MASTERS_OUTPUT": "/masters"})
        dark._active_calibrations = {
            "bias": True,
            "dark": "/p/master_dark.fits",
            "flat": False,
        }
        frame_in = MagicMock(name="l1_in")
        associated = MagicMock(name="l1_associated")
        processed = MagicMock(name="l1_processed")
        loaded = {
            "bias": MagicMock(name="loaded_bias"),
            "dark": MagicMock(name="loaded_dark"),
        }

        with (
            patch("kpfpipe.modules.masters.base.CalibrationAssociation") as mock_ca,
            patch("kpfpipe.modules.masters.base.ImageProcessing") as mock_ip,
            patch.object(
                dark,
                "_load_calibration",
                side_effect=lambda l1, cal: loaded.get(cal, False),
            ),
        ):
            mock_ca.return_value.perform.return_value = associated
            mock_ip.return_value.perform.return_value = processed
            mock_ip.calibration_applied.return_value = False  # frame not yet calibrated
            result = dark._process_frame(frame_in)

        # Only the True (bias) calibration is associated; the explicit dark is not.
        mock_ca.return_value.perform.assert_called_once_with(["bias"])
        mock_ip.assert_called_once_with(associated)
        mock_ip.return_value.perform.assert_called_once_with(
            bias=loaded["bias"], dark=loaded["dark"], flat=False
        )
        assert result is processed

    def test_skips_frame_already_calibrated(self):
        # A frame whose active calibrations are already flagged applied (e.g. a
        # cached frame revisited by the streaming pass) is returned untouched,
        # so calibrations are never subtracted twice.
        dark = Dark(FILE_LIST, config={"KPF_MASTERS_OUTPUT": "/masters"})
        frame_in = MagicMock(name="l1_in")

        with (
            patch("kpfpipe.modules.masters.base.CalibrationAssociation") as mock_ca,
            patch("kpfpipe.modules.masters.base.ImageProcessing") as mock_ip,
        ):
            mock_ip.calibration_applied.return_value = True  # bias already applied
            result = dark._process_frame(frame_in)

        mock_ca.assert_not_called()
        mock_ip.assert_not_called()
        assert result is frame_in


# ---------------------------------------------------------------------------
# _load_calibration: resolve a calibration to a master, caching one per type
# ---------------------------------------------------------------------------


class TestLoadMaster:
    """A master shared across a stack is read from disk once; a different
    associated master replaces the cached one (reload-and-replace)."""

    @staticmethod
    def _frame(biasfile="master_bias.fits", biasdir="/m"):
        frame = MagicMock(name="l1")
        frame.headers = {"PRIMARY": {"BIASFILE": biasfile, "BIASDIR": biasdir}}
        return frame

    def test_falsy_value_returns_unchanged_without_loading(self, monkeypatch):
        dark = Dark(FILE_LIST)
        dark._active_calibrations = {"bias": False, "dark": False, "flat": False}
        reads = []
        monkeypatch.setattr(KPFMasterL1, "from_fits", staticmethod(reads.append))
        assert dark._load_calibration(self._frame(), "bias") is False
        assert reads == []

    def test_kpfmaster_object_passes_through_without_loading(self, monkeypatch):
        dark = Dark(FILE_LIST)
        obj = KPFMasterL1()
        dark._active_calibrations = {"bias": obj, "dark": False, "flat": False}
        reads = []
        monkeypatch.setattr(KPFMasterL1, "from_fits", staticmethod(reads.append))
        assert dark._load_calibration(self._frame(), "bias") is obj
        assert reads == []

    def test_str_path_loaded_and_cached(self, monkeypatch):
        dark = Dark(FILE_LIST)
        dark._active_calibrations = {"bias": "/m/b.fits", "dark": False, "flat": False}
        reads = []
        sentinel = MagicMock(name="bias_ml1")
        monkeypatch.setattr(os.path, "isfile", lambda p: True)
        monkeypatch.setattr(
            KPFMasterL1,
            "from_fits",
            staticmethod(lambda p: reads.append(p) or sentinel),
        )
        first = dark._load_calibration(self._frame(), "bias")
        second = dark._load_calibration(self._frame(), "bias")
        assert first is sentinel and second is sentinel
        assert reads == ["/m/b.fits"]  # read once

    def test_header_master_loaded_once_across_frames(self, monkeypatch):
        dark = Dark(FILE_LIST)
        dark._active_calibrations = {"bias": True, "dark": False, "flat": False}
        reads = []
        monkeypatch.setattr(os.path, "isfile", lambda p: True)
        monkeypatch.setattr(
            KPFMasterL1,
            "from_fits",
            staticmethod(lambda p: reads.append(p) or MagicMock()),
        )
        dark._load_calibration(self._frame(), "bias")
        dark._load_calibration(self._frame(), "bias")  # same associated path
        assert reads == ["/m/master_bias.fits"]

    def test_reloads_when_associated_master_changes(self, monkeypatch):
        dark = Dark(FILE_LIST)
        dark._active_calibrations = {"bias": True, "dark": False, "flat": False}
        reads = []
        monkeypatch.setattr(os.path, "isfile", lambda p: True)
        monkeypatch.setattr(
            KPFMasterL1,
            "from_fits",
            staticmethod(lambda p: reads.append(p) or MagicMock()),
        )
        dark._load_calibration(self._frame(biasfile="b1.fits"), "bias")
        dark._load_calibration(self._frame(biasfile="b2.fits"), "bias")
        assert reads == ["/m/b1.fits", "/m/b2.fits"]

    def test_missing_file_raises(self, monkeypatch):
        dark = Dark(FILE_LIST)
        dark._active_calibrations = {
            "bias": "/m/missing.fits",
            "dark": False,
            "flat": False,
        }
        monkeypatch.setattr(os.path, "isfile", lambda p: False)
        with pytest.raises(FileNotFoundError, match="Master file not found"):
            dark._load_calibration(self._frame(), "bias")


# ---------------------------------------------------------------------------
# Rate estimator: master IMG = total counts / total exposure time
# ---------------------------------------------------------------------------


def _stack_frame(exptime, ccd_val, var_val, shape=(2, 2)):
    """A synthetic assembled frame with uniform CCD/VAR and a given EXPTIME."""
    frame = MagicMock()
    frame.headers = {"PRIMARY": {"EXPTIME": exptime}}
    frame.data = {
        "GREEN_CCD": np.full(shape, ccd_val, dtype=np.float32),
        "GREEN_VAR": np.full(shape, var_val, dtype=np.float32),
    }
    return frame


class TestRateEstimator:
    """The master IMG is the exposure-weighted rate (total counts / total
    exposure time), correct even when the stack mixes exposure times. Outlier
    rejection is disabled (large sigma) so the arithmetic is exact."""

    @staticmethod
    def _stacked_img(frames, *, nframe_stream=6):
        file_list = sorted(f"f{i}.fits" for i in range(len(frames)))
        dark = Dark(file_list)
        dark.chips = ["GREEN"]
        dark.ccd = {"nrow": 2, "ncol": 2}
        dark.stack_sigma = 1e6  # effectively no clipping
        dark.nframe_stream = nframe_stream
        # Keyed by filename: the streaming path re-reads its first frames for
        # the approximate (clip-bound) pass, so a frame may be loaded twice.
        by_fn = dict(zip(file_list, frames, strict=True))
        with (
            patch.object(dark, "_load_frame", lambda fn, **k: (by_fn[fn], True)),
            patch.object(dark, "_process_frame", lambda l1: l1),
        ):
            return dark.stack_frames()["GREEN_IMG"]

    def test_mixed_exptime_is_exposure_weighted(self):
        # rates per frame are 10 and 3.33; an equal-weight mean would give
        # ~6.67, but the correct estimate is (100+100)/(10+30) = 5.0 e-/sec.
        frames = [_stack_frame(10.0, 100.0, 10.0), _stack_frame(30.0, 100.0, 10.0)]
        np.testing.assert_allclose(self._stacked_img(frames), 5.0, rtol=1e-5)

    def test_equal_exptime_is_counts_over_exptime(self):
        # (100 + 140) / (20 + 20) = 6.0 e-/sec.
        frames = [_stack_frame(20.0, 100.0, 10.0), _stack_frame(20.0, 140.0, 10.0)]
        np.testing.assert_allclose(self._stacked_img(frames), 6.0, rtol=1e-5)

    def test_zero_exptime_is_mean_counts(self):
        # bias: T = 1, so the estimate is the mean in electrons: (100 + 140)/2.
        frames = [_stack_frame(0.0, 100.0, 10.0), _stack_frame(0.0, 140.0, 10.0)]
        np.testing.assert_allclose(self._stacked_img(frames), 120.0, rtol=1e-5)

    def test_streaming_path_matches(self):
        # Force the streaming path (nframe >= nframe_stream) with mixed
        # exposures: (4*100) / (10+30+10+30) = 5.0 e-/sec.
        frames = [
            _stack_frame(10.0, 100.0, 10.0),
            _stack_frame(30.0, 100.0, 10.0),
            _stack_frame(10.0, 100.0, 10.0),
            _stack_frame(30.0, 100.0, 10.0),
        ]
        img = self._stacked_img(frames, nframe_stream=3)
        np.testing.assert_allclose(img, 5.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Per-pixel-per-frame rejection in the final normalization
# ---------------------------------------------------------------------------


class TestPerPixelRejection:
    """Counts and exposure time are summed over the SAME per-pixel survivor set,
    so a pixel with fewer good frames yields the same rate as a fully-sampled
    pixel — only its SNR drops (photon statistics)."""

    def test_datacube_partial_rejection_preserves_rate(self, monkeypatch):
        # 5 identical frames (100 e- over 10 s -> 10 e-/sec). flag_outliers is
        # stubbed to reject frame 4 at pixel (0, 0) only; all other pixels keep
        # all 5 frames.
        n, nrow, ncol = 5, 2, 2
        frames = [
            _stack_frame(10.0, 100.0, 100.0, shape=(nrow, ncol)) for _ in range(n)
        ]
        outlier = np.zeros((n, nrow, ncol), dtype=bool)
        outlier[4, 0, 0] = True
        # Reject frame 4 at (0, 0) during stacking (3D cube input); return no
        # outliers for the 2D bad-pixel pass in _clean_l1_arrays.
        monkeypatch.setattr(
            "kpfpipe.modules.masters.base.flag_outliers",
            lambda arr, sigma, axis=0: (
                outlier if arr.ndim == 3 else np.zeros(arr.shape, dtype=bool)
            ),
        )

        dark = Dark(sorted(f"f{i}.fits" for i in range(n)))
        dark.chips = ["GREEN"]
        dark.ccd = {"nrow": nrow, "ncol": ncol}
        by_fn = dict(zip(dark.l0_file_list, frames, strict=True))
        with (
            patch.object(dark, "_load_frame", lambda fn, **k: (by_fn[fn], True)),
            patch.object(dark, "_process_frame", lambda l1: l1),
        ):
            arrays = dark.stack_frames()

        img, snr = arrays["GREEN_IMG"], arrays["GREEN_SNR"]
        # Rate is the same at the 4/5 pixel (400/40) as at the 5/5 pixels
        # (500/50): both 10 e-/sec. A bug summing exptime over all frames would
        # give 400/50 = 8 at (0, 0).
        np.testing.assert_allclose(img, 10.0, rtol=1e-5)
        # SNR drops at the rejected pixel by sqrt(4/5) (one fewer frame's
        # photons): 400/sqrt(400) vs 500/sqrt(500).
        assert snr[0, 0] < snr[1, 1]
        np.testing.assert_allclose(snr[0, 0] / snr[1, 1], np.sqrt(4 / 5), rtol=1e-4)

    def test_streaming_rejection_keeps_rate_consistent(self, monkeypatch):
        # Force the streaming path; a clear outlier frame must drop out of BOTH
        # the counts sum and the exposure-time sum, leaving the rate correct.
        n = 4
        counts = [95.0, 105.0, 100.0, 1e5]  # last frame is a gross outlier
        frames = [_stack_frame(10.0, c, 100.0) for c in counts]
        # No outliers in the approx pass; the exact pass clips via rate bounds.
        monkeypatch.setattr(
            "kpfpipe.modules.masters.base.flag_outliers",
            lambda arr, sigma, axis=0: np.zeros(arr.shape, dtype=bool),
        )

        dark = Dark(sorted(f"f{i}.fits" for i in range(n)))
        dark.chips = ["GREEN"]
        dark.ccd = {"nrow": 2, "ncol": 2}
        dark.nframe_stream = 3  # ndirect = 2 -> approx from frames 0, 1
        dark.stack_sigma = 5.0
        by_fn = dict(zip(dark.l0_file_list, frames, strict=True))
        with (
            patch.object(dark, "_load_frame", lambda fn, **k: (by_fn[fn], True)),
            patch.object(dark, "_process_frame", lambda l1: l1),
        ):
            img = dark.stack_frames()["GREEN_IMG"]

        # (95 + 105 + 100) / (3 * 10) = 10 e-/sec; the outlier is excluded from
        # numerator and denominator alike. A mismatch would give 300/40 = 7.5.
        np.testing.assert_allclose(img, 10.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# _resolve_calibrations: standard ∩ resolved(bias/dark/flat) clamp
# ---------------------------------------------------------------------------


class TestResolveCalibrations:
    """Effective calibrations = the per-master standard intersected with the
    resolved flags; flags can only turn standard calibrations off."""

    def test_default_is_bias_only(self):
        # Dark's standard is ("bias",); the enabled defaults are (bias, dark).
        dark = Dark(FILE_LIST)
        assert dark._resolve_calibrations() == {
            "bias": True,
            "dark": False,
            "flat": False,
        }

    def test_kwarg_can_disable_standard_calibration(self):
        dark = Dark(FILE_LIST)
        assert dark._resolve_calibrations(bias=False)["bias"] is False

    def test_illogical_enable_is_clamped(self):
        # dark/flat are outside a dark master's standard -> stay off.
        dark = Dark(FILE_LIST)
        assert dark._resolve_calibrations(dark=True, flat=True) == {
            "bias": True,
            "dark": False,
            "flat": False,
        }

    def test_config_can_disable_standard_bias(self):
        dark = Dark(FILE_LIST, config={"bias": False})
        assert dark._resolve_calibrations()["bias"] is False

    def test_str_override_passes_through_for_standard_calibration(self):
        dark = Dark(FILE_LIST)
        assert dark._resolve_calibrations(bias="/p/master_bias.fits") == {
            "bias": "/p/master_bias.fits",
            "dark": False,
            "flat": False,
        }

    def test_object_override_passes_through_for_standard_calibration(self):
        dark = Dark(FILE_LIST)
        sentinel = object()  # stands in for a KPFMasterL1 instance
        assert dark._resolve_calibrations(bias=sentinel)["bias"] is sentinel

    def test_str_override_outside_standard_is_clamped(self):
        # flat is not in a dark master's standard -> ignored even as a path.
        dark = Dark(FILE_LIST)
        assert dark._resolve_calibrations(flat="/p/master_flat.fits")["flat"] is False


# ---------------------------------------------------------------------------
# Signature: a dark is only bias-subtracted, so make_master_l1 takes bias only
# ---------------------------------------------------------------------------


class TestMasterDarkSignature:
    @pytest.mark.parametrize("kwarg", ["dark", "flat"])
    def test_dark_flat_kwargs_rejected(self, kwarg):
        with pytest.raises(TypeError):
            Dark(FILE_LIST).make_master_l1(**{kwarg: True})

    def test_bias_kwarg_accepted(self):
        synthetic = make_l1_arrays()
        dark = Dark(FILE_LIST)
        with patch.object(dark, "stack_frames", return_value=synthetic):
            ml1 = dark.make_master_l1(bias=False)
        assert isinstance(ml1, KPFMasterL1)
