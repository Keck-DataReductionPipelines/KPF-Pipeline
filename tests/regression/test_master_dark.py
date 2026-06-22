"""
Unit and regression tests for the master dark module.

Unit tests use mocked stack_frames (no real data needed); the orchestration
hooks (_process_frame / _load_calibration) are covered with mocked
CalibrationAssociation/ImageProcessing. TestMasterDarkRegression builds a real
master dark from the bundled L0 darks: the five frames span two default-gap
clusters, so it widens cluster_gap_seconds to group them into one stack.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.modules.masters.dark import Dark
from kpfpipe.utils.io import build_l0_file_lists

from ._masters import make_l1_arrays

CHIPS = ["GREEN", "RED"]
NROW, NCOL = 10, 10  # small arrays for unit tests
# make_l1_arrays() — shared synthetic stack_frames builder — lives in _masters.py

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"
TESTDATA_L0_DIR = TESTDATA_DIR / "L0" / "20240405"

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

    @pytest.mark.parametrize("ext", ["GREEN_IMG", "RED_IMG", "GREEN_SNR", "RED_SNR"])
    def test_extension_shape(self, master_dark, ext):
        assert master_dark.data[ext].shape == (NROW, NCOL)

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
# Base-class fail-loudly paths (construction + frame loading)
# ---------------------------------------------------------------------------


class TestMasterBaseErrors:
    """Error/guard paths shared by all master modules (BaseMasterModule)."""

    def test_unsorted_l0_list_raises(self):
        # The base requires a sorted L0 list so stacking order is deterministic.
        with pytest.raises(ValueError, match="sorted in ascending order"):
            Dark(["KP.20240101.00002.00.fits", "KP.20240101.00001.00.fits"])

    def test_bad_config_type_raises(self):
        with pytest.raises(
            TypeError, match="config must be None, dict, or ConfigHandler"
        ):
            Dark(FILE_LIST, config="not-a-config")

    def test_load_frame_missing_file_warns_and_skips(self):
        # A missing/unreadable L0 frame warns and returns (None, failure), not a crash.
        fn = "/nonexistent/KP.20240101.00001.00.fits"
        m = Dark([fn])
        with pytest.warns(UserWarning, match="Failed to load"):
            l1_obj, success = m._load_frame(fn, cache=False, verbose=True)
        assert l1_obj is None and success is False

    def test_load_frame_exptime_failure_warns_and_skips(self, monkeypatch):
        # A frame failing the exptime-vs-elapsed check is warned and skipped.
        m = Dark(FILE_LIST)
        fn = FILE_LIST[0]
        m._l1_obj_cache[fn] = object()  # cache hit bypasses real I/O

        def bad_check(l1_obj, exptime_tolerance):
            raise ValueError("elapsed exceeds exptime")

        monkeypatch.setattr(m, "_check_exptime_vs_elapsed", bad_check)
        with pytest.warns(UserWarning, match="Exptime check failed"):
            l1_obj, success = m._load_frame(fn, cache=False, verbose=True)
        assert l1_obj is None and success is False


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

    def test_forwards_configured_search_window(self):
        # A configured search window must reach CalibrationAssociation rather
        # than silently falling back to the module default.
        dark = Dark(
            FILE_LIST,
            config={
                "KPF_MASTERS_OUTPUT": "/masters",
                "masters_search_window_days": [-3, 1],
            },
        )
        frame_in = MagicMock(name="l1_in")

        with (
            patch("kpfpipe.modules.masters.base.CalibrationAssociation") as mock_ca,
            patch("kpfpipe.modules.masters.base.ImageProcessing") as mock_ip,
            patch.object(dark, "_load_calibration", lambda l1, cal: False),
        ):
            mock_ip.calibration_applied.return_value = False
            dark._process_frame(frame_in)

        ca_config = mock_ca.call_args[0][1]
        assert ca_config["masters_search_window_days"] == [-3, 1]

    def test_omits_search_window_when_unset(self):
        # Without a configured window, the key is left out so CalibrationAssociation
        # applies its own default.
        dark = Dark(FILE_LIST, config={"KPF_MASTERS_OUTPUT": "/masters"})
        frame_in = MagicMock(name="l1_in")

        with (
            patch("kpfpipe.modules.masters.base.CalibrationAssociation") as mock_ca,
            patch("kpfpipe.modules.masters.base.ImageProcessing") as mock_ip,
            patch.object(dark, "_load_calibration", lambda l1, cal: False),
        ):
            mock_ip.calibration_applied.return_value = False
            dark._process_frame(frame_in)

        ca_config = mock_ca.call_args[0][1]
        assert "masters_search_window_days" not in ca_config


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
    def _stacked_img(frames, *, nstream=6):
        file_list = sorted(f"f{i}.fits" for i in range(len(frames)))
        dark = Dark(file_list)
        dark.chips = ["GREEN"]
        dark.ccd = {"nrow": 2, "ncol": 2}
        dark.stack_sigma = 1e6  # effectively no clipping
        # Keyed by filename: the streaming path re-reads its first frames for
        # the approximate (clip-bound) pass, so a frame may be loaded twice.
        by_fn = dict(zip(file_list, frames, strict=True))
        with (
            patch.object(dark, "_load_frame", lambda fn, **k: (by_fn[fn], True)),
            patch.object(dark, "_process_frame", lambda l1: l1),
        ):
            return dark.stack_frames(nstream=nstream)["GREEN_IMG"]

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
        # Zero-exptime (bias-like) branch: T = 1, so the estimate is the mean
        # in electrons: (100 + 140) / 2.
        frames = [_stack_frame(0.0, 100.0, 10.0), _stack_frame(0.0, 140.0, 10.0)]
        np.testing.assert_allclose(self._stacked_img(frames), 120.0, rtol=1e-5)

    def test_streaming_path_matches(self):
        # Force the streaming path (nframe >= nstream) with mixed
        # exposures: (4*100) / (10+30+10+30) = 5.0 e-/sec.
        frames = [
            _stack_frame(10.0, 100.0, 10.0),
            _stack_frame(30.0, 100.0, 10.0),
            _stack_frame(10.0, 100.0, 10.0),
            _stack_frame(30.0, 100.0, 10.0),
        ]
        img = self._stacked_img(frames, nstream=3)
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
            lambda arr, sigma, axis=0, **kwargs: (
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
            lambda arr, sigma, axis=0, **kwargs: np.zeros(arr.shape, dtype=bool),
        )

        dark = Dark(sorted(f"f{i}.fits" for i in range(n)))
        dark.chips = ["GREEN"]
        dark.ccd = {"nrow": 2, "ncol": 2}
        dark.stack_sigma = 5.0
        by_fn = dict(zip(dark.l0_file_list, frames, strict=True))
        with (
            patch.object(dark, "_load_frame", lambda fn, **k: (by_fn[fn], True)),
            patch.object(dark, "_process_frame", lambda l1: l1),
        ):
            # nstream = 3 -> ndirect = 2 -> approx from frames 0, 1
            img = dark.stack_frames(nstream=3)["GREEN_IMG"]

        # (95 + 105 + 100) / (3 * 10) = 10 e-/sec; the outlier is excluded from
        # numerator and denominator alike. A mismatch would give 300/40 = 7.5.
        np.testing.assert_allclose(img, 10.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Real sigma-clipping in the datacube path (un-stubbed flag_outliers)
# ---------------------------------------------------------------------------


class TestDatacubeClipping:
    """The datacube path runs the real flag_outliers rejection (unlike
    TestRateEstimator, which disables clipping, and TestPerPixelRejection, which
    stubs flag_outliers). A gross outlier frame is dropped from both the counts
    and exposure-time sums, leaving the surviving rate correct."""

    def test_outlier_frame_is_rejected(self):
        # Five identical frames (100 e- over 10 s -> 10 e-/sec) plus one gross
        # outlier frame; nstream is set high to stay on the datacube path.
        nrow, ncol = 3, 3
        good = [_stack_frame(10.0, 100.0, 100.0, shape=(nrow, ncol)) for _ in range(5)]
        outlier = _stack_frame(10.0, 1e5, 100.0, shape=(nrow, ncol))
        frames = good[:2] + [outlier] + good[2:]  # outlier in the middle

        dark = Dark(sorted(f"f{i}.fits" for i in range(len(frames))))
        dark.chips = ["GREEN"]
        dark.ccd = {"nrow": nrow, "ncol": ncol}
        dark.stack_sigma = 5.0
        by_fn = dict(zip(dark.l0_file_list, frames, strict=True))
        with (
            patch.object(dark, "_load_frame", lambda fn, **k: (by_fn[fn], True)),
            patch.object(dark, "_process_frame", lambda l1: l1),
        ):
            arrays = dark.stack_frames(nstream=10)  # > nframe, so datacube path

        # Outlier excluded from numerator and denominator: (5*100)/(5*10) = 10.
        # Without rejection it would be (5*100 + 1e5) / (6*10) ~= 1675.
        np.testing.assert_allclose(arrays["GREEN_IMG"], 10.0, rtol=1e-5)
        # Dropping one of six frames still leaves every pixel well-sampled.
        assert np.all(arrays["GREEN_MASK"])


# ---------------------------------------------------------------------------
# _clean_l1_arrays: bad-pixel interpolation and mask recompute
# ---------------------------------------------------------------------------


class TestCleanL1Arrays:
    """_clean_l1_arrays interpolates masked-bad pixels for the final IMG/SNR, but
    the recomputed mask is the union of all rejections (across-stack, bad
    SNR/IMG, FFI outliers): a repaired pixel keeps its filled value yet stays
    flagged, preserving provenance."""

    @staticmethod
    def _dark():
        dark = Dark(FILE_LIST)
        dark.chips = ["GREEN"]
        return dark

    @staticmethod
    def _arrays(img, snr, mask):
        return {"GREEN_IMG": img, "GREEN_SNR": snr, "GREEN_MASK": mask}

    def test_rejected_pixel_is_interpolated_but_stays_masked(self):
        # A pixel rejected by stacking (IMG/SNR = 0, mask False) is filled from
        # its neighbors rather than left at zero, but it stays flagged: the
        # across-stack rejection is one of the three checks the final mask
        # unions, so a repair does not restore it to good.
        img = np.full((5, 5), 10.0, dtype=np.float32)
        snr = np.full((5, 5), 20.0, dtype=np.float32)
        mask = np.ones((5, 5), dtype=bool)
        img[2, 2], snr[2, 2], mask[2, 2] = 0.0, 0.0, False

        out = self._dark()._clean_l1_arrays(self._arrays(img, snr, mask), sigma=5.0)

        np.testing.assert_allclose(out["GREEN_IMG"][2, 2], 10.0, rtol=1e-5)
        np.testing.assert_allclose(out["GREEN_SNR"][2, 2], 20.0, rtol=1e-5)
        assert bool(out["GREEN_MASK"][2, 2]) is False

    def test_final_image_outlier_is_flagged(self):
        # A pixel consistent across frames (so it survives stacking, mask True)
        # but extreme in the combined image is caught by the final outlier pass.
        img = np.full((5, 5), 10.0, dtype=np.float32)
        snr = np.full((5, 5), 20.0, dtype=np.float32)
        mask = np.ones((5, 5), dtype=bool)
        img[1, 3] = 1000.0

        out = self._dark()._clean_l1_arrays(self._arrays(img, snr, mask), sigma=5.0)

        assert bool(out["GREEN_MASK"][1, 3]) is False
        assert bool(out["GREEN_MASK"][0, 0]) is True

    def test_zero_value_pixel_is_flagged(self):
        # A good-masked zero pixel is not interpolated (only masked-bad pixels
        # are) and is flagged by the IMG == 0 rule in the mask recompute.
        img = np.full((5, 5), 10.0, dtype=np.float32)
        snr = np.full((5, 5), 20.0, dtype=np.float32)
        mask = np.ones((5, 5), dtype=bool)
        img[4, 4] = 0.0

        out = self._dark()._clean_l1_arrays(self._arrays(img, snr, mask), sigma=5.0)

        assert bool(out["GREEN_MASK"][4, 4]) is False


# ---------------------------------------------------------------------------
# Stacking input validation (the ValueErrors raised by stack_frames /
# _compute_stats_from_datacube on malformed stacks)
# ---------------------------------------------------------------------------


class TestStackingValidation:
    """Malformed stacks must raise rather than silently produce a bad master."""

    @staticmethod
    def _dark(n):
        dark = Dark(sorted(f"f{i}.fits" for i in range(n)))
        dark.chips = ["GREEN"]
        dark.ccd = {"nrow": 2, "ncol": 2}
        return dark

    def _stack_with_frames(self, frames):
        dark = self._dark(len(frames))
        by_fn = dict(zip(dark.l0_file_list, frames, strict=True))
        with (
            patch.object(dark, "_load_frame", lambda fn, **k: (by_fn[fn], True)),
            patch.object(dark, "_process_frame", lambda l1: l1),
        ):
            return dark.stack_frames()

    def test_fewer_than_two_frames_raises(self):
        dark = self._dark(1)
        with pytest.raises(ValueError, match="at least two frames"):
            dark.stack_frames()

    def test_negative_exptime_raises(self):
        frames = [_stack_frame(-5.0, 100.0, 10.0), _stack_frame(-5.0, 100.0, 10.0)]
        with pytest.raises(ValueError, match="cannot be negative"):
            self._stack_with_frames(frames)

    def test_mixed_zero_and_nonzero_exptime_raises(self):
        frames = [_stack_frame(0.0, 100.0, 10.0), _stack_frame(10.0, 100.0, 10.0)]
        with pytest.raises(ValueError, match="all zero or all non-zero"):
            self._stack_with_frames(frames)

    def test_excessive_load_failures_raises(self):
        dark = self._dark(5)
        with (
            patch.object(dark, "_load_frame", lambda fn, **k: (None, False)),
            patch.object(dark, "_process_frame", lambda l1: l1),
        ):
            with pytest.raises(ValueError, match="more than 20%"):
                dark.stack_frames()

    def test_ccd_var_frame_count_mismatch_raises(self):
        # CCD and VAR normally share a survivor mask; a mismatch signals a bug
        # and must not be averaged into a master.
        dark = self._dark(3)
        ones = np.ones((2, 2), dtype=np.float32)
        stats = {
            "GREEN_CCD": {
                "counts_sum": ones.copy(),
                "exptime_sum": ones * 3,
                "nframe": np.full((2, 2), 3, dtype=np.int32),
            },
            "GREEN_VAR": {
                "counts_sum": ones.copy(),
                "nframe": np.full((2, 2), 2, dtype=np.int32),
            },
        }
        with patch.object(
            dark, "_compute_stats_from_datacube", return_value=(stats, False)
        ):
            with pytest.raises(ValueError, match="mismatched frame count"):
                dark.stack_frames()


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


# ---------------------------------------------------------------------------
# Regression: a real master dark from the bundled L0 darks (bias-subtracted,
# real flag_outliers rejection, real bad-pixel cleaning)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMasterDarkRegression:
    """End-to-end master dark from real L0 frames. The five bundled darks span
    two default-gap clusters, so cluster_gap_seconds is widened to stack them;
    each frame is bias-subtracted against the bundled master bias."""

    @pytest.fixture(scope="class")
    def master_dark(self):
        files = build_l0_file_lists(
            "dark",
            data_dir=str(TESTDATA_L0_DIR),
            min_file_count=5,
            cluster_gap_seconds=24 * 3600,
        )
        assert len(files) == 1 and len(files[0]) == 5
        config = {"KPF_MASTERS_OUTPUT": str(TESTDATA_DIR)}
        return Dark(files[0], config=config).make_master_l1()

    def test_returns_kpf_master_l1(self, master_dark):
        assert isinstance(master_dark, KPFMasterL1)

    def test_dark_current_is_small_and_positive(self, master_dark):
        # KPF dark current is a small positive rate (electrons/sec).
        for chip in CHIPS:
            median = np.nanmedian(master_dark.data[f"{chip}_IMG"])
            assert 0 < median < 1.0

    def test_bunit_is_rate(self, master_dark):
        for chip in CHIPS:
            bunit = _header_value(master_dark.headers[f"{chip}_IMG"]["BUNIT"])
            assert bunit == "electrons/sec"

    def test_snr_never_negative(self, master_dark):
        for chip in CHIPS:
            assert np.all(master_dark.data[f"{chip}_SNR"] >= 0)

    def test_mask_mostly_good(self, master_dark):
        # A clean detector stack should keep the large majority of pixels.
        for chip in CHIPS:
            mask = master_dark.data[f"{chip}_MASK"]
            assert mask.dtype == bool
            assert np.mean(mask) > 0.9

    def test_bias_subtracted_via_receipt(self, master_dark):
        assert "master_dark" in master_dark.receipt["Module_Name"].values
