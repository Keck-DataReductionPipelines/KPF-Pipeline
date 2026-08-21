"""Unit tests for the shared master-frame engine, `BaseMasterModule`.

The base is abstract, so these drive it through the simplest concrete vehicles:
`Bias` (no calibrations) for the L1 output/dtype/save path, and `Dark`
(bias-subtracted) for the calibration orchestration path. Behavior specific to a
concrete module lives in test_master_<type>.py.
"""

import logging
import os
import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import kpfpipe.modules.masters.base as masters_base
from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.modules.masters.bias import Bias
from kpfpipe.modules.masters.dark import Dark
from kpfpipe.modules.masters.flat import Flat
from kpfpipe.modules.masters.wls import WLS
from kpfpipe.utils.config import ConfigHandler

from ._dtype_policy import (
    L1_IMAGE,
    MASK_DISK,
    MASK_MEM,
    assert_dtype,
    assert_roundtrip_dtype,
)
from ._masters import (
    FILE_LIST,
    MASTER_NAME,
    make_mocked_master,
    mocked_stack,
)

MASTERS_CONFIG_PATH = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    + "/configs/kpf_drp_masters.toml"
)

# ---------------------------------------------------------------------------
# Base-class fail-loudly paths (construction + frame loading)
# ---------------------------------------------------------------------------


class TestMasterBaseErrors:
    def test_unsorted_l0_list_raises(self):
        # The L0 list must be sorted so stacking order is deterministic.
        with pytest.raises(ValueError, match="sorted in ascending order"):
            Dark(["KP.20240101.00002.00.fits", "KP.20240101.00001.00.fits"])

    def test_bad_config_type_raises(self):
        with pytest.raises(
            TypeError, match="config must be None, dict, or ConfigHandler"
        ):
            Dark(FILE_LIST, config="not-a-config")

    def test_load_frame_missing_file_warns_and_skips(self, caplog):
        fn = "/nonexistent/KP.20240101.00001.00.fits"
        m = Dark([fn])
        with caplog.at_level(logging.WARNING):
            l1_obj = m._load_frame(fn, cache=False)
        assert "Failed to load" in caplog.text
        assert l1_obj is None

    def test_load_frame_qc_failure_warns_and_skips(self, caplog, monkeypatch):
        # EXPTIMOK is the EXPTIME/ELAPSED consistency flag; failing a required
        # QCL0 flag drops the frame before assembly instead of raising.
        m = Dark(FILE_LIST)
        fn = FILE_LIST[0]
        qc_result = {kw: (kw != "EXPTIMOK", "") for kw in Dark._REQUIRED_L0_QC_FLAGS}
        monkeypatch.setattr(
            "kpfpipe.modules.masters.base.KPF0.from_fits", lambda fn: object()
        )
        monkeypatch.setattr(
            "kpfpipe.modules.masters.base.QCL0",
            lambda l0: MagicMock(run=lambda: qc_result),
        )
        with caplog.at_level(logging.WARNING):
            l1_obj = m._load_frame(fn, cache=False)
        assert re.search(r"QC failed.*EXPTIMOK", caplog.text)
        assert l1_obj is None

    def test_load_frame_qc_pass_returns_assembled(self, monkeypatch):
        # The gate's pass-through branch, otherwise only exercised in slow tests.
        m = Dark(FILE_LIST)
        fn = FILE_LIST[0]
        qc_result = {kw: (True, "") for kw in Dark._REQUIRED_L0_QC_FLAGS}
        assembled = object()
        raw_l0 = object()
        seen = {}
        monkeypatch.setattr(
            "kpfpipe.modules.masters.base.KPF0.from_fits", lambda fn: raw_l0
        )

        def _qc(l0):
            seen["qc"] = l0
            return MagicMock(run=lambda: qc_result)

        def _assembly(l0):
            seen["assembly"] = l0
            return MagicMock(perform=lambda: assembled)

        monkeypatch.setattr("kpfpipe.modules.masters.base.QCL0", _qc)
        monkeypatch.setattr("kpfpipe.modules.masters.base.ImageAssembly", _assembly)

        assert m._load_frame(fn, cache=False) is assembled
        # Both stages must see the frame that was read, not some other object:
        # assembling the pre-QC frame, or QC-ing the wrong one, still returns
        # `assembled` if only the return value is checked.
        assert seen["qc"] is raw_l0
        assert seen["assembly"] is raw_l0


# ---------------------------------------------------------------------------
# _process_frame: bias subtraction via CalibrationAssociation + ImageProcessing
# ---------------------------------------------------------------------------


class TestProcessFrame:
    """Dark subtracts the master bias by reusing the standard science modules
    through `_process_frame`, not by hand-rolling the math."""

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

        mock_ca.assert_called_once_with(frame_in, {"KPF_MASTERS_OUTPUT": "/masters"})
        mock_ca.return_value.perform.assert_called_once_with(["bias"])

        mock_load.assert_any_call(associated, "bias")

        # Only the bias is subtracted.
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
        # True calibrations are associated; explicit paths skip association but are
        # still loaded before ImageProcessing runs.
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

        mock_ca.return_value.perform.assert_called_once_with(["bias"])
        mock_ip.assert_called_once_with(associated)
        mock_ip.return_value.perform.assert_called_once_with(
            bias=loaded["bias"], dark=loaded["dark"], flat=False
        )
        assert result is processed

    def test_skips_frame_already_calibrated(self):
        # The streaming pass revisits cached frames, so an already-calibrated frame
        # must pass through untouched rather than be subtracted twice.
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
        # A configured window must reach CalibrationAssociation rather than
        # silently falling back to the module default.
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
        # With no configured window the key is omitted, so CalibrationAssociation
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
    associated master replaces the cached one."""

    @staticmethod
    def _frame(biasfile="master_bias.fits", biasdir="/m"):
        frame = MagicMock(name="l1")
        # BIASFILE holds the master's full path and lives on RECEIPT.
        frame.headers = {"RECEIPT": {"BIASFILE": os.path.join(biasdir, biasfile)}}
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
    # Stacking reads PRIMARY EXPTIME, the actual elapsed time (mapped from the
    # native WMKO ELAPSED), not the requested exposure.
    frame.headers = {"PRIMARY": {"EXPTIME": exptime}, "INSTRUMENT_HEADER": {}}
    frame.data = {
        "GREEN_CCD": np.full(shape, ccd_val, dtype=np.float32),
        "GREEN_VAR": np.full(shape, var_val, dtype=np.float32),
    }
    return frame


class TestRateEstimator:
    """The master IMG is the exposure-weighted rate (total counts / total exposure
    time), correct even when the stack mixes exposure times. Outlier rejection is
    disabled here (large sigma) so the arithmetic is exact."""

    @staticmethod
    def _stacked_img(frames, *, nstream=6):
        file_list = sorted(f"f{i}.fits" for i in range(len(frames)))
        dark = Dark(file_list)
        dark.chips = ["GREEN"]
        dark.ccd = {"nrow": 2, "ncol": 2}
        dark.stack_sigma = 1e6  # effectively no clipping
        # Keyed by filename because the streaming path re-reads its first frames
        # for the approximate (clip-bound) pass, loading a frame twice.
        by_fn = dict(zip(file_list, frames, strict=True))
        with (
            patch.object(dark, "_load_frame", lambda fn, **k: by_fn[fn]),
            patch.object(dark, "_process_frame", lambda l1: l1),
        ):
            return dark.stack_frames(nstream=nstream)["GREEN_IMG"]

    def test_mixed_exptime_is_exposure_weighted(self):
        # Per-frame rates are 10 and 3.33; an equal-weight mean would give ~6.67,
        # but the correct estimate is (100+100)/(10+30) = 5.0 e-/sec.
        frames = [_stack_frame(10.0, 100.0, 10.0), _stack_frame(30.0, 100.0, 10.0)]
        np.testing.assert_allclose(self._stacked_img(frames), 5.0, rtol=1e-5)

    def test_equal_exptime_is_counts_over_exptime(self):
        # (100 + 140) / (20 + 20) = 6.0 e-/sec.
        frames = [_stack_frame(20.0, 100.0, 10.0), _stack_frame(20.0, 140.0, 10.0)]
        np.testing.assert_allclose(self._stacked_img(frames), 6.0, rtol=1e-5)

    def test_zero_exptime_is_mean_counts(self):
        # Zero-exptime (bias-like) branch: T = 1, so the estimate is the mean in
        # electrons, (100 + 140) / 2.
        frames = [_stack_frame(0.0, 100.0, 10.0), _stack_frame(0.0, 140.0, 10.0)]
        np.testing.assert_allclose(self._stacked_img(frames), 120.0, rtol=1e-5)

    def test_streaming_path_matches(self):
        # Streaming path (nframe >= nstream), mixed exposures:
        # (4*100) / (10+30+10+30) = 5.0 e-/sec.
        frames = [
            _stack_frame(10.0, 100.0, 10.0),
            _stack_frame(30.0, 100.0, 10.0),
            _stack_frame(10.0, 100.0, 10.0),
            _stack_frame(30.0, 100.0, 10.0),
        ]
        img = self._stacked_img(frames, nstream=3)
        np.testing.assert_allclose(img, 5.0, rtol=1e-5)

    def test_flat_is_total_electrons_not_a_rate(self):
        # A master flat is the total electrons summed over the stack, not a
        # rate: two frames of 100 e- over 10 s give 200, not 10. Deleting that
        # branch rescales every master flat by 1/exptime_sum while leaving
        # BUNIT ("electrons") and every other assertion in the suite true.
        # The final cleaning pass is bypassed so the arithmetic is exact; the
        # flat-mode cleaner is covered by TestCleanL1Arrays and by the real
        # master flat in test_master_flat.py.
        frames = [_stack_frame(10.0, 100.0, 10.0), _stack_frame(10.0, 100.0, 10.0)]
        file_list = sorted(f"f{i}.fits" for i in range(len(frames)))
        dark = Dark(file_list)
        dark.chips = ["GREEN"]
        dark.ccd = {"nrow": 2, "ncol": 2}
        dark.stack_sigma = 1e6
        by_fn = dict(zip(file_list, frames, strict=True))
        with (
            patch.object(dark, "_load_frame", lambda fn, **k: by_fn[fn]),
            patch.object(dark, "_process_frame", lambda l1: l1),
            patch.object(dark, "_clean_l1_arrays", lambda arrays, *a, **k: arrays),
        ):
            arrays = dark.stack_frames(cal_type="flat")
        np.testing.assert_allclose(arrays["GREEN_IMG"], 200.0, rtol=1e-5)

        # Same frames without the flat token take the rate branch: 200/20 = 10.
        with (
            patch.object(dark, "_load_frame", lambda fn, **k: by_fn[fn]),
            patch.object(dark, "_process_frame", lambda l1: l1),
            patch.object(dark, "_clean_l1_arrays", lambda arrays, *a, **k: arrays),
        ):
            arrays = dark.stack_frames()
        np.testing.assert_allclose(arrays["GREEN_IMG"], 10.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Per-pixel-per-frame rejection in the final normalization
# ---------------------------------------------------------------------------


class TestPerPixelRejection:
    """Counts and exposure time are summed over the SAME per-pixel survivor set, so
    a pixel with fewer good frames yields the same rate as a fully-sampled pixel --
    only its SNR drops (photon statistics)."""

    def test_datacube_partial_rejection_preserves_rate(self, monkeypatch):
        # 5 identical frames (100 e- over 10 s -> 10 e-/sec), with flag_outliers
        # stubbed to reject frame 4 at pixel (0, 0) only.
        n, nrow, ncol = 5, 2, 2
        frames = [
            _stack_frame(10.0, 100.0, 100.0, shape=(nrow, ncol)) for _ in range(n)
        ]
        outlier = np.zeros((n, nrow, ncol), dtype=bool)
        outlier[4, 0, 0] = True
        # 3D input is the stacking pass; 2D is the bad-pixel pass in
        # _clean_l1_arrays, which must stay clean here.
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
            patch.object(dark, "_load_frame", lambda fn, **k: by_fn[fn]),
            patch.object(dark, "_process_frame", lambda l1: l1),
        ):
            arrays = dark.stack_frames()

        img, snr = arrays["GREEN_IMG"], arrays["GREEN_SNR"]
        # The 4/5 pixel (400/40) and the 5/5 pixels (500/50) both give 10 e-/sec.
        # A bug summing exptime over all frames would give 400/50 = 8 at (0, 0).
        np.testing.assert_allclose(img, 10.0, rtol=1e-5)
        # One fewer frame's photons drops SNR by sqrt(4/5): 400/sqrt(400) vs
        # 500/sqrt(500).
        assert snr[0, 0] < snr[1, 1]
        np.testing.assert_allclose(snr[0, 0] / snr[1, 1], np.sqrt(4 / 5), rtol=1e-4)

    def test_streaming_rejection_keeps_rate_consistent(self, monkeypatch):
        # On the streaming path a clear outlier must drop out of BOTH the counts
        # sum and the exposure-time sum, leaving the rate correct.
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
            patch.object(dark, "_load_frame", lambda fn, **k: by_fn[fn]),
            patch.object(dark, "_process_frame", lambda l1: l1),
        ):
            # nstream = 3 -> ndirect = 2 -> approx from frames 0, 1
            img = dark.stack_frames(nstream=3)["GREEN_IMG"]

        # (95 + 105 + 100) / (3 * 10) = 10 e-/sec, the outlier excluded from
        # numerator and denominator alike. A mismatch would give 300/40 = 7.5.
        np.testing.assert_allclose(img, 10.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Real sigma-clipping in the datacube path (un-stubbed flag_outliers)
# ---------------------------------------------------------------------------


class TestDatacubeClipping:
    """The datacube path runs the real flag_outliers rejection, unlike
    TestRateEstimator (clipping disabled) and TestPerPixelRejection (stubbed)."""

    def test_outlier_frame_is_rejected(self):
        # Five identical frames (100 e- over 10 s -> 10 e-/sec) plus one gross
        # outlier; nstream is set high to stay on the datacube path.
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
            patch.object(dark, "_load_frame", lambda fn, **k: by_fn[fn]),
            patch.object(dark, "_process_frame", lambda l1: l1),
        ):
            arrays = dark.stack_frames(nstream=10)  # > nframe, so datacube path

        # Outlier excluded from numerator and denominator: (5*100)/(5*10) = 10.
        # Without rejection it would be (5*100 + 1e5) / (6*10) ~= 1675.
        np.testing.assert_allclose(arrays["GREEN_IMG"], 10.0, rtol=1e-5)
        # Dropping one of six frames still leaves every pixel well-sampled.
        assert np.all(arrays["GREEN_MASK"])

    def test_var_outlier_frame_is_not_rejected(self):
        # Rejection is CCD-only: VAR = |CCD| + RN carries no independent
        # information, so a gross VAR outlier alone must not drop a frame.
        nrow, ncol = 3, 3
        good = [_stack_frame(10.0, 100.0, 100.0, shape=(nrow, ncol)) for _ in range(5)]
        var_outlier = _stack_frame(10.0, 100.0, 1e5, shape=(nrow, ncol))
        frames = good[:2] + [var_outlier] + good[2:]

        dark = Dark(sorted(f"f{i}.fits" for i in range(len(frames))))
        dark.chips = ["GREEN"]
        dark.ccd = {"nrow": nrow, "ncol": ncol}
        dark.stack_sigma = 5.0
        by_fn = dict(zip(dark.l0_file_list, frames, strict=True))
        with (
            patch.object(dark, "_load_frame", lambda fn, **k: by_fn[fn]),
            patch.object(dark, "_process_frame", lambda l1: l1),
        ):
            arrays = dark.stack_frames(nstream=10)  # > nframe, so datacube path

        # CCD normal -> IMG unaffected at 10 e-/sec, and the VAR outlier stays in
        # the variance sum: SNR = 6*100 / sqrt(5*100 + 1e5) ~= 1.89. Had the frame
        # been rejected, SNR would be ~22.4.
        np.testing.assert_allclose(arrays["GREEN_IMG"], 10.0, rtol=1e-5)
        np.testing.assert_allclose(
            arrays["GREEN_SNR"], 600.0 / np.sqrt(100500.0), rtol=1e-5
        )


# ---------------------------------------------------------------------------
# Per-pixel inclusion gates in the final normalization
# ---------------------------------------------------------------------------


class TestSurvivorGate:
    """The last gate before a pixel enters a master: it must have survived in a
    majority of the requested frames, and carry non-zero variance and exposure.
    Otherwise it is zeroed and flagged rather than reconstructed from a
    minority of the stack and published as if fully sampled."""

    @staticmethod
    def _stats(nframe_ccd, var_sum, exptime_sum):
        counts = np.full((2, 2), 300.0, dtype=np.float32)
        return {
            "GREEN_CCD": {
                "counts_sum": counts,
                "exptime_sum": np.asarray(exptime_sum, dtype=np.float32),
                "nframe": np.asarray(nframe_ccd, dtype=np.int32),
            },
            "GREEN_VAR": {
                "counts_sum": np.asarray(var_sum, dtype=np.float32),
                "nframe": np.asarray(nframe_ccd, dtype=np.int32),
            },
        }

    def _stack(self, stats, nrequested=4):
        dark = Dark(sorted(f"f{i}.fits" for i in range(nrequested)))
        dark.chips = ["GREEN"]
        dark.ccd = {"nrow": 2, "ncol": 2}
        # The gate is what is under test, so the later cleaning pass -- which
        # would interpolate the zeroed pixels back in -- is bypassed.
        with (
            patch.object(
                dark, "_compute_stats_from_datacube", return_value=(stats, False)
            ),
            patch.object(dark, "_clean_l1_arrays", lambda arrays, *a, **k: arrays),
        ):
            return dark.stack_frames()

    def test_pixel_surviving_a_minority_of_frames_is_dropped(self):
        # Four frames requested, so the threshold is > 2.0: three survivors pass,
        # two do not. Weakening this to >= would admit the half-sampled pixel.
        stats = self._stats(
            nframe_ccd=[[2, 3], [4, 4]],
            var_sum=[[9.0, 9.0], [9.0, 9.0]],
            exptime_sum=[[3.0, 3.0], [3.0, 3.0]],
        )
        arrays = self._stack(stats)
        assert bool(arrays["GREEN_MASK"][0, 0]) is False
        assert arrays["GREEN_IMG"][0, 0] == 0.0
        assert bool(arrays["GREEN_MASK"][0, 1]) is True
        assert arrays["GREEN_IMG"][0, 1] > 0.0

    def test_zero_variance_or_zero_exposure_pixels_are_dropped(self):
        # Both guards sit beside the majority test and are otherwise unexercised:
        # a pixel with no variance has no SNR, and one with no exposure has no
        # rate, so neither may be published as good.
        stats = self._stats(
            nframe_ccd=[[4, 4], [4, 4]],
            var_sum=[[0.0, 9.0], [9.0, 9.0]],
            exptime_sum=[[3.0, 0.0], [3.0, 3.0]],
        )
        arrays = self._stack(stats)
        assert bool(arrays["GREEN_MASK"][0, 0]) is False  # var_sum == 0
        assert bool(arrays["GREEN_MASK"][0, 1]) is False  # exptime_sum == 0
        assert bool(arrays["GREEN_MASK"][1, 1]) is True


# ---------------------------------------------------------------------------
# _clean_l1_arrays: bad-pixel interpolation and mask recompute
# ---------------------------------------------------------------------------


class TestCleanL1Arrays:
    """_clean_l1_arrays interpolates masked-bad pixels for the final IMG/SNR, but
    the recomputed mask unions all rejections (across-stack, bad SNR/IMG, FFI
    outliers): a repaired pixel keeps its filled value yet stays flagged."""

    @staticmethod
    def _dark():
        dark = Dark(FILE_LIST)
        dark.chips = ["GREEN"]
        return dark

    @staticmethod
    def _arrays(img, snr, mask):
        return {"GREEN_IMG": img, "GREEN_SNR": snr, "GREEN_MASK": mask}

    def test_rejected_pixel_is_interpolated_but_stays_masked(self):
        # A pixel rejected by stacking is filled from its neighbors rather than
        # left at zero, but the across-stack rejection is one of the checks the
        # final mask unions, so the repair does not restore it to good.
        img = np.full((5, 5), 10.0, dtype=np.float32)
        snr = np.full((5, 5), 20.0, dtype=np.float32)
        mask = np.ones((5, 5), dtype=bool)
        img[2, 2], snr[2, 2], mask[2, 2] = 0.0, 0.0, False

        out = self._dark()._clean_l1_arrays(self._arrays(img, snr, mask), sigma=5.0)

        np.testing.assert_allclose(out["GREEN_IMG"][2, 2], 10.0, rtol=1e-5)
        np.testing.assert_allclose(out["GREEN_SNR"][2, 2], 20.0, rtol=1e-5)
        assert bool(out["GREEN_MASK"][2, 2]) is False

    def test_final_image_outlier_is_flagged(self):
        # A pixel consistent across frames survives stacking, so only the final
        # outlier pass can catch it being extreme in the combined image.
        img = np.full((5, 5), 10.0, dtype=np.float32)
        snr = np.full((5, 5), 20.0, dtype=np.float32)
        mask = np.ones((5, 5), dtype=bool)
        img[1, 3] = 1000.0

        out = self._dark()._clean_l1_arrays(self._arrays(img, snr, mask), sigma=5.0)

        assert bool(out["GREEN_MASK"][1, 3]) is False
        assert bool(out["GREEN_MASK"][0, 0]) is True

    def test_zero_value_pixel_is_flagged(self):
        # Only masked-bad pixels are interpolated, so this one is left at zero and
        # flagged by the IMG == 0 rule in the mask recompute.
        img = np.full((5, 5), 10.0, dtype=np.float32)
        snr = np.full((5, 5), 20.0, dtype=np.float32)
        mask = np.ones((5, 5), dtype=bool)
        img[4, 4] = 0.0

        out = self._dark()._clean_l1_arrays(self._arrays(img, snr, mask), sigma=5.0)

        assert bool(out["GREEN_MASK"][4, 4]) is False

    @pytest.mark.parametrize(
        "cal_type,expected",
        [
            # The detector is illuminated only for flats, so each master type
            # judges a pixel against a different reference: a bias against its
            # own cross-dispersion column (axis=0), a flat against the smooth
            # illumination trend along dispersion (axis=1, windowed), a dark
            # against the global median. Collapsing this dispatch to one mode
            # over-flags bias column structure and the flat's blaze while
            # leaving mean(mask) > 0.9 true.
            (None, {"method": "median"}),
            ("dark", {"method": "median"}),
            ("bias", {"axis": 0, "method": "median"}),
            ("flat", {"axis": 1, "kernel_size": 32, "method": "trend"}),
        ],
    )
    def test_cal_type_selects_the_flagging_mode(self, cal_type, expected, monkeypatch):
        seen = {}

        def _spy(data, sigma, **kwargs):
            seen.update(kwargs)
            return np.zeros(data.shape, dtype=bool)

        monkeypatch.setattr(masters_base, "flag_outliers", _spy)
        img = np.full((5, 5), 10.0, dtype=np.float32)
        snr = np.full((5, 5), 20.0, dtype=np.float32)
        mask = np.ones((5, 5), dtype=bool)

        self._dark()._clean_l1_arrays(
            self._arrays(img, snr, mask), sigma=5.0, cal_type=cal_type
        )

        assert seen == expected

    def test_unknown_cal_type_raises(self):
        # Fail loud rather than silently falling back to the dark mode.
        img = np.full((5, 5), 10.0, dtype=np.float32)
        snr = np.full((5, 5), 20.0, dtype=np.float32)
        mask = np.ones((5, 5), dtype=bool)
        with pytest.raises(ValueError, match="unknown cal_type"):
            self._dark()._clean_l1_arrays(
                self._arrays(img, snr, mask), sigma=5.0, cal_type="wls"
            )


# ---------------------------------------------------------------------------
# Stacking input validation (stack_frames / _compute_stats_from_datacube)
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
            patch.object(dark, "_load_frame", lambda fn, **k: by_fn[fn]),
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
            patch.object(dark, "_load_frame", lambda fn, **k: None),
            patch.object(dark, "_process_frame", lambda l1: l1),
        ):
            with pytest.raises(ValueError, match="too many frames failed to load"):
                dark.stack_frames()

    def test_ccd_var_frame_count_mismatch_raises(self):
        # CCD and VAR share a survivor mask, so a mismatch signals a bug and must
        # not be averaged into a master.
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
# _resolve_calibrations: standard intersected with resolved(bias/dark/flat)
# ---------------------------------------------------------------------------


class TestResolveCalibrations:
    """Effective calibrations are the per-master standard intersected with the
    resolved flags; flags can only turn standard calibrations off."""

    def test_default_is_bias_only(self):
        # Dark's standard is ("bias",), but the enabled defaults are bias and dark.
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
        # dark/flat are outside a dark master's standard, so they stay off.
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
        # flat is outside a dark master's standard, so even a path is ignored.
        dark = Dark(FILE_LIST)
        assert dark._resolve_calibrations(flat="/p/master_flat.fits")["flat"] is False


# ---------------------------------------------------------------------------
# L1 output dtype provenance (shared master-L1 path; see _dtype_policy.py)
# ---------------------------------------------------------------------------


class TestDtypeProvenance:
    """Master IMG/SNR are float32; MASK is bool in memory, uint8 (8-bit) on disk."""

    @pytest.fixture(scope="class")
    def master(self):
        return make_mocked_master(Bias)

    def test_img_snr_float32(self, master):
        for ext in ("GREEN_IMG", "RED_IMG", "GREEN_SNR", "RED_SNR"):
            assert_dtype(master.data[ext], L1_IMAGE, ext)

    def test_mask_bool_in_memory(self, master):
        for ext in ("GREEN_MASK", "RED_MASK"):
            assert_dtype(master.data[ext], MASK_MEM, ext)

    def test_roundtrip_img_float32_mask_uint8(self, master, tmp_path):
        assert_roundtrip_dtype(
            KPFMasterL1,
            master,
            "GREEN_IMG",
            L1_IMAGE,
            tmp_path,
            name=MASTER_NAME,
        )
        assert_roundtrip_dtype(
            KPFMasterL1,
            master,
            "GREEN_MASK",
            MASK_MEM,
            tmp_path,
            name=MASTER_NAME,
            expected_disk=MASK_DISK,
        )
        # to_fits casts the masks to uint8 and restores them in a finally:.
        # Losing that restore would turn boolean masking into fancy integer
        # indexing for every in-process consumer of this object.
        assert_dtype(master.data["GREEN_MASK"], MASK_MEM, "GREEN_MASK after to_fits")


# ---------------------------------------------------------------------------
# save_master / make_master_l1(master_path=...) -- the shared write path
# ---------------------------------------------------------------------------


class TestSaveMaster:
    """The shared write path, exercised through Bias (the simplest master)."""

    def test_master_path_writes_fits(self, tmp_path):
        master_path = tmp_path / MASTER_NAME
        make_mocked_master(Bias, master_path=str(master_path))
        assert master_path.exists()

    def test_master_path_creates_parent_dir(self, tmp_path):
        master_path = tmp_path / "nested" / "subdir" / MASTER_NAME
        make_mocked_master(Bias, master_path=str(master_path))
        assert master_path.exists()

    def test_master_path_overwrites_existing(self, tmp_path):
        master_path = tmp_path / MASTER_NAME
        master_path.touch()
        make_mocked_master(Bias, master_path=str(master_path))
        assert master_path.read_bytes()[:6] == b"SIMPLE"

    def test_save_master_before_make_raises(self, tmp_path):
        bias = Bias(FILE_LIST)
        with pytest.raises(RuntimeError, match="run make_master_l1"):
            bias.save_master("L1", str(tmp_path / "should_not_be_created.fits"))

    def test_save_master_refuses_overwrite_by_default(self, tmp_path):
        bias = Bias(FILE_LIST)
        master_path = tmp_path / MASTER_NAME
        master_path.touch()
        with mocked_stack(bias):
            bias.make_master_l1()  # populates ml1_obj
        with pytest.raises(FileExistsError, match="overwrite=True"):
            bias.save_master("L1", str(master_path))


# ---------------------------------------------------------------------------
# Config plumbing: each module reads the sections it names, and only those
# ---------------------------------------------------------------------------


class TestConfigSectionPlumbing:
    """Every masters subclass names the config sections it reads. Dropping one
    from that list is a silent fallback to _DEFAULTS -- a stack_sigma of 5.0
    instead of the configured rejection threshold, or a search window reset --
    with the master still building and only the numbers changing."""

    # One config carrying a different stack_sigma in every section, so a module
    # reading the wrong section is caught by the value, not merely by its absence.
    SIGMAS = {"BIAS": 1.5, "DARK": 7.5, "FLAT": 3.25, "WLS": 9.0}

    @classmethod
    def _config(cls, **extra):
        overrides = {section: {"stack_sigma": v} for section, v in cls.SIGMAS.items()}
        for section, values in extra.items():
            overrides.setdefault(section, {}).update(values)
        return ConfigHandler(MASTERS_CONFIG_PATH, overrides=overrides)

    @pytest.mark.parametrize(
        "cls,section",
        [(Bias, "BIAS"), (Dark, "DARK"), (Flat, "FLAT"), (WLS, "WLS")],
    )
    def test_stack_sigma_comes_from_the_modules_own_section(self, cls, section):
        module = cls(FILE_LIST, self._config())
        assert module.stack_sigma == self.SIGMAS[section]

    @pytest.mark.parametrize("cls", [Bias, Dark, Flat, WLS])
    def test_masters_output_comes_from_data_dirs(self, cls, tmp_path):
        config = self._config(DATA_DIRS={"KPF_MASTERS_OUTPUT": str(tmp_path)})
        assert cls(FILE_LIST, config)._masters_output == str(tmp_path)

    @pytest.mark.parametrize("cls", [Dark, Flat, WLS])
    def test_search_window_comes_from_the_calibration_association_section(self, cls):
        # Bias takes no calibrations, so only the three that associate masters
        # need this section; a reset to the default window would silently change
        # which master calibrates which frame.
        config = self._config(
            MODULE_CALIBRATION_ASSOCIATION={"masters_search_window_days": [-3, 1]}
        )
        assert cls(FILE_LIST, config)._masters_search_window_days == [-3, 1]


# ---------------------------------------------------------------------------
# INPUT_FILES provenance
# ---------------------------------------------------------------------------


class TestInputFilesProvenance:
    def test_records_the_input_list_and_master_type(self):
        # INPUT_FILES and MASTYPE are the master's provenance anchor, and
        # MASTYPE is what generate_standard_filename() needs to build a
        # DRP-RUN-05 name after a round trip.
        ml1 = make_mocked_master(Bias)
        assert ml1.data["INPUT_FILES"]["FILENAME"].tolist() == FILE_LIST
        assert ml1.headers["PRIMARY"]["MASTYPE"] == "bias"

    def test_records_only_the_frames_that_stacked(self):
        # A frame _load_frame drops contributed nothing to the master, so it
        # must not appear in INPUT_FILES -- and, dropped first, must not name
        # the master via generate_standard_filename()'s INPUT_FILES[0].
        # Eight frames, so one drop stays inside the tolerated failure budget.
        dark = Dark(sorted(f"f{i}.fits" for i in range(8)))
        dark.chips = ["GREEN"]
        dark.ccd = {"nrow": 2, "ncol": 2}

        dropped = dark.l0_file_list[0]
        with (
            patch.object(
                dark,
                "_load_frame",
                lambda fn, **k: (
                    None if fn == dropped else _stack_frame(10.0, 100.0, 1.0)
                ),
            ),
            patch.object(dark, "_process_frame", lambda l1: l1),
        ):
            l1_arrays = dark.stack_frames()

        ml1 = dark._build_ml1_obj(l1_arrays, master_type="dark")
        assert ml1.data["INPUT_FILES"]["FILENAME"].tolist() == dark.l0_file_list[1:]
