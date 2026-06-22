"""
Unit tests for the WLS module.

All sub-module I/O is mocked; no real data or FITS files are required.
"""

from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest

import kpfpipe.modules.masters.base as base_module
from kpfpipe import DETECTOR
from kpfpipe.data_models.masters import KPFMasterL2
from kpfpipe.modules.masters.wls import WLS

from ._dtype_policy import WAVE, assert_dtype, assert_roundtrip_dtype

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
NCOL_TEST = 16

FILE_LIST = sorted([f"KP.20240101.{i:05d}.00.fits" for i in range(8)])


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class MockL1:
    pass


class MockL2:
    pass


def _linelist_df(chip, norder, waves):
    """Stub line list (CHIP, ORDER, WAVE): `waves` repeated for every order."""
    return pd.DataFrame(
        [(chip, o, w) for o in range(norder) for w in waves],
        columns=["CHIP", "ORDER", "WAVE"],
    )


@pytest.fixture
def mock_pipeline(monkeypatch):
    """
    Patch CalibrationAssociation, ImageProcessing, and SpectralExtraction so
    that _process_frame and _extract_frame run without touching disk or real
    data.

    Returns the MockL2 instance that SpectralExtraction.perform() will return.
    """
    l2 = MockL2()

    mock_ca = MagicMock()
    mock_ca.return_value.perform.return_value = MockL1()

    mock_ip = MagicMock()
    mock_ip.return_value.perform.return_value = MockL1()
    mock_ip.calibration_applied.return_value = False  # frame not yet calibrated

    mock_se = MagicMock()
    mock_se.return_value.perform.return_value = l2

    monkeypatch.setattr(base_module, "CalibrationAssociation", mock_ca)
    monkeypatch.setattr(base_module, "ImageProcessing", mock_ip)
    monkeypatch.setattr(base_module, "SpectralExtraction", mock_se)
    monkeypatch.setattr(
        base_module.BaseMasterModule, "_load_calibration", lambda self, l1, cal: False
    )

    return l2


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_none_config_sets_masters_root_none(self):
        wls = WLS(FILE_LIST)
        assert wls._masters_root is None

    def test_reads_masters_root_from_config(self):
        # WLS associates the master bias from KPF_MASTERS_OUTPUT -- where the
        # masters recipe writes it and where CalibrationAssociation reads it.
        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": "/masters"})
        assert wls._masters_root == "/masters"

    def test_ignores_data_input_for_masters_root(self):
        # The raw input root is not where masters live; only KPF_MASTERS_OUTPUT
        # drives the masters search.
        wls = WLS(FILE_LIST, config={"KPF_DATA_INPUT": "/in"})
        assert wls._masters_root is None

    def test_invalid_config_raises(self):
        with pytest.raises(TypeError):
            WLS(FILE_LIST, config=42)


# ---------------------------------------------------------------------------
# TestExtractFrame
# ---------------------------------------------------------------------------


class TestExtractFrame:
    def test_returns_l2_obj(self, mock_pipeline):
        wls = WLS(FILE_LIST)
        result = wls._extract_frame(MockL1())
        assert result is mock_pipeline

    def test_passes_masters_root_to_calibration_association(self, monkeypatch):
        # _process_frame (run before _extract_frame) associates the bias from
        # KPF_MASTERS_OUTPUT; _extract_frame itself no longer does calibration.
        mock_ca = MagicMock()
        mock_ca.return_value.perform.return_value = MockL1()
        mock_ip = MagicMock()
        mock_ip.return_value.perform.return_value = MockL1()
        mock_ip.calibration_applied.return_value = False  # frame not yet calibrated
        mock_se = MagicMock()
        mock_se.return_value.perform.return_value = MockL2()

        monkeypatch.setattr(base_module, "CalibrationAssociation", mock_ca)
        monkeypatch.setattr(base_module, "ImageProcessing", mock_ip)
        monkeypatch.setattr(base_module, "SpectralExtraction", mock_se)
        monkeypatch.setattr(
            base_module.BaseMasterModule,
            "_load_calibration",
            lambda self, l1, cal: False,
        )

        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": "/masters"})
        wls._process_frame(MockL1())

        call_args = mock_ca.call_args[0]
        assert call_args[1].get("KPF_MASTERS_OUTPUT") == "/masters"


# ---------------------------------------------------------------------------
# TestProcessIndividualFrames
# ---------------------------------------------------------------------------


class TestProcessIndividualFrames:
    def test_returns_l2_objects_for_all_frames(self, mock_pipeline, monkeypatch):
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(
            wls, "_load_frame", lambda fn, cache=False, **kwargs: (MockL1(), True)
        )
        result = wls._process_stack_l0_to_l2()
        assert len(result) == len(FILE_LIST)
        assert all(r is mock_pipeline for r in result)

    def test_file_list_override(self, mock_pipeline, monkeypatch):
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(
            wls, "_load_frame", lambda fn, cache=False, **kwargs: (MockL1(), True)
        )
        result = wls._process_stack_l0_to_l2(l0_file_list=FILE_LIST[:3])
        assert len(result) == 3

    def test_raises_when_failures_exceed_threshold(self, monkeypatch):
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(
            wls, "_load_frame", lambda fn, cache=False, **kwargs: (None, False)
        )
        with pytest.raises(ValueError, match="20%"):
            wls._process_stack_l0_to_l2()

    def test_tolerates_minority_failure(self, mock_pipeline, monkeypatch):
        # 1 failure out of 8 = 12.5%, below the 20% threshold
        calls = iter([(None, False)] + [(MockL1(), True)] * 7)
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(
            wls, "_load_frame", lambda fn, cache=False, **kwargs: next(calls)
        )
        result = wls._process_stack_l0_to_l2()
        assert len(result) == 7


# ---------------------------------------------------------------------------
# TestMakeMasterL2
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_make_master_l2(monkeypatch):
    """
    Patch frame loading and _compute_wls_from_stack so make_master_l2 runs
    without touching disk or real spectra. _compute_wls_from_stack returns
    synthetic W and coefficient arrays with chip-correct shapes.
    """
    monkeypatch.setattr(
        WLS, "_load_frame", lambda self, fn, cache=False, **kwargs: (MockL1(), True)
    )
    monkeypatch.setattr(WLS, "_process_frame", lambda self, l1, **kwargs: l1)
    monkeypatch.setattr(WLS, "_extract_frame", lambda self, l1, **kwargs: MockL2())

    def mock_compute(
        self,
        chip,
        fibers,
        lineprofile=None,
        polyorder_x=None,
        polyorder_m=None,
        polyorder_f=None,
        **kwargs,
    ):
        polyorder_x = polyorder_x if polyorder_x is not None else self.polyorder_x
        polyorder_m = polyorder_m if polyorder_m is not None else self.polyorder_m
        polyorder_f = polyorder_f if polyorder_f is not None else self.polyorder_f

        norder = NORDER_GREEN if chip == "GREEN" else NORDER_RED
        nfibers = len(fibers)

        if nfibers == 1:
            W = np.full((norder, NCOL_TEST), 5500.0)
            coeffs = np.zeros((polyorder_x + 1, polyorder_m + 1))
        else:
            W = np.full((norder, NCOL_TEST, nfibers), 5500.0)
            coeffs = np.zeros((polyorder_x + 1, polyorder_m + 1, polyorder_f + 1))

        coeffs_stack = np.array([coeffs] * 3)
        lines_stack = [
            {
                "wav": np.array([5500.0, 5501.0]),
                "pix": np.array([100.5, 200.5]),
                "order": np.array([1, 2]),
                "fiber": np.array([fibers[0]] * 2),
                "bad": np.array([False, False]),
                "std": np.array([0.5, 0.5]),
                "amp": np.array([1.0, 1.0]),
            }
            for _ in range(3)
        ]
        return W, coeffs, coeffs_stack, lines_stack

    monkeypatch.setattr(WLS, "_compute_wls_from_stack", mock_compute)


def _header_value(header, key):
    """Extract a header value whether stored as tuple or plain."""
    val = header[key]
    return val[0] if isinstance(val, tuple) else val


class TestMakeMasterL2:
    def test_returns_kpf_master_l2(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        result = wls.make_master_l2()
        assert isinstance(result, KPFMasterL2)

    def test_wave_extensions_populated(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        for chip in wls.chips:
            for fiber in wls.fibers:
                wave = ml2.data[f"{chip}_{fiber}_WAVE"]
                assert np.size(wave) > 0
                assert np.all(wave == 5500.0)

    def test_wave_is_float64_and_survives_roundtrip(
        self, mock_make_master_l2, tmp_path
    ):
        # Master WLS wavelengths are born-64 (EPRV); must not downcast on disk.
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        assert_dtype(ml2.data["TRACE3_WAVE"], WAVE, "master TRACE3_WAVE")
        ml2.headers["PRIMARY"]["DATE-OBS"] = "2024-01-13T10:26:56"
        assert_roundtrip_dtype(KPFMasterL2, ml2, "TRACE3_WAVE", WAVE, tmp_path)

    def test_coeffs_extensions_populated(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        for chip in wls.chips:
            ext = f"{chip}_WLS_COEFFS"
            assert ext in ml2.extensions
            coeffs = ml2.data[ext]
            assert coeffs is not None
            assert coeffs.shape == (
                wls.polyorder_x + 1,
                wls.polyorder_m + 1,
                wls.polyorder_f + 1,
            )

    def test_primary_header_keywords(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        primary = ml2.headers["PRIMARY"]
        for key in [
            "ROUGHWLS",
            "LINELIST",
            "LINEPROF",
            "POLYORDX",
            "POLYORDM",
            "POLYORDF",
            "CHIPS",
            "FIBERS",
        ]:
            assert key in primary

    def test_coeffs_extension_header_keywords(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        for chip in wls.chips:
            hdr = ml2.headers[f"{chip}_WLS_COEFFS"]
            for key in ["POLYORDX", "POLYORDM", "POLYORDF"]:
                assert key in hdr

    def test_to_fits_round_trip(self, mock_make_master_l2, tmp_path):
        # Regression: rvdata builds non-PRIMARY headers via fits.Header(dict),
        # which rejects (value, comment) tuple values. Make sure every header
        # we set survives the round-trip.
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        out_path = tmp_path / "round_trip_master.fits"
        ml2.to_fits(str(out_path))
        assert out_path.exists()

    def test_polyorder_override_stamped(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        override_x = wls.polyorder_x + 4  # ensure different from default
        ml2 = wls.make_master_l2(polyorder_x=override_x)
        assert _header_value(ml2.headers["PRIMARY"], "POLYORDX") == override_x
        for chip in wls.chips:
            assert (
                _header_value(ml2.headers[f"{chip}_WLS_COEFFS"], "POLYORDX")
                == override_x
            )
            # verify the override actually propagated into the fit, not just the header
            coeffs = ml2.data[f"{chip}_WLS_COEFFS"]
            assert coeffs.shape[0] == override_x + 1

    def test_input_files_recorded(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        assert ml2.data["INPUT_FILES"]["FILENAME"].tolist() == FILE_LIST

    def test_receipt_entry(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        assert "master_wls" in ml2.receipt["Module_Name"].tolist()

    def test_resets_l2_cache(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        wls._l2_obj_cache = [MockL2(), MockL2(), MockL2()]
        wls.make_master_l2()
        assert len(wls._l2_obj_cache) == len(FILE_LIST)

    def test_stacks_stashed_on_self(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()
        assert wls._coeffs_stack is not None
        assert wls._lines_stack is not None
        for chip in wls.chips:
            assert chip in wls._coeffs_stack
            assert chip in wls._lines_stack

    def test_save_diagnostics_before_make_raises(self):
        wls = WLS(FILE_LIST)
        with pytest.raises(RuntimeError, match="run make_master_l2"):
            wls.save_diagnostics("/tmp/should_not_be_created.h5")

    def test_save_diagnostics_with_empty_stash_raises(self, tmp_path):
        # make_master_l2 initialises both stash dicts to {} before populating
        # them. If the chip loop raises before any chip is added, the dicts
        # stay empty — save_diagnostics must refuse rather than write an
        # empty HDF5.
        wls = WLS(FILE_LIST)
        wls._coeffs_stack = {}
        wls._lines_stack = {}
        out_path = tmp_path / "empty.h5"
        with pytest.raises(RuntimeError, match="run make_master_l2"):
            wls.save_diagnostics(str(out_path))
        assert not out_path.exists()

    def test_diagnostics_path_writes_hdf5(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST)
        diagnostics_path = tmp_path / "diagnostics.h5"
        wls.make_master_l2(diagnostics_path=str(diagnostics_path))
        assert diagnostics_path.exists()

    def test_save_diagnostics_post_hoc(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()  # no diagnostics_path; stacks stashed on self
        diagnostics_path = tmp_path / "diagnostics.h5"
        wls.save_diagnostics(str(diagnostics_path))
        assert diagnostics_path.exists()

    def test_save_master_before_make_raises(self):
        wls = WLS(FILE_LIST)
        with pytest.raises(RuntimeError, match="run make_master_l2"):
            wls.save_master("L2", "/tmp/should_not_be_created.fits")

    def test_save_master_rejects_unknown_level(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()
        with pytest.raises(ValueError, match="level"):
            wls.save_master("L4", str(tmp_path / "master.fits"))

    def test_master_path_writes_fits(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST)
        master_path = tmp_path / "master.fits"
        wls.make_master_l2(master_path=str(master_path))
        assert master_path.exists()

    def test_save_master_post_hoc(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()  # no master_path; ml2_obj stashed on self
        master_path = tmp_path / "master.fits"
        wls.save_master("L2", str(master_path))
        assert master_path.exists()

    def test_save_master_creates_parent_dir(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()
        master_path = tmp_path / "nested" / "subdir" / "master.fits"
        wls.save_master("L2", str(master_path))
        assert master_path.exists()

    def test_save_master_refuses_overwrite_by_default(
        self, mock_make_master_l2, tmp_path
    ):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()
        master_path = tmp_path / "master.fits"
        master_path.touch()
        with pytest.raises(FileExistsError, match="overwrite=True"):
            wls.save_master("L2", str(master_path))

    def test_save_master_overwrite_true_replaces(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()
        master_path = tmp_path / "master.fits"
        master_path.write_bytes(b"stale")
        wls.save_master("L2", str(master_path), overwrite=True)
        assert master_path.read_bytes()[:6] == b"SIMPLE"

    def test_master_path_overwrites_existing(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST)
        master_path = tmp_path / "master.fits"
        master_path.touch()
        wls.make_master_l2(master_path=str(master_path))
        assert master_path.read_bytes()[:6] == b"SIMPLE"

    def test_hdf5_structure(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST)
        diagnostics_path = str(tmp_path / "diagnostics.h5")
        wls.make_master_l2(diagnostics_path=diagnostics_path)

        # mock_compute returns three synthetic frames per chip
        expected_nframes = 3
        expected_coeffs_shape = (
            expected_nframes,
            wls.polyorder_x + 1,
            wls.polyorder_m + 1,
            wls.polyorder_f + 1,
        )

        with h5py.File(diagnostics_path, "r") as h5:
            for chip in wls.chips:
                assert chip in h5
                cs = h5[chip]["coeffs_stack"]
                assert cs.shape == expected_coeffs_shape
                assert np.issubdtype(cs.dtype, np.floating)

                assert "lines_stack" in h5[chip]
                frame_keys = sorted(h5[chip]["lines_stack"].keys())
                assert len(frame_keys) == expected_nframes

                sample = h5[chip]["lines_stack"][frame_keys[0]]
                for key in ["wav", "pix", "order", "fiber", "bad", "std", "amp"]:
                    assert key in sample
                assert np.issubdtype(sample["wav"].dtype, np.floating)
                assert np.issubdtype(sample["pix"].dtype, np.floating)
                assert np.issubdtype(sample["order"].dtype, np.integer)
                assert sample["bad"].dtype == bool
                assert h5py.check_string_dtype(sample["fiber"].dtype) is not None
                # finite values for all numeric per-line arrays
                for key in ["wav", "pix", "std", "amp"]:
                    assert np.all(np.isfinite(sample[key][...]))

    def test_nan_orderlet_emits_warning_and_does_not_crash(self):
        """
        A NaN-filled orderlet (extraction failure) should be skipped with a
        warning rather than crashing scipy's least_squares.
        """
        wls = WLS(FILE_LIST)
        ncol = wls.ccd["ncol"]
        norder = wls.norder["RED"]

        flux = np.ones((norder, ncol))
        flux[0, :] = np.nan  # simulate failed-extraction orderlet

        class StubL2:
            data = {"RED_SCI1_FLUX": flux}

        wls.rough_wls["RED_SCI1_WAVE"] = np.tile(
            np.linspace(6500.0, 6510.0, ncol), (norder, 1)
        )
        wls._linelist_df = _linelist_df("RED", norder, [6502.0, 6505.0, 6508.0])

        with pytest.warns(UserWarning, match=r"RED SCI1 order 1: orderlet skipped"):
            result = wls._fit_line_positions_ffi(
                StubL2(),
                "RED",
                ["SCI1"],
            )

        # The NaN order contributed no lines; remaining orders did.
        assert result["order"].min() >= 2
        assert len(result["wav"]) > 0

    def test_linelist_override(self, mock_make_master_l2, tmp_path, monkeypatch):
        """Override file is loaded into the cache and stamped to the header."""
        override = tmp_path / "alt_linelist.csv"
        override.write_text(
            "CHIP,ORDER,WAVE\nGREEN,0,4500.0\nGREEN,1,5500.0\nRED,0,6500.0\n"
        )

        wls = WLS(FILE_LIST)
        original_path = wls.linelist
        original_df = wls._linelist_df.copy()

        ml2 = wls.make_master_l2(linelist=str(override))

        assert wls.linelist == str(override)
        assert wls.linelist != original_path
        assert not wls._linelist_df.equals(original_df)
        np.testing.assert_array_equal(
            wls._linelist_df["WAVE"].values, np.array([4500.0, 5500.0, 6500.0])
        )
        assert _header_value(ml2.headers["PRIMARY"], "LINELIST") == str(override)

    def test_single_frame_stack(self, mock_make_master_l2):
        """A 1-frame stack should still produce a valid master L2."""
        wls = WLS(FILE_LIST[:1])
        ml2 = wls.make_master_l2()
        assert isinstance(ml2, KPFMasterL2)
        assert len(wls._l2_obj_cache) == 1
        for chip in wls.chips:
            for fiber in wls.fibers:
                assert f"{chip}_{fiber}_WAVE" in ml2.data

    def test_empty_file_list_raises(self, mock_make_master_l2):
        wls = WLS([])
        with pytest.raises(ValueError, match=r"Empty l0_file_list"):
            wls.make_master_l2()

    def test_single_chip_config(self, mock_make_master_l2):
        """A WLS configured for one chip only should not populate the other."""
        wls = WLS(FILE_LIST, config={"chips": ["GREEN"]})
        ml2 = wls.make_master_l2()
        for fiber in wls.fibers:
            # GREEN populated by the mock (5500.0); RED left at the schema-
            # default zeros since RED was not in self.chips.
            assert np.all(ml2.data[f"GREEN_{fiber}_WAVE"] == 5500.0)
            assert np.all(ml2.data[f"RED_{fiber}_WAVE"] == 0.0)
        assert "GREEN_WLS_COEFFS" in ml2.extensions
        assert "RED_WLS_COEFFS" not in ml2.extensions

    def test_results_populated(self, mock_make_master_l2):
        """make_master_l2 should populate self._results with per-chip line yields."""
        wls = WLS(FILE_LIST)
        assert wls._results is None
        wls.make_master_l2()
        assert wls._results is not None
        for chip in wls.chips:
            assert chip in wls._results
            assert "n_total" in wls._results[chip]
            assert "n_fit" in wls._results[chip]
            assert isinstance(wls._results[chip]["n_total"], int)
            assert isinstance(wls._results[chip]["n_fit"], int)

    def test_info_before_make_master_l2(self, capsys):
        """info() before perform should print config and the not-called message."""
        wls = WLS(FILE_LIST)
        wls.info()
        out = capsys.readouterr().out
        assert "WLS" in out
        assert "make_master_l2() has not been called" in out

    def test_info_after_make_master_l2(self, mock_make_master_l2, capsys):
        """info() after perform should print the per-chip line yield table."""
        wls = WLS(FILE_LIST)
        wls.make_master_l2()
        capsys.readouterr()  # discard any output from make_master_l2 itself
        wls.info()
        out = capsys.readouterr().out
        assert "WLS" in out
        assert "make_master_l2() has not been called" not in out
        for chip in wls.chips:
            assert chip in out


# ---------------------------------------------------------------------------
# TestCalculateWlsCoeffs
# ---------------------------------------------------------------------------


class TestCalculateWlsCoeffs:
    def _make_lines(self, n, fibers=("SCI1",)):
        """Build a minimal `lines` dict with `n` lines per fiber."""
        wav, pix, ord_, fib = [], [], [], []
        for f in fibers:
            wav.extend(5000.0 + np.arange(n))
            pix.extend(np.linspace(10.0, 4000.0, n))
            ord_.extend(np.linspace(1, 30, n).astype(int))
            fib.extend([f] * n)
        return {
            "wav": np.asarray(wav, dtype=float),
            "pix": np.asarray(pix, dtype=float),
            "order": np.asarray(ord_, dtype=int),
            "fiber": np.asarray(fib),
            "bad": np.zeros(n * len(fibers), dtype=bool),
        }

    def test_underconstrained_single_fiber_raises(self):
        # default polyorder is (6, 3, 2) → 7*4 = 28 free params single-fiber
        wls = WLS(FILE_LIST)
        lines = self._make_lines(5, fibers=("SCI1",))
        with pytest.raises(ValueError, match=r"underconstrained"):
            wls._calculate_wls_coeffs(lines, norder=30)

    def test_underconstrained_multi_fiber_raises(self):
        # 5-fiber → 7*4*3 = 84 free params; 10 lines per fiber * 5 = 50 < 84
        wls = WLS(FILE_LIST)
        lines = self._make_lines(10, fibers=("SKY", "SCI1", "SCI2", "SCI3", "CAL"))
        with pytest.raises(ValueError, match=r"underconstrained"):
            wls._calculate_wls_coeffs(lines, norder=30)

    def test_sufficient_lines_does_not_raise(self):
        wls = WLS(FILE_LIST)
        lines = self._make_lines(50, fibers=("SCI1",))
        coeffs = wls._calculate_wls_coeffs(lines, norder=30)
        assert coeffs.shape == (wls.polyorder_x + 1, wls.polyorder_m + 1)


# ---------------------------------------------------------------------------
# TestComputeWlsFrameRejection
# ---------------------------------------------------------------------------


class TestComputeWlsFrameRejection:
    """Frame-level QC in _compute_wls_from_stack: drop frames whose line-fit
    failure fraction exceeds max_bad_frac, and error if more than one is dropped."""

    def _setup(self, monkeypatch, bad_fracs, nlines=100):
        """Build a WLS whose stack yields one fake `lines` dict per entry in
        `bad_fracs`, each with the requested fraction of bad line fits."""
        wls = WLS(FILE_LIST)
        wls._l2_obj_cache = [MockL2() for _ in bad_fracs]

        frames = []
        for frac in bad_fracs:
            bad = np.zeros(nlines, dtype=bool)
            bad[: int(round(frac * nlines))] = True
            frames.append({"wav": np.zeros(nlines), "bad": bad})
        it = iter(frames)

        monkeypatch.setattr(
            WLS, "_fit_line_positions_ffi", lambda self, *a, **k: next(it)
        )
        monkeypatch.setattr(
            WLS, "_calculate_wls_coeffs", lambda self, *a, **k: np.ones((2, 2))
        )
        monkeypatch.setattr(
            WLS, "_evaluate_wls_coeffs", staticmethod(lambda *a, **k: np.zeros((3, 3)))
        )
        return wls

    def test_all_clean_frames_kept(self, monkeypatch):
        wls = self._setup(monkeypatch, [0.01, 0.02, 0.0, 0.03, 0.01])
        _, _, coeffs_stack, lines_stack = wls._compute_wls_from_stack(
            "GREEN", ["SCI1"], verbose=False
        )
        assert len(coeffs_stack) == 5
        assert len(lines_stack) == 5

    def test_single_bad_frame_dropped(self, monkeypatch):
        wls = self._setup(monkeypatch, [0.01, 0.22, 0.0, 0.03, 0.01])
        _, _, coeffs_stack, lines_stack = wls._compute_wls_from_stack(
            "GREEN", ["SCI1"], verbose=False
        )
        # the 22%-bad frame is excluded from both stacks
        assert len(coeffs_stack) == 4
        assert len(lines_stack) == 4

    def test_two_bad_frames_raises(self, monkeypatch):
        wls = self._setup(monkeypatch, [0.22, 0.01, 0.34, 0.01, 0.01])
        with pytest.raises(ValueError, match=r"more than one frame rejected"):
            wls._compute_wls_from_stack("GREEN", ["SCI1"], verbose=False)

    def test_threshold_is_inclusive_at_max_bad_frac(self, monkeypatch):
        # exactly 5% bad is not > 5%, so the frame is kept
        wls = self._setup(monkeypatch, [0.05, 0.01], nlines=100)
        _, _, coeffs_stack, _ = wls._compute_wls_from_stack(
            "GREEN", ["SCI1"], verbose=False
        )
        assert len(coeffs_stack) == 2

    def test_nonfinite_coeffs_raise(self, monkeypatch):
        # a frame whose per-frame fit yields a NaN coefficient must fail loudly
        # rather than silently poison the combined solution
        wls = WLS(FILE_LIST)
        wls._l2_obj_cache = [MockL2() for _ in range(3)]
        lines = {"wav": np.zeros(100), "bad": np.zeros(100, dtype=bool)}
        coeffs = iter(
            [np.ones((2, 2)), np.array([[1.0, np.nan], [1.0, 1.0]]), np.ones((2, 2))]
        )
        monkeypatch.setattr(WLS, "_fit_line_positions_ffi", lambda self, *a, **k: lines)
        monkeypatch.setattr(
            WLS, "_calculate_wls_coeffs", lambda self, *a, **k: next(coeffs)
        )
        monkeypatch.setattr(
            WLS, "_evaluate_wls_coeffs", staticmethod(lambda *a, **k: np.zeros((3, 3)))
        )
        with pytest.raises(ValueError, match=r"non-finite Legendre coefficients"):
            wls._compute_wls_from_stack("GREEN", ["SCI1"], verbose=False)


# ---------------------------------------------------------------------------
# TestFitLinePositions
# ---------------------------------------------------------------------------


class TestFitLinePositions:
    def test_no_lines_returns_empty(self):
        """No reference lines for the order yields empty arrays."""
        wls = WLS(FILE_LIST)
        flux = np.ones(100)
        wave = np.linspace(5000.0, 5100.0, 100)

        result = wls._fit_line_positions_1d(flux, wave, np.array([]))
        for key in ["wav", "pix", "std", "amp", "bad"]:
            assert len(result[key]) == 0

    def test_line_outside_rough_wls_range_raises(self):
        """A reference line outside the order's rough WLS span is a CHIP/ORDER
        labeling inconsistency and must fail loudly."""
        wls = WLS(FILE_LIST)
        flux = np.ones(100)
        wave = np.linspace(5000.0, 5100.0, 100)
        with pytest.raises(ValueError, match="outside the rough"):
            wls._fit_line_positions_1d(flux, wave, np.array([6000.0]))

    def test_line_fit_qc_flags_centroid_outside_window(self):
        """A fitted centroid more than `window` pixels from its window center
        is flagged bad, even when amplitude and width are in range."""
        wls = WLS(FILE_LIST)
        lines = {
            "amp": np.array([1.0, 1.0, 1.0]),
            "std": np.array([1.0, 1.0, 1.0]),
            "pix": np.array([100.0, 103.0, 120.0]),  # offsets 0, 3, 20
        }
        loc = np.full(3, 100.0)
        bad = wls._line_fit_qc(lines, "gaussian", window=5, loc=loc)
        assert list(bad) == [False, False, True]

    def test_fit_returns_float64_from_float32_inputs(self):
        """float32 flux/wave inputs must not drag the line fit into float32."""
        wls = WLS(FILE_LIST)
        ncol = 100
        x = np.arange(ncol)
        wave = np.linspace(5000.0, 5100.0, ncol).astype(np.float32)
        flux = (1.0 + 50.0 * np.exp(-0.5 * ((x - 50) / 2.0) ** 2)).astype(np.float32)
        line_waves = np.array([wave[50]], dtype=float)

        result = wls._fit_line_positions_1d(
            flux, wave, line_waves, lineprofile="gaussian"
        )
        assert len(result["wav"]) == 1
        for key in ["wav", "pix", "std", "amp"]:
            assert result[key].dtype == np.float64

    def test_all_nan_fiber_emits_fiber_level_warning(self):
        """A fiber whose every order is NaN should emit a fiber-level warning."""
        wls = WLS(FILE_LIST)
        ncol = wls.ccd["ncol"]
        norder = wls.norder["RED"]

        flux = np.full((norder, ncol), np.nan)  # entire fiber NaN

        class StubL2:
            data = {"RED_SCI1_FLUX": flux}

        wls.rough_wls["RED_SCI1_WAVE"] = np.tile(
            np.linspace(6500.0, 6510.0, ncol), (norder, 1)
        )
        wls._linelist_df = _linelist_df("RED", norder, [6502.0, 6505.0])

        with pytest.warns(UserWarning, match=r"RED SCI1: no good lines retained"):
            result = wls._fit_line_positions_ffi(
                StubL2(),
                "RED",
                ["SCI1"],
            )
        assert len(result["wav"]) == 0

    def test_nan_orderlet_warning_suppressed_when_verbose_false(self, recwarn):
        """verbose=False should silence the per-orderlet skip warning."""
        wls = WLS(FILE_LIST)
        ncol = wls.ccd["ncol"]
        norder = wls.norder["RED"]

        flux = np.ones((norder, ncol))
        flux[0, :] = np.nan  # one orderlet NaN-filled

        class StubL2:
            data = {"RED_SCI1_FLUX": flux}

        wls.rough_wls["RED_SCI1_WAVE"] = np.tile(
            np.linspace(6500.0, 6510.0, ncol), (norder, 1)
        )
        wls._linelist_df = _linelist_df("RED", norder, [6502.0, 6505.0, 6508.0])

        wls._fit_line_positions_ffi(StubL2(), "RED", ["SCI1"], verbose=False)

        skipped = [w for w in recwarn if "orderlet skipped" in str(w.message)]
        assert len(skipped) == 0

    def test_all_nan_fiber_warning_suppressed_when_verbose_false(self, recwarn):
        """verbose=False should silence the fiber-level no-good-lines warning."""
        wls = WLS(FILE_LIST)
        ncol = wls.ccd["ncol"]
        norder = wls.norder["RED"]

        flux = np.full((norder, ncol), np.nan)

        class StubL2:
            data = {"RED_SCI1_FLUX": flux}

        wls.rough_wls["RED_SCI1_WAVE"] = np.tile(
            np.linspace(6500.0, 6510.0, ncol), (norder, 1)
        )
        wls._linelist_df = _linelist_df("RED", norder, [6502.0, 6505.0])

        wls._fit_line_positions_ffi(StubL2(), "RED", ["SCI1"], verbose=False)

        fiber_level = [w for w in recwarn if "no good lines retained" in str(w.message)]
        assert len(fiber_level) == 0


# ---------------------------------------------------------------------------
# TestRoughWlsLoading
# ---------------------------------------------------------------------------


class TestRoughWlsLoading:
    def test_missing_rough_wls_file_raises(self):
        wls = WLS(FILE_LIST)
        with pytest.raises(FileNotFoundError):
            wls._load_rough_wls(rough_wls_file="/nonexistent/path.csv")
