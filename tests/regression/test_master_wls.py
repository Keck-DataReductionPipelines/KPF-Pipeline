"""Unit tests for the WLS module.

All sub-module I/O is mocked; no real data or FITS files are required.
"""

import logging
from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest

import kpfpipe.modules.masters.base as base_module
from kpfpipe import DETECTOR
from kpfpipe.data_models.masters import KPFMasterL2
from kpfpipe.modules.masters.wls import WLS
from kpfpipe.utils.kpf import get_obs_id

from ._dtype_policy import MASK_MEM, WAVE, assert_dtype, assert_roundtrip_dtype
from ._masters import FILE_LIST

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
NCOL_TEST = 16

DATECODE = "20240101"  # shared by FILE_LIST and the master obs_ids below


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class MockL1:
    pass


class MockL2:
    def __init__(self, obs_id=None):
        self.obs_id = obs_id

    def to_fits(self, path):
        open(path, "w").close()


def _linelist_df(chip, norder, waves):
    """Stub line list: `waves` repeated for every 0-based order index."""
    echelle = np.linspace(*DETECTOR["echelle_orders"][chip], norder).round().astype(int)
    return pd.DataFrame(
        [(chip, o, int(echelle[o]), w) for o in range(norder) for w in waves],
        columns=["CHIP", "INDEX", "ECHELLE", "WAVE"],
    )


def _master_and_stack(masters_output, obs_id="KP.20240101.00000.00"):
    """The (master_path, stack_dir) pair for a ThAr master keyed on `obs_id`.

    The stack subdir, beside the master, is where WLS writes its per-frame L2s
    and diagnostics.
    """
    night = masters_output / "masters" / DATECODE
    return night / f"{obs_id}_master_thar_L2.fits", night / "thar_L2"


@pytest.fixture
def mock_pipeline(monkeypatch):
    """Run _process_frame and _extract_frame without touching disk or real data.

    Returns the MockL2 instance that SpectralExtraction.perform() will return.
    """
    l2 = MockL2()

    mock_ca = MagicMock()
    mock_ca.return_value.perform.return_value = MockL1()

    mock_ip = MagicMock()
    mock_ip.return_value.perform.return_value = MockL1()
    mock_ip.calibration_applied.return_value = False

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


def _stub_frame_pipeline(monkeypatch, l1=None, extract=None):
    """Short-circuit load -> process -> extract so a test drives only the fit.

    ``_load_frame`` threads each source filename through as the opaque L1 token
    unless ``l1`` overrides it, so the extracted L2 stub carries that frame's
    obs_id (which drives per-frame naming). The fit/combine stubs stay per-test:
    their survivor and rejection structure IS what each test asserts.
    """
    load = (lambda self, fn, cache=False, **kw: fn) if l1 is None else l1
    extract = extract or (lambda self, l1_obj, **kw: MockL2(get_obs_id(l1_obj)))
    monkeypatch.setattr(WLS, "_load_frame", load)
    monkeypatch.setattr(WLS, "_process_frame", lambda self, l1_obj, **kw: l1_obj)
    monkeypatch.setattr(WLS, "_extract_frame", extract)


class TestInit:
    def test_none_config_sets_masters_output_none(self):
        wls = WLS(FILE_LIST)
        assert wls._masters_output is None

    def test_reads_masters_output_from_config(self):
        # WLS associates the master bias from KPF_MASTERS_OUTPUT, where the
        # masters recipe writes it.
        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": "/masters"})
        assert wls._masters_output == "/masters"

    def test_ignores_data_input_for_masters_output(self):
        # The raw input root is not where masters live.
        wls = WLS(FILE_LIST, config={"KPF_DATA_INPUT": "/in"})
        assert wls._masters_output is None

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

    def test_passes_masters_output_to_calibration_association(self, monkeypatch):
        # _process_frame, not _extract_frame, is what associates the bias.
        mock_ca = MagicMock()
        mock_ca.return_value.perform.return_value = MockL1()
        mock_ip = MagicMock()
        mock_ip.return_value.perform.return_value = MockL1()
        mock_ip.calibration_applied.return_value = False
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
            wls, "_load_frame", lambda fn, cache=False, **kwargs: MockL1()
        )
        result = wls._process_stack_l0_to_l2()
        assert len(result) == len(FILE_LIST)
        assert all(r is mock_pipeline for r in result)

    def test_file_list_override(self, mock_pipeline, monkeypatch):
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(
            wls, "_load_frame", lambda fn, cache=False, **kwargs: MockL1()
        )
        result = wls._process_stack_l0_to_l2(l0_file_list=FILE_LIST[:3])
        assert len(result) == 3

    def test_raises_when_failures_exceed_threshold(self, monkeypatch):
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(wls, "_load_frame", lambda fn, cache=False, **kwargs: None)
        with pytest.raises(ValueError, match="too many frames failed to load"):
            wls._process_stack_l0_to_l2()

    def test_tolerates_minority_failure(self, mock_pipeline, monkeypatch):
        # 1 failure out of 8 = 12.5%, below the 20% threshold
        calls = iter([None] + [MockL1()] * 7)
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
    """Run make_master_l2 without touching disk or real spectra.

    The fitting stub reports one non-rejected frame per input (FILE_LIST has 8,
    so the default min_stack_size=5 gate passes); the combine stub returns
    synthetic W and coefficient arrays with chip-correct shapes.
    """
    _stub_frame_pipeline(monkeypatch)

    def mock_fit_and_qc(
        self,
        chip,
        fibers,
        lineprofile=None,
        poly_degree_x=None,
        poly_degree_m=None,
        poly_degree_f=None,
        **kwargs,
    ):
        poly_degree_x = (
            poly_degree_x if poly_degree_x is not None else self.poly_degree_x
        )
        poly_degree_m = (
            poly_degree_m if poly_degree_m is not None else self.poly_degree_m
        )
        poly_degree_f = (
            poly_degree_f if poly_degree_f is not None else self.poly_degree_f
        )

        nfibers = len(fibers)
        if nfibers == 1:
            coeffs = np.zeros((poly_degree_x + 1, poly_degree_m + 1))
        else:
            coeffs = np.zeros((poly_degree_x + 1, poly_degree_m + 1, poly_degree_f + 1))

        return [
            {
                "obs_id": l2_obj.obs_id,
                "lines": {
                    "chip": np.array([chip] * 2),
                    "fiber": np.array([fibers[0]] * 2),
                    "index": np.array([0, 1]),
                    "echelle": self._echelle_orders[chip][:2].astype(int),
                    "wav": np.array([5500.0, 5501.0]),
                    "pix": np.array([100.5, 200.5]),
                    "std": np.array([0.5, 0.5]),
                    "amp": np.array([1.0, 1.0]),
                    "isgood": np.array([True, True]),
                },
                "coeffs": coeffs,
                "rejected": False,
            }
            for l2_obj in self._l2_obj_cache
        ]

    def mock_combine(self, frames, chip, nfibers, **kwargs):
        norder = NORDER_GREEN if chip == "GREEN" else NORDER_RED
        coeffs = next(fr["coeffs"] for fr in frames if not fr["rejected"])
        if nfibers == 1:
            W = np.full((norder, NCOL_TEST), 5500.0)
        else:
            W = np.full((norder, NCOL_TEST, nfibers), 5500.0)
        return W, coeffs

    monkeypatch.setattr(WLS, "_fit_and_qc_lines_stack", mock_fit_and_qc)
    monkeypatch.setattr(WLS, "_combine_coeffs_stack", mock_combine)


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

    def test_wave_routing_uses_physical_position_not_fiber_order(self, monkeypatch):
        # W's planes are ordered by fiber_positions (SKY=0..CAL=4), so the write
        # loop must sort by that, not trust self.fibers' config-overridable
        # order -- else SKY's solution lands on CAL and vice versa.
        _stub_frame_pipeline(
            monkeypatch,
            l1=lambda self, fn, cache=False, **kw: MockL1(),
            extract=lambda self, l1_obj, **kw: MockL2(),
        )

        def mock_fit_and_qc(self, chip, fibers, **kwargs):
            coeffs = np.zeros((self.poly_degree_x + 1, self.poly_degree_m + 1, 1))
            return [
                {
                    "obs_id": None,
                    "lines": {"wav": np.array([5500.0]), "isgood": np.array([True])},
                    "coeffs": coeffs,
                    "rejected": False,
                }
                for _ in self._l2_obj_cache
            ]

        def mock_combine(self, frames, chip, nfibers, **kwargs):
            # Plane i carries the constant value i (its physical-position rank).
            norder = NORDER_GREEN if chip == "GREEN" else NORDER_RED
            W = np.empty((norder, NCOL_TEST, nfibers))
            for i in range(nfibers):
                W[:, :, i] = float(i)
            coeffs = np.zeros((self.poly_degree_x + 1, self.poly_degree_m + 1, 1))
            return W, coeffs

        monkeypatch.setattr(WLS, "_fit_and_qc_lines_stack", mock_fit_and_qc)
        monkeypatch.setattr(WLS, "_combine_coeffs_stack", mock_combine)

        wls = WLS(FILE_LIST)
        # Reorder away from physical-slicer order.
        wls.fibers = ["CAL", "SCI1", "SCI2", "SCI3", "SKY"]
        ml2 = wls.make_master_l2()

        # Ground truth from detector.toml [fiber_positions].
        physical_rank = {"SKY": 0, "SCI1": 1, "SCI2": 2, "SCI3": 3, "CAL": 4}
        for chip in wls.chips:
            for fiber, rank in physical_rank.items():
                wave = ml2.data[f"{chip}_{fiber}_WAVE"]
                assert np.all(wave == float(rank)), (
                    f"{chip}_{fiber}_WAVE routed to the wrong W plane"
                )

    def test_wave_is_float64_and_survives_roundtrip(
        self, mock_make_master_l2, tmp_path
    ):
        # Master WLS wavelengths are born-64 (EPRV); must not downcast on disk.
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        assert_dtype(ml2.data["TRACE3_WAVE"], WAVE, "master TRACE3_WAVE")
        ml2.headers["PRIMARY"]["DATE-OBS"] = "2024-01-13T10:26:56"
        assert_roundtrip_dtype(
            KPFMasterL2,
            ml2,
            "TRACE3_WAVE",
            WAVE,
            tmp_path,
            name="KP.20240113.23249.10_master_thar_L2.fits",
        )

    def test_coeffs_extensions_populated(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        for chip in wls.chips:
            ext = f"{chip}_WLS_COEFFS"
            assert ext in ml2.extensions
            coeffs = ml2.data[ext]
            assert coeffs is not None
            assert coeffs.shape == (
                wls.poly_degree_x + 1,
                wls.poly_degree_m + 1,
                wls.poly_degree_f + 1,
            )

    def test_primary_header_keywords(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        primary = ml2.headers["PRIMARY"]
        for key in [
            "ROUGHWLS",
            "LINELIST",
            "LINEPROF",
            "POLYDEGX",
            "POLYDEGM",
            "POLYDEGF",
        ]:
            assert key in primary

    def test_primary_header_keyword_comments_from_registry(self, mock_make_master_l2):
        # WLS metadata routes through set_keyword, so the FITS comments come from
        # the keyword registry, not code-local strings.
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        primary = ml2.headers["PRIMARY"]
        assert primary.comments["MASTYPE"] == "Master calibration type"
        assert primary.comments["POLYDEGX"] == "WLS polynomial degree, pixel axis"

    def test_to_fits_round_trip(self, mock_make_master_l2, tmp_path):
        # Regression: rvdata builds non-PRIMARY headers via fits.Header(dict),
        # which rejects (value, comment) tuple values. Make sure every header
        # we set survives the round-trip.
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        out_path = tmp_path / "KP.20240113.23249.10_master_thar_L2.fits"
        ml2.to_fits(str(out_path))
        assert out_path.exists()

    def test_ml2_datalvl_and_minimal_primary(self, mock_make_master_l2, tmp_path):
        # Regression: ML2 must not inherit RV2's EPRV science PRIMARY skeleton, and
        # DATALVL must be "ML2" both in memory and on disk -- rvdata's to_fits
        # never re-stamps DATALVL.
        from kpfpipe.data_models.masters.level2 import KPFMasterL2

        ml2 = WLS(FILE_LIST).make_master_l2()
        assert ml2.headers["PRIMARY"].get("DATALVL") == "ML2"
        seeded = set(ml2.keyword_registry.eprv_primary_seed)
        assert not (seeded & set(ml2.headers["PRIMARY"])) - {"DATALVL"}

        out_path = tmp_path / "KP.20240113.23249.10_master_thar_L2.fits"
        ml2.to_fits(str(out_path))
        read_back = KPFMasterL2.from_fits(str(out_path))
        assert read_back.headers["PRIMARY"].get("DATALVL") == "ML2"

    def test_poly_degree_override_stamped(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        override_x = wls.poly_degree_x + 4
        ml2 = wls.make_master_l2(poly_degree_x=override_x)
        assert ml2.headers["PRIMARY"].get("POLYDEGX") == override_x
        for chip in wls.chips:
            # Verify the override propagated into the fit, not just the header.
            coeffs = ml2.data[f"{chip}_WLS_COEFFS"]
            assert coeffs.shape[0] == override_x + 1

    def test_input_files_recorded(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        assert ml2.data["INPUT_FILES"]["FILENAME"].tolist() == FILE_LIST

    def test_receipt_entry(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        assert "master_wls" in ml2.receipt["FUNCTION"].tolist()

    def test_resets_l2_cache(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        wls._l2_obj_cache = [MockL2(), MockL2(), MockL2()]
        wls.make_master_l2()
        assert len(wls._l2_obj_cache) == len(FILE_LIST)

    def test_frame_diagnostics_stashed_on_self(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()
        assert wls._frame_diagnostics is not None
        for chip in wls.chips:
            assert chip in wls._frame_diagnostics
            assert len(wls._frame_diagnostics[chip]) == len(FILE_LIST)

    def test_save_diagnostics_before_make_raises(self, tmp_path):
        wls = WLS(FILE_LIST)
        with pytest.raises(RuntimeError, match="run make_master_l2"):
            wls.save_diagnostics(
                str(tmp_path / "KP.20240101.00000.00_master_thar_L2.fits")
            )

    def test_save_diagnostics_with_empty_stash_raises(self, tmp_path):
        # An empty (not None) stash means the chip loop aborted before any chip
        # was added; save_diagnostics must refuse rather than write empty HDF5.
        wls = WLS(FILE_LIST)
        wls._frame_diagnostics = {}
        master_path = tmp_path / "KP.20240101.00000.00_master_thar_L2.fits"
        with pytest.raises(RuntimeError, match="run make_master_l2"):
            wls.save_diagnostics(str(master_path))
        assert not (tmp_path / "thar_L2").exists()

    def test_master_path_writes_diagnostics_hdf5(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": str(tmp_path)})
        master_path, thar_dir = _master_and_stack(tmp_path)
        wls.make_master_l2(master_path=str(master_path))
        h5_path = thar_dir / "KP.20240101.00000.00_master_thar_diagnostics.h5"
        assert h5_path.exists()

    def test_save_diagnostics_post_hoc(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": str(tmp_path)})
        wls.make_master_l2()  # no master_path; diagnostics stashed on self
        master_path, thar_dir = _master_and_stack(tmp_path)
        wls.save_diagnostics(str(master_path))
        h5_path = thar_dir / "KP.20240101.00000.00_master_thar_diagnostics.h5"
        assert h5_path.exists()

    def test_save_master_rejects_unknown_level(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()
        with pytest.raises(ValueError, match="level"):
            wls.save_master(
                "L4", str(tmp_path / "KP.20240113.23249.10_master_thar_L2.fits")
            )

    def test_master_path_writes_fits(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": str(tmp_path)})
        master_path, _ = _master_and_stack(tmp_path)
        wls.make_master_l2(master_path=str(master_path))
        assert master_path.exists()

    def test_save_master_post_hoc(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()  # no master_path; ml2_obj stashed on self
        master_path = tmp_path / "KP.20240113.23249.10_master_thar_L2.fits"
        wls.save_master("L2", str(master_path))
        assert master_path.exists()

    def test_master_path_writes_per_frame_thar_l2(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": str(tmp_path)})
        master_path, thar_dir = _master_and_stack(tmp_path)
        wls.make_master_l2(master_path=str(master_path))

        written = sorted(p.name for p in thar_dir.glob("*_thar_L2.fits"))
        expected = sorted(f"{get_obs_id(fn)}_thar_L2.fits" for fn in FILE_LIST)
        assert written == expected

    def test_no_master_path_writes_no_per_frame_thar_l2(
        self, mock_make_master_l2, tmp_path
    ):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()  # no master_path
        assert not (tmp_path / "thar_L2").exists()

    def test_save_reduced_frames_before_make_raises(self, tmp_path):
        wls = WLS(FILE_LIST)
        with pytest.raises(RuntimeError, match="run make_master_l2"):
            wls.save_reduced_frames(
                str(tmp_path / "KP.20240113.23249.10_master_thar_L2.fits")
            )

    def test_save_reduced_frames_refuses_overwrite(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": str(tmp_path)})
        wls.make_master_l2()  # populates _l2_obj_cache, writes nothing
        master_path = str(_master_and_stack(tmp_path)[0])
        wls.save_reduced_frames(master_path)  # first write
        with pytest.raises(FileExistsError, match="overwrite=True"):
            wls.save_reduced_frames(master_path)

    def test_save_master_overwrite_true_replaces(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()
        master_path = tmp_path / "KP.20240113.23249.10_master_thar_L2.fits"
        master_path.write_bytes(b"stale")
        wls.save_master("L2", str(master_path), overwrite=True)
        assert master_path.read_bytes()[:6] == b"SIMPLE"

    def test_hdf5_structure(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": str(tmp_path)})
        master_path, thar_dir = _master_and_stack(tmp_path)
        wls.make_master_l2(master_path=str(master_path))
        h5_path = thar_dir / "KP.20240101.00000.00_master_thar_diagnostics.h5"

        expected_coeffs_shape = (
            wls.poly_degree_x + 1,
            wls.poly_degree_m + 1,
            wls.poly_degree_f + 1,
        )
        expected_obs_ids = {get_obs_id(fn) for fn in FILE_LIST}

        with h5py.File(h5_path, "r") as h5:
            assert set(h5.keys()) == expected_obs_ids
            for obs_id in expected_obs_ids:
                for chip in wls.chips:
                    grp = h5[obs_id][chip]
                    assert grp.attrs["rejected"] == np.False_

                    assert grp["coeffs"].shape == expected_coeffs_shape
                    assert np.issubdtype(grp["coeffs"].dtype, np.floating)

                    lines = grp["lines"]
                    for key in [
                        "chip",
                        "fiber",
                        "index",
                        "echelle",
                        "wav",
                        "pix",
                        "std",
                        "amp",
                        "isgood",
                    ]:
                        assert key in lines
                    assert np.issubdtype(lines["wav"].dtype, np.floating)
                    assert np.issubdtype(lines["pix"].dtype, np.floating)
                    assert np.issubdtype(lines["index"].dtype, np.integer)
                    assert np.issubdtype(lines["echelle"].dtype, np.integer)
                    assert_dtype(lines["isgood"], MASK_MEM, "isgood")
                    assert h5py.check_string_dtype(lines["chip"].dtype) is not None
                    assert h5py.check_string_dtype(lines["fiber"].dtype) is not None
                    for key in ["wav", "pix", "std", "amp"]:
                        assert np.all(np.isfinite(lines[key][...]))

    def test_rejected_frame_written_with_flag_and_no_coeffs(self, tmp_path):
        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": str(tmp_path)})
        lines = {
            "chip": np.array(["GREEN"]),
            "fiber": np.array(["SCI1"]),
            "index": np.array([0]),
            "echelle": np.array([100]),
            "wav": np.array([5500.0]),
            "pix": np.array([100.0]),
            "std": np.array([0.5]),
            "amp": np.array([1.0]),
            "isgood": np.array([False]),
        }
        wls._frame_diagnostics = {
            "GREEN": [
                {
                    "obs_id": "KP.20240101.00001.00",
                    "lines": lines,
                    "coeffs": np.zeros((2, 2)),
                    "rejected": False,
                },
                {
                    "obs_id": "KP.20240101.00002.00",
                    "lines": lines,
                    "coeffs": None,
                    "rejected": True,
                },
            ]
        }
        master_path, thar_dir = _master_and_stack(tmp_path)
        wls.save_diagnostics(str(master_path))
        h5_path = thar_dir / "KP.20240101.00000.00_master_thar_diagnostics.h5"
        with h5py.File(h5_path, "r") as h5:
            kept = h5["KP.20240101.00001.00"]["GREEN"]
            assert kept.attrs["rejected"] == np.False_
            assert "coeffs" in kept

            rej = h5["KP.20240101.00002.00"]["GREEN"]
            assert rej.attrs["rejected"] == np.True_
            assert "coeffs" not in rej
            assert "lines" in rej  # rejected frame's line diagnostics still written

    def test_nan_orderlet_emits_warning_and_does_not_crash(self, caplog):
        # A NaN-filled orderlet (extraction failure) is skipped rather than
        # crashing scipy's least_squares.
        wls = WLS(FILE_LIST)
        ncol = wls.ccd["ncol"]
        norder = wls.norder["RED"]

        flux = np.ones((norder, ncol))
        flux[0, :] = np.nan

        class StubL2:
            data = {"RED_SCI1_FLUX": flux}

        wls.rough_wls["RED_SCI1_WAVE"] = np.tile(
            np.linspace(6500.0, 6510.0, ncol), (norder, 1)
        )
        wls._linelist_df = _linelist_df("RED", norder, [6502.0, 6505.0, 6508.0])

        with caplog.at_level(logging.DEBUG):
            result = wls._fit_line_positions_ffi(
                StubL2(),
                "RED",
                ["SCI1"],
            )
        assert "RED SCI1 order 102: orderlet skipped" in caplog.text

        # The NaN order (bluest RED, echelle 102) contributed no lines; rest did.
        assert 102 not in result["echelle"]
        assert len(result["wav"]) > 0

    def test_linelist_override(self, mock_make_master_l2, tmp_path, monkeypatch):
        override = tmp_path / "alt_linelist.csv"
        override.write_text(
            "CHIP,INDEX,ECHELLE,WAVE\n"
            "GREEN,0,137,4500.0\nGREEN,1,136,5500.0\nRED,0,102,6500.0\n"
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
        assert ml2.headers["PRIMARY"].get("LINELIST") == str(override)

    def test_single_frame_stack(self, mock_make_master_l2):
        # A 1-frame stack is valid only once the survivor gate is lowered to 1.
        wls = WLS(FILE_LIST[:1])
        wls.min_stack_size = 1
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
        wls = WLS(FILE_LIST, config={"chips": ["GREEN"]})
        ml2 = wls.make_master_l2()
        for fiber in wls.fibers:
            # RED stays at the schema-default zeros: it is not in self.chips.
            assert np.all(ml2.data[f"GREEN_{fiber}_WAVE"] == 5500.0)
            assert np.all(ml2.data[f"RED_{fiber}_WAVE"] == 0.0)
        assert "GREEN_WLS_COEFFS" in ml2.extensions
        assert "RED_WLS_COEFFS" not in ml2.extensions

    def test_info_before_make_master_l2(self, capsys):
        wls = WLS(FILE_LIST)
        wls.info()
        out = capsys.readouterr().out
        assert "WLS" in out
        assert "make_master_l2() has not been called" in out

    def test_info_after_make_master_l2(self, mock_make_master_l2, capsys):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()
        capsys.readouterr()  # discard make_master_l2's own output
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
            ord_.extend(np.linspace(137, 103, n).astype(int))  # GREEN echelle orders
            fib.extend([f] * n)
        return {
            "wav": np.asarray(wav, dtype=float),
            "pix": np.asarray(pix, dtype=float),
            "echelle": np.asarray(ord_, dtype=int),
            "fiber": np.asarray(fib),
            "isgood": np.ones(n * len(fibers), dtype=bool),
        }

    def test_underconstrained_single_fiber_raises(self):
        # default poly_degree is (6, 6, 2): 7*7 = 49 free params single-fiber
        wls = WLS(FILE_LIST)
        lines = self._make_lines(5, fibers=("SCI1",))
        with pytest.raises(ValueError, match=r"underconstrained"):
            wls._calculate_wls_coeffs(lines, wls._echelle_orders["GREEN"])

    def test_underconstrained_multi_fiber_raises(self):
        # 5-fiber: 7*7*3 = 147 free params; 10 lines per fiber * 5 = 50 < 147
        wls = WLS(FILE_LIST)
        lines = self._make_lines(10, fibers=("SKY", "SCI1", "SCI2", "SCI3", "CAL"))
        with pytest.raises(ValueError, match=r"underconstrained"):
            wls._calculate_wls_coeffs(lines, wls._echelle_orders["GREEN"])

    def test_sufficient_lines_does_not_raise(self):
        wls = WLS(FILE_LIST)
        lines = self._make_lines(50, fibers=("SCI1",))
        coeffs = wls._calculate_wls_coeffs(lines, wls._echelle_orders["GREEN"])
        assert coeffs.shape == (wls.poly_degree_x + 1, wls.poly_degree_m + 1)

    def test_mlambda_roundtrip_recovers_wavelength(self):
        wls = WLS(FILE_LIST)
        orders = wls._echelle_orders["GREEN"]
        ncol = wls.ccd["ncol"]

        # The fit models m*lambda as a Legendre surface, so a surface built from
        # known low-degree coefficients must be recovered exactly.
        coeffs_true = np.zeros((wls.poly_degree_x + 1, wls.poly_degree_m + 1))
        coeffs_true[0, 0], coeffs_true[1, 0] = 8.0e5, 300.0  # mean, pixel slope
        coeffs_true[0, 1], coeffs_true[1, 1] = 5.0e4, 5.0  # order slope, cross term
        wave_true = wls._evaluate_wls_coeffs(coeffs_true, orders, nfiber=1)

        cols = np.linspace(0, ncol - 1, 60).astype(int)
        lines = {
            "wav": np.array(
                [wave_true[j, c] for j in range(len(orders)) for c in cols]
            ),
            "pix": np.array([float(c) for _ in orders for c in cols]),
            "echelle": np.array([o for o in orders for _ in cols]),
            "fiber": np.array(["SCI1"] * (len(orders) * len(cols))),
            "isgood": np.ones(len(orders) * len(cols), dtype=bool),
        }
        coeffs_fit = wls._calculate_wls_coeffs(lines, orders)
        wave_fit = wls._evaluate_wls_coeffs(coeffs_fit, orders, nfiber=1)
        np.testing.assert_allclose(wave_fit, wave_true, rtol=1e-9)


# ---------------------------------------------------------------------------
# Stack QC: line-fit QC, coefficient combination, min_stack_size gate
# ---------------------------------------------------------------------------


class TestFitAndQcStack:
    """A frame failing line-fit QC is warned and dropped, never raised.

    Dropped frames are still reported (rejected=True, coeffs=None) so the
    diagnostics stay fully populated.
    """

    def _setup(self, monkeypatch, bad_fracs, nlines=100):
        """Build a WLS whose stack yields one fake `lines` dict per bad_fracs entry."""
        wls = WLS(FILE_LIST)
        wls._l2_obj_cache = [MockL2() for _ in bad_fracs]

        frames = []
        for frac in bad_fracs:
            isgood = np.ones(nlines, dtype=bool)
            isgood[: int(round(frac * nlines))] = False
            frames.append({"wav": np.zeros(nlines), "isgood": isgood})
        it = iter(frames)

        monkeypatch.setattr(
            WLS, "_fit_line_positions_ffi", lambda self, *a, **k: next(it)
        )
        monkeypatch.setattr(
            WLS, "_calculate_wls_coeffs", lambda self, *a, **k: np.ones((2, 2))
        )
        return wls

    def test_all_clean_frames_kept(self, monkeypatch):
        wls = self._setup(monkeypatch, [0.01, 0.02, 0.0, 0.03, 0.01])
        frames = wls._fit_and_qc_lines_stack("GREEN", ["SCI1"])
        assert len(frames) == 5
        assert [fr["rejected"] for fr in frames] == [False] * 5
        assert all(fr["coeffs"] is not None for fr in frames)

    def test_single_bad_frame_dropped(self, caplog, monkeypatch):
        wls = self._setup(monkeypatch, [0.01, 0.22, 0.0, 0.03, 0.01])
        with caplog.at_level(logging.WARNING):
            frames = wls._fit_and_qc_lines_stack("GREEN", ["SCI1"])
        assert "line fits failed QC" in caplog.text
        # The 22%-bad frame is rejected but still reported alongside the rest.
        assert len(frames) == 5
        assert [fr["rejected"] for fr in frames] == [False, True, False, False, False]
        assert frames[1]["coeffs"] is None

    def test_many_bad_frames_all_dropped_no_raise(self, caplog, monkeypatch):
        # More than one over-threshold frame does not abort the build.
        wls = self._setup(monkeypatch, [0.22, 0.01, 0.34, 0.01, 0.01])
        with caplog.at_level(logging.WARNING):
            frames = wls._fit_and_qc_lines_stack("GREEN", ["SCI1"])
        assert "line fits failed QC" in caplog.text
        assert [fr["rejected"] for fr in frames] == [True, False, True, False, False]
        assert sum(not fr["rejected"] for fr in frames) == 3

    def test_threshold_is_inclusive_at_max_bad_frac(self, monkeypatch):
        # Exactly max_bad_frac (5%) is not > 5%, so the frame is kept.
        wls = self._setup(monkeypatch, [0.05, 0.01], nlines=100)
        frames = wls._fit_and_qc_lines_stack("GREEN", ["SCI1"])
        assert [fr["rejected"] for fr in frames] == [False, False]

    def test_nonfinite_coeffs_frame_rejected_not_raised(self, caplog, monkeypatch):
        # A NaN coefficient rejects that frame, not the whole build.
        wls = WLS(FILE_LIST)
        wls._l2_obj_cache = [MockL2() for _ in range(3)]
        lines = {"wav": np.zeros(100), "isgood": np.ones(100, dtype=bool)}
        coeffs = iter(
            [np.ones((2, 2)), np.array([[1.0, np.nan], [1.0, 1.0]]), np.ones((2, 2))]
        )
        monkeypatch.setattr(WLS, "_fit_line_positions_ffi", lambda self, *a, **k: lines)
        monkeypatch.setattr(
            WLS, "_calculate_wls_coeffs", lambda self, *a, **k: next(coeffs)
        )
        with caplog.at_level(logging.WARNING):
            frames = wls._fit_and_qc_lines_stack("GREEN", ["SCI1"])
        assert "non-finite Legendre" in caplog.text
        assert [fr["rejected"] for fr in frames] == [False, True, False]
        assert frames[1]["coeffs"] is None

    def test_underconstrained_frame_rejected_not_raised(self, caplog, monkeypatch):
        # A per-frame fit that raises rejects that frame, not the whole build.
        wls = WLS(FILE_LIST)
        wls._l2_obj_cache = [MockL2() for _ in range(3)]
        lines = {"wav": np.zeros(100), "isgood": np.ones(100, dtype=bool)}
        results = iter([np.ones((2, 2)), "raise", np.ones((2, 2))])

        def calc(self, *a, **k):
            result = next(results)
            if isinstance(result, str):
                raise ValueError("WLS fit underconstrained: 2 good lines < 28")
            return result

        monkeypatch.setattr(WLS, "_fit_line_positions_ffi", lambda self, *a, **k: lines)
        monkeypatch.setattr(WLS, "_calculate_wls_coeffs", calc)
        with caplog.at_level(logging.WARNING):
            frames = wls._fit_and_qc_lines_stack("GREEN", ["SCI1"])
        assert "WLS fit failed" in caplog.text
        assert [fr["rejected"] for fr in frames] == [False, True, False]
        assert frames[1]["coeffs"] is None


class TestCombineCoeffs:
    """_combine_coeffs_stack averages the surviving per-frame coeffs."""

    def _frames(self, coeffs_list):
        return [{"coeffs": c, "rejected": False} for c in coeffs_list]

    def test_averages_surviving_coeffs(self, monkeypatch):
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(
            WLS, "_evaluate_wls_coeffs", lambda self, coeffs, *a, **k: coeffs
        )
        frames = self._frames([np.full((2, 2), 4.0), np.full((2, 2), 6.0)])
        W, coeffs_mean = wls._combine_coeffs_stack(frames, "GREEN", 1)
        np.testing.assert_allclose(coeffs_mean, 5.0)

    def test_rejected_frames_excluded_from_average(self, monkeypatch):
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(
            WLS, "_evaluate_wls_coeffs", lambda self, coeffs, *a, **k: coeffs
        )
        frames = [
            {"coeffs": np.full((2, 2), 5.0), "rejected": False},
            {"coeffs": None, "rejected": True},
            {"coeffs": np.full((2, 2), 5.0), "rejected": False},
        ]
        _, coeffs_mean = wls._combine_coeffs_stack(frames, "GREEN", 1)
        np.testing.assert_allclose(coeffs_mean, 5.0)

    def test_all_outliers_raises(self):
        # With no frame surviving for some coefficient the stack is uncombinable,
        # so it raises rather than producing a spurious master. The default
        # qc_sigma keeps at least half the frames, hence qc_sigma=0 here.
        wls = WLS(FILE_LIST)
        frames = self._frames([np.full((2, 2), 0.0), np.full((2, 2), 10.0)])
        with pytest.raises(ValueError, match=r"all frames rejected as outliers"):
            wls._combine_coeffs_stack(frames, "GREEN", 1, qc_sigma=0.0)


class TestMinStackSizeGate:
    """make_master_l2 gates the master on min_stack_size survivors per chip.

    Below the threshold it raises, but the per-frame diagnostics and L2s are
    still written.
    """

    def _mock(self, monkeypatch, n_survivors, n_frames=8, masters_output=None):
        # n_survivors is an int (same for every chip) or a {chip: count} dict, so
        # a test can make GREEN and RED pass/fail the gate independently.
        files = sorted(f"KP.20240101.{i:05d}.00.fits" for i in range(n_frames))
        _stub_frame_pipeline(monkeypatch)

        def mock_fit(self, chip, fibers, **kwargs):
            n = n_survivors[chip] if isinstance(n_survivors, dict) else n_survivors
            frames = []
            for j, l2 in enumerate(self._l2_obj_cache):
                rejected = j >= n
                coeffs = (
                    None
                    if rejected
                    else np.zeros(
                        (
                            self.poly_degree_x + 1,
                            self.poly_degree_m + 1,
                            self.poly_degree_f + 1,
                        )
                    )
                )
                frames.append(
                    {
                        "obs_id": l2.obs_id,
                        "lines": {
                            "wav": np.array([5500.0]),
                            "isgood": np.array([not rejected]),
                        },
                        "coeffs": coeffs,
                        "rejected": rejected,
                    }
                )
            return frames

        def mock_combine(self, frames, chip, nfibers, **kwargs):
            norder = NORDER_GREEN if chip == "GREEN" else NORDER_RED
            coeffs = np.zeros(
                (self.poly_degree_x + 1, self.poly_degree_m + 1, self.poly_degree_f + 1)
            )
            return np.full((norder, NCOL_TEST, nfibers), 5500.0), coeffs

        monkeypatch.setattr(WLS, "_fit_and_qc_lines_stack", mock_fit)
        monkeypatch.setattr(WLS, "_combine_coeffs_stack", mock_combine)
        config = {"KPF_MASTERS_OUTPUT": str(masters_output)} if masters_output else None
        return WLS(files, config=config)

    def test_below_min_raises_but_diagnostics_written(self, monkeypatch, tmp_path):
        wls = self._mock(monkeypatch, n_survivors=2, masters_output=tmp_path)
        wls.min_stack_size = 5
        master_path, thar_dir = _master_and_stack(tmp_path)
        with pytest.raises(ValueError, match=r"passed line-fit QC"):
            wls.make_master_l2(master_path=str(master_path))

        # diagnostics + per-frame L2s persist despite the abort ...
        assert (thar_dir / "KP.20240101.00000.00_master_thar_diagnostics.h5").exists()
        assert list(thar_dir.glob("*_thar_L2.fits"))
        # ... but the master itself is withheld.
        assert not master_path.exists()

    def test_at_min_writes_master(self, monkeypatch, tmp_path):
        wls = self._mock(monkeypatch, n_survivors=5, masters_output=tmp_path)
        wls.min_stack_size = 5
        master_path, _ = _master_and_stack(tmp_path)
        ml2 = wls.make_master_l2(master_path=str(master_path))
        assert isinstance(ml2, KPFMasterL2)
        assert master_path.exists()

    def test_gate_is_per_chip(self, monkeypatch, tmp_path):
        # GREEN clears the gate but RED does not: the master is withheld even
        # though one chip passed. Diagnostics are saved before the gate, so they
        # still persist.
        wls = self._mock(monkeypatch, {"GREEN": 5, "RED": 2}, masters_output=tmp_path)
        wls.min_stack_size = 5
        master_path, thar_dir = _master_and_stack(tmp_path)
        with pytest.raises(ValueError, match=r"RED: only 2 frame"):
            wls.make_master_l2(master_path=str(master_path))
        assert not master_path.exists()
        assert (thar_dir / "KP.20240101.00000.00_master_thar_diagnostics.h5").exists()


# ---------------------------------------------------------------------------
# TestFitLinePositions
# ---------------------------------------------------------------------------


class TestFitLinePositions:
    def test_no_lines_returns_empty(self):
        wls = WLS(FILE_LIST)
        flux = np.ones(100)
        wave = np.linspace(5000.0, 5100.0, 100)

        result = wls._fit_line_positions_1d(flux, wave, np.array([]))
        for key in ["wav", "pix", "std", "amp", "isgood"]:
            assert len(result[key]) == 0

    def test_line_outside_rough_wls_range_raises(self):
        # A line outside the order's rough WLS span means the CHIP/ORDER labels
        # are inconsistent, which must fail loudly rather than fit garbage.
        wls = WLS(FILE_LIST)
        flux = np.ones(100)
        wave = np.linspace(5000.0, 5100.0, 100)
        with pytest.raises(ValueError, match="outside the rough"):
            wls._fit_line_positions_1d(flux, wave, np.array([6000.0]))

    def test_line_fit_qc_flags_centroid_outside_window(self):
        # Centroid distance from the window center is an independent QC axis:
        # amplitude and width are in range for all three lines here.
        wls = WLS(FILE_LIST)
        lines = {
            "amp": np.array([1.0, 1.0, 1.0]),
            "std": np.array([1.0, 1.0, 1.0]),
            "pix": np.array([100.0, 103.0, 120.0]),  # offsets 0, 3, 20
        }
        loc = np.full(3, 100.0)
        isgood = wls._line_fit_qc(lines, "gaussian", window=5, loc=loc)
        assert list(isgood) == [True, True, False]

    def test_fit_returns_float64_from_float32_inputs(self):
        # float32 inputs must not drag the line fit down to float32.
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
            assert_dtype(result[key], WAVE, f"_fit_line_positions_1d {key}")

    def test_all_nan_fiber_emits_fiber_level_warning(self, caplog):
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

        # Capture at WARNING to isolate the fiber-level message from the
        # per-order skips, which log at DEBUG.
        with caplog.at_level(logging.WARNING):
            result = wls._fit_line_positions_ffi(
                StubL2(),
                "RED",
                ["SCI1"],
            )
        assert "RED SCI1: no good lines retained" in caplog.text
        assert len(result["wav"]) == 0

    def test_nan_orderlet_emits_skip_warning(self, caplog):
        wls = WLS(FILE_LIST)
        ncol = wls.ccd["ncol"]
        norder = wls.norder["RED"]

        flux = np.ones((norder, ncol))
        flux[0, :] = np.nan

        class StubL2:
            data = {"RED_SCI1_FLUX": flux}

        wls.rough_wls["RED_SCI1_WAVE"] = np.tile(
            np.linspace(6500.0, 6510.0, ncol), (norder, 1)
        )
        wls._linelist_df = _linelist_df("RED", norder, [6502.0, 6505.0, 6508.0])

        with caplog.at_level(logging.DEBUG):
            wls._fit_line_positions_ffi(StubL2(), "RED", ["SCI1"])

        assert "orderlet skipped" in caplog.text


# ---------------------------------------------------------------------------
# TestRoughWlsLoading
# ---------------------------------------------------------------------------


class TestRoughWlsLoading:
    def test_missing_rough_wls_file_raises(self):
        wls = WLS(FILE_LIST)
        with pytest.raises(FileNotFoundError):
            wls._load_rough_wls(rough_wls_file="/nonexistent/path.csv")
