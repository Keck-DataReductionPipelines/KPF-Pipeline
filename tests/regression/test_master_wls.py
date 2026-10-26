"""Unit tests for the WLS module.

All sub-module I/O is mocked; no real data or FITS files are required.
"""

import logging
import os
from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest

import kpfpipe.modules.masters.base as base_module
import kpfpipe.modules.masters.wls as wls_module
from kpfpipe import DETECTOR, FIBERS, REPO_ROOT
from kpfpipe.data_models.masters import KPFMasterL2
from kpfpipe.modules.masters.wls import WLS
from kpfpipe.utils.kpf import get_obs_id, select_era_reference

from ._dtype_policy import (
    MASK_MEM,
    WAVE,
    WLS_COEFFS,
    assert_dtype,
    assert_roundtrip_dtype,
)
from ._masters import FILE_LIST

# Captured before the memoizing fixture below patches the method.
_REAL_LOAD_ROUGH_WLS = WLS._load_rough_wls

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
NCOL_TEST = 16

DATECODE = "20240101"  # shared by FILE_LIST and the master obs_ids below
# The rough WLS of DATECODE's era (1.0), served to every frame unless a test
# drives the real per-frame selection.
ROUGH_WLS_FILE = f"{REPO_ROOT}/reference/rough_wls/rough_wls_20221207.csv"


def _fake_lines(chip, fiber, echelle):
    """A per-line fit dict with every _LINE_FIT_COLUMNS key, one row per order
    in ``echelle``, all fit and centered on PIX_GUESS."""
    n = len(echelle)
    lines = {name: np.full(n, 0.5) for name in wls_module._LINE_FIT_COLUMNS}
    lines.update(
        CHIP=np.full(n, chip),
        FIBER=np.full(n, fiber),
        ORDER_INDEX=np.arange(n),
        ECHELLE_ORDER=np.asarray(echelle, dtype=int),
        WAVE=5500.0 + np.arange(n, dtype=float),
        PIX_GUESS=100 * (np.arange(n) + 1),
        NPIX=np.full(n, 11),
        FIT_MEAN=100.0 * (np.arange(n) + 1),
        CONVERGED=np.ones(n, dtype=bool),
        FINITE=np.ones(n, dtype=bool),
    )
    return lines


@pytest.fixture(scope="module")
def cached_rough_wls():
    """The rough-WLS grid of ROUGH_WLS_FILE, built once for the whole module."""
    wls = WLS(FILE_LIST)
    _REAL_LOAD_ROUGH_WLS(wls, ROUGH_WLS_FILE)
    return wls.rough_wls


@pytest.fixture(autouse=True)
def _memoize_rough_wls(cached_rough_wls, monkeypatch):
    """Serve every frame in this file ROUGH_WLS_FILE's grid, built once above.

    The test frames carry no headers, so the per-frame era selection returns
    ROUGH_WLS_FILE, and loading it -- measured at 0.15 s a call, in a file that
    processes roughly sixty stacks -- hands each instance a fresh dict wrapping
    the shared arrays; the tests that override a channel rebind the key rather
    than writing through it.

    Any other path still reaches the real loader, so the missing-file path is
    unaffected, and ``test_cached_grid_matches_the_real_loader`` drives the real
    loader once per run to keep this cache honest.
    """

    def _cached(self, rough_wls_file=None):
        if rough_wls_file not in (None, ROUGH_WLS_FILE):
            return _REAL_LOAD_ROUGH_WLS(self, rough_wls_file)
        if self.rough_wls_file != ROUGH_WLS_FILE:
            self.rough_wls_file = ROUGH_WLS_FILE
            self.rough_wls = dict(cached_rough_wls)
        return self.rough_wls

    monkeypatch.setattr(WLS, "_load_rough_wls", _cached)
    monkeypatch.setattr(
        wls_module, "select_era_reference", lambda *args, **kwargs: ROUGH_WLS_FILE
    )


@pytest.fixture(autouse=True)
def frame_qc(monkeypatch):
    """Stub the per-frame DiagL1/QCL1 and DiagL2/QCL2 runs, which need real
    frames; returns the stubs so a test can check what they were run on."""
    stubs = {}
    for module, name in (
        (base_module, "DiagL1"),
        (base_module, "QCL1"),
        (wls_module, "DiagL2"),
        (wls_module, "QCL2"),
    ):
        stubs[name] = MagicMock()
        monkeypatch.setattr(module, name, stubs[name])
    return stubs


class MockL1:
    headers = {"PRIMARY": {}}  # rough-WLS selection is stubbed (_memoize_rough_wls)

    def __init__(self, obs_id=None):
        self.obs_id = obs_id


class MockL2:
    level = 2

    def __init__(self, obs_id=None):
        self.obs_id = obs_id
        self.data = {}

    def set_data(self, ext, value):
        self.data[ext] = value

    def to_fits(self, path):
        # mirrors the real to_fits, which creates the parent directory
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()


def _linelist_df(chip, norder, waves):
    """Stub line list: `waves` repeated for every 0-based order index."""
    echelle = np.linspace(*DETECTOR["echelle_orders"][chip], norder).round().astype(int)
    return pd.DataFrame(
        [(chip, o, int(echelle[o]), w) for o in range(norder) for w in waves],
        columns=["CHIP", "INDEX", "ECHELLE", "WAVE"],
    )


def _master_and_stack(masters_output, obs_id="KP.20240101.00000.00"):
    """The (night_dir, master_path, stack_dir) triple for a ThAr master keyed on
    `obs_id`.

    The night dir is what WLS is given as ``output_dir``; the stack subdir,
    beside the master, is where WLS writes its per-frame L2s and diagnostics.
    """
    night = masters_output / "masters" / DATECODE
    return night, night / f"{obs_id}_master_thar_L2.fits", night / "thar_L2"


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


def _stub_frame_pipeline(monkeypatch, l1=None, extract=None):
    """Short-circuit load -> process -> extract so a test drives only the fit.

    ``_load_frame`` returns a MockL1 carrying each source file's obs_id unless
    ``l1`` overrides it, so the extracted L2 stub carries that frame's
    obs_id (which drives per-frame naming). The fit/combine stubs stay per-test:
    their survivor and rejection structure IS what each test asserts.
    """
    load = (
        (lambda self, fn, cache=False, **kw: MockL1(get_obs_id(fn)))
        if l1 is None
        else l1
    )
    extract = extract or (lambda self, l1_obj, **kw: MockL2(l1_obj.obs_id))
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
        with pytest.raises(
            TypeError, match="config must be None, dict, or ConfigHandler"
        ):
            WLS(FILE_LIST, config=42)


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


class TestProcessIndividualFrames:
    def test_returns_l2_objects_for_all_frames(self, mock_pipeline, monkeypatch):
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(
            wls, "_load_frame", lambda fn, cache=False, **kwargs: MockL1()
        )
        result = wls._process_stack_l0_to_l2(FILE_LIST)
        assert len(result) == len(FILE_LIST)
        assert all(r is mock_pipeline for r in result)

    def test_runs_l2_qc_with_rough_wave(self, mock_pipeline, monkeypatch, frame_qc):
        # Extraction writes no WAVE, so the rough WLS stands in before DiagL2
        # and QCL2 run on the frame.
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(
            wls, "_load_frame", lambda fn, cache=False, **kwargs: MockL1()
        )
        wls._process_stack_l0_to_l2(FILE_LIST)
        assert mock_pipeline.data.keys() == wls.rough_wls.keys()
        for ext, wave in wls.rough_wls.items():
            assert mock_pipeline.data[ext] is wave
        for name in ("DiagL2", "QCL2"):
            frame_qc[name].assert_called_with(mock_pipeline)
            assert frame_qc[name].call_count == len(FILE_LIST)

    @pytest.mark.parametrize(
        "date_obs, instera, datecode",
        [
            ("2023-06-01T05:00:00", "1.0", "20221207"),
            ("2024-06-01T05:00:00", "2.0", "20240224"),
        ],
    )
    def test_selects_the_rough_wls_of_the_frames_era(
        self, monkeypatch, date_obs, instera, datecode
    ):
        # Chosen like the order trace: the latest reference within the frame's era.
        monkeypatch.setattr(wls_module, "select_era_reference", select_era_reference)
        l1 = MockL1()
        l1.headers = {
            "PRIMARY": {
                "JD_UTC": pd.Timestamp(date_obs).to_julian_date(),
                "INSTERA": instera,
            }
        }
        _stub_frame_pipeline(
            monkeypatch,
            l1=lambda self, fn, **kw: l1,
            extract=lambda self, l1_obj, **kw: MockL2(),
        )
        wls = WLS(FILE_LIST)
        wls._process_stack_l0_to_l2(FILE_LIST)
        assert wls.rough_wls_file == (
            f"{REPO_ROOT}/reference/rough_wls/rough_wls_{datecode}.csv"
        )

    def test_file_list_override(self, mock_pipeline, monkeypatch):
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(
            wls, "_load_frame", lambda fn, cache=False, **kwargs: MockL1()
        )
        wls.min_stack_size = 3
        result = wls._process_stack_l0_to_l2(FILE_LIST[:3])
        assert len(result) == 3

    def test_raises_when_failures_exceed_threshold(self, monkeypatch):
        wls = WLS(FILE_LIST)
        attempted = []

        def load(fn, cache=False, **kwargs):
            attempted.append(fn)

        monkeypatch.setattr(wls, "_load_frame", load)
        with pytest.raises(ValueError, match="too many frames failed to load/process"):
            wls._process_stack_l0_to_l2(FILE_LIST)
        # the limit is applied once, after every frame has been attempted
        assert attempted == FILE_LIST

    def test_tolerates_minority_failure(self, mock_pipeline, monkeypatch):
        # 1 failure out of 8 = 12.5%, below the 20% threshold
        calls = iter([None] + [MockL1()] * 7)
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(
            wls, "_load_frame", lambda fn, cache=False, **kwargs: next(calls)
        )
        result = wls._process_stack_l0_to_l2(FILE_LIST)
        assert len(result) == 7

    def test_extract_failure_drops_frame_and_continues(
        self, mock_pipeline, monkeypatch, tmp_path
    ):
        # The failed frame is skipped; every other frame's L2 is still written.
        mock_se = MagicMock()
        mock_se.return_value.perform.side_effect = [LookupError("orderlets")] + [
            MockL2(get_obs_id(fn)) for fn in FILE_LIST[1:]
        ]
        monkeypatch.setattr(base_module, "SpectralExtraction", mock_se)
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(
            wls, "_load_frame", lambda fn, cache=False, **kwargs: MockL1()
        )
        result = wls._process_stack_l0_to_l2(FILE_LIST, output_dir=tmp_path)
        assert len(result) == 7
        assert wls._stacked_files == FILE_LIST[1:]
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
            f"{get_obs_id(fn)}_thar_L2.fits" for fn in FILE_LIST[1:]
        )

    def test_write_failure_raises(self, monkeypatch, tmp_path):
        _stub_frame_pipeline(monkeypatch)

        def write(self, obj, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(WLS, "_write_frame", write)
        with pytest.raises(OSError, match="disk full"):
            WLS(FILE_LIST)._process_stack_l0_to_l2(FILE_LIST, output_dir=tmp_path)


@pytest.fixture
def mock_make_master_l2(monkeypatch):
    """Run make_master_l2 without touching disk or real spectra.

    The fitting stub stashes and returns one frame per input (FILE_LIST has 8,
    so the default min_stack_size=5 gate passes); the combine stub returns
    synthetic W and coefficient arrays with chip-correct shapes.
    """
    _stub_frame_pipeline(monkeypatch)

    def mock_fit_lines(
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

        frames = [
            {
                "obs_id": l2_obj.obs_id,
                "lines": _fake_lines(chip, fibers[0], self._echelle_orders[chip][:2]),
                "coeffs": coeffs,
                "window": 5,
                "lineprofile": "gaussian",
            }
            for l2_obj in self._l2_obj_cache
        ]
        self._frame_diagnostics[chip] = frames
        return frames

    def mock_combine(self, frames, chip, nfibers, **kwargs):
        norder = NORDER_GREEN if chip == "GREEN" else NORDER_RED
        coeffs = frames[0]["coeffs"]
        if nfibers == 1:
            W = np.full((norder, NCOL_TEST), 5500.0)
        else:
            W = np.full((norder, NCOL_TEST, nfibers), 5500.0)
        return W, coeffs

    monkeypatch.setattr(WLS, "_fit_lines_stack", mock_fit_lines)
    monkeypatch.setattr(WLS, "_combine_coeffs_stack", mock_combine)


class TestMakeMasterL2:
    def test_returns_kpf_master_l2(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        result = wls.make_master_l2()
        assert isinstance(result, KPFMasterL2)

    def test_wave_routing_uses_physical_position_not_fiber_order(self, monkeypatch):
        # W's planes are ordered by fiber_positions (SKY=0..CAL=4), so the write
        # loop must sort by that, not trust self.fibers' config-overridable
        # order -- else SKY's solution lands on CAL and vice versa.
        _stub_frame_pipeline(
            monkeypatch,
            l1=lambda self, fn, cache=False, **kw: MockL1(),
            extract=lambda self, l1_obj, **kw: MockL2(),
        )

        def mock_fit_lines(self, chip, fibers, **kwargs):
            coeffs = np.zeros((self.poly_degree_x + 1, self.poly_degree_m + 1, 1))
            frames = [
                {
                    "obs_id": None,
                    "lines": _fake_lines(chip, fibers[0], [137]),
                    "coeffs": coeffs,
                }
                for _ in self._l2_obj_cache
            ]
            self._frame_diagnostics[chip] = frames
            return frames

        def mock_combine(self, frames, chip, nfibers, **kwargs):
            # Plane i carries the constant value i (its physical-position rank).
            norder = NORDER_GREEN if chip == "GREEN" else NORDER_RED
            W = np.empty((norder, NCOL_TEST, nfibers))
            for i in range(nfibers):
                W[:, :, i] = float(i)
            coeffs = np.zeros((self.poly_degree_x + 1, self.poly_degree_m + 1, 1))
            return W, coeffs

        monkeypatch.setattr(WLS, "_fit_lines_stack", mock_fit_lines)
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

    def test_coeffs_extensions_exist_before_any_fit(self):
        # The manifest builds both chips' coefficient extensions at
        # construction; make_master_l2 only fills them.
        bare = KPFMasterL2(kind="wls")
        assert "GREEN_WLS_COEFFS" in bare.extensions
        assert "RED_WLS_COEFFS" in bare.extensions

    def test_coeffs_extensions_populated(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        for chip in wls.chips:
            ext = f"{chip}_WLS_COEFFS"
            assert ext in ml2.extensions
            coeffs = ml2.data[ext]
            assert coeffs is not None
            assert_dtype(coeffs, WLS_COEFFS, ext)
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
        # which rejects (value, comment) tuple values. Every header this module
        # sets must come back off disk, not merely survive the write.
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        out_path = tmp_path / "KP.20240113.23249.10_master_thar_L2.fits"
        ml2.to_fits(str(out_path))

        read_back = KPFMasterL2.from_fits(str(out_path))
        primary = read_back.headers["PRIMARY"]
        assert primary["MASTYPE"] == "thar"
        assert primary["ROUGHWLS"] == wls.rough_wls_file
        assert primary["LINELIST"] == wls.linelist
        assert primary["LINEPROF"] == wls.lineprofile
        assert primary["POLYDEGX"] == wls.poly_degree_x
        assert primary["POLYDEGM"] == wls.poly_degree_m
        assert primary["POLYDEGF"] == wls.poly_degree_f

    def test_ml2_datalvl_and_master_primary(self, mock_make_master_l2, tmp_path):
        # Regression: ML2 seeds its own ML2-wls PRIMARY set, not the EPRV L2
        # science skeleton, and DATALVL must be "ML2" both in memory and on disk
        # -- rvdata's to_fits never re-stamps DATALVL.
        from kpfpipe.data_models.masters.level2 import KPFMasterL2

        ml2 = WLS(FILE_LIST).make_master_l2()
        assert ml2.headers["PRIMARY"].get("DATALVL") == "ML2"
        primary = set(ml2.headers["PRIMARY"])
        assert set(ml2.keyword_registry.primary_seed("ML2-wls")) <= primary
        science = set(ml2.keyword_registry.primary_seed("L2"))
        # DATALVL and DRPSTATU are stamped, not seeded: every product carries
        # them, masters included.
        assert not (science & primary) - {"DATALVL", "DRPSTATU"}

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

    def test_save_diagnostics_before_make_raises(self):
        wls = WLS(FILE_LIST)
        with pytest.raises(RuntimeError, match="run make_master_l2"):
            wls.save_diagnostics("/unused")

    def test_save_diagnostics_with_empty_stash_raises(self, tmp_path):
        # An empty (not None) stash means the chip loop aborted before any chip
        # was added; save_diagnostics must refuse rather than write empty HDF5.
        wls = WLS(FILE_LIST)
        wls._frame_diagnostics = {}
        with pytest.raises(RuntimeError, match="run make_master_l2"):
            wls.save_diagnostics(str(tmp_path / "thar_L2"))
        assert not (tmp_path / "thar_L2").exists()

    def test_output_dir_writes_diagnostics_hdf5(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": str(tmp_path)})
        night, _, thar_dir = _master_and_stack(tmp_path)
        wls.make_master_l2(output_dir=str(night))
        h5_path = thar_dir / "KP.20240101.00000.00_thar_l2_diagnostics.h5"
        assert h5_path.exists()

    def test_save_diagnostics_post_hoc(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": str(tmp_path)})
        wls.make_master_l2()  # no output_dir; diagnostics stashed on self
        _, _, thar_dir = _master_and_stack(tmp_path)
        wls.save_diagnostics(str(thar_dir))
        h5_path = thar_dir / "KP.20240101.00000.00_thar_l2_diagnostics.h5"
        assert h5_path.exists()

    def test_save_master_rejects_unknown_level(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()
        with pytest.raises(ValueError, match="level"):
            wls.save_master(
                "L4", str(tmp_path / "KP.20240113.23249.10_master_thar_L2.fits")
            )

    def test_output_dir_writes_the_standard_name(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": str(tmp_path)})
        night, master_path, _ = _master_and_stack(tmp_path)
        wls.make_master_l2(output_dir=str(night))
        assert master_path.exists()

    def test_file_list_override_replaces_l0_file_list(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        wls.make_master_l2(l0_file_list=FILE_LIST[:6])
        assert wls.l0_file_list == FILE_LIST[:6]
        assert len(wls._l2_obj_cache) == 6

    def test_save_master_post_hoc(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()  # no output_dir; ml2_obj stashed on self
        master_path = tmp_path / "KP.20240113.23249.10_master_thar_L2.fits"
        wls.save_master("L2", str(master_path))
        assert master_path.exists()

    def test_output_dir_writes_per_frame_thar_l2(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": str(tmp_path)})
        night, _, thar_dir = _master_and_stack(tmp_path)
        wls.make_master_l2(output_dir=str(night))

        written = sorted(p.name for p in thar_dir.glob("*_thar_L2.fits"))
        expected = sorted(f"{get_obs_id(fn)}_thar_L2.fits" for fn in FILE_LIST)
        assert written == expected

    def test_no_output_dir_writes_no_per_frame_thar_l2(
        self, mock_make_master_l2, tmp_path
    ):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()  # no output_dir
        assert not (tmp_path / "thar_L2").exists()

    def test_per_frame_thar_l2_written_before_line_fitting(
        self, mock_make_master_l2, monkeypatch, tmp_path
    ):
        def fit_raises(self, *args, **kwargs):
            raise ValueError("line fitting failed")

        monkeypatch.setattr(WLS, "_fit_lines_stack", fit_raises)
        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": str(tmp_path)})
        night, _, thar_dir = _master_and_stack(tmp_path)
        with pytest.raises(ValueError, match="line fitting failed"):
            wls.make_master_l2(output_dir=str(night))
        assert len(list(thar_dir.glob("*_thar_L2.fits"))) == len(FILE_LIST)

    def test_save_master_overwrite_true_replaces(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST)
        wls.make_master_l2()
        master_path = tmp_path / "KP.20240113.23249.10_master_thar_L2.fits"
        master_path.write_bytes(b"stale")
        wls.save_master("L2", str(master_path), overwrite=True)
        assert master_path.read_bytes()[:6] == b"SIMPLE"

    def test_hdf5_structure(self, mock_make_master_l2, tmp_path):
        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": str(tmp_path)})
        night, _, thar_dir = _master_and_stack(tmp_path)
        wls.make_master_l2(output_dir=str(night))
        h5_path = thar_dir / "KP.20240101.00000.00_thar_l2_diagnostics.h5"

        expected_coeffs_shape = (
            wls.poly_degree_x + 1,
            wls.poly_degree_m + 1,
            wls.poly_degree_f + 1,
        )
        expected_obs_ids = {get_obs_id(fn) for fn in FILE_LIST}

        with h5py.File(h5_path, "r") as h5:
            assert set(h5.keys()) == expected_obs_ids
            assert set(h5.attrs) == {"LINEPROF", "FITWIN", "LINELIST", "ROUGHWLS"}
            for obs_id in expected_obs_ids:
                assert set(h5[obs_id]) == {
                    f"{chip}_{node}"
                    for chip in wls.chips
                    for node in ("LINES", "WLS_COEFFS")
                }
                for chip in wls.chips:
                    coeffs = h5[obs_id][f"{chip}_WLS_COEFFS"]
                    assert coeffs.shape == expected_coeffs_shape
                    assert_dtype(coeffs[...], WAVE, "h5 coeffs")

                    lines = h5[obs_id][f"{chip}_LINES"]
                    assert set(lines) == set(wls_module._LINE_FIT_COLUMNS)
                    assert len({lines[k].shape for k in lines}) == 1
                    for name, (units, _) in wls_module._LINE_FIT_COLUMNS.items():
                        assert lines[name].attrs["units"] == units
                    assert_dtype(lines["WAVE"][...], WAVE, "h5 WAVE")
                    assert_dtype(lines["FIT_MEAN"][...], WAVE, "h5 FIT_MEAN")
                    assert np.issubdtype(lines["ORDER_INDEX"].dtype, np.integer)
                    assert np.issubdtype(lines["ECHELLE_ORDER"].dtype, np.integer)
                    assert_dtype(lines["FINITE"], MASK_MEM, "FINITE")
                    assert h5py.check_string_dtype(lines["CHIP"].dtype) is not None
                    assert h5py.check_string_dtype(lines["FIBER"].dtype) is not None

    def test_wls_fit_failure_still_writes_hdf5(
        self, mock_make_master_l2, monkeypatch, tmp_path
    ):
        # The first frame's WLS fit succeeds and the second's raises: every
        # frame's lines are written, coeffs only for the frame that was fit.
        def fit_then_raise(self, chip, fibers, **kwargs):
            frames = [
                {
                    "obs_id": l2_obj.obs_id,
                    "lines": _fake_lines(chip, fibers[0], [100, 101]),
                    "coeffs": np.zeros((2, 2)) if i == 0 else None,
                    "window": 5,
                    "lineprofile": "gaussian",
                }
                for i, l2_obj in enumerate(self._l2_obj_cache)
            ]
            self._frame_diagnostics[chip] = frames
            raise ValueError(f"{chip} frame {frames[1]['obs_id']}: WLS fit failed")

        monkeypatch.setattr(WLS, "_fit_lines_stack", fit_then_raise)
        wls = WLS(FILE_LIST, config={"KPF_MASTERS_OUTPUT": str(tmp_path)})
        night, master_path, thar_dir = _master_and_stack(tmp_path)
        with pytest.raises(ValueError, match="WLS fit failed"):
            wls.make_master_l2(output_dir=str(night))
        assert not master_path.exists()

        h5_path = thar_dir / "KP.20240101.00000.00_thar_l2_diagnostics.h5"
        obs_ids = [get_obs_id(fn) for fn in FILE_LIST]
        with h5py.File(h5_path, "r") as h5:
            assert set(h5[obs_ids[0]]) == {"GREEN_LINES", "GREEN_WLS_COEFFS"}
            assert set(h5[obs_ids[1]]) == {"GREEN_LINES"}

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
        # Both coefficient extensions come from the manifest, so an unprocessed
        # chip's ships at the schema default rather than being absent.
        assert "GREEN_WLS_COEFFS" in ml2.extensions
        assert "RED_WLS_COEFFS" in ml2.extensions

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
            "WAVE": np.asarray(wav, dtype=float),
            "FIT_MEAN": np.asarray(pix, dtype=float),
            "PIX_GUESS": np.round(pix).astype(int),
            "ECHELLE_ORDER": np.asarray(ord_, dtype=int),
            "FIBER": np.asarray(fib),
        }

    def _roundtrip(self, wls):
        """A known low-degree m*lambda surface and lines sampled from it."""
        orders = wls._echelle_orders["GREEN"]
        ncol = DETECTOR["ccd"]["ncol"]

        coeffs_true = np.zeros((wls.poly_degree_x + 1, wls.poly_degree_m + 1))
        coeffs_true[0, 0], coeffs_true[1, 0] = 8.0e5, 300.0  # mean, pixel slope
        coeffs_true[0, 1], coeffs_true[1, 1] = 5.0e4, 5.0  # order slope, cross term
        wave_true = wls._evaluate_wls_coeffs(coeffs_true, orders, nfiber=1)

        cols = np.linspace(0, ncol - 1, 60).astype(int)
        lines = {
            "WAVE": np.array(
                [wave_true[j, c] for j in range(len(orders)) for c in cols]
            ),
            "FIT_MEAN": np.array([float(c) for _ in orders for c in cols]),
            "PIX_GUESS": np.array([c for _ in orders for c in cols]),
            "ECHELLE_ORDER": np.array([o for o in orders for _ in cols]),
            "FIBER": np.array(["SCI1"] * (len(orders) * len(cols))),
        }
        return orders, wave_true, lines

    def test_underconstrained_single_fiber_raises(self):
        # default poly_degree is (6, 6, 2): 7*7 = 49 free params single-fiber
        wls = WLS(FILE_LIST)
        lines = self._make_lines(5, fibers=("SCI1",))
        with pytest.raises(ValueError, match=r"underconstrained"):
            wls._calculate_wls_coeffs(lines, wls._echelle_orders["GREEN"])

    def test_underconstrained_multi_fiber_raises(self):
        # 5-fiber: 7*7*3 = 147 free params; 10 lines per fiber * 5 = 50 < 147
        wls = WLS(FILE_LIST)
        lines = self._make_lines(10, fibers=FIBERS)
        with pytest.raises(ValueError, match=r"underconstrained"):
            wls._calculate_wls_coeffs(lines, wls._echelle_orders["GREEN"])

    def test_sufficient_lines_does_not_raise(self):
        wls = WLS(FILE_LIST)
        lines = self._make_lines(50, fibers=("SCI1",))
        coeffs = wls._calculate_wls_coeffs(lines, wls._echelle_orders["GREEN"])
        assert coeffs.shape == (wls.poly_degree_x + 1, wls.poly_degree_m + 1)

    def test_mlambda_roundtrip_recovers_wavelength(self):
        # The fit models m*lambda as a Legendre surface, so a surface built from
        # known low-degree coefficients must be recovered exactly.
        wls = WLS(FILE_LIST)
        orders, wave_true, lines = self._roundtrip(wls)
        coeffs_fit = wls._calculate_wls_coeffs(lines, orders)
        wave_fit = wls._evaluate_wls_coeffs(coeffs_fit, orders, nfiber=1)
        np.testing.assert_allclose(wave_fit, wave_true, rtol=1e-9)

    def test_centroid_cut_excludes_off_window_and_unfit_lines(self):
        # A line fit > window px from its guess and an unfit (NaN) line carry
        # wrong wavelengths; both must be excluded, leaving the exact solution.
        wls = WLS(FILE_LIST)
        orders, wave_true, lines = self._roundtrip(wls)
        lines = {k: np.append(v, v[:2]) for k, v in lines.items()}
        lines["WAVE"][-2:] = 9000.0
        lines["FIT_MEAN"][-2] = lines["PIX_GUESS"][-2] + 6.0
        lines["FIT_MEAN"][-1] = np.nan

        coeffs_fit = wls._calculate_wls_coeffs(lines, orders, window=5)
        wave_fit = wls._evaluate_wls_coeffs(coeffs_fit, orders, nfiber=1)
        np.testing.assert_allclose(wave_fit, wave_true, rtol=1e-9)


class TestFitLinesStack:
    """Every frame is line-fit and gets a per-frame WLS; nothing is rejected."""

    def _setup(self, monkeypatch, nframes, calc):
        wls = WLS(FILE_LIST)
        wls._l2_obj_cache = [MockL2(f"KP.20240101.0000{i}.00") for i in range(nframes)]
        monkeypatch.setattr(
            WLS,
            "_fit_line_positions_ffi",
            lambda self, *a, **k: _fake_lines("GREEN", "SCI1", [137, 136]),
        )
        monkeypatch.setattr(WLS, "_calculate_wls_coeffs", calc)
        return wls

    def test_every_frame_gets_coeffs(self, monkeypatch):
        wls = self._setup(monkeypatch, 5, lambda self, *a, **k: np.ones((2, 2)))
        frames = wls._fit_lines_stack("GREEN", ["SCI1"])
        assert len(frames) == 5
        assert all(fr["coeffs"] is not None for fr in frames)
        assert wls._frame_diagnostics["GREEN"] is frames

    def test_wls_fit_failure_raises_naming_frame(self, monkeypatch):
        results = iter([np.ones((2, 2)), "raise", np.ones((2, 2))])

        def calc(self, *a, **k):
            result = next(results)
            if isinstance(result, str):
                raise ValueError("WLS fit underconstrained: 2 good lines < 28")
            return result

        wls = self._setup(monkeypatch, 3, calc)
        with pytest.raises(
            ValueError, match=r"GREEN frame KP.20240101.00001.00: WLS fit failed"
        ):
            wls._fit_lines_stack("GREEN", ["SCI1"])
        # Every frame's lines were stashed before the per-frame WLS fits.
        frames = wls._frame_diagnostics["GREEN"]
        assert len(frames) == 3
        assert [fr["coeffs"] is None for fr in frames] == [False, True, True]


class TestCombineCoeffs:
    """_combine_coeffs_stack averages every frame's coeffs."""

    def test_plain_mean_includes_outlier_frame(self, monkeypatch):
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(
            WLS, "_evaluate_wls_coeffs", lambda self, coeffs, *a, **k: coeffs
        )
        frames = [{"coeffs": np.full((2, 2), c)} for c in (4.0, 4.0, 4.0, 100.0)]
        _, coeffs_mean = wls._combine_coeffs_stack(frames, "GREEN", 1)
        np.testing.assert_allclose(coeffs_mean, 28.0)


class TestMinStackSizeGate:
    """make_master_l2 gates the master on min_stack_size frames extracted to L2.

    Below the threshold it raises before line fitting, but the per-frame L2s of
    the frames that did load are still written.
    """

    def _mock(self, monkeypatch, n_loaded, n_frames=8, masters_output=None):
        # Frames past the first n_loaded fail to load (the first frame loads, so
        # it still names the master).
        files = sorted(f"KP.20240101.{i:05d}.00.fits" for i in range(n_frames))
        _stub_frame_pipeline(
            monkeypatch,
            l1=lambda self, fn, cache=False, **kw: (
                None if files.index(fn) >= n_loaded else MockL1(get_obs_id(fn))
            ),
        )

        def mock_fit(self, chip, fibers, **kwargs):
            coeffs = np.zeros(
                (self.poly_degree_x + 1, self.poly_degree_m + 1, self.poly_degree_f + 1)
            )
            frames = [
                {
                    "obs_id": l2.obs_id,
                    "lines": _fake_lines(chip, fibers[0], [137]),
                    "coeffs": coeffs,
                    "window": 5,
                    "lineprofile": "gaussian",
                }
                for l2 in self._l2_obj_cache
            ]
            self._frame_diagnostics[chip] = frames
            return frames

        def mock_combine(self, frames, chip, nfibers, **kwargs):
            norder = NORDER_GREEN if chip == "GREEN" else NORDER_RED
            coeffs = np.zeros(
                (self.poly_degree_x + 1, self.poly_degree_m + 1, self.poly_degree_f + 1)
            )
            return np.full((norder, NCOL_TEST, nfibers), 5500.0), coeffs

        monkeypatch.setattr(WLS, "_fit_lines_stack", mock_fit)
        monkeypatch.setattr(WLS, "_combine_coeffs_stack", mock_combine)
        config = {"KPF_MASTERS_OUTPUT": str(masters_output)} if masters_output else None
        return WLS(files, config=config)

    def test_below_min_raises_but_l2s_written(self, monkeypatch, tmp_path):
        # 7 of 8 load: within the failure limits, but below min_stack_size=8.
        wls = self._mock(monkeypatch, n_loaded=7, masters_output=tmp_path)
        wls.min_stack_size = 8
        night, master_path, thar_dir = _master_and_stack(tmp_path)
        with pytest.raises(ValueError, match=r"only 7 of 8 frames processed"):
            wls.make_master_l2(output_dir=str(night))

        # the loaded frames' L2s persist despite the abort ...
        assert len(list(thar_dir.glob("*_thar_L2.fits"))) == 7
        # ... but the master itself is withheld.
        assert not master_path.exists()

    def test_at_min_writes_master(self, monkeypatch, tmp_path):
        wls = self._mock(monkeypatch, n_loaded=7, masters_output=tmp_path)
        wls.min_stack_size = 7
        night, master_path, _ = _master_and_stack(tmp_path)
        ml2 = wls.make_master_l2(output_dir=str(night))
        assert isinstance(ml2, KPFMasterL2)
        assert master_path.exists()


class TestFitLinePositions:
    def _stub_ffi(self, wls, flux, waves):
        ncol = DETECTOR["ccd"]["ncol"]
        norder = DETECTOR["norder"]["RED"]

        class StubL2:
            data = {"RED_SCI1_FLUX": flux, "RED_SCI1_VAR": np.ones_like(flux)}

        wls.rough_wls = {
            "RED_SCI1_WAVE": np.tile(np.linspace(6500.0, 6510.0, ncol), (norder, 1))
        }
        wls._linelist_df = _linelist_df("RED", norder, waves)
        return StubL2()

    def test_nan_orderlet_emits_warning_and_does_not_crash(self, caplog):
        # A NaN-filled orderlet (extraction failure) is skipped rather than
        # crashing scipy's least_squares, but its lines still get rows.
        wls = WLS(FILE_LIST)
        ncol = DETECTOR["ccd"]["ncol"]
        norder = DETECTOR["norder"]["RED"]

        flux = np.ones((norder, ncol))
        flux[0, :] = np.nan
        l2 = self._stub_ffi(wls, flux, [6502.0, 6505.0, 6508.0])

        with caplog.at_level(logging.DEBUG):
            result = wls._fit_line_positions_ffi(l2, "RED", ["SCI1"])
        assert "RED SCI1 order 102: orderlet skipped" in caplog.text

        # The NaN order (bluest RED, echelle 102) has unfit rows; the rest fit.
        nan_rows = result["ECHELLE_ORDER"] == 102
        assert nan_rows.sum() == 3
        assert not result["FINITE"][nan_rows].any()
        assert not result["CONVERGED"][nan_rows].any()
        assert np.isnan(result["FIT_MEAN"][nan_rows]).all()
        assert result["FINITE"][~nan_rows].all()
        assert set(result) == set(wls_module._LINE_FIT_COLUMNS)

    def test_no_lines_returns_empty(self):
        wls = WLS(FILE_LIST)
        flux = np.ones(100)
        wave = np.linspace(5000.0, 5100.0, 100)

        result = wls._fit_line_positions_1d(flux, np.ones(100), wave, np.array([]))
        for key in ["WAVE", "FIT_MEAN", "FIT_STD", "FIT_AMP", "FINITE"]:
            assert len(result[key]) == 0

    def test_line_outside_rough_wls_range_raises(self):
        # A line outside the order's rough WLS span means the CHIP/ORDER labels
        # are inconsistent, which must fail loudly rather than fit garbage.
        wls = WLS(FILE_LIST)
        flux = np.ones(100)
        wave = np.linspace(5000.0, 5100.0, 100)
        with pytest.raises(ValueError, match="outside the rough"):
            wls._fit_line_positions_1d(flux, np.ones(100), wave, np.array([6000.0]))

    def test_nan_var_window_not_fit(self):
        wls = WLS(FILE_LIST)
        x = np.arange(100)
        wave = np.linspace(5000.0, 5100.0, 100)
        flux = 1.0 + 50.0 * np.exp(-0.5 * ((x - 50) / 2.0) ** 2)
        var = np.ones(100)
        var[50] = np.nan
        result = wls._fit_line_positions_1d(flux, var, wave, np.array([wave[50]]))
        assert not result["FINITE"][0]
        assert np.isnan(result["SNR"][0])

    def test_fit_returns_float64_from_float32_inputs(self):
        # float32 inputs must not drag the line fit down to float32.
        wls = WLS(FILE_LIST)
        ncol = 100
        x = np.arange(ncol)
        wave = np.linspace(5000.0, 5100.0, ncol).astype(np.float32)
        flux = (1.0 + 50.0 * np.exp(-0.5 * ((x - 50) / 2.0) ** 2)).astype(np.float32)
        var = np.ones(ncol, dtype=np.float32)
        line_waves = np.array([wave[50]], dtype=float)

        result = wls._fit_line_positions_1d(
            flux, var, wave, line_waves, lineprofile="gaussian"
        )
        assert len(result["WAVE"]) == 1
        assert result["FINITE"][0] and result["CONVERGED"][0]
        assert abs(result["FIT_MEAN"][0] - 50.0) < 0.01
        for key in ["WAVE", "FIT_MEAN", "FIT_STD", "FIT_AMP", "FIT_MEAN_ERR", "SNR"]:
            assert_dtype(result[key], WAVE, f"_fit_line_positions_1d {key}")

    def test_all_nan_fiber_emits_fiber_level_warning(self, caplog):
        wls = WLS(FILE_LIST)
        ncol = DETECTOR["ccd"]["ncol"]
        norder = DETECTOR["norder"]["RED"]
        l2 = self._stub_ffi(wls, np.full((norder, ncol), np.nan), [6502.0, 6505.0])

        # Capture at WARNING to isolate the fiber-level message from the
        # per-order skips, which log at DEBUG.
        with caplog.at_level(logging.WARNING):
            result = wls._fit_line_positions_ffi(l2, "RED", ["SCI1"])
        assert "RED SCI1: no lines fit" in caplog.text
        assert len(result["WAVE"]) == 2 * norder
        assert not result["FINITE"].any()


class TestRoughWlsLoading:
    def test_missing_rough_wls_file_raises(self):
        wls = WLS(FILE_LIST)
        with pytest.raises(FileNotFoundError, match="/nonexistent/path.csv"):
            wls._load_rough_wls(rough_wls_file="/nonexistent/path.csv")

    def test_cached_grid_matches_the_real_loader(self, cached_rough_wls):
        # The one test that runs the real Legendre evaluation, so the module's
        # memoized grid cannot drift from what production would have built.
        wls = WLS(FILE_LIST)
        _REAL_LOAD_ROUGH_WLS(wls, ROUGH_WLS_FILE)
        assert set(wls.rough_wls) == set(cached_rough_wls)
        for channel_ext, grid in wls.rough_wls.items():
            np.testing.assert_array_equal(grid, cached_rough_wls[channel_ext])
