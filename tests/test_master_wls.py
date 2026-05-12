"""
Unit tests for the WLS module.

All sub-module I/O is mocked; no real data or FITS files are required.
"""
import h5py
import numpy as np
import pytest
from unittest.mock import MagicMock

from kpfpipe import DETECTOR
from kpfpipe.data_models.masters import KPFMasterL2
import kpfpipe.modules.masters.wls as wls_module
from kpfpipe.modules.masters.wls import WLS

NORDER_GREEN = DETECTOR['norder']['GREEN']
NORDER_RED = DETECTOR['norder']['RED']
NCOL_TEST = 16

FILE_LIST = sorted([f"KP.20240101.{i:05d}.00.fits" for i in range(8)])


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

class MockL1:
    pass


class MockL2:
    pass


@pytest.fixture
def mock_pipeline(monkeypatch):
    """
    Patch CalibrationAssociation, ImageProcessing, and SpectralExtraction so
    that _extract_frame runs without touching disk or real data.

    Returns the MockL2 instance that SpectralExtraction.perform() will return.
    """
    l2 = MockL2()

    mock_ca = MagicMock()
    mock_ca.return_value.perform.return_value = MockL1()

    mock_ip = MagicMock()
    mock_ip.return_value.perform.return_value = MockL1()

    mock_se = MagicMock()
    mock_se.return_value.perform.return_value = l2

    monkeypatch.setattr(wls_module, 'CalibrationAssociation', mock_ca)
    monkeypatch.setattr(wls_module, 'ImageProcessing', mock_ip)
    monkeypatch.setattr(wls_module, 'SpectralExtraction', mock_se)

    return l2


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:

    def test_none_config_sets_data_input_none(self):
        wls = WLS(FILE_LIST)
        assert wls._data_root is None

    def test_dict_config_sets_data_input(self):
        wls = WLS(FILE_LIST, config={'KPF_DATA_INPUT': '/data/kpf'})
        assert wls._data_root == '/data/kpf'

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

    def test_passes_data_input_to_calibration_association(self, monkeypatch):
        mock_ca = MagicMock()
        mock_ca.return_value.perform.return_value = MockL1()
        mock_ip = MagicMock()
        mock_ip.return_value.perform.return_value = MockL1()
        mock_se = MagicMock()
        mock_se.return_value.perform.return_value = MockL2()

        monkeypatch.setattr(wls_module, 'CalibrationAssociation', mock_ca)
        monkeypatch.setattr(wls_module, 'ImageProcessing', mock_ip)
        monkeypatch.setattr(wls_module, 'SpectralExtraction', mock_se)

        wls = WLS(FILE_LIST, config={'KPF_DATA_INPUT': '/data/kpf'})
        wls._extract_frame(MockL1())

        call_args = mock_ca.call_args[0]
        assert call_args[1].get('KPF_DATA_INPUT') == '/data/kpf'


# ---------------------------------------------------------------------------
# TestProcessIndividualFrames
# ---------------------------------------------------------------------------

class TestProcessIndividualFrames:

    def test_returns_l2_objects_for_all_frames(self, mock_pipeline, monkeypatch):
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(wls, '_load_frame', lambda fn, ncache=0: (MockL1(), True))
        result = wls.process_stack_l0_to_l2()
        assert len(result) == len(FILE_LIST)
        assert all(r is mock_pipeline for r in result)

    def test_file_list_override(self, mock_pipeline, monkeypatch):
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(wls, '_load_frame', lambda fn, ncache=0: (MockL1(), True))
        result = wls.process_stack_l0_to_l2(l0_file_list=FILE_LIST[:3])
        assert len(result) == 3

    def test_raises_when_failures_exceed_threshold(self, monkeypatch):
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(wls, '_load_frame', lambda fn, ncache=0: (None, False))
        with pytest.raises(ValueError, match="20%"):
            wls.process_stack_l0_to_l2()

    def test_tolerates_minority_failure(self, mock_pipeline, monkeypatch):
        # 1 failure out of 8 = 12.5%, below the 20% threshold
        calls = iter([(None, False)] + [(MockL1(), True)] * 7)
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(wls, '_load_frame', lambda fn, ncache=0: next(calls))
        result = wls.process_stack_l0_to_l2()
        assert len(result) == 7


# ---------------------------------------------------------------------------
# TestMakeMasterL2
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_make_master_l2(monkeypatch):
    """
    Patch frame loading and compute_wls_from_stack so make_master_l2 runs
    without touching disk or real spectra. compute_wls_from_stack returns
    synthetic W and coefficient arrays with chip-correct shapes.
    """
    monkeypatch.setattr(WLS, '_load_frame',
                        lambda self, fn, ncache=0: (MockL1(), True))
    monkeypatch.setattr(WLS, '_extract_frame',
                        lambda self, l1: MockL2())

    def mock_compute(self, chip, fibers, lineprofile=None,
                     polyorder_x=None, polyorder_m=None, polyorder_f=None,
                     return_stacks=False, **kwargs):
        polyorder_x = polyorder_x if polyorder_x is not None else self.polyorder_x
        polyorder_m = polyorder_m if polyorder_m is not None else self.polyorder_m
        polyorder_f = polyorder_f if polyorder_f is not None else self.polyorder_f

        norder = NORDER_GREEN if chip == 'GREEN' else NORDER_RED
        nfibers = len(fibers)

        if nfibers == 1:
            W = np.full((norder, NCOL_TEST), 5500.0)
            coeffs = np.zeros((polyorder_x + 1, polyorder_m + 1))
        else:
            W = np.full((norder, NCOL_TEST, nfibers), 5500.0)
            coeffs = np.zeros((polyorder_x + 1, polyorder_m + 1, polyorder_f + 1))

        if return_stacks:
            coeffs_stack = np.array([coeffs] * 3)
            lines_stack = [
                {'wav': np.array([5500.0, 5501.0]),
                 'pix': np.array([100.5, 200.5]),
                 'ord': np.array([1, 2]),
                 'fib': np.array([fibers[0]] * 2),
                 'bad': np.array([False, False]),
                 'std': np.array([0.5, 0.5]),
                 'amp': np.array([1.0, 1.0]),
                 'rms': np.array([0.01, 0.01])}
                for _ in range(3)
            ]
            return W, coeffs, coeffs_stack, lines_stack
        return W, coeffs

    monkeypatch.setattr(WLS, 'compute_wls_from_stack', mock_compute)


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
                wave = ml2.data[f'{chip}_{fiber}_WAVE']
                assert np.size(wave) > 0
                assert np.all(wave == 5500.0)

    def test_coeffs_extensions_populated(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        for chip in wls.chips:
            ext = f'{chip}_WLS_COEFFS'
            assert ext in ml2.extensions
            coeffs = ml2.data[ext]
            assert coeffs is not None
            assert coeffs.shape == (wls.polyorder_x + 1,
                                    wls.polyorder_m + 1,
                                    wls.polyorder_f + 1)

    def test_primary_header_keywords(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        primary = ml2.headers['PRIMARY']
        for key in ['ROUGHWLS', 'LINELIST', 'LINEPROF',
                    'POLYORDX', 'POLYORDM', 'POLYORDF',
                    'CHIPS', 'FIBERS']:
            assert key in primary

    def test_coeffs_extension_header_keywords(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        for chip in wls.chips:
            hdr = ml2.headers[f'{chip}_WLS_COEFFS']
            for key in ['POLYORDX', 'POLYORDM', 'POLYORDF']:
                assert key in hdr

    def test_polyorder_override_stamped(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2(polyorder_x=10)
        assert _header_value(ml2.headers['PRIMARY'], 'POLYORDX') == 10
        for chip in wls.chips:
            assert _header_value(ml2.headers[f'{chip}_WLS_COEFFS'], 'POLYORDX') == 10

    def test_input_files_recorded(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        assert ml2.data['INPUT_FILES']['FILENAME'].tolist() == FILE_LIST

    def test_receipt_entry(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2 = wls.make_master_l2()
        assert 'master_wls' in ml2.receipt['Module_Name'].tolist()

    def test_resets_l2_cache(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        wls._l2_obj_cache = [MockL2(), MockL2(), MockL2()]
        wls.make_master_l2()
        assert len(wls._l2_obj_cache) == len(FILE_LIST)

    def test_return_stacks_false_returns_single(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        result = wls.make_master_l2(return_stacks=False)
        assert isinstance(result, KPFMasterL2)

    def test_return_stacks_true_returns_tuple(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        result = wls.make_master_l2(return_stacks=True)
        assert isinstance(result, tuple) and len(result) == 2
        ml2, h5 = result
        assert isinstance(ml2, KPFMasterL2)
        assert isinstance(h5, h5py.File)

    def test_hdf5_structure(self, mock_make_master_l2):
        wls = WLS(FILE_LIST)
        ml2, h5 = wls.make_master_l2(return_stacks=True)
        for chip in wls.chips:
            assert chip in h5
            assert 'coeffs_stack' in h5[chip]
            assert 'lines_stack' in h5[chip]
            frame_keys = list(h5[chip]['lines_stack'].keys())
            assert len(frame_keys) > 0
            sample = h5[chip]['lines_stack'][frame_keys[0]]
            for key in ['wav', 'pix', 'ord', 'fib', 'bad']:
                assert key in sample
