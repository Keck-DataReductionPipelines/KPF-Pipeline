"""
Unit tests for the WLS module.

All sub-module I/O is mocked; no real data or FITS files are required.
"""
import pytest
from unittest.mock import MagicMock

import kpfpipe.modules.masters.wls as wls_module
from kpfpipe.modules.masters.wls import WLS


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
