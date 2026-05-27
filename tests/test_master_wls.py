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
        monkeypatch.setattr(wls, '_load_frame', lambda fn, ncache=0, **kwargs:(MockL1(), True))
        result = wls.process_stack_l0_to_l2()
        assert len(result) == len(FILE_LIST)
        assert all(r is mock_pipeline for r in result)

    def test_file_list_override(self, mock_pipeline, monkeypatch):
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(wls, '_load_frame', lambda fn, ncache=0, **kwargs:(MockL1(), True))
        result = wls.process_stack_l0_to_l2(l0_file_list=FILE_LIST[:3])
        assert len(result) == 3

    def test_raises_when_failures_exceed_threshold(self, monkeypatch):
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(wls, '_load_frame', lambda fn, ncache=0, **kwargs:(None, False))
        with pytest.raises(ValueError, match="20%"):
            wls.process_stack_l0_to_l2()

    def test_tolerates_minority_failure(self, mock_pipeline, monkeypatch):
        # 1 failure out of 8 = 12.5%, below the 20% threshold
        calls = iter([(None, False)] + [(MockL1(), True)] * 7)
        wls = WLS(FILE_LIST)
        monkeypatch.setattr(wls, '_load_frame', lambda fn, ncache=0, **kwargs:next(calls))
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
                        lambda self, fn, ncache=0, **kwargs: (MockL1(), True))
    monkeypatch.setattr(WLS, '_extract_frame',
                        lambda self, l1, **kwargs: MockL2())

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
        override_x = wls.polyorder_x + 4   # ensure different from default
        ml2 = wls.make_master_l2(polyorder_x=override_x)
        assert _header_value(ml2.headers['PRIMARY'], 'POLYORDX') == override_x
        for chip in wls.chips:
            assert _header_value(ml2.headers[f'{chip}_WLS_COEFFS'], 'POLYORDX') == override_x
            # verify the override actually propagated into the fit, not just the header
            coeffs = ml2.data[f'{chip}_WLS_COEFFS']
            assert coeffs.shape[0] == override_x + 1

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
            wls.save_diagnostics('/tmp/should_not_be_created.h5')

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

        with h5py.File(diagnostics_path, 'r') as h5:
            for chip in wls.chips:
                assert chip in h5
                cs = h5[chip]['coeffs_stack']
                assert cs.shape == expected_coeffs_shape
                assert np.issubdtype(cs.dtype, np.floating)

                assert 'lines_stack' in h5[chip]
                frame_keys = sorted(h5[chip]['lines_stack'].keys())
                assert len(frame_keys) == expected_nframes

                sample = h5[chip]['lines_stack'][frame_keys[0]]
                for key in ['wav', 'pix', 'ord', 'fib', 'bad', 'std', 'amp', 'rms']:
                    assert key in sample
                assert np.issubdtype(sample['wav'].dtype, np.floating)
                assert np.issubdtype(sample['pix'].dtype, np.floating)
                assert np.issubdtype(sample['ord'].dtype, np.integer)
                assert sample['bad'].dtype == bool
                assert h5py.check_string_dtype(sample['fib'].dtype) is not None
                # finite values for all numeric per-line arrays
                for key in ['wav', 'pix', 'std', 'amp', 'rms']:
                    assert np.all(np.isfinite(sample[key][...]))

    def test_nan_orderlet_emits_warning_and_does_not_crash(self):
        """
        A NaN-filled orderlet (extraction failure) should be skipped with a
        warning rather than crashing scipy's least_squares.
        """
        wls = WLS(FILE_LIST)
        ncol = wls.ccd['ncol']
        norder = wls.norder['RED']

        flux = np.ones((norder, ncol))
        flux[0, :] = np.nan  # simulate failed-extraction orderlet

        class StubL2:
            data = {f'RED_SCI1_FLUX': flux}

        wls.rough_wls['RED_SCI1_WAVE'] = np.tile(
            np.linspace(6500.0, 6510.0, ncol), (norder, 1)
        )
        wls._linelist_array = np.array([6502.0, 6505.0, 6508.0])

        with pytest.warns(UserWarning, match=r"RED SCI1 order 1: orderlet skipped"):
            result = wls.fit_line_positions_ffi(
                StubL2(), 'RED', ['SCI1'],
            )

        # The NaN order contributed no lines; remaining orders did.
        assert result['ord'].min() >= 2
        assert len(result['wav']) > 0

    def test_linelist_override(self, mock_make_master_l2, tmp_path, monkeypatch):
        """Override file is loaded into the cache and stamped to the header."""
        override = tmp_path / "alt_linelist.csv"
        override.write_text("Wavelength\n4500.0\n5500.0\n6500.0\n")

        wls = WLS(FILE_LIST)
        original_path = wls.linelist
        original_array = wls._linelist_array.copy()

        ml2 = wls.make_master_l2(linelist=str(override))

        assert wls.linelist == str(override)
        assert wls.linelist != original_path
        assert not np.array_equal(wls._linelist_array, original_array)
        np.testing.assert_array_equal(
            wls._linelist_array, np.array([4500.0, 5500.0, 6500.0])
        )
        assert _header_value(ml2.headers['PRIMARY'], 'LINELIST') == str(override)

    def test_single_frame_stack(self, mock_make_master_l2):
        """A 1-frame stack should still produce a valid master L2."""
        wls = WLS(FILE_LIST[:1])
        ml2 = wls.make_master_l2()
        assert isinstance(ml2, KPFMasterL2)
        assert len(wls._l2_obj_cache) == 1
        for chip in wls.chips:
            for fiber in wls.fibers:
                assert f'{chip}_{fiber}_WAVE' in ml2.data

    def test_empty_file_list_raises(self, mock_make_master_l2):
        wls = WLS([])
        with pytest.raises(ValueError, match=r"Empty l0_file_list"):
            wls.make_master_l2()

    def test_single_chip_config(self, mock_make_master_l2):
        """A WLS configured for one chip only should not populate the other."""
        wls = WLS(FILE_LIST, config={'chips': ['GREEN']})
        ml2 = wls.make_master_l2()
        for fiber in wls.fibers:
            # GREEN populated by the mock (5500.0); RED left at the schema-
            # default zeros since RED was not in self.chips.
            assert np.all(ml2.data[f'GREEN_{fiber}_WAVE'] == 5500.0)
            assert np.all(ml2.data[f'RED_{fiber}_WAVE'] == 0.0)
        assert 'GREEN_WLS_COEFFS' in ml2.extensions
        assert 'RED_WLS_COEFFS' not in ml2.extensions

    def test_results_populated(self, mock_make_master_l2):
        """make_master_l2 should populate self._results with per-chip line yields."""
        wls = WLS(FILE_LIST)
        assert wls._results is None
        wls.make_master_l2()
        assert wls._results is not None
        for chip in wls.chips:
            assert chip in wls._results
            assert 'n_total' in wls._results[chip]
            assert 'n_fit' in wls._results[chip]
            assert isinstance(wls._results[chip]['n_total'], int)
            assert isinstance(wls._results[chip]['n_fit'], int)

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

    def _make_lines(self, n, fibers=('SCI1',)):
        """Build a minimal `lines` dict with `n` lines per fiber."""
        wav, pix, ord_, fib = [], [], [], []
        for f in fibers:
            wav.extend(5000.0 + np.arange(n))
            pix.extend(np.linspace(10.0, 4000.0, n))
            ord_.extend(np.linspace(1, 30, n).astype(int))
            fib.extend([f] * n)
        return {
            'wav': np.asarray(wav, dtype=float),
            'pix': np.asarray(pix, dtype=float),
            'ord': np.asarray(ord_, dtype=int),
            'fib': np.asarray(fib),
            'bad': np.zeros(n * len(fibers), dtype=bool),
        }

    def test_underconstrained_single_fiber_raises(self):
        # default polyorder is (6, 3, 2) → 7*4 = 28 free params single-fiber
        wls = WLS(FILE_LIST)
        lines = self._make_lines(5, fibers=('SCI1',))
        with pytest.raises(ValueError, match=r"underconstrained"):
            wls.calculate_wls_coeffs(lines, norder=30)

    def test_underconstrained_multi_fiber_raises(self):
        # 5-fiber → 7*4*3 = 84 free params; 10 lines per fiber * 5 = 50 < 84
        wls = WLS(FILE_LIST)
        lines = self._make_lines(10, fibers=('SKY', 'SCI1', 'SCI2', 'SCI3', 'CAL'))
        with pytest.raises(ValueError, match=r"underconstrained"):
            wls.calculate_wls_coeffs(lines, norder=30)

    def test_sufficient_lines_does_not_raise(self):
        wls = WLS(FILE_LIST)
        lines = self._make_lines(50, fibers=('SCI1',))
        coeffs = wls.calculate_wls_coeffs(lines, norder=30)
        assert coeffs.shape == (wls.polyorder_x + 1, wls.polyorder_m + 1)


# ---------------------------------------------------------------------------
# TestFitLinePositions
# ---------------------------------------------------------------------------

class TestFitLinePositions:

    def test_linelist_no_overlap_returns_empty(self):
        """Linelist with no entries inside `wave1d` range yields empty arrays."""
        wls = WLS(FILE_LIST)
        flux = np.ones(100)
        wave = np.linspace(5000.0, 5100.0, 100)
        wls._linelist_array = np.array([6000.0, 6100.0])  # entirely outside

        result = wls.fit_line_positions_1d(flux, wave)
        for key in ['wav', 'pix', 'std', 'amp', 'rms', 'bad']:
            assert len(result[key]) == 0

    def test_all_nan_fiber_emits_fiber_level_warning(self):
        """A fiber whose every order is NaN should emit a fiber-level warning."""
        wls = WLS(FILE_LIST)
        ncol = wls.ccd['ncol']
        norder = wls.norder['RED']

        flux = np.full((norder, ncol), np.nan)  # entire fiber NaN

        class StubL2:
            data = {'RED_SCI1_FLUX': flux}

        wls.rough_wls['RED_SCI1_WAVE'] = np.tile(
            np.linspace(6500.0, 6510.0, ncol), (norder, 1)
        )
        wls._linelist_array = np.array([6502.0, 6505.0])

        with pytest.warns(UserWarning, match=r"RED SCI1: no good lines retained"):
            result = wls.fit_line_positions_ffi(
                StubL2(), 'RED', ['SCI1'],
            )
        assert len(result['wav']) == 0

    def test_nan_orderlet_warning_suppressed_when_verbose_false(self, recwarn):
        """verbose=False should silence the per-orderlet skip warning."""
        wls = WLS(FILE_LIST)
        ncol = wls.ccd['ncol']
        norder = wls.norder['RED']

        flux = np.ones((norder, ncol))
        flux[0, :] = np.nan  # one orderlet NaN-filled

        class StubL2:
            data = {'RED_SCI1_FLUX': flux}

        wls.rough_wls['RED_SCI1_WAVE'] = np.tile(
            np.linspace(6500.0, 6510.0, ncol), (norder, 1)
        )
        wls._linelist_array = np.array([6502.0, 6505.0, 6508.0])

        wls.fit_line_positions_ffi(StubL2(), 'RED', ['SCI1'], verbose=False)

        skipped = [w for w in recwarn if "orderlet skipped" in str(w.message)]
        assert len(skipped) == 0

    def test_all_nan_fiber_warning_suppressed_when_verbose_false(self, recwarn):
        """verbose=False should silence the fiber-level no-good-lines warning."""
        wls = WLS(FILE_LIST)
        ncol = wls.ccd['ncol']
        norder = wls.norder['RED']

        flux = np.full((norder, ncol), np.nan)

        class StubL2:
            data = {'RED_SCI1_FLUX': flux}

        wls.rough_wls['RED_SCI1_WAVE'] = np.tile(
            np.linspace(6500.0, 6510.0, ncol), (norder, 1)
        )
        wls._linelist_array = np.array([6502.0, 6505.0])

        wls.fit_line_positions_ffi(StubL2(), 'RED', ['SCI1'], verbose=False)

        fiber_level = [w for w in recwarn if "no good lines retained" in str(w.message)]
        assert len(fiber_level) == 0


# ---------------------------------------------------------------------------
# TestRoughWlsLoading
# ---------------------------------------------------------------------------

class TestRoughWlsLoading:

    def test_missing_rough_wls_file_raises(self):
        wls = WLS(FILE_LIST)
        with pytest.raises(FileNotFoundError):
            wls._load_rough_wls(rough_wls_file='/nonexistent/path.csv')
