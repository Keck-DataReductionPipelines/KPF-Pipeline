"""Tests for the Diagnostics framework and per-level subclasses."""

import numpy as np
import pytest
from astropy.io import fits

from kpfpipe import DETECTOR
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.quality_control.diagnostics import Diagnostics, DiagL0, DiagL1, DiagL2

NORDER_GREEN = DETECTOR['norder']['GREEN']
NORDER_RED   = DETECTOR['norder']['RED']
NCOL         = DETECTOR['ccd']['ncol']

_FIBERS = ('SCI1', 'SCI2', 'SCI3', 'SKY', 'CAL')
_NAN_KEYS = ('NANSCI1', 'NANSCI2', 'NANSCI3', 'NANSKY', 'NANCAL')


def _hval(raw):
    """Unpack header value from (value, comment) tuple or plain value."""
    return raw[0] if isinstance(raw, tuple) else raw


# ---------------------------------------------------------------------------
# Diagnostics base class
# ---------------------------------------------------------------------------

class TestDiagnosticsBase:

    def _make_obj(self):
        class _FakeObj:
            headers = {"PRIMARY": {}}
            data = {}
        return _FakeObj()

    def test_writes_returned_keys_to_primary(self):
        obj = self._make_obj()

        class MyDiag(Diagnostics):
            def metric_a(self):
                return {"KEYA": (3.14, "metric a")}
            metric_a._diag_name = "metric_a"

        results = MyDiag(obj).run()
        assert obj.headers["PRIMARY"]["KEYA"] == (3.14, "metric a")
        assert results["KEYA"] == (3.14, "metric a")

    def test_method_can_emit_multiple_keys(self):
        obj = self._make_obj()

        class MyDiag(Diagnostics):
            def multi(self):
                return {"K1": (1, "one"), "K2": (2, "two")}
            multi._diag_name = "multi"

        MyDiag(obj).run()
        assert obj.headers["PRIMARY"]["K1"] == (1, "one")
        assert obj.headers["PRIMARY"]["K2"] == (2, "two")

    def test_empty_dict_writes_nothing(self):
        obj = self._make_obj()

        class MyDiag(Diagnostics):
            def skipped(self):
                return {}
            skipped._diag_name = "skipped"

        results = MyDiag(obj).run()
        assert results == {}
        assert obj.headers["PRIMARY"] == {}

    def test_raising_method_propagates_runtime_error(self):
        obj = self._make_obj()

        class MyDiag(Diagnostics):
            def boom(self):
                raise ValueError("boom!")
            boom._diag_name = "boom"

        with pytest.raises(RuntimeError, match="Diagnostic 'boom' raised"):
            MyDiag(obj).run()

    def test_repeated_run_resets_results(self):
        obj = self._make_obj()
        obj.value = 1

        class MyDiag(Diagnostics):
            def metric(self):
                return {"VAL": (self.kpf.value, "value")}
            metric._diag_name = "metric"

        d = MyDiag(obj)
        d.run()
        assert d.results == {"VAL": (1, "value")}

        obj.value = 99
        d.run()
        assert d.results == {"VAL": (99, "value")}

    def test_empty_subclass_runs_cleanly(self):
        obj = self._make_obj()

        class EmptyDiag(Diagnostics):
            pass

        results = EmptyDiag(obj).run()
        assert results == {}


# ---------------------------------------------------------------------------
# DiagL0 / DiagL1 — currently empty placeholders
# ---------------------------------------------------------------------------

class TestEmptyLevels:

    def _make_obj(self):
        class _FakeObj:
            headers = {"PRIMARY": {}}
            data = {}
        return _FakeObj()

    def test_diag_l0_runs_cleanly(self):
        results = DiagL0(self._make_obj()).run()
        assert results == {}

    def test_diag_l1_runs_cleanly(self):
        results = DiagL1(self._make_obj()).run()
        assert results == {}


# ---------------------------------------------------------------------------
# DiagL2 — NaN counts + zero-flux fraction
# ---------------------------------------------------------------------------

def _make_kpf2_with_flux(nan_frac=0.0, zero_frac=0.0):
    """Build a minimal KPF1, convert to KPF2, populate FLUX/VAR extensions
    with controllable NaN and zero fractions across all (chip, fiber) pairs.

    Each FLUX extension has shape (norder[chip], NCOL). Each is initialized
    to ones, then a fraction is replaced with NaN, then a fraction with 0.0.
    """
    from io import BytesIO
    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-01-01T00:00:00"
    green_ccd = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="GREEN_CCD")
    green_var = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="GREEN_VAR")
    red_ccd   = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="RED_CCD")
    red_var   = fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="RED_VAR")

    # Round-trip via in-memory FITS to produce a valid KPF1 → to_kpf2().
    buf = BytesIO()
    fits.HDUList([primary, green_ccd, green_var, red_ccd, red_var]).writeto(buf)
    buf.seek(0)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
        tmp.write(buf.read())
        tmp_path = tmp.name
    try:
        l1 = KPF1.from_fits(tmp_path)
    finally:
        os.unlink(tmp_path)
    kpf2 = l1.to_kpf2()

    norder = {'GREEN': NORDER_GREEN, 'RED': NORDER_RED}
    rng = np.random.default_rng(42)
    for chip in ('GREEN', 'RED'):
        for fiber in _FIBERS:
            n = norder[chip]
            arr = np.ones((n, NCOL), dtype=np.float32)
            mask = rng.random(arr.shape)
            if nan_frac > 0:
                arr[mask < nan_frac] = np.nan
            if zero_frac > 0:
                # Zeros fall in the band [nan_frac, nan_frac+zero_frac)
                arr[(mask >= nan_frac) & (mask < nan_frac + zero_frac)] = 0.0
            kpf2.set_data(f'{chip}_{fiber}_FLUX', arr)
    return kpf2


class TestDiagL2NanCounts:

    def test_writes_all_five_keys_with_zero_when_clean(self):
        kpf2 = _make_kpf2_with_flux(nan_frac=0.0)
        DiagL2(kpf2).run()
        for key in _NAN_KEYS:
            assert key in kpf2.headers['PRIMARY'], f"missing {key}"
            assert _hval(kpf2.headers['PRIMARY'][key]) == 0

    def test_counts_injected_nans_per_fiber(self):
        kpf2 = _make_kpf2_with_flux(nan_frac=0.0)
        # Inject one NaN into GREEN_SCI1_FLUX; expect NANSCI1==1, others==0.
        kpf2.data['GREEN_SCI1_FLUX'][0, 0] = np.nan
        DiagL2(kpf2).run()
        assert _hval(kpf2.headers['PRIMARY']['NANSCI1']) == 1
        for key in ('NANSCI2', 'NANSCI3', 'NANSKY', 'NANCAL'):
            assert _hval(kpf2.headers['PRIMARY'][key]) == 0

    def test_writes_keys_even_when_no_data(self):
        """KPF2 with no FLUX extensions populated should still write all 5
        keys with value 0 (consistent header schema)."""
        # Build a KPF2 without populating any FLUX arrays.
        from io import BytesIO
        primary = fits.PrimaryHDU()
        primary.header["INSTRUME"] = "KPF"
        primary.header["DATE-OBS"] = "2024-01-01T00:00:00"
        for ext in ("GREEN_CCD", "GREEN_VAR", "RED_CCD", "RED_VAR"):
            primary  # noop
        hdul = fits.HDUList([
            primary,
            fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="GREEN_CCD"),
            fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="GREEN_VAR"),
            fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="RED_CCD"),
            fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="RED_VAR"),
        ])
        buf = BytesIO()
        hdul.writeto(buf)
        buf.seek(0)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            tmp.write(buf.read())
            tmp_path = tmp.name
        try:
            l1 = KPF1.from_fits(tmp_path)
        finally:
            os.unlink(tmp_path)
        kpf2 = l1.to_kpf2()
        DiagL2(kpf2).run()
        for key in _NAN_KEYS:
            assert _hval(kpf2.headers['PRIMARY'][key]) == 0


class TestDiagL2ZeroFlux:

    def test_zerofrac_written_when_data_present(self):
        kpf2 = _make_kpf2_with_flux(zero_frac=0.0)  # all ones
        DiagL2(kpf2).run()
        assert 'ZEROFRAC' in kpf2.headers['PRIMARY']
        assert _hval(kpf2.headers['PRIMARY']['ZEROFRAC']) == pytest.approx(0.0)

    def test_zerofrac_one_when_all_zero(self):
        kpf2 = _make_kpf2_with_flux(zero_frac=1.0)
        DiagL2(kpf2).run()
        assert _hval(kpf2.headers['PRIMARY']['ZEROFRAC']) == pytest.approx(1.0)

    def test_zerofrac_approximate_when_partial(self):
        """50% zeros sprinkled randomly → ZEROFRAC ≈ 0.5 within sampling error."""
        kpf2 = _make_kpf2_with_flux(zero_frac=0.5)
        DiagL2(kpf2).run()
        assert _hval(kpf2.headers['PRIMARY']['ZEROFRAC']) == pytest.approx(0.5, abs=0.01)

    def test_zerofrac_skipped_when_no_data(self):
        """KPF2 with no populated FLUX extensions → no ZEROFRAC key written."""
        from io import BytesIO
        primary = fits.PrimaryHDU()
        primary.header["DATE-OBS"] = "2024-01-01T00:00:00"
        hdul = fits.HDUList([
            primary,
            fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="GREEN_CCD"),
            fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="GREEN_VAR"),
            fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="RED_CCD"),
            fits.ImageHDU(data=np.zeros((4, 4), dtype=np.float32), name="RED_VAR"),
        ])
        buf = BytesIO()
        hdul.writeto(buf)
        buf.seek(0)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
            tmp.write(buf.read())
            tmp_path = tmp.name
        try:
            l1 = KPF1.from_fits(tmp_path)
        finally:
            os.unlink(tmp_path)
        kpf2 = l1.to_kpf2()
        DiagL2(kpf2).run()
        assert 'ZEROFRAC' not in kpf2.headers['PRIMARY']
