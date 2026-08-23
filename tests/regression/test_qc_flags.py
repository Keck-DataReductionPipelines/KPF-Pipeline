"""Tests for the KPF QC framework (quality_control/qc_flags).

All tests use synthetic in-memory data -- no real KPF files required. The qc.py
CLI smoke tests live in test_qc_script.py.
"""

import logging
import os
import types

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.quality_control.qc_flags.base import QC
from kpfpipe.quality_control.qc_flags.level0 import QCL0
from kpfpipe.quality_control.qc_flags.level1 import QCL1
from kpfpipe.quality_control.qc_flags.level2 import QCL2
from kpfpipe.quality_control.qc_flags.level4 import QCL4

from ._data_models import (
    GOOD_DATES,
    NORDER,
    NORDER_TOTAL,
    make_l4,
    seed_required_primary,
    set_fiber_arrays,
    write_amp_l0,
)

_NCOLS = 10  # small column count for fast tests


# ---------------------------------------------------------------------------
# Helpers: build minimal synthetic KPF objects in memory
# ---------------------------------------------------------------------------


def _make_kpf0(
    tmp_path,
    *,
    with_amps=True,
    exptime=60.0,
    obs_id="KP.20240405.00001.00",
    dates=None,
    expmeter=None,
    imtype="Object",
):
    """Minimal 4-amp KPF0 object with required headers.

    ``dates`` seeds raw DATE-BEG/MID/END/ELAPSED cards on PRIMARY. ``expmeter``
    maps an EXPMETER_SCI/SKY extension name to its table; without it the frame has
    no EM data at all, like a calibration. ``imtype`` sets PRIMARY IMTYPE
    ('Object' is a science frame, anything else a calibration).
    """
    fn = write_amp_l0(
        tmp_path / f"{obs_id}.fits",
        namps=4 if with_amps else 0,
        shape=(10, 10),
        primary_cards={
            "DATE-OBS": "2024-04-05T01:00:37",
            "EXPTIME": exptime,
            "IMTYPE": imtype,
            **(dates or {}),
        },
        extra_hdus=[
            fits.BinTableHDU(table, name=name)
            for name, table in (expmeter or {}).items()
        ],
    )
    return KPF0.from_fits(fn)


# Clean exposure-meter flux for the EMFLUXOK tests: 4 readings x 25 wavelength
# channels (more than the 20-channel negative run the check looks for).
_EM_CLEAN_FLUX = np.full((4, 25), 1000.0)


def _make_kpf1(
    tmp_path,
    *,
    with_rn=True,
    oscansub=True,
    biassub=True,
    darksub=True,
    flatdiv=True,
    agebias=1.0,
    agedark=5.0,
    ageflat=10.0,
    finite_ccd=True,
    shape=(20, 20),
):
    """Minimal KPF1 with all L1 QC-relevant headers.

    QC-relevant keywords live on their registry-home extensions, not PRIMARY: the
    applied-step flags (OSCANSUB/BIASSUB/DARKSUB/FLATDIV) on RECEIPT, and the
    read-noise and master-age keywords on QUALITY_CONTROL. They are seeded on the
    loaded object after ``from_fits`` returns.
    """
    fn = str(tmp_path / "kpf_L1_20240405T010037.fits")
    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-04-05T01:00:37"
    primary.header["EXPTIME"] = 300.0

    hdus = [primary]
    for chip in ["GREEN", "RED"]:
        data = np.ones(shape, dtype=np.float32)
        if not finite_ccd:
            data[0, 0] = np.nan
        hdus.append(fits.ImageHDU(data=data, name=f"{chip}_CCD"))
        hdus.append(fits.ImageHDU(data=data, name=f"{chip}_VAR"))

    fits.HDUList(hdus).writeto(fn, overwrite=True)
    l1 = KPF1.from_fits(fn)

    receipt = l1.headers["RECEIPT"]
    receipt["OSCANSUB"] = (oscansub, "Overscan subtraction applied")
    receipt["BIASSUB"] = (biassub, "Bias subtraction applied")
    receipt["DARKSUB"] = (darksub, "Dark subtraction applied")
    receipt["FLATDIV"] = (flatdiv, "Flat division applied")

    qc = l1.headers["QUALITY_CONTROL"]
    qc["BIASAGE"] = (agebias, "Age of bias master [days]")
    qc["DARKAGE"] = (agedark, "Age of dark master [days]")
    qc["FLATAGE"] = (ageflat, "Age of flat master [days]")

    if with_rn:
        for i in range(1, 5):
            qc[f"RNGREEN{i}"] = (3.5, "RN e-")
            qc[f"RNRED{i}"] = (4.0, "RN e-")
            qc[f"RNNGGR{i}"] = (1.0, "RNNG")
            qc[f"RNNGRD{i}"] = (1.0, "RNNG")

    # KPF1.__init__ seeds the EPRV skeleton, but KPF1._read replaces PRIMARY with
    # the file's sparse header, so a synthetic from_fits L1 would fail KWRDPRL1.
    seed_required_primary(l1, QCL1)
    return l1


def _make_kpf2_nan_headers(*, nan_frac=0.0, zero_frac=0.1, missing_ext=None):
    """Minimal KPF2 with all 10 {CHIP}_{FIBER}_FLUX extensions populated.

    The arrays stay clean; the NaN counts and zero fraction are written as HEADERS,
    because QCL2 reads them from the header rather than measuring pixels. Not the
    same as test_diagnostics.py's ``_make_kpf2_nan_pixels``, which injects real
    NaN/zero PIXELS for DiagL2 to measure. Do not merge them.

    Per-chip row counts must match NORDER_GREEN/NORDER_RED because KPF2's
    chip-prefix __setitem__ rejects any other shape. ``nan_frac`` is the fraction
    of total pixels reported NaN via the NANSCI* headers, ``zero_frac`` the value
    written to ZEROFRAC, and ``missing_ext`` a chip_fiber key to leave empty.
    """
    chips = ["GREEN", "RED"]
    fibers = ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]
    ncols = _NCOLS

    kpf2 = KPF2()

    total_pixels = 0
    for chip in chips:
        nrows = NORDER[chip]
        for fiber in fibers:
            ext = f"{chip}_{fiber}_FLUX"
            if ext == missing_ext:
                continue
            arr = np.ones((nrows, ncols), dtype=np.float32)
            kpf2.set_data(ext, arr)
            total_pixels += nrows * ncols

    nan_count = int(nan_frac * total_pixels / 5)  # spread evenly across 5 nan keys
    for k in ["NANSCI1", "NANSCI2", "NANSCI3", "NANSKY", "NANCAL"]:
        kpf2.headers["QUALITY_CONTROL"][k] = (nan_count, f"NaN count {k}")
    kpf2.headers["QUALITY_CONTROL"]["ZEROFRAC"] = (zero_frac, "Zero-flux fraction")

    return kpf2


# ---------------------------------------------------------------------------
# QC base class runner
# ---------------------------------------------------------------------------


class TestQCBase:
    """Runner behaviour: aggregation, failure, raises, empty."""

    def _make_obj(self):
        """Minimal object with a headers dict and a set_keyword router.

        QC.run() reads each check's comment off ``keyword_registry.routing`` and
        derives ISGOOD as the AND over the ``keyword_registry.qc_flag_keywords``
        present on QUALITY_CONTROL, so the stub declares the synthetic check keys
        as the QC-flag set and stores every keyword on QUALITY_CONTROL.
        """
        qc_keys = frozenset({"CHECKA", "CHECKB", "CHKOK", "CHKFAIL", "FLAG", "ISGOOD"})

        class _FakeObj:
            headers = {"PRIMARY": {}, "QUALITY_CONTROL": {}}
            keyword_registry = types.SimpleNamespace(
                routing={k: ("QUALITY_CONTROL", "") for k in qc_keys},
                qc_flag_keywords=qc_keys,
            )

            def set_keyword(self, key, value):
                self.headers["QUALITY_CONTROL"][key] = value

        return _FakeObj()

    def test_all_passing_isgood_1(self):
        obj = self._make_obj()

        class MyQC(QC):
            def check_a(self):
                return True

            check_a._qc_key = "CHECKA"

            def check_b(self):
                return True

            check_b._qc_key = "CHECKB"

        results = MyQC(obj).run()
        assert obj.headers["QUALITY_CONTROL"]["ISGOOD"] == 1
        assert obj.headers["QUALITY_CONTROL"]["CHECKA"] == 1
        assert obj.headers["QUALITY_CONTROL"]["CHECKB"] == 1
        assert results["CHECKA"][0] is True
        assert results["CHECKB"][0] is True

    def test_one_failing_isgood_0(self):
        obj = self._make_obj()

        class MyQC(QC):
            def check_ok(self):
                return True

            check_ok._qc_key = "CHKOK"

            def check_fail(self):
                return False

            check_fail._qc_key = "CHKFAIL"

        MyQC(obj).run()
        assert obj.headers["QUALITY_CONTROL"]["ISGOOD"] == 0
        assert obj.headers["QUALITY_CONTROL"]["CHKOK"] == 1
        assert obj.headers["QUALITY_CONTROL"]["CHKFAIL"] == 0

    def test_raising_check_propagates_and_logs(self, caplog):
        obj = self._make_obj()

        class MyQC(QC):
            LEVEL = "L0"

            def check_boom(self):
                raise ValueError("boom!")

            check_boom._qc_key = "BOOM"

        # Fail-fast: the exception propagates unwrapped, logged at ERROR.
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="boom!"):
                MyQC(obj).run()
        assert "QC check 'check_boom' raised" in caplog.text

    def test_empty_subclass_isgood_1(self):
        obj = self._make_obj()

        class EmptyQC(QC):
            pass

        results = EmptyQC(obj).run()
        assert results == {}
        assert obj.headers["QUALITY_CONTROL"]["ISGOOD"] == 1

    def test_repeated_run_resets_results(self):
        # Without the per-run reset, the first run's failed result would linger in
        # self.results and ISGOOD would stay 0 once the check starts passing.
        obj = self._make_obj()
        obj.flag = False

        class MyQC(QC):
            def check_flag(self):
                return self.kpf_obj.flag

            check_flag._qc_key = "FLAG"

        qc = MyQC(obj)
        qc.run()
        assert obj.headers["QUALITY_CONTROL"]["ISGOOD"] == 0
        assert list(qc.results) == ["FLAG"]
        assert qc.results["FLAG"][0] is False

        obj.flag = True
        qc.run()
        assert obj.headers["QUALITY_CONTROL"]["ISGOOD"] == 1
        assert list(qc.results) == ["FLAG"]
        assert qc.results["FLAG"][0] is True

    def test_run_logs_pass_debug_fail_warning(self, caplog):
        obj = self._make_obj()

        class MyQC(QC):
            LEVEL = "L2"

            def check_ok(self):
                return True

            check_ok._qc_key = "CHKOK"

            def check_fail(self):
                return False

            check_fail._qc_key = "CHKFAIL"

        with caplog.at_level(logging.DEBUG):
            MyQC(obj).run()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("CHKFAIL = 0" in r.getMessage() for r in warnings)
        assert any("CHKOK = 1" in r.getMessage() for r in debugs)
        # A passing flag never warns.
        assert not any("CHKOK" in r.getMessage() for r in warnings)

    def test_hdr_float_absent_empty_none_corrupt_raises(self):
        # Absent and valueless cards degrade to None so checks can skip or fail
        # gracefully; a present-but-non-numeric card is malformed and raises.
        hdr = fits.Header()
        hdr["NUM"] = 3.5
        hdr["EMPTY"] = None  # a valueless card
        hdr["STR"] = "not-a-number"
        assert QC._hdr_float(hdr, "NUM") == 3.5
        assert QC._hdr_float(hdr, "MISSING") is None
        assert QC._hdr_float(hdr, "EMPTY") is None
        with pytest.raises(ValueError, match="could not convert string to float"):
            QC._hdr_float(hdr, "STR")


# ---------------------------------------------------------------------------
# QCL0 checks
# ---------------------------------------------------------------------------


class TestQCL0:
    def test_data_l0_red_green_pass(self, tmp_path):
        l0 = _make_kpf0(tmp_path, with_amps=True)
        assert QCL0(l0).data_l0_red_green() is True

    def test_data_l0_red_green_fail_missing(self, tmp_path):
        l0 = _make_kpf0(tmp_path, with_amps=False)
        assert QCL0(l0).data_l0_red_green() is False

    def test_data_l0_red_green_fail_empty(self, tmp_path):
        fn = str(tmp_path / "KP.20240405.00002.00.fits")
        primary = fits.PrimaryHDU()
        primary.header["DATE-OBS"] = "2024-04-05T01:00:37"
        primary.header["OFNAME"] = os.path.basename(fn)
        primary.header["PROGNAME"] = "K123"
        hdus = [primary]
        for chip in ["GREEN", "RED"]:
            for amp in range(1, 5):
                # data=None: KPF0 stores array(None, dtype=object), treated as absent.
                hdus.append(fits.ImageHDU(data=None, name=f"{chip}_AMP{amp}"))
        fits.HDUList(hdus).writeto(fn, overwrite=True)
        l0 = KPF0.from_fits(fn)
        assert QCL0(l0).data_l0_red_green() is False

    def test_data_l0_red_green_pass_two_amp(self, tmp_path):
        # 2-amp readout (AMP1/AMP2 only) is the truth-frame layout and must pass.
        fn = str(tmp_path / "KP.20240405.00003.00.fits")
        primary = fits.PrimaryHDU()
        primary.header["DATE-OBS"] = "2024-04-05T01:00:37"
        primary.header["OFNAME"] = os.path.basename(fn)
        primary.header["PROGNAME"] = "K123"
        hdus = [primary]
        for chip in ["GREEN", "RED"]:
            for amp in (1, 2):
                data = np.ones((10, 10), dtype=np.float32)
                hdus.append(fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}"))
        fits.HDUList(hdus).writeto(fn, overwrite=True)
        l0 = KPF0.from_fits(fn)
        assert QCL0(l0).data_l0_red_green() is True

    def test_data_l0_red_green_fail_partial_amp(self, tmp_path):
        fn = str(tmp_path / "KP.20240405.00004.00.fits")
        primary = fits.PrimaryHDU()
        primary.header["DATE-OBS"] = "2024-04-05T01:00:37"
        primary.header["OFNAME"] = os.path.basename(fn)
        primary.header["PROGNAME"] = "K123"
        hdus = [primary]
        for chip in ["GREEN", "RED"]:
            for amp in (1, 2, 3):  # 3 amps -> not a valid 2/4-amp readout
                data = np.ones((10, 10), dtype=np.float32)
                hdus.append(fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}"))
        fits.HDUList(hdus).writeto(fn, overwrite=True)
        l0 = KPF0.from_fits(fn)
        assert QCL0(l0).data_l0_red_green() is False

    def test_header_keywords_present_pass(self, tmp_path):
        l0 = _make_kpf0(tmp_path)
        assert QCL0(l0).header_keywords_present() is True

    def test_header_keywords_present_fail(self, tmp_path):
        l0 = _make_kpf0(tmp_path)
        del l0.headers["PRIMARY"]["OFNAME"]
        assert QCL0(l0).header_keywords_present() is False

    def test_times_consistent_pass(self, tmp_path):
        l0 = _make_kpf0(tmp_path, dates=GOOD_DATES)
        assert QCL0(l0).times_consistent() is True

    def test_times_out_of_order_fails(self, tmp_path):
        bad = dict(GOOD_DATES, **{"DATE-MID": "2024-09-23T09:12:25.000"})  # mid > end
        assert QCL0(_make_kpf0(tmp_path, dates=bad)).times_consistent() is False

    def test_times_elapsed_mismatch_fails(self, tmp_path):
        bad = dict(GOOD_DATES, ELAPSED=99.0)  # END-BEG != ELAPSED
        assert QCL0(_make_kpf0(tmp_path, dates=bad)).times_consistent() is False

    def test_times_elapsed_missing_raises(self, tmp_path):
        missing = {k: v for k, v in GOOD_DATES.items() if k != "ELAPSED"}
        with pytest.raises(KeyError):
            QCL0(_make_kpf0(tmp_path, dates=missing)).times_consistent()

    def test_times_missing_keys_fail(self, tmp_path):
        # Raw L0 PRIMARY without DATE-BEG/MID/END -> cannot verify -> fail.
        assert QCL0(_make_kpf0(tmp_path)).times_consistent() is False

    @pytest.mark.parametrize("key", ["GRDATE-B", "GRDATE-E", "RDDATE-B", "RDDATE-E"])
    def test_times_shutter_offset_fails(self, tmp_path, key):
        # 0.5 s off the window edge it bounds, past the 0.1 s tolerance.
        bad = dict(GOOD_DATES, **{key: "2024-09-23T09:12:30.000"})
        assert QCL0(_make_kpf0(tmp_path, dates=bad)).times_consistent() is False

    @pytest.mark.parametrize("key", ["GRDATE-B", "GRDATE-E", "RDDATE-B", "RDDATE-E"])
    def test_times_shutter_missing_fails(self, tmp_path, key):
        missing = {k: v for k, v in GOOD_DATES.items() if k != key}
        assert QCL0(_make_kpf0(tmp_path, dates=missing)).times_consistent() is False

    def test_times_shutter_within_tolerance_passes(self, tmp_path):
        near = dict(GOOD_DATES, **{"GRDATE-B": "2024-09-23T09:12:09.534"})  # +50 ms
        assert QCL0(_make_kpf0(tmp_path, dates=near)).times_consistent() is True

    def test_timechk_key_present(self):
        assert QCL0.__dict__["times_consistent"]._qc_key == "DATTIMOK"

    def test_ntp_timing_pass(self, tmp_path):
        assert QCL0(_make_kpf0(tmp_path)).ntp_timing() is True

    @pytest.mark.parametrize(
        "timeerr",
        [
            "NTP time correct to within 100.0 ms",  # at the limit
            "NTP time correct to within 250.0 ms",
            "NTP is not synchronised",  # unparseable
            None,  # absent
        ],
    )
    def test_ntp_timing_fail(self, tmp_path, timeerr):
        l0 = _make_kpf0(tmp_path)
        if timeerr is None:
            del l0.headers["PRIMARY"]["TIMEERR"]
        else:
            l0.headers["PRIMARY"]["TIMEERR"] = timeerr
        assert QCL0(l0).ntp_timing() is False

    def test_ntp_key_present(self):
        assert QCL0.__dict__["ntp_timing"]._qc_key == "NTPOK"

    def test_exptime_sane_pass_positive(self, tmp_path):
        l0 = _make_kpf0(tmp_path, exptime=300.0)
        assert QCL0(l0).exptime_sane() is True

    def test_exptime_sane_pass_zero(self, tmp_path):
        # Bias frames legitimately have EXPTIME=0.
        l0 = _make_kpf0(tmp_path, exptime=0.0)
        assert QCL0(l0).exptime_sane() is True

    def test_exptime_sane_fail_negative(self, tmp_path):
        l0 = _make_kpf0(tmp_path, exptime=-1.0)
        assert QCL0(l0).exptime_sane() is False

    def test_exptime_sane_fail_missing(self, tmp_path):
        l0 = _make_kpf0(tmp_path)
        del l0.headers["PRIMARY"]["EXPTIME"]
        assert QCL0(l0).exptime_sane() is False

    def test_exptime_sane_pass_elapsed_within_tol(self, tmp_path):
        # ELAPSED may exceed the requested EXPTIME by up to the timing tolerance.
        l0 = _make_kpf0(tmp_path, exptime=300.0, dates={"ELAPSED": 300.05})
        assert QCL0(l0).exptime_sane() is True

    def test_exptime_sane_pass_elapsed_absent(self, tmp_path):
        # No ELAPSED card -> the consistency comparison is skipped.
        l0 = _make_kpf0(tmp_path, exptime=300.0)
        assert "ELAPSED" not in l0.headers["PRIMARY"]
        assert QCL0(l0).exptime_sane() is True

    def test_exptime_sane_fail_premature_readout(self, tmp_path):
        # ELAPSED shorter than the requested EXPTIME (premature readout).
        l0 = _make_kpf0(tmp_path, exptime=300.0, dates={"ELAPSED": 299.0})
        assert QCL0(l0).exptime_sane() is False

    def test_exptime_sane_fail_elapsed_exceeds_tol(self, tmp_path):
        # ELAPSED over the requested EXPTIME by more than the tolerance.
        l0 = _make_kpf0(tmp_path, exptime=300.0, dates={"ELAPSED": 301.0})
        assert QCL0(l0).exptime_sane() is False

    # not_junk delegates all file I/O to load_junk_obs_ids, so these tests
    # monkeypatch it (no junk file is ever written to a data tree) and cover only
    # not_junk's own logic. The CSV parsing is covered in test_masters_recipe.py.
    def test_not_junk_pass_empty_list(self, tmp_path, monkeypatch):
        # An empty set is what an absent junk list on disk yields.
        import kpfpipe.quality_control.qc_flags.level0 as mod

        monkeypatch.setattr(mod, "load_junk_obs_ids", lambda data_input: set())
        l0 = _make_kpf0(tmp_path)
        l0.dirname = str(tmp_path / "L0" / "20240405")
        assert QCL0(l0).not_junk() is True

    def test_not_junk_pass_not_in_list(self, tmp_path, monkeypatch):
        import kpfpipe.quality_control.qc_flags.level0 as mod

        monkeypatch.setattr(
            mod, "load_junk_obs_ids", lambda data_input: {"KP.20240101.99999.00"}
        )
        l0 = _make_kpf0(tmp_path)
        l0.dirname = str(tmp_path / "L0" / "20240405")
        assert QCL0(l0).not_junk() is True

    def test_not_junk_fail_in_list(self, tmp_path, monkeypatch):
        import kpfpipe.quality_control.qc_flags.level0 as mod

        obs_id = "KP.20240405.00001.00"
        monkeypatch.setattr(mod, "load_junk_obs_ids", lambda data_input: {obs_id})
        l0 = _make_kpf0(tmp_path, obs_id=obs_id)
        l0.dirname = str(tmp_path / "L0" / "20240405")
        assert QCL0(l0).not_junk() is False

    def test_not_junk_recovers_data_input_from_dirname(self, tmp_path, monkeypatch):
        # data_input is the L0 dir's grandparent: {root}/L0/{datecode} -> {root}.
        import kpfpipe.quality_control.qc_flags.level0 as mod

        seen = {}

        def _fake(data_input):
            seen["data_input"] = data_input
            return set()

        monkeypatch.setattr(mod, "load_junk_obs_ids", _fake)
        l0 = _make_kpf0(tmp_path)
        l0.dirname = str(tmp_path / "L0" / "20240405")
        QCL0(l0).not_junk()
        assert seen["data_input"] == str(tmp_path)

    def test_not_junk_pass_none_obs_id(self, tmp_path, monkeypatch):
        # obs_id=None is on no junk list, so an unresolved obs_id passes here.
        import kpfpipe.quality_control.qc_flags.level0 as mod

        monkeypatch.setattr(
            mod, "load_junk_obs_ids", lambda data_input: {"KP.20240101.99999.00"}
        )
        l0 = _make_kpf0(tmp_path)
        l0.dirname = str(tmp_path / "L0" / "20240405")
        l0.obs_id = None
        assert QCL0(l0).not_junk() is True

    def test_not_junk_raises_on_unknown_dirname(self, tmp_path):
        # dirname is set on every L0 read, so an absent one is a broken upstream
        # invariant and must fail loud rather than silently pass.
        l0 = _make_kpf0(tmp_path)
        l0.dirname = None
        with pytest.raises(TypeError, match="NoneType"):
            QCL0(l0).not_junk()

    def test_not_junk_key_present(self):
        qc = QCL0.__dict__["not_junk"]
        assert qc._qc_key == "NOTJUNK"

    def test_radec_consistent_pass(self, tmp_path):
        l0 = _make_kpf0(tmp_path)
        for k, v in (("TCSOFF", 0.02), ("OBJOFF", 2.3), ("GAIAOFF", 2.3)):
            l0.set_keyword(k, v)  # routes to QUALITY_CONTROL
        assert QCL0(l0).radec_consistent() is True

    def test_radec_gaiaoff_fail(self, tmp_path):
        # The 20240816 signature: pointing/target/OBJECT fine, GAIAID ~25 deg off.
        l0 = _make_kpf0(tmp_path)
        for k, v in (("TCSOFF", 0.02), ("OBJOFF", 2.3), ("GAIAOFF", 91004.96)):
            l0.set_keyword(k, v)
        assert QCL0(l0).radec_consistent() is False

    def test_radec_tcsoff_fail(self, tmp_path):
        l0 = _make_kpf0(tmp_path)
        l0.set_keyword("TCSOFF", 1.5)  # > 1" pointing-vs-target
        assert QCL0(l0).radec_consistent() is False

    def test_radec_objoff_fail(self, tmp_path):
        l0 = _make_kpf0(tmp_path)
        l0.set_keyword("TCSOFF", 0.02)  # internal pointing OK
        l0.set_keyword("OBJOFF", 6.0)  # > 5" pointing-vs-OBJECT
        assert QCL0(l0).radec_consistent() is False

    def test_radec_tcsoff_required_empty_fails(self, tmp_path):
        # TCSOFF is internal pointing consistency: absent or present-but-empty fails.
        l0 = _make_kpf0(tmp_path)
        assert QCL0(l0).radec_consistent() is False
        l0.set_keyword("TCSOFF", None)  # present-but-empty (astrometry unavailable)
        assert QCL0(l0).radec_consistent() is False

    def test_radec_external_offsets_optional(self, tmp_path):
        # TCSOFF within budget; GAIAOFF/OBJOFF unavailable (empty) -> still pass.
        l0 = _make_kpf0(tmp_path)
        l0.set_keyword("TCSOFF", 0.02)
        l0.set_keyword("GAIAOFF", None)
        l0.set_keyword("OBJOFF", None)
        assert QCL0(l0).radec_consistent() is True

    @pytest.mark.parametrize("imtype", ["Bias", "Dark", "Flatlamp", "Arclamp"])
    def test_radec_calibration_frames_pass(self, tmp_path, imtype):
        # A calibration frame has no target, so its blank TCSOFF must not fail.
        l0 = _make_kpf0(tmp_path, imtype=imtype)
        assert QCL0(l0).radec_consistent() is True

    def test_radec_calibration_ignores_offsets(self, tmp_path):
        # Even a badly-off offset on a calibration frame is not a pointing fault.
        l0 = _make_kpf0(tmp_path, imtype="Bias")
        for k, v in (("TCSOFF", 1.5), ("OBJOFF", 6.0), ("GAIAOFF", 91004.96)):
            l0.set_keyword(k, v)
        assert QCL0(l0).radec_consistent() is True

    def test_radecok_key_present(self):
        assert QCL0.__dict__["radec_consistent"]._qc_key == "RADECOK"

    # --- catalog_astrometry_sane (ASTROMOK): physical range of the canonical
    # CATALOG_RECORD astrometry.

    def _make_kpf0_with_canonical(self, tmp_path, **overrides):
        """KPF0 whose CATALOG_RECORD holds a merged 'kpf-drp' row.

        Overrides patch individual fields of a physically-sane base.
        """
        from kpfpipe.modules.astro_query import AstroQuery

        l0 = _make_kpf0(tmp_path)
        record = {
            "object": "test",
            "ra": "12:00:00.00",
            "dec": "+40:00:00.0",
            "pmra": 0.1,
            "pmdec": -0.2,
            "parallax": 100.0,
            "rv": 10.0,
            "frame": "icrs",
            "epoch": 2016.0,
            "equinox": 2000.0,
            "color": 0.823,  # solar Gaia BP-RP
            "color_name": "Gaia BP-RP",
        }
        record.update(overrides)
        AstroQuery(l0)._write_catalog_record("kpf-drp", record)
        return l0

    def test_catalog_astrometry_sane_pass(self, tmp_path):
        l0 = self._make_kpf0_with_canonical(tmp_path)
        assert QCL0(l0).catalog_astrometry_sane() is True

    def test_catalog_astrometry_sane_no_record_passes(self, tmp_path):
        # CATALOG_RECORD is empty when AstroQuery never ran; value sanity is then
        # N/A and presence is enforced elsewhere.
        l0 = _make_kpf0(tmp_path)
        assert not l0.data["CATALOG_RECORD"].colnames
        assert QCL0(l0).catalog_astrometry_sane() is True

    def test_catalog_astrometry_sane_fail_epoch_zero(self, tmp_path):
        # A WMKO TARGEPOC=0.0 placeholder clears the merge's is-not-None gate, so
        # only the range check catches it.
        l0 = self._make_kpf0_with_canonical(tmp_path, epoch=0.0)
        assert QCL0(l0).catalog_astrometry_sane() is False

    def test_catalog_astrometry_sane_fail_epoch_high(self, tmp_path):
        l0 = self._make_kpf0_with_canonical(tmp_path, epoch=2100.0)
        assert QCL0(l0).catalog_astrometry_sane() is False

    def test_catalog_astrometry_sane_epoch_boundaries(self, tmp_path):
        # Window is (1950, 2050]: 1950 fails (exclusive low), 2050 passes (inclusive).
        assert (
            QCL0(
                self._make_kpf0_with_canonical(tmp_path, epoch=1950.0)
            ).catalog_astrometry_sane()
            is False
        )
        assert (
            QCL0(
                self._make_kpf0_with_canonical(tmp_path, epoch=2050.0)
            ).catalog_astrometry_sane()
            is True
        )

    def test_catalog_astrometry_sane_fail_equinox_out_of_range(self, tmp_path):
        l0 = self._make_kpf0_with_canonical(tmp_path, equinox=1900.0)
        assert QCL0(l0).catalog_astrometry_sane() is False

    def test_catalog_astrometry_sane_rv_high_but_in_bound_passes(self, tmp_path):
        # A fast star: |rv| = 150 is well within the 350 km/s bound (Chubak 2012).
        l0 = self._make_kpf0_with_canonical(tmp_path, rv=150.0)
        assert QCL0(l0).catalog_astrometry_sane() is True

    def test_catalog_astrometry_sane_fail_rv_too_large(self, tmp_path):
        l0 = self._make_kpf0_with_canonical(tmp_path, rv=400.0)
        assert QCL0(l0).catalog_astrometry_sane() is False

    def test_catalog_astrometry_sane_rv_absent_passes(self, tmp_path):
        # No catalog rv (NaN cell) -> the rv bound is skipped (check-when-present).
        l0 = self._make_kpf0_with_canonical(tmp_path, rv=None)
        assert QCL0(l0).catalog_astrometry_sane() is True

    def test_catalog_astrometry_sane_fail_parallax_negative(self, tmp_path):
        # A negative Gaia parallax is routine for faint sources but nonphysical.
        l0 = self._make_kpf0_with_canonical(tmp_path, parallax=-0.3)
        assert QCL0(l0).catalog_astrometry_sane() is False

    def test_catalog_astrometry_sane_fail_parallax_zero(self, tmp_path):
        l0 = self._make_kpf0_with_canonical(tmp_path, parallax=0.0)
        assert QCL0(l0).catalog_astrometry_sane() is False

    def test_catalog_astrometry_sane_fail_parallax_too_large(self, tmp_path):
        # >= 1000 mas is nearer than 1 pc, which nothing observable is.
        l0 = self._make_kpf0_with_canonical(tmp_path, parallax=1000.0)
        assert QCL0(l0).catalog_astrometry_sane() is False

    def test_catalog_astrometry_sane_high_pm_in_bound_passes(self, tmp_path):
        # Barnard's Star (~10.4"/yr, the highest real PM) is within the 15"/yr bound.
        l0 = self._make_kpf0_with_canonical(tmp_path, pmra=10.4, pmdec=-8.0)
        assert QCL0(l0).catalog_astrometry_sane() is True

    def test_catalog_astrometry_sane_fail_pmra_too_large(self, tmp_path):
        # A corrupt PM would misplace the source at the obs epoch.
        l0 = self._make_kpf0_with_canonical(tmp_path, pmra=25.0)
        assert QCL0(l0).catalog_astrometry_sane() is False

    def test_catalog_astrometry_sane_fail_pmdec_too_large(self, tmp_path):
        l0 = self._make_kpf0_with_canonical(tmp_path, pmdec=-30.0)
        assert QCL0(l0).catalog_astrometry_sane() is False

    def test_astromok_key_present(self):
        assert QCL0.__dict__["catalog_astrometry_sane"]._qc_key == "ASTROMOK"

    # --- catalog_color_sane (COLOROK): the canonical color must be present,
    # labeled, and on the dwarf sequence for CrossCorrelation to turn it into a
    # line-mask Teff.

    @pytest.mark.parametrize(
        ("color", "color_name"),
        [(0.823, "Gaia BP-RP"), (0.650, "B-V"), (1.035, "G-J")],
    )
    def test_catalog_color_sane_pass(self, tmp_path, color, color_name):
        l0 = self._make_kpf0_with_canonical(
            tmp_path, color=color, color_name=color_name
        )
        assert QCL0(l0).catalog_color_sane() is True

    def test_catalog_color_sane_no_record_passes(self, tmp_path):
        # Calibration frame: no target, so no color to check.
        l0 = _make_kpf0(tmp_path)
        assert not l0.data["CATALOG_RECORD"].colnames
        assert QCL0(l0).catalog_color_sane() is True

    def test_catalog_color_sane_fail_color_absent(self, tmp_path):
        # All three catalogs lacked a usable magnitude pair -> NaN cell.
        l0 = self._make_kpf0_with_canonical(tmp_path, color=None, color_name=None)
        assert QCL0(l0).catalog_color_sane() is False

    def test_catalog_color_sane_fail_unlabeled_color(self, tmp_path):
        # A color with no label cannot be placed on any sequence.
        l0 = self._make_kpf0_with_canonical(tmp_path, color_name=None)
        assert QCL0(l0).catalog_color_sane() is False

    def test_catalog_color_sane_fail_unrecognized_name(self, tmp_path):
        l0 = self._make_kpf0_with_canonical(tmp_path, color=1.0, color_name="V-Ks")
        assert QCL0(l0).catalog_color_sane() is False

    @pytest.mark.parametrize(
        ("color", "color_name"),
        [(-1.0, "B-V"), (3.0, "B-V"), (5.5, "Gaia BP-RP"), (-0.5, "G-J")],
    )
    def test_catalog_color_sane_fail_out_of_range(self, tmp_path, color, color_name):
        l0 = self._make_kpf0_with_canonical(
            tmp_path, color=color, color_name=color_name
        )
        assert QCL0(l0).catalog_color_sane() is False

    def test_catalog_color_sane_bounds_are_inclusive(self, tmp_path):
        # The endpoints are real tabulated stars (O3V, M9V), so they must pass.
        for color in (-0.33, 2.17):
            l0 = self._make_kpf0_with_canonical(tmp_path, color=color, color_name="B-V")
            assert QCL0(l0).catalog_color_sane() is True

    def test_colorok_key_present(self):
        assert QCL0.__dict__["catalog_color_sane"]._qc_key == "COLOROK"

    def test_dataprl0_key_and_comment(self):
        fn = QCL0.__dict__["data_l0_red_green"]
        assert fn._qc_key == "DATAPRL0"
        # The comment lives in the registry Description, not on the method.
        assert "GREEN" in KPF0.keyword_registry.routing["DATAPRL0"][1]

    # --- expmeter_times_consistent (EMTIMEOK) and expmeter_flux_sane (EMFLUXOK).
    # Four readings tile the GOOD_DATES shutter window.

    _EM_BEGS = [
        "2024-09-23T09:12:09.484",
        "2024-09-23T09:12:12.484",
        "2024-09-23T09:12:15.484",
        "2024-09-23T09:12:18.484",
    ]
    _EM_ENDS = [
        "2024-09-23T09:12:12.484",
        "2024-09-23T09:12:15.484",
        "2024-09-23T09:12:18.484",
        "2024-09-23T09:12:21.554",
    ]

    def _make_kpf0_with_expmeter(
        self,
        tmp_path,
        *,
        begs=None,
        ends=None,
        flux=None,
        sky_flux=None,
        corrected=False,
    ):
        """KPF0 carrying an EXPMETER_SCI table, plus EXPMETER_SKY when ``sky_flux``
        is given.

        Built from per-reading timestamps and an (nreading, nchannel) flux array
        whose column labels are wavelengths.
        """
        suffix = "-Corr" if corrected else ""

        def table(values):
            columns = {
                f"Date-Beg{suffix}": begs or self._EM_BEGS,
                f"Date-End{suffix}": ends or self._EM_ENDS,
            }
            for i in range(values.shape[1]):
                columns[str(5000.0 + i)] = values[:, i]
            return Table(columns)

        extensions = {"EXPMETER_SCI": table(_EM_CLEAN_FLUX if flux is None else flux)}
        if sky_flux is not None:
            extensions["EXPMETER_SKY"] = table(sky_flux)
        return _make_kpf0(tmp_path, dates=GOOD_DATES, expmeter=extensions)

    def test_expmeter_times_consistent_pass(self, tmp_path):
        l0 = self._make_kpf0_with_expmeter(tmp_path)
        assert QCL0(l0).expmeter_times_consistent() is True

    def test_expmeter_times_no_em_data_passes(self, tmp_path):
        # Calibration frames carry no EM extension; there is nothing to check.
        l0 = _make_kpf0(tmp_path, dates=GOOD_DATES)
        assert l0.data.get("EXPMETER_SCI") is None
        assert QCL0(l0).expmeter_times_consistent() is True

    def test_expmeter_times_within_tolerance_passes(self, tmp_path):
        # Sub-second EM dead time is routine and must not trip the check.
        begs = ["2024-09-23T09:12:09.984"] + self._EM_BEGS[1:]
        l0 = self._make_kpf0_with_expmeter(tmp_path, begs=begs)
        assert QCL0(l0).expmeter_times_consistent() is True

    def test_expmeter_times_late_start_fails(self, tmp_path):
        begs = ["2024-09-23T09:12:14.484"] + self._EM_BEGS[1:]  # 5 s after DATE-BEG
        l0 = self._make_kpf0_with_expmeter(tmp_path, begs=begs)
        assert QCL0(l0).expmeter_times_consistent() is False

    def test_expmeter_times_early_end_fails(self, tmp_path):
        ends = self._EM_ENDS[:-1] + ["2024-09-23T09:12:16.554"]  # 5 s before DATE-END
        l0 = self._make_kpf0_with_expmeter(tmp_path, ends=ends)
        assert QCL0(l0).expmeter_times_consistent() is False

    def test_expmeter_times_prefers_corrected_columns(self, tmp_path):
        # Only the -Corr columns bracket the window, so plain ones would fail.
        l0 = self._make_kpf0_with_expmeter(tmp_path, corrected=True)
        assert "Date-Beg-Corr" in l0.data["EXPMETER_SCI"].colnames
        assert QCL0(l0).expmeter_times_consistent() is True

    def test_expmeter_times_missing_shutter_window_fails(self, tmp_path):
        # EM data present but no DATE-BEG/DATE-END to compare against.
        l0 = self._make_kpf0_with_expmeter(tmp_path)
        del l0.headers["PRIMARY"]["DATE-BEG"]
        assert QCL0(l0).expmeter_times_consistent() is False

    def test_emtimeok_key_present(self):
        assert QCL0.__dict__["expmeter_times_consistent"]._qc_key == "EMTIMEOK"

    def test_expmeter_flux_sane_pass(self, tmp_path):
        l0 = self._make_kpf0_with_expmeter(tmp_path, sky_flux=_EM_CLEAN_FLUX)
        assert QCL0(l0).expmeter_flux_sane() is True

    def test_expmeter_flux_no_em_data_passes(self, tmp_path):
        l0 = _make_kpf0(tmp_path, dates=GOOD_DATES)
        assert QCL0(l0).expmeter_flux_sane() is True

    def test_expmeter_flux_saturated_fails(self, tmp_path):
        # 2 channels saturated in each of the 2 interior readings -> 4 elements,
        # over the 1.5-per-reading allowance.
        flux = _EM_CLEAN_FLUX.copy()
        flux[1:3, :2] = 0.95 * 1.93e6
        l0 = self._make_kpf0_with_expmeter(tmp_path, flux=flux)
        assert QCL0(l0).expmeter_flux_sane() is False

    def test_expmeter_flux_saturated_edge_readings_pass(self, tmp_path):
        # The first and last readings are partial, so saturation there is dropped.
        flux = _EM_CLEAN_FLUX.copy()
        flux[[0, -1], :] = 0.95 * 1.93e6
        l0 = self._make_kpf0_with_expmeter(tmp_path, flux=flux)
        assert QCL0(l0).expmeter_flux_sane() is True

    def test_expmeter_flux_negative_run_fails(self, tmp_path):
        # 20 consecutive channels summing negative: bias over-subtraction.
        flux = _EM_CLEAN_FLUX.copy()
        flux[:, 5:25] = -1000.0
        l0 = self._make_kpf0_with_expmeter(tmp_path, flux=flux)
        assert QCL0(l0).expmeter_flux_sane() is False

    def test_expmeter_flux_short_negative_run_passes(self, tmp_path):
        # 19 consecutive is under the run length; isolated negatives are noise.
        flux = _EM_CLEAN_FLUX.copy()
        flux[:, 5:24] = -1000.0
        l0 = self._make_kpf0_with_expmeter(tmp_path, flux=flux)
        assert QCL0(l0).expmeter_flux_sane() is True

    def test_expmeter_flux_checks_sky_fiber(self, tmp_path):
        # The SKY fiber is checked too, so a bad SKY fails an otherwise clean frame.
        sky_flux = _EM_CLEAN_FLUX.copy()
        sky_flux[:, 5:25] = -1000.0
        l0 = self._make_kpf0_with_expmeter(tmp_path, sky_flux=sky_flux)
        assert QCL0(l0).expmeter_flux_sane() is False

    def test_emfluxok_key_present(self):
        assert QCL0.__dict__["expmeter_flux_sane"]._qc_key == "EMFLUXOK"


# ---------------------------------------------------------------------------
# QCL1 checks
# ---------------------------------------------------------------------------


class TestQCL1:
    def test_data_present_pass(self, tmp_path):
        l1 = _make_kpf1(tmp_path)
        assert QCL1(l1).data_present() is True

    def test_data_present_fail_missing(self, tmp_path):
        l1 = _make_kpf1(tmp_path)
        l1.data["GREEN_CCD"] = None
        assert QCL1(l1).data_present() is False

    def test_data_present_fail_empty(self, tmp_path):
        l1 = _make_kpf1(tmp_path)
        l1.data["RED_CCD"] = np.array([], dtype=np.float32)
        assert QCL1(l1).data_present() is False

    def test_required_keywords_present_pass(self, tmp_path):
        # _make_kpf1 seeds the full required-PRIMARY set.
        l1 = _make_kpf1(tmp_path)
        assert QCL1(l1).required_keywords_present() is True

    def test_required_keywords_present_fail_missing(self, tmp_path):
        l1 = _make_kpf1(tmp_path)
        del l1.headers["PRIMARY"]["INSTRUME"]  # a registry-required PRIMARY keyword
        assert QCL1(l1).required_keywords_present() is False

    def test_read_noise_ok_pass(self, tmp_path):
        l1 = _make_kpf1(tmp_path, with_rn=True)
        assert QCL1(l1).read_noise_ok() is True

    def test_read_noise_ok_fail_too_high(self, tmp_path):
        l1 = _make_kpf1(tmp_path, with_rn=True)
        l1.headers["QUALITY_CONTROL"]["RNGREEN1"] = (99.0, "bad RN")
        assert QCL1(l1).read_noise_ok() is False

    def test_read_noise_ok_fail_too_low(self, tmp_path):
        # The realistic failure signature: a broken or short-circuited RN
        # estimator returns ~0, not 99. Only the upper bound was ever crossed,
        # so rewriting the predicate as `v <= hi` left the whole suite green.
        l1 = _make_kpf1(tmp_path, with_rn=True)
        l1.headers["QUALITY_CONTROL"]["RNGREEN1"] = (0.5, "collapsed RN")
        assert QCL1(l1).read_noise_ok() is False

    def test_read_noise_ok_boundaries_pass(self, tmp_path):
        # The range is inclusive at both ends.
        l1 = _make_kpf1(tmp_path, with_rn=True)
        qc = l1.headers["QUALITY_CONTROL"]
        qc["RNGREEN1"] = (2.0, "lower bound")
        qc["RNRED1"] = (6.0, "upper bound")
        assert QCL1(l1).read_noise_ok() is True

    def test_read_noise_ok_fail_missing(self, tmp_path):
        l1 = _make_kpf1(tmp_path, with_rn=False)
        assert QCL1(l1).read_noise_ok() is False

    def test_read_noise_ok_pass_2amp(self, tmp_path):
        # 2-amp readout: only AMP1/AMP2 keys present (AMP3/4 absent).
        l1 = _make_kpf1(tmp_path, with_rn=True)
        for i in (3, 4):
            del l1.headers["QUALITY_CONTROL"][f"RNGREEN{i}"]
            del l1.headers["QUALITY_CONTROL"][f"RNRED{i}"]
        assert QCL1(l1).read_noise_ok() is True

    def test_read_noise_nongauss_ok_pass(self, tmp_path):
        l1 = _make_kpf1(tmp_path, with_rn=True)
        assert QCL1(l1).read_noise_nongauss_ok() is True

    def test_read_noise_nongauss_ok_fail_too_high(self, tmp_path):
        l1 = _make_kpf1(tmp_path, with_rn=True)
        l1.headers["QUALITY_CONTROL"]["RNNGGR1"] = (9.9, "bad RNNG")
        assert QCL1(l1).read_noise_nongauss_ok() is False

    def test_read_noise_nongauss_ok_fail_too_low(self, tmp_path):
        l1 = _make_kpf1(tmp_path, with_rn=True)
        l1.headers["QUALITY_CONTROL"]["RNNGGR1"] = (0.1, "collapsed RNNG")
        assert QCL1(l1).read_noise_nongauss_ok() is False

    def test_read_noise_nongauss_ok_fail_missing(self, tmp_path):
        l1 = _make_kpf1(tmp_path, with_rn=False)
        assert QCL1(l1).read_noise_nongauss_ok() is False

    def test_read_noise_nongauss_ok_pass_2amp(self, tmp_path):
        # 2-amp readout: only AMP1/AMP2 keys present (AMP3/4 absent).
        l1 = _make_kpf1(tmp_path, with_rn=True)
        for i in (3, 4):
            del l1.headers["QUALITY_CONTROL"][f"RNNGGR{i}"]
            del l1.headers["QUALITY_CONTROL"][f"RNNGRD{i}"]
        assert QCL1(l1).read_noise_nongauss_ok() is True

    def test_bias_ok_pass(self, tmp_path):
        l1 = _make_kpf1(tmp_path, biassub=True, agebias=3.0)
        assert QCL1(l1).bias_ok() is True

    def test_bias_ok_pass_future_dated_master(self, tmp_path):
        # DiagL1 writes a SIGNED age (master_dt - obs_dt), and the check uses
        # abs(), so a master processed after the science frame -- routine in the
        # masters pipeline -- is still in range. Dropping the abs() would start
        # failing BIASOK on every same-night master.
        l1 = _make_kpf1(tmp_path, biassub=True, agebias=-3.0)
        assert QCL1(l1).bias_ok() is True

    def test_bias_ok_boundary(self, tmp_path):
        # 7 days exactly is inside the gate; just past it is not.
        assert QCL1(_make_kpf1(tmp_path, agebias=7.0)).bias_ok() is True
        assert QCL1(_make_kpf1(tmp_path, agebias=7.5)).bias_ok() is False

    def test_bias_ok_fail_not_subtracted(self, tmp_path):
        l1 = _make_kpf1(tmp_path, biassub=False, agebias=3.0)
        assert QCL1(l1).bias_ok() is False

    def test_bias_ok_fail_subtract_flag_missing(self, tmp_path):
        l1 = _make_kpf1(tmp_path, agebias=3.0)
        del l1.headers["RECEIPT"]["BIASSUB"]
        assert QCL1(l1).bias_ok() is False

    def test_bias_ok_fail_too_old(self, tmp_path):
        l1 = _make_kpf1(tmp_path, biassub=True, agebias=10.0)
        assert QCL1(l1).bias_ok() is False

    def test_bias_ok_fail_age_missing(self, tmp_path):
        l1 = _make_kpf1(tmp_path, biassub=True)
        del l1.headers["QUALITY_CONTROL"]["BIASAGE"]
        assert QCL1(l1).bias_ok() is False

    def test_dark_ok_pass(self, tmp_path):
        l1 = _make_kpf1(tmp_path, darksub=True, agedark=7.0)
        assert QCL1(l1).dark_ok() is True

    def test_dark_ok_fail_not_subtracted(self, tmp_path):
        l1 = _make_kpf1(tmp_path, darksub=False, agedark=7.0)
        assert QCL1(l1).dark_ok() is False

    def test_dark_ok_fail_too_old(self, tmp_path):
        l1 = _make_kpf1(tmp_path, darksub=True, agedark=20.0)
        assert QCL1(l1).dark_ok() is False

    def test_dark_ok_fail_age_missing(self, tmp_path):
        l1 = _make_kpf1(tmp_path, darksub=True)
        del l1.headers["QUALITY_CONTROL"]["DARKAGE"]
        assert QCL1(l1).dark_ok() is False

    def test_flat_ok_pass(self, tmp_path):
        l1 = _make_kpf1(tmp_path, flatdiv=True, ageflat=15.0)
        assert QCL1(l1).flat_ok() is True

    def test_flat_ok_fail_not_divided(self, tmp_path):
        l1 = _make_kpf1(tmp_path, flatdiv=False, ageflat=15.0)
        assert QCL1(l1).flat_ok() is False

    def test_flat_ok_fail_too_old(self, tmp_path):
        l1 = _make_kpf1(tmp_path, flatdiv=True, ageflat=45.0)
        assert QCL1(l1).flat_ok() is False

    def test_flat_ok_fail_age_missing(self, tmp_path):
        l1 = _make_kpf1(tmp_path, flatdiv=True)
        del l1.headers["QUALITY_CONTROL"]["FLATAGE"]
        assert QCL1(l1).flat_ok() is False

    def test_ffi_finite_pass(self, tmp_path):
        l1 = _make_kpf1(tmp_path, finite_ccd=True)
        assert QCL1(l1).ffi_finite() is True

    def test_ffi_finite_fail_nan(self, tmp_path):
        l1 = _make_kpf1(tmp_path, finite_ccd=False)
        assert QCL1(l1).ffi_finite() is False

    def test_ffi_finite_fail_missing(self, tmp_path):
        l1 = _make_kpf1(tmp_path)
        l1.data["GREEN_CCD"] = None
        assert QCL1(l1).ffi_finite() is False

    def test_qc_keys_correct(self):
        expected = {
            "data_present": "DATAPRL1",
            "required_keywords_present": "KWRDPRL1",
            "read_noise_ok": "RNOK",
            "read_noise_nongauss_ok": "RNNGOK",
            "bias_ok": "BIASOK",
            "dark_ok": "DARKOK",
            "flat_ok": "FLATOK",
            "ffi_finite": "L1NANOK",
            "nonzero_flux": "L1FLXOK",
            "variance_positive": "L1VAROK",
            "negative_snr_fraction": "L1SNROK",
            "saturated_fraction": "L1SATOK",
        }
        for method_name, key in expected.items():
            fn = QCL1.__dict__[method_name]
            assert fn._qc_key == key, (
                f"{method_name}: expected {key!r}, got {fn._qc_key!r}"
            )


# ---------------------------------------------------------------------------
# Full QCL1 run on an all-good synthetic L1
# ---------------------------------------------------------------------------


class TestQCL1Run:
    def test_all_good_isgood_1(self, tmp_path):
        l1 = _make_kpf1(tmp_path)
        results = QCL1(l1).run()

        isgood = l1.headers["QUALITY_CONTROL"].get("ISGOOD")
        assert isgood == 1

        # BIASOK/DARKOK/FLATOK read the RECEIPT *SUB flags and DiagL1 *AGE values
        # but are themselves QUALITY_CONTROL keywords; the applied-step flags
        # (OSCANSUB/BIASSUB/DARKSUB/FLATDIV) stay RECEIPT-only provenance.
        qc_keys = [
            "DATAPRL1",
            "KWRDPRL1",
            "RNOK",
            "RNNGOK",
            "BIASOK",
            "DARKOK",
            "FLATOK",
            "L1NANOK",
        ]
        for k in qc_keys:
            v = l1.headers["QUALITY_CONTROL"].get(k)
            assert v == 1, f"{k} should be 1 but is {v}"
            assert k in results

        # A placeholder check raising NotImplementedError writes no flag, so the
        # keyword stays absent and ISGOOD is unaffected.
        for k in ("L1FLXOK", "L1VAROK", "L1SNROK", "L1SATOK"):
            assert l1.headers["QUALITY_CONTROL"].get(k) is None
            assert k not in results

    def test_one_bad_check_isgood_0(self, tmp_path):
        l1 = _make_kpf1(tmp_path, biassub=False)
        QCL1(l1).run()
        isgood = l1.headers["QUALITY_CONTROL"].get("ISGOOD")
        assert isgood == 0

        # QC writes the BIASOK flag and never touches RECEIPT's BIASSUB.
        assert l1.headers["QUALITY_CONTROL"].get("BIASOK") == 0

    def test_isgood_aggregates_propagated_flag(self, tmp_path):
        # ISGOOD aggregates every QC flag on QUALITY_CONTROL, including ones
        # propagated from a lower level, not just this level's checks.
        l1 = _make_kpf1(tmp_path)
        l1.headers["QUALITY_CONTROL"]["DATAPRL0"] = (0, "L0 data present (propagated)")
        QCL1(l1).run()
        assert l1.headers["QUALITY_CONTROL"].get("DATAPRL1") == 1
        assert l1.headers["QUALITY_CONTROL"].get("ISGOOD") == 0


# ---------------------------------------------------------------------------
# QCL2 checks
# ---------------------------------------------------------------------------


class TestQCL2:
    def test_extraction_present_pass(self):
        kpf2 = _make_kpf2_nan_headers()
        assert QCL2(kpf2).extraction_present() is True

    def test_required_keywords_present_pass(self):
        # A fresh KPF2 carries only the rvdata-seeded EPRV keywords, not the KPF
        # provenance cards.
        kpf2 = _make_kpf2_nan_headers()
        seed_required_primary(kpf2, QCL2)
        assert QCL2(kpf2).required_keywords_present() is True

    def test_required_keywords_present_fail_missing(self):
        kpf2 = _make_kpf2_nan_headers()
        seed_required_primary(kpf2, QCL2)
        del kpf2.headers["PRIMARY"]["INSTRUME"]
        assert QCL2(kpf2).required_keywords_present() is False

    def test_extraction_present_fail_empty_kpf2(self):
        kpf2 = KPF2()
        for k in ["NANSCI1", "NANSCI2", "NANSCI3", "NANSKY", "NANCAL"]:
            kpf2.headers["QUALITY_CONTROL"][k] = (0, k)
        kpf2.headers["QUALITY_CONTROL"]["ZEROFRAC"] = (0.0, "z")
        assert QCL2(kpf2).extraction_present() is False

    def test_extraction_present_fail_one_trace_cleared(self):
        kpf2 = _make_kpf2_nan_headers()
        # The SKY_FLUX alias resolves to TRACE1_FLUX; emptying it zeroes the size
        # of every chip view over it.
        kpf2.data["SKY_FLUX"] = np.array([], dtype=np.float32)
        assert QCL2(kpf2).extraction_present() is False

    def test_flux_finite_fraction_pass(self):
        kpf2 = _make_kpf2_nan_headers(nan_frac=0.0)
        assert QCL2(kpf2).flux_finite_fraction() is True

    def test_flux_finite_fraction_fail_too_many_nans(self):
        kpf2 = _make_kpf2_nan_headers()
        # total_pixels = (35 + 32) * 5 fibers * _NCOLS = 3350, so 5 x 200 NaN is
        # ~30%, well over the 1% limit.
        for k in ["NANSCI1", "NANSCI2", "NANSCI3", "NANSKY", "NANCAL"]:
            kpf2.headers["QUALITY_CONTROL"][k] = (200, k)
        assert QCL2(kpf2).flux_finite_fraction() is False

    def test_flux_finite_fraction_fail_missing_header(self):
        kpf2 = _make_kpf2_nan_headers(nan_frac=0.0)
        del kpf2.headers["QUALITY_CONTROL"]["NANSCI1"]
        assert QCL2(kpf2).flux_finite_fraction() is False

    def test_flux_finite_fraction_fail_no_extensions(self):
        # No flux arrays means zero total pixels, so no fraction can be formed.
        kpf2 = KPF2()
        for k in ["NANSCI1", "NANSCI2", "NANSCI3", "NANSKY", "NANCAL"]:
            kpf2.headers["QUALITY_CONTROL"][k] = (0, k)
        assert QCL2(kpf2).flux_finite_fraction() is False

    def test_nonzero_flux_pass(self):
        kpf2 = _make_kpf2_nan_headers(zero_frac=0.1)
        assert QCL2(kpf2).nonzero_flux() is True

    def test_nonzero_flux_fail_high_frac(self):
        kpf2 = _make_kpf2_nan_headers(zero_frac=0.75)
        assert QCL2(kpf2).nonzero_flux() is False

    def test_nonzero_flux_fail_missing(self):
        kpf2 = _make_kpf2_nan_headers()
        del kpf2.headers["QUALITY_CONTROL"]["ZEROFRAC"]
        assert QCL2(kpf2).nonzero_flux() is False

    def test_nonzero_flux_exactly_half(self):
        # The check is strictly < 0.5.
        kpf2 = _make_kpf2_nan_headers(zero_frac=0.5)
        assert QCL2(kpf2).nonzero_flux() is False

    # --- variance_positive (L2VAROK) ---

    def test_variance_positive_pass(self):
        kpf2 = _make_kpf2_nan_headers()
        set_fiber_arrays(kpf2, "VAR", 1.0, ncol=_NCOLS)
        assert QCL2(kpf2).variance_positive() is True

    def test_variance_positive_tolerates_zero(self):
        kpf2 = _make_kpf2_nan_headers()
        set_fiber_arrays(kpf2, "VAR", 0.0, ncol=_NCOLS)  # zero variance is allowed
        assert QCL2(kpf2).variance_positive() is True

    def test_variance_positive_fail_negative(self):
        kpf2 = _make_kpf2_nan_headers()
        set_fiber_arrays(kpf2, "VAR", 1.0, ncol=_NCOLS)
        var = np.full((NORDER["GREEN"], _NCOLS), 1.0, dtype=np.float32)
        var[0, 0] = -1.0  # negative variance where flux is finite
        kpf2.set_data("GREEN_SCI1_VAR", var)
        assert QCL2(kpf2).variance_positive() is False

    def test_variance_positive_raises_on_shape_mismatch(self):
        # FLUX populated with VAR left at its default empty (0,) shape is a
        # malformed product, so the shape mismatch raises rather than skipping.
        kpf2 = _make_kpf2_nan_headers()  # no VAR populated
        with pytest.raises(ValueError, match="could not be broadcast"):
            QCL2(kpf2).variance_positive()

    # --- science_snr (L2SNROK) ---

    def test_science_snr_pass(self):
        kpf2 = _make_kpf2_nan_headers()
        kpf2.headers["QUALITY_CONTROL"]["GSNRSCI"] = (20.0, "g snr")
        kpf2.headers["QUALITY_CONTROL"]["RSNRSCI"] = (18.0, "r snr")
        assert QCL2(kpf2).science_snr() is True

    def test_science_snr_fail_missing(self):
        kpf2 = _make_kpf2_nan_headers()  # no GSNRSCI/RSNRSCI headers
        assert QCL2(kpf2).science_snr() is False

    def test_science_snr_fail_below_floor(self):
        kpf2 = _make_kpf2_nan_headers()
        kpf2.headers["QUALITY_CONTROL"]["GSNRSCI"] = (0.5, "g snr")  # below floor
        kpf2.headers["QUALITY_CONTROL"]["RSNRSCI"] = (18.0, "r snr")
        assert QCL2(kpf2).science_snr() is False

    def test_qc_keys_correct(self):
        expected = {
            "extraction_present": "DATAPRL2",
            "required_keywords_present": "KWRDPRL2",
            "flux_finite_fraction": "L2NANOK",
            "nonzero_flux": "L2FLXOK",
            "variance_positive": "L2VAROK",
            "science_snr": "L2SNROK",
            "saturated_fraction": "L2SATOK",
        }
        for method_name, key in expected.items():
            fn = QCL2.__dict__[method_name]
            assert fn._qc_key == key, (
                f"{method_name}: expected {key!r}, got {fn._qc_key!r}"
            )


# ---------------------------------------------------------------------------
# QCL4 -- CCF/RV presence and per-order BERV/BJD dispersion
# ---------------------------------------------------------------------------


class TestQCL4:
    def test_ccf_rv_present_pass(self):
        assert QCL4(make_l4()).ccf_rv_present() is True

    def test_ccf_rv_present_fail_when_missing(self):
        assert QCL4(make_l4(sci=False)).ccf_rv_present() is False

    def test_ccf_rv_present_fail_when_rvs_unfilled(self):
        # The RVs are the L4 product, so an all-NaN RV column fails even with the
        # CCFs in place.
        assert QCL4(make_l4(rv_filled=False)).ccf_rv_present() is False

    def test_ccf_rv_present_fail_when_columns_missing(self):
        # The per-order BJD_TDB/BERV/WEIGHT columns the DiagL4 dispersion metrics
        # consume are absent, so the product is incomplete.
        l4 = KPF4()
        for fiber in ("SCI1", "SCI2", "SCI3"):
            l4.set_data(f"{fiber}_CCF", np.ones((NORDER_TOTAL, 5)))
            l4.set_data(
                f"{fiber}_RV",
                Table(
                    {
                        "ORDER_INDEX": np.arange(NORDER_TOTAL),
                        "RV": np.zeros(NORDER_TOTAL),
                    }
                ),
            )
        assert QCL4(l4).ccf_rv_present() is False

    def test_berv_within_tolerance_pass(self):
        l4 = make_l4(sci_obj="target", bervrng=0.02)
        assert QCL4(l4).berv_within_tolerance() is True

    def test_berv_within_tolerance_fail(self):
        l4 = make_l4(sci_obj="target", bervrng=0.5)
        assert QCL4(l4).berv_within_tolerance() is False

    def test_berv_within_tolerance_non_target_passes(self):
        # SCI2 is not star-illuminated, so the check is N/A however bad BERVRNG is.
        l4 = make_l4(sci_obj="etalon", bervrng=0.5)
        assert QCL4(l4).berv_within_tolerance() is True

    def test_berv_within_tolerance_raises_when_sci_obj_absent(self):
        # CrossCorrelation requires SCI-OBJ upstream, so a frame without one is
        # malformed and must not pass as a non-target source.
        with pytest.raises(ValueError, match="SCI-OBJ not in INSTRUMENT_HEADER"):
            QCL4(make_l4(bervrng=0.02)).berv_within_tolerance()

    def test_berv_within_tolerance_target_absent_fails(self):
        # DiagL4 skips the metric on degenerate weights or a NaN barycorr; on a
        # target frame that is malformed, not a vacuous pass.
        assert QCL4(make_l4(sci_obj="target")).berv_within_tolerance() is False

    def test_bjd_within_tolerance_pass(self):
        l4 = make_l4(sci_obj="target", bjdrng=0.5)
        assert QCL4(l4).bjd_within_tolerance() is True

    def test_bjd_within_tolerance_fail(self):
        l4 = make_l4(sci_obj="target", bjdrng=2.0)
        assert QCL4(l4).bjd_within_tolerance() is False

    def test_bjd_within_tolerance_non_target_passes(self):
        assert (
            QCL4(make_l4(sci_obj="etalon", bjdrng=2.0)).bjd_within_tolerance() is True
        )

    def test_bjd_within_tolerance_raises_when_sci_obj_absent(self):
        with pytest.raises(ValueError, match="SCI-OBJ not in INSTRUMENT_HEADER"):
            QCL4(make_l4(bjdrng=0.5)).bjd_within_tolerance()

    def test_bjd_within_tolerance_target_absent_fails(self):
        assert QCL4(make_l4(sci_obj="target")).bjd_within_tolerance() is False

    def test_required_keywords_present(self):
        l4 = make_l4()
        req = QCL4(l4)._required_primary_keywords()
        for kw in req:
            l4.headers["PRIMARY"][kw] = 1.0
        assert QCL4(l4).required_keywords_present() is True
        if req:
            del l4.headers["PRIMARY"][sorted(req)[0]]
            assert QCL4(l4).required_keywords_present() is False

    def test_run_all_good_isgood(self):
        l4 = make_l4(sci_obj="target", bervrng=0.02, bjdrng=0.5)
        for kw in QCL4(l4)._required_primary_keywords():
            l4.headers["PRIMARY"][kw] = 1.0
        results = QCL4(l4).run()
        assert set(results) >= {"DATAPRL4", "KWRDPRL4", "BERVOK", "BJDOK"}
        qc = l4.headers["QUALITY_CONTROL"]
        assert qc["DATAPRL4"] == 1 and qc["BERVOK"] == 1 and qc["BJDOK"] == 1
        assert qc["ISGOOD"] == 1

    def test_run_flags_failure_in_isgood(self):
        # no CCF/RV, and out-of-tolerance BERV/BJD ranges on a target frame
        l4 = make_l4(sci=False, sci_obj="target", bervrng=0.5, bjdrng=2.0)
        for kw in QCL4(l4)._required_primary_keywords():
            l4.headers["PRIMARY"][kw] = 1.0
        QCL4(l4).run()
        qc = l4.headers["QUALITY_CONTROL"]
        assert qc["DATAPRL4"] == 0 and qc["BERVOK"] == 0 and qc["BJDOK"] == 0
        assert qc["ISGOOD"] == 0


# ---------------------------------------------------------------------------
# Every check's _qc_key must be a registered keyword
# ---------------------------------------------------------------------------


class TestQCKeyRegistration:
    """Each ``_qc_key`` tag resolves in the keyword registry.

    The per-class ``test_qc_keys_correct`` tests above compare each tag against a
    literal dict in this file, which restates the tag rather than checking it is
    registered. ``QC.run`` looks the tag up in ``keyword_registry.routing`` to
    fetch its comment, so an unregistered key is a ``KeyError`` at run time --
    and QCL0/QCL2 have no in-process ``run()`` test to surface it.
    """

    _CLASSES = {"L0": QCL0, "L1": QCL1, "L2": QCL2, "L4": QCL4}

    @pytest.mark.parametrize("level", sorted(_CLASSES))
    def test_qc_keys_are_registered(self, level):
        qc_cls = self._CLASSES[level]
        registry = KPF0().keyword_registry
        tagged = {
            attr._qc_key
            for cls in qc_cls.__mro__
            for attr in cls.__dict__.values()
            if getattr(attr, "_qc_key", None) is not None
        }
        assert tagged, f"{qc_cls.__name__} exposes no tagged checks"
        for key in sorted(tagged):
            assert key in registry.routing, (
                f"{qc_cls.__name__} writes {key}, which no config/L*-headers.csv "
                "row registers"
            )
            assert key in registry.qc_flag_keywords_by_level[level], (
                f"{key} is not tagged as a {level} QC flag in the registry, so "
                "Checkpoint.qc_flags would never scan it"
            )
