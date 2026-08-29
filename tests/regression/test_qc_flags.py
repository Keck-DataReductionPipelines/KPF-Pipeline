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
    set_fiber_arrays,
    standardized_l0,
    telemetry_hdu,
    write_amp_l0,
)

_NCOLS = 20  # small column count for fast tests; matches the mini_detector ncol,
# which the DATAPRL2 shape check reads


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
    return standardized_l0(fn)


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
    agedark=3.0,
    ageflat=3.0,
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

    return l1


def _make_kpf2_nan_headers(*, nan_frac=0.0, zero_frac=0.1):
    """Minimal KPF2 with all 10 {CHIP}_{FIBER}_FLUX extensions populated.

    The arrays stay clean; the NaN and non-positive counts are written as HEADERS,
    because QCL2 reads them from the header rather than measuring pixels. Not the
    same as test_diagnostics.py's ``_make_kpf2_nan_pixels``, which injects real
    NaN/zero PIXELS for DiagL2 to measure. Do not merge them.

    Per-chip row counts must match NORDER_GREEN/NORDER_RED because KPF2's
    chip-prefix __setitem__ rejects any other shape. ``nan_frac`` and ``zero_frac``
    are the fractions of total pixels reported via the NAN* and ZERO* headers.
    """
    chips = ["GREEN", "RED"]
    fibers = ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]
    ncols = _NCOLS

    kpf2 = KPF2()

    total_pixels = 0
    for chip in chips:
        nrows = NORDER[chip]
        for fiber in fibers:
            arr = np.ones((nrows, ncols), dtype=np.float32)
            kpf2.set_data(f"{chip}_{fiber}_FLUX", arr)
            total_pixels += nrows * ncols

    # The companions DATAPRL2 requires: VAR from extraction, WAVE from the WLS
    # (float64 per its EPRV MinBitDepth), and the per-order barycentric arrays.
    set_fiber_arrays(kpf2, "VAR", 1.0, ncol=ncols)
    set_fiber_arrays(kpf2, "WAVE", 5000.0, ncol=ncols, dtype=np.float64)
    for ext in ("BJD_TDB", "BARYCORR_KMS", "BARYCORR_Z"):
        kpf2.set_data(ext, np.zeros(NORDER_TOTAL, dtype=np.float64))

    nan_count = int(nan_frac * total_pixels / 5)  # spread evenly across 5 nan keys
    for k in ["NANSCI1", "NANSCI2", "NANSCI3", "NANSKY", "NANCAL"]:
        kpf2.headers["QUALITY_CONTROL"][k] = (nan_count, f"NaN count {k}")
    zero_count = int(zero_frac * total_pixels / 5)  # likewise across 5 zero keys
    for k in ["ZEROSCI1", "ZEROSCI2", "ZEROSCI3", "ZEROSKY", "ZEROCAL"]:
        kpf2.headers["QUALITY_CONTROL"][k] = (zero_count, f"Non-positive count {k}")

    return kpf2


# ---------------------------------------------------------------------------
# QC base class runner
# ---------------------------------------------------------------------------


class TestQCBase:
    """Runner behaviour: writing, failure, raises, empty."""

    def _make_obj(self):
        """Minimal object with a headers dict and a set_keyword router.

        QC.run() reads each check's comment off ``keyword_registry.routing``, so
        the stub routes the synthetic check keys to QUALITY_CONTROL and stores
        every keyword there.
        """
        qc_keys = frozenset({"CHECKA", "CHECKB", "CHKOK", "CHKFAIL", "FLAG", "BOOM"})

        class _FakeObj:
            headers = {"PRIMARY": {}, "QUALITY_CONTROL": {}}
            keyword_registry = types.SimpleNamespace(
                routing=dict.fromkeys(qc_keys, "QUALITY_CONTROL"),
                comment_for=lambda kw, ext: "",
            )

            def set_keyword(self, key, value):
                self.headers["QUALITY_CONTROL"][key] = value

        return _FakeObj()

    def test_all_passing_write_1(self):
        obj = self._make_obj()

        class MyQC(QC):
            def check_a(self):
                return True

            check_a._qc_key = "CHECKA"

            def check_b(self):
                return True

            check_b._qc_key = "CHECKB"

        results = MyQC(obj).run()
        assert obj.headers["QUALITY_CONTROL"]["CHECKA"] == 1
        assert obj.headers["QUALITY_CONTROL"]["CHECKB"] == 1
        assert results["CHECKA"][0] is True
        assert results["CHECKB"][0] is True

    def test_one_failing_writes_0(self):
        obj = self._make_obj()

        class MyQC(QC):
            def check_ok(self):
                return True

            check_ok._qc_key = "CHKOK"

            def check_fail(self):
                return False

            check_fail._qc_key = "CHKFAIL"

        MyQC(obj).run()
        assert obj.headers["QUALITY_CONTROL"]["CHKOK"] == 1
        assert obj.headers["QUALITY_CONTROL"]["CHKFAIL"] == 0

    def test_raising_check_writes_zero_and_continues(self, caplog):
        obj = self._make_obj()

        class MyQC(QC):
            LEVEL = "L0"

            def check_boom(self):
                raise ValueError("boom!")

            check_boom._qc_key = "BOOM"

            def check_ok(self):
                return True

            check_ok._qc_key = "CHKOK"

        # Informational layer: a check that cannot run is logged at ERROR (vs the
        # WARNING an ordinary fail gets), counted as a fail, and the run continues.
        with caplog.at_level(logging.ERROR):
            results = MyQC(obj).run()
        assert "QC check 'check_boom' raised" in caplog.text
        assert "boom!" in caplog.text
        assert obj.headers["QUALITY_CONTROL"]["BOOM"] == 0
        assert results["BOOM"][0] is False
        assert obj.headers["QUALITY_CONTROL"]["CHKOK"] == 1

    def test_empty_subclass_writes_nothing(self):
        obj = self._make_obj()

        class EmptyQC(QC):
            pass

        results = EmptyQC(obj).run()
        assert results == {}
        assert obj.headers["QUALITY_CONTROL"] == {}

    def test_repeated_run_resets_results(self):
        # Without the per-run reset, the first run's failed result would linger in
        # self.results once the check starts passing.
        obj = self._make_obj()
        obj.flag = False

        class MyQC(QC):
            def check_flag(self):
                return self.kpf_obj.flag

            check_flag._qc_key = "FLAG"

        qc = MyQC(obj)
        qc.run()
        assert obj.headers["QUALITY_CONTROL"]["FLAG"] == 0
        assert list(qc.results) == ["FLAG"]
        assert qc.results["FLAG"][0] is False

        obj.flag = True
        qc.run()
        assert obj.headers["QUALITY_CONTROL"]["FLAG"] == 1
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


# ---------------------------------------------------------------------------
# QCL0 checks
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mini_detector")
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
        # Two amps split the detector by column alone, so each is full height.
        fn = str(tmp_path / "KP.20240405.00003.00.fits")
        primary = fits.PrimaryHDU()
        primary.header["DATE-OBS"] = "2024-04-05T01:00:37"
        primary.header["OFNAME"] = os.path.basename(fn)
        primary.header["PROGNAME"] = "K123"
        hdus = [primary]
        for chip in ["GREEN", "RED"]:
            for amp in (1, 2):
                data = np.ones((20, 10), dtype=np.float32)
                hdus.append(fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}"))
        fits.HDUList(hdus).writeto(fn, overwrite=True)
        l0 = KPF0.from_fits(fn)
        assert QCL0(l0).data_l0_red_green() is True

    def test_data_l0_red_green_fail_wrong_amp_shape(self, tmp_path):
        # Four amps that do not tile the detector: a truncated readout.
        l0 = _make_kpf0(tmp_path, with_amps=True)
        l0.data["GREEN_AMP2"] = np.ones((10, 9), dtype=np.float32)
        assert QCL0(l0).data_l0_red_green() is False

    def test_data_l0_red_green_fail_transposed_amp(self, tmp_path):
        l0 = _make_kpf0(tmp_path, with_amps=True)
        l0.data["RED_AMP1"] = np.ones((20, 5), dtype=np.float32)
        assert QCL0(l0).data_l0_red_green() is False

    def test_data_l0_red_green_fail_all_nan_amp(self, tmp_path):
        # Correctly shaped but carrying no readout at all.
        l0 = _make_kpf0(tmp_path, with_amps=True)
        l0.data["GREEN_AMP1"][:] = np.nan
        assert QCL0(l0).data_l0_red_green() is False

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

    def test_header_keywords_present_is_stubbed(self, tmp_path):
        # KWRDPRL0 is pending a KPF-owned definition of "required"; until then it
        # raises NotImplementedError, which QC.run treats as "write no flag".
        l0 = _make_kpf0(tmp_path)
        with pytest.raises(NotImplementedError, match="KWRDPRL0"):
            QCL0(l0).header_keywords_present()

    def _make_kpf0_with_telemetry(self, tmp_path, nrows):
        fn = write_amp_l0(
            tmp_path / "KP.20240405.00003.00.fits",
            shape=(10, 10),
            extra_hdus=[telemetry_hdu(nrows)],
        )
        return KPF0.from_fits(fn)

    def test_telemetry_present_pass(self, tmp_path):
        l0 = self._make_kpf0_with_telemetry(tmp_path, nrows=1)
        assert QCL0(l0).telemetry_present() is True

    def test_telemetry_present_empty_table_fails(self, tmp_path):
        l0 = self._make_kpf0_with_telemetry(tmp_path, nrows=0)
        assert QCL0(l0).telemetry_present() is False

    def test_telemetry_all_nan_fails(self, tmp_path):
        # A table of rows whose averages never recorded is not a recording.
        l0 = self._make_kpf0_with_telemetry(tmp_path, nrows=2)
        l0.data["TELEMETRY"]["average"] = np.nan
        assert QCL0(l0).telemetry_present() is False

    def test_telemetry_one_finite_average_passes(self, tmp_path):
        l0 = self._make_kpf0_with_telemetry(tmp_path, nrows=2)
        l0.data["TELEMETRY"]["average"] = [np.nan, 20.0]
        assert QCL0(l0).telemetry_present() is True

    def test_telemetry_absent_fails(self, tmp_path):
        assert QCL0(_make_kpf0(tmp_path)).telemetry_present() is False

    def test_teleprl0_key_present(self):
        assert QCL0.__dict__["telemetry_present"]._qc_key == "TELEPR"

    def _make_kpf0_with_cahk(self, tmp_path, shape):
        fn = write_amp_l0(
            tmp_path / "KP.20240405.00004.00.fits",
            shape=(10, 10),
            extra_hdus=[fits.ImageHDU(np.ones(shape, dtype=np.float32), name="CA_HK")],
        )
        return KPF0.from_fits(fn)

    def test_cahk_present_pass(self, tmp_path):
        assert (
            QCL0(self._make_kpf0_with_cahk(tmp_path, (16, 16))).cahk_present() is True
        )

    def test_cahk_present_empty_fails(self, tmp_path):
        l0 = self._make_kpf0_with_cahk(tmp_path, (0,))
        assert QCL0(l0).cahk_present() is False

    def test_cahk_absent_fails(self, tmp_path):
        assert QCL0(_make_kpf0(tmp_path)).cahk_present() is False

    def test_cahk_all_non_finite_fails(self, tmp_path):
        # An all-NaN image is a placeholder, not a readout.
        l0 = self._make_kpf0_with_cahk(tmp_path, (16, 16))
        l0.data["CA_HK"][:] = np.nan
        assert QCL0(l0).cahk_present() is False

    def test_cahkprl0_key_present(self):
        assert QCL0.__dict__["cahk_present"]._qc_key == "CAHKPR"

    def _make_kpf0_with_em_table(self, tmp_path, columns, ext="EXPMETER_SCI"):
        other = "EXPMETER_SKY" if ext == "EXPMETER_SCI" else "EXPMETER_SCI"
        fn = write_amp_l0(
            tmp_path / "KP.20240405.00007.00.fits",
            shape=(10, 10),
            extra_hdus=[
                fits.BinTableHDU(Table(columns), name=ext),
                fits.BinTableHDU(
                    Table({"Date-Beg": ["2024-09-23T09:12:09.484"], "5000.0": [1.0]}),
                    name=other,
                ),
            ],
        )
        return KPF0.from_fits(fn)

    _EM_GOOD_COLUMNS = {
        "Date-Beg": ["2024-09-23T09:12:09.484"],
        "5000.0": [1000.0],
        "5001.0": [2000.0],
    }

    def test_expmeter_present_pass(self, tmp_path):
        l0 = self._make_kpf0_with_em_table(tmp_path, self._EM_GOOD_COLUMNS)
        assert QCL0(l0).expmeter_sci_present() is True
        assert QCL0(l0).expmeter_sky_present() is True

    def test_expmeter_absent_fails(self, tmp_path):
        l0 = _make_kpf0(tmp_path)
        assert QCL0(l0).expmeter_sci_present() is False
        assert QCL0(l0).expmeter_sky_present() is False

    def test_expmeter_no_readings_fails(self, tmp_path):
        columns = {k: [] for k in self._EM_GOOD_COLUMNS}
        l0 = self._make_kpf0_with_em_table(tmp_path, columns)
        assert QCL0(l0).expmeter_sci_present() is False

    def test_expmeter_no_channel_columns_fails(self, tmp_path):
        # Timestamps alone are not a readout.
        columns = {"Date-Beg": ["2024-09-23T09:12:09.484"]}
        l0 = self._make_kpf0_with_em_table(tmp_path, columns)
        assert QCL0(l0).expmeter_sci_present() is False

    def test_expmeter_all_non_finite_fails(self, tmp_path):
        columns = dict(
            self._EM_GOOD_COLUMNS, **{"5000.0": [np.nan], "5001.0": [np.nan]}
        )
        l0 = self._make_kpf0_with_em_table(tmp_path, columns)
        assert QCL0(l0).expmeter_sci_present() is False

    def test_expmeter_fibers_judged_separately(self, tmp_path):
        # A broken SKY leaves the SCI verdict alone.
        columns = {"Date-Beg": ["2024-09-23T09:12:09.484"]}
        l0 = self._make_kpf0_with_em_table(tmp_path, columns, ext="EXPMETER_SKY")
        assert QCL0(l0).expmeter_sci_present() is True
        assert QCL0(l0).expmeter_sky_present() is False

    def test_expmeter_presence_keys_present(self):
        assert QCL0.__dict__["expmeter_sci_present"]._qc_key == "EMSCIPR"
        assert QCL0.__dict__["expmeter_sky_present"]._qc_key == "EMSKYPR"

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

    def test_times_missing_keys_raise(self, tmp_path):
        # Raw L0 PRIMARY without DATE-BEG/MID/END -> cannot be computed.
        with pytest.raises(KeyError, match="DATE-BEG"):
            QCL0(_make_kpf0(tmp_path)).times_consistent()

    @pytest.mark.parametrize("key", ["GRDATE-B", "GRDATE-E", "RDDATE-B", "RDDATE-E"])
    def test_times_shutter_offset_fails(self, tmp_path, key):
        # 0.5 s off the window edge it bounds, past the 0.1 s tolerance.
        bad = dict(GOOD_DATES, **{key: "2024-09-23T09:12:30.000"})
        assert QCL0(_make_kpf0(tmp_path, dates=bad)).times_consistent() is False

    @pytest.mark.parametrize("key", ["GRDATE-B", "GRDATE-E", "RDDATE-B", "RDDATE-E"])
    def test_times_shutter_missing_raises(self, tmp_path, key):
        missing = dict(GOOD_DATES, **{key: None})
        with pytest.raises(KeyError, match=key):
            QCL0(_make_kpf0(tmp_path, dates=missing)).times_consistent()

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
            "NTP is not synchronised",  # reports no error
        ],
    )
    def test_ntp_timing_fail(self, tmp_path, timeerr):
        l0 = _make_kpf0(tmp_path)
        l0.headers["INSTRUMENT_HEADER"]["TIMEERR"] = timeerr
        assert QCL0(l0).ntp_timing() is False

    def test_ntp_timing_missing_raises(self, tmp_path):
        l0 = _make_kpf0(tmp_path)
        del l0.headers["INSTRUMENT_HEADER"]["TIMEERR"]
        with pytest.raises(KeyError, match="TIMEERR"):
            QCL0(l0).ntp_timing()

    def test_ntp_key_present(self):
        assert QCL0.__dict__["ntp_timing"]._qc_key == "NTPOK"

    def test_exptime_sane_pass_positive(self, tmp_path):
        l0 = _make_kpf0(tmp_path, exptime=300.0, dates={"ELAPSED": 300.0})
        assert QCL0(l0).exptime_sane() is True

    def test_exptime_sane_pass_zero(self, tmp_path):
        # Bias frames legitimately have EXPTIME=0.
        l0 = _make_kpf0(tmp_path, exptime=0.0, dates={"ELAPSED": 0.0})
        assert QCL0(l0).exptime_sane() is True

    def test_exptime_sane_fail_negative(self, tmp_path):
        l0 = _make_kpf0(tmp_path, exptime=-1.0, dates={"ELAPSED": -1.0})
        assert QCL0(l0).exptime_sane() is False

    def test_exptime_sane_missing_raises(self, tmp_path):
        l0 = _make_kpf0(tmp_path, dates={"ELAPSED": 60.0})
        del l0.headers["INSTRUMENT_HEADER"]["EXPTIME"]
        with pytest.raises(KeyError, match="EXPTIME"):
            QCL0(l0).exptime_sane()

    def test_exptime_sane_pass_elapsed_within_tol(self, tmp_path):
        # ELAPSED may exceed the requested EXPTIME by up to the timing tolerance.
        l0 = _make_kpf0(tmp_path, exptime=300.0, dates={"ELAPSED": 300.05})
        assert QCL0(l0).exptime_sane() is True

    def test_exptime_sane_elapsed_absent_raises(self, tmp_path):
        # ELAPSED is the readout evidence; without it there is nothing to compare.
        l0 = _make_kpf0(tmp_path, exptime=300.0)
        with pytest.raises(KeyError, match="ELAPSED"):
            QCL0(l0).exptime_sane()

    def test_exptime_sane_fail_premature_readout(self, tmp_path):
        # ELAPSED shorter than the requested EXPTIME (premature readout).
        l0 = _make_kpf0(tmp_path, exptime=300.0, dates={"ELAPSED": 299.0})
        assert QCL0(l0).exptime_sane() is False

    def test_exptime_sane_fail_elapsed_exceeds_tol(self, tmp_path):
        # ELAPSED over the requested EXPTIME by more than the tolerance.
        l0 = _make_kpf0(tmp_path, exptime=300.0, dates={"ELAPSED": 301.0})
        assert QCL0(l0).exptime_sane() is False

    @pytest.mark.parametrize("elapsed", [5.9, 6.8, 300.0])
    def test_good_readout_pass(self, tmp_path, elapsed):
        l0 = _make_kpf0(tmp_path, exptime=300.0, dates={"ELAPSED": elapsed})
        assert QCL0(l0).good_readout() is True

    def test_good_readout_pass_short_request(self, tmp_path):
        # A sub-7 s request may legitimately elapse inside the smear window.
        l0 = _make_kpf0(tmp_path, exptime=6.5, dates={"ELAPSED": 6.5})
        assert QCL0(l0).good_readout() is True

    @pytest.mark.parametrize("elapsed", [6.0, 6.35, 6.7])
    def test_good_readout_fail_smeared(self, tmp_path, elapsed):
        l0 = _make_kpf0(tmp_path, exptime=300.0, dates={"ELAPSED": elapsed})
        assert QCL0(l0).good_readout() is False

    def test_good_readout_missing_raises(self, tmp_path):
        l0 = _make_kpf0(tmp_path, exptime=300.0)
        with pytest.raises(KeyError, match="ELAPSED"):
            QCL0(l0).good_readout()

    def test_goodread_key_present(self):
        assert QCL0.__dict__["good_readout"]._qc_key == "READOK"

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

    def test_radec_tcsoff_required_absent_raises(self, tmp_path):
        # TCSOFF is internal pointing consistency and DiagL0 always emits it.
        l0 = _make_kpf0(tmp_path)
        with pytest.raises(KeyError, match="TCSOFF"):
            QCL0(l0).radec_consistent()

    def test_radec_external_offsets_optional(self, tmp_path):
        # TCSOFF within budget; Gaia/SIMBAD unmatched, so DiagL0 emitted no
        # GAIAOFF/OBJOFF and there is no external bound to apply.
        l0 = _make_kpf0(tmp_path)
        l0.set_keyword("TCSOFF", 0.02)
        assert QCL0(l0).radec_consistent() is True

    def test_radecok_key_present(self):
        assert QCL0.__dict__["radec_consistent"]._qc_key == "TARGETOK"

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

    def test_catalog_astrometry_sane_no_record_raises(self, tmp_path):
        # CATALOG_RECORD is empty when AstroQuery never ran: nothing to check.
        l0 = _make_kpf0(tmp_path)
        assert not l0.data["CATALOG_RECORD"].colnames
        with pytest.raises(KeyError, match="source"):
            QCL0(l0).catalog_astrometry_sane()

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

    def test_catalog_color_sane_no_record_raises(self, tmp_path):
        l0 = _make_kpf0(tmp_path)
        assert not l0.data["CATALOG_RECORD"].colnames
        with pytest.raises(KeyError, match="source"):
            QCL0(l0).catalog_color_sane()

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
        registry = KPF0.keyword_registry
        assert "GREEN" in registry.comment_for("DATAPRL0", registry.routing["DATAPRL0"])

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
        """KPF0 carrying the EXPMETER_SCI and EXPMETER_SKY tables.

        Built from per-reading timestamps and an (nreading, nchannel) flux array
        whose column labels are wavelengths. ``sky_flux`` defaults to the same
        clean flux as SCI; a real science L0 always carries both fibers.
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

        extensions = {
            "EXPMETER_SCI": table(_EM_CLEAN_FLUX if flux is None else flux),
            "EXPMETER_SKY": table(_EM_CLEAN_FLUX if sky_flux is None else sky_flux),
        }
        return _make_kpf0(tmp_path, dates=GOOD_DATES, expmeter=extensions)

    def test_expmeter_times_consistent_pass(self, tmp_path):
        l0 = self._make_kpf0_with_expmeter(tmp_path)
        assert QCL0(l0).expmeter_times_consistent() is True

    def test_expmeter_times_no_em_data_raises(self, tmp_path):
        # A frame with no EM readings (e.g. a calibration) cannot be checked. The
        # manifest creates EXPMETER_SCI on every L0, so the table is present and
        # empty rather than absent, and the raise lands on its missing columns.
        l0 = _make_kpf0(tmp_path, dates=GOOD_DATES)
        with pytest.raises(KeyError, match="Date-Beg"):
            QCL0(l0).expmeter_times_consistent()

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

    def test_expmeter_times_missing_shutter_window_raises(self, tmp_path):
        # EM data present but no DATE-BEG/DATE-END to compare against.
        l0 = self._make_kpf0_with_expmeter(tmp_path)
        del l0.headers["INSTRUMENT_HEADER"]["DATE-BEG"]
        with pytest.raises(KeyError, match="DATE-BEG"):
            QCL0(l0).expmeter_times_consistent()

    def test_emtimeok_key_present(self):
        assert QCL0.__dict__["expmeter_times_consistent"]._qc_key == "EMTIMEOK"

    # expmeter_flux_sane reads the DiagL0 channel metrics, so these seed them
    # directly; the measurement itself is covered in test_diagnostics.py.
    def _make_kpf0_with_em_metrics(self, tmp_path, **metrics):
        l0 = _make_kpf0(tmp_path, dates=GOOD_DATES)
        l0.headers["QUALITY_CONTROL"].update(
            {
                f"EM{fiber}{m}": 0
                for fiber in ("SCI", "SKY")
                for m in ("SAT", "NEG", "INF")
            }
        )
        l0.headers["QUALITY_CONTROL"].update(metrics)
        return l0

    def test_expmeter_flux_sane_pass(self, tmp_path):
        l0 = self._make_kpf0_with_em_metrics(tmp_path)
        assert QCL0(l0).expmeter_flux_sane() is True

    def test_expmeter_flux_missing_metric_raises(self, tmp_path):
        # DiagL0 did not run (e.g. a frame with no EM data), so nothing to judge.
        l0 = _make_kpf0(tmp_path, dates=GOOD_DATES)
        with pytest.raises(KeyError, match="EMSCISAT"):
            QCL0(l0).expmeter_flux_sane()

    def test_expmeter_flux_pass_at_limits(self, tmp_path):
        # 1.5 saturated elements per reading and a 19-channel run are the limits.
        l0 = self._make_kpf0_with_em_metrics(
            tmp_path, EMSCISAT=1.5, EMSCINEG=19, EMSCIINF=19
        )
        assert QCL0(l0).expmeter_flux_sane() is True

    def test_expmeter_flux_saturated_fails(self, tmp_path):
        l0 = self._make_kpf0_with_em_metrics(tmp_path, EMSCISAT=1.6)
        assert QCL0(l0).expmeter_flux_sane() is False

    def test_expmeter_flux_negative_run_fails(self, tmp_path):
        l0 = self._make_kpf0_with_em_metrics(tmp_path, EMSCINEG=20)
        assert QCL0(l0).expmeter_flux_sane() is False

    def test_expmeter_flux_non_finite_run_fails(self, tmp_path):
        l0 = self._make_kpf0_with_em_metrics(tmp_path, EMSCIINF=20)
        assert QCL0(l0).expmeter_flux_sane() is False

    def test_expmeter_flux_checks_sky_fiber(self, tmp_path):
        # v2.12 fails on either fiber, so a bad SKY fails an otherwise clean frame.
        l0 = self._make_kpf0_with_em_metrics(tmp_path, EMSKYNEG=20)
        assert QCL0(l0).expmeter_flux_sane() is False

    def test_emfluxok_key_present(self):
        assert QCL0.__dict__["expmeter_flux_sane"]._qc_key == "EMFLUXOK"


class TestQCL0PixelQuality:
    """Per-chip pixel quality: the two v2.12 infobits folded into one flag per chip.

    The fractions themselves come from DiagL0 (tested there); these cover only the
    limits QC applies to them, so each test seeds QUALITY_CONTROL directly.
    """

    def _make_kpf0_with_fractions(self, tmp_path, **fractions):
        l0 = _make_kpf0(tmp_path)
        l0.headers["QUALITY_CONTROL"].update(
            {"DEADPXFG": 0.0, "DEADPXFR": 0.0, "SATPXFG": 0.0, "SATPXFR": 0.0}
        )
        l0.headers["QUALITY_CONTROL"].update(fractions)
        return l0

    def test_pixels_ok_pass(self, tmp_path):
        l0 = self._make_kpf0_with_fractions(tmp_path)
        assert QCL0(l0).green_pixels_ok() is True
        assert QCL0(l0).red_pixels_ok() is True

    def test_pass_at_limits(self, tmp_path):
        # 5% dead and 15% saturated are the limits; a fraction must exceed to fail.
        l0 = self._make_kpf0_with_fractions(tmp_path, DEADPXFG=0.05, SATPXFG=0.15)
        assert QCL0(l0).green_pixels_ok() is True

    def test_dead_fail_past_limit(self, tmp_path):
        l0 = self._make_kpf0_with_fractions(tmp_path, DEADPXFG=0.06)
        assert QCL0(l0).green_pixels_ok() is False

    def test_saturated_fail_past_limit(self, tmp_path):
        l0 = self._make_kpf0_with_fractions(tmp_path, SATPXFR=0.16)
        assert QCL0(l0).red_pixels_ok() is False

    def test_chips_judged_separately(self, tmp_path):
        # A dead GREEN chip does not drag down the RED verdict.
        l0 = self._make_kpf0_with_fractions(tmp_path, DEADPXFG=0.5)
        assert QCL0(l0).green_pixels_ok() is False
        assert QCL0(l0).red_pixels_ok() is True

    def test_missing_fraction_raises(self, tmp_path):
        # DiagL0 did not run, so there is nothing to judge.
        with pytest.raises(KeyError, match="DEADPXFG"):
            QCL0(_make_kpf0(tmp_path)).green_pixels_ok()

    def test_qc_keys_correct(self):
        expected = {
            "green_pixels_ok": "GREENL0",
            "red_pixels_ok": "REDL0",
        }
        for method_name, key in expected.items():
            fn = QCL0.__dict__[method_name]
            assert fn._qc_key == key, (
                f"{method_name}: expected {key!r}, got {fn._qc_key!r}"
            )


class TestQCL0Telemetry:
    """Instrument-state checks drawn from telemetry and the guide camera.

    The CCD-offset and guider metrics come from DiagL0 (tested there); these cover
    only the limits QC applies, so each test seeds QUALITY_CONTROL directly.
    """

    def _make_kpf0_with_temps(self, tmp_path, **offsets):
        l0 = _make_kpf0(tmp_path)
        l0.headers["QUALITY_CONTROL"].update({"GTEMPOFF": 0.0, "RTEMPOFF": 0.0})
        l0.headers["QUALITY_CONTROL"].update(offsets)
        return l0

    def _make_kpf0_with_guider(self, tmp_path, **metrics):
        l0 = _make_kpf0(tmp_path)
        l0.headers["QUALITY_CONTROL"].update(
            {
                "GDRXRMS": 10.0,
                "GDRYRMS": 10.0,
                "GDRXBIAS": 1.0,
                "GDRYBIAS": 1.0,
                "GDRNSAT": 0,
                "GDRFRSAT": 0.0,
                "GDRSEEV": 0.5,
            }
        )
        l0.headers["QUALITY_CONTROL"].update(metrics)
        return l0

    def test_ccd_temps_pass(self, tmp_path):
        l0 = self._make_kpf0_with_temps(tmp_path, GTEMPOFF=-9.9, RTEMPOFF=9.9)
        assert QCL0(l0).green_ccd_temp_ok() is True
        assert QCL0(l0).red_ccd_temp_ok() is True

    def test_ccd_temp_fail_either_direction(self, tmp_path):
        # The limit is on the magnitude, so a cold CCD fails like a warm one.
        assert (
            QCL0(
                self._make_kpf0_with_temps(tmp_path, GTEMPOFF=-10.5)
            ).green_ccd_temp_ok()
            is False
        )
        assert (
            QCL0(self._make_kpf0_with_temps(tmp_path, RTEMPOFF=10.5)).red_ccd_temp_ok()
            is False
        )

    def test_chips_judged_separately(self, tmp_path):
        l0 = self._make_kpf0_with_temps(tmp_path, GTEMPOFF=50.0)
        assert QCL0(l0).green_ccd_temp_ok() is False
        assert QCL0(l0).red_ccd_temp_ok() is True

    def test_guiding_ok_pass(self, tmp_path):
        assert QCL0(self._make_kpf0_with_guider(tmp_path)).guiding_ok() is True

    def test_guiding_rms_fail(self, tmp_path):
        l0 = self._make_kpf0_with_guider(tmp_path, GDRYRMS=51.0)
        assert QCL0(l0).guiding_ok() is False

    def test_guiding_bias_judged_on_magnitude(self, tmp_path):
        # v2.12 compared the signed bias and let a large negative offset pass.
        l0 = self._make_kpf0_with_guider(tmp_path, GDRXBIAS=-60.0)
        assert QCL0(l0).guiding_ok() is False

    def test_guider_saturation_fails(self, tmp_path):
        assert (
            QCL0(self._make_kpf0_with_guider(tmp_path, GDRNSAT=4)).guiding_ok() is False
        )
        assert (
            QCL0(self._make_kpf0_with_guider(tmp_path, GDRFRSAT=0.11)).guiding_ok()
            is False
        )

    def test_guiding_missing_metric_raises(self, tmp_path):
        # DiagL0 emits no guiding error when the camera was not tracking.
        with pytest.raises(KeyError, match="GDRXRMS"):
            QCL0(_make_kpf0(tmp_path)).guiding_ok()

    def test_seeing_ok_pass(self, tmp_path):
        l0 = self._make_kpf0_with_guider(tmp_path, GDRSEEV=0.999)
        assert QCL0(l0).seeing_ok() is True

    def test_seeing_ok_fail_at_one_arcsec(self, tmp_path):
        l0 = self._make_kpf0_with_guider(tmp_path, GDRSEEV=1.0)
        assert QCL0(l0).seeing_ok() is False

    def test_seeing_missing_metric_raises(self, tmp_path):
        # An unconverged guider Moffat fit emits no GDRSEEV at all.
        with pytest.raises(KeyError, match="GDRSEEV"):
            QCL0(_make_kpf0(tmp_path)).seeing_ok()

    def test_elevation_ok(self, tmp_path):
        l0 = _make_kpf0(tmp_path)
        native = l0.headers["INSTRUMENT_HEADER"]
        native["EL"] = 30.0  # the ADC limit itself passes
        assert QCL0(l0).elevation_ok() is True
        native["EL"] = 29.9
        assert QCL0(l0).elevation_ok() is False

    def _make_kpf0_with_etalon(self, tmp_path, offset):
        l0 = _make_kpf0(tmp_path)
        l0.headers["QUALITY_CONTROL"]["ETATOFF"] = offset
        return l0

    def test_etalon_at_temp_pass(self, tmp_path):
        # 0.5 mK is the limit itself; an offset must exceed it to fail.
        assert QCL0(self._make_kpf0_with_etalon(tmp_path, 0.5)).etalon_at_temp() is True
        assert (
            QCL0(self._make_kpf0_with_etalon(tmp_path, -0.5)).etalon_at_temp() is True
        )

    def test_etalon_off_setpoint_fails(self, tmp_path):
        # Judged on magnitude, so a cold chamber fails like a warm one.
        assert (
            QCL0(self._make_kpf0_with_etalon(tmp_path, 0.6)).etalon_at_temp() is False
        )
        assert (
            QCL0(self._make_kpf0_with_etalon(tmp_path, -0.6)).etalon_at_temp() is False
        )

    def test_etalon_missing_offset_raises(self, tmp_path):
        with pytest.raises(KeyError, match="ETATOFF"):
            QCL0(_make_kpf0(tmp_path)).etalon_at_temp()

    def _make_kpf0_with_agitator(self, tmp_path, *, status="Running", speed=2000.0):
        fn = write_amp_l0(
            tmp_path / "KP.20240405.00002.00.fits",
            shape=(10, 10),
            primary_cards={
                "DATE-OBS": "2024-04-05T01:00:37",
                "EXPTIME": 60.0,
                "IMTYPE": "Object",
                "AGITSTA": status,
            },
            extra_hdus=[
                fits.BinTableHDU(
                    Table({"keyword": ["kpfmot.AGITSPD"], "average": [speed]}),
                    name="TELEMETRY",
                )
            ],
        )
        return standardized_l0(fn)

    def test_agitator_running_above_minimum(self, tmp_path):
        l0 = self._make_kpf0_with_agitator(tmp_path)
        assert QCL0(l0).agitator_operating() is True

    def test_agitator_speed_judged_on_magnitude(self, tmp_path):
        l0 = self._make_kpf0_with_agitator(tmp_path, speed=-2000.0)
        assert QCL0(l0).agitator_operating() is True

    def test_agitator_stalled_fails(self, tmp_path):
        l0 = self._make_kpf0_with_agitator(tmp_path, speed=900.0)
        assert QCL0(l0).agitator_operating() is False

    def test_agitator_not_running_fails(self, tmp_path):
        l0 = self._make_kpf0_with_agitator(tmp_path, status="Stopped")
        assert QCL0(l0).agitator_operating() is False

    def test_qc_keys_correct(self):
        expected = {
            "green_ccd_temp_ok": "GTEMPOK",
            "red_ccd_temp_ok": "RTEMPOK",
            "guiding_ok": "GUIDEROK",
            "seeing_ok": "SEEINGOK",
            "elevation_ok": "ELEVOK",
            "etalon_at_temp": "ETATMPOK",
            "agitator_operating": "AGITOK",
        }
        for method_name, key in expected.items():
            fn = QCL0.__dict__[method_name]
            assert fn._qc_key == key, (
                f"{method_name}: expected {key!r}, got {fn._qc_key!r}"
            )


# ---------------------------------------------------------------------------
# QCL1 checks
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mini_detector")
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

    def test_data_present_fail_variance_missing(self, tmp_path):
        # Assembly writes CCD and VAR together, so a flux without its variance is
        # a malformed product, not a lesser one.
        l1 = _make_kpf1(tmp_path)
        l1.data["GREEN_VAR"] = np.array([], dtype=np.float32)
        assert QCL1(l1).data_present() is False

    def test_data_present_fail_wrong_shape(self, tmp_path):
        l1 = _make_kpf1(tmp_path, shape=(20, 10))  # half the detector width
        assert QCL1(l1).data_present() is False

    def test_data_present_fail_all_nan(self, tmp_path):
        l1 = _make_kpf1(tmp_path)
        l1.data["RED_CCD"][:] = np.nan
        assert QCL1(l1).data_present() is False

    def test_required_keywords_present_is_stubbed(self, tmp_path):
        # KWRDPRL1 is pending a KPF-owned definition of "required"; until then
        # it raises NotImplementedError, which QC.run treats as "write no flag".
        with pytest.raises(NotImplementedError, match="KWRDPRL1"):
            QCL1(_make_kpf1(tmp_path)).required_keywords_present()

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
        # CalibrationAssociation writes a SIGNED age (master_dt - obs_dt), and the
        # check uses abs(), so a master processed after the science frame --
        # routine in the masters pipeline -- is still in range. Dropping the abs()
        # would start failing BIASOK on every same-night master.
        l1 = _make_kpf1(tmp_path, biassub=True, agebias=-3.0)
        assert QCL1(l1).bias_ok() is True

    def test_bias_ok_boundary(self, tmp_path):
        # 5 days exactly is inside the gate; just past it is not.
        assert QCL1(_make_kpf1(tmp_path, agebias=5.0)).bias_ok() is True
        assert QCL1(_make_kpf1(tmp_path, agebias=5.5)).bias_ok() is False

    def test_bias_ok_fail_not_subtracted(self, tmp_path):
        l1 = _make_kpf1(tmp_path, biassub=False, agebias=3.0)
        assert QCL1(l1).bias_ok() is False

    def test_bias_ok_subtract_flag_missing_raises(self, tmp_path):
        # ImageProcessing writes BIASSUB on every run, 0 or 1; an absent card is
        # a broken upstream invariant, not a not-subtracted frame.
        l1 = _make_kpf1(tmp_path, agebias=3.0)
        del l1.headers["RECEIPT"]["BIASSUB"]
        with pytest.raises(KeyError, match="BIASSUB"):
            QCL1(l1).bias_ok()

    def test_bias_ok_fail_too_old(self, tmp_path):
        l1 = _make_kpf1(tmp_path, biassub=True, agebias=10.0)
        assert QCL1(l1).bias_ok() is False

    def test_bias_ok_age_missing_raises(self, tmp_path):
        l1 = _make_kpf1(tmp_path, biassub=True)
        del l1.headers["QUALITY_CONTROL"]["BIASAGE"]
        with pytest.raises(KeyError, match="BIASAGE"):
            QCL1(l1).bias_ok()

    def test_dark_ok_pass(self, tmp_path):
        l1 = _make_kpf1(tmp_path, darksub=True, agedark=3.0)
        assert QCL1(l1).dark_ok() is True

    def test_dark_ok_fail_not_subtracted(self, tmp_path):
        l1 = _make_kpf1(tmp_path, darksub=False, agedark=3.0)
        assert QCL1(l1).dark_ok() is False

    def test_dark_ok_fail_too_old(self, tmp_path):
        l1 = _make_kpf1(tmp_path, darksub=True, agedark=10.0)
        assert QCL1(l1).dark_ok() is False

    def test_dark_ok_age_missing_raises(self, tmp_path):
        l1 = _make_kpf1(tmp_path, darksub=True)
        del l1.headers["QUALITY_CONTROL"]["DARKAGE"]
        with pytest.raises(KeyError, match="DARKAGE"):
            QCL1(l1).dark_ok()

    def test_flat_ok_pass(self, tmp_path):
        l1 = _make_kpf1(tmp_path, flatdiv=True, ageflat=3.0)
        assert QCL1(l1).flat_ok() is True

    def test_flat_ok_fail_not_divided(self, tmp_path):
        l1 = _make_kpf1(tmp_path, flatdiv=False, ageflat=3.0)
        assert QCL1(l1).flat_ok() is False

    def test_flat_ok_fail_too_old(self, tmp_path):
        l1 = _make_kpf1(tmp_path, flatdiv=True, ageflat=10.0)
        assert QCL1(l1).flat_ok() is False

    def test_flat_ok_age_missing_raises(self, tmp_path):
        l1 = _make_kpf1(tmp_path, flatdiv=True)
        del l1.headers["QUALITY_CONTROL"]["FLATAGE"]
        with pytest.raises(KeyError, match="FLATAGE"):
            QCL1(l1).flat_ok()

    def test_ffi_finite_pass(self, tmp_path):
        l1 = _make_kpf1(tmp_path, finite_ccd=True)
        assert QCL1(l1).ffi_finite() is True

    def test_ffi_finite_fail_nan(self, tmp_path):
        l1 = _make_kpf1(tmp_path, finite_ccd=False)
        assert QCL1(l1).ffi_finite() is False

    def test_ffi_finite_missing_raises(self, tmp_path):
        l1 = _make_kpf1(tmp_path)
        del l1.data["GREEN_CCD"]
        with pytest.raises(KeyError, match="GREEN_CCD"):
            QCL1(l1).ffi_finite()

    # --- variance_positive (L1VAROK) and negative_snr_fraction (L1SNROK).
    # _make_kpf1 writes all-ones CCD and VAR, so SNR == the CCD value and each
    # chip is 20x20 = 400 pixels, putting the 1% limit at 4.

    def test_variance_positive_pass(self, tmp_path):
        assert QCL1(_make_kpf1(tmp_path)).variance_positive() is True

    def test_variance_positive_fail_negative_var(self, tmp_path):
        l1 = _make_kpf1(tmp_path)
        l1.data["GREEN_VAR"][0, 0] = -1.0
        assert QCL1(l1).variance_positive() is False

    def test_variance_positive_pass_zero_var(self, tmp_path):
        # Zero variance at a masked column is tolerated; only negative is unphysical.
        l1 = _make_kpf1(tmp_path)
        l1.data["GREEN_VAR"][0, 0] = 0.0
        assert QCL1(l1).variance_positive() is True

    def test_variance_positive_pass_negative_var_under_nan_flux(self, tmp_path):
        # Scoped to pixels whose flux is finite, matching QCL2.variance_positive.
        l1 = _make_kpf1(tmp_path)
        l1.data["GREEN_CCD"][0, 0] = np.nan
        l1.data["GREEN_VAR"][0, 0] = -1.0
        assert QCL1(l1).variance_positive() is True

    def test_negative_snr_fraction_pass(self, tmp_path):
        assert QCL1(_make_kpf1(tmp_path)).negative_snr_fraction() is True

    def test_negative_snr_fraction_pass_at_limit(self, tmp_path):
        # 4 of 400 pixels is exactly 1%; the fraction must exceed it to fail.
        l1 = _make_kpf1(tmp_path)
        l1.data["GREEN_CCD"].flat[:4] = -10.0
        assert QCL1(l1).negative_snr_fraction() is True

    def test_negative_snr_fraction_fail_past_limit(self, tmp_path):
        l1 = _make_kpf1(tmp_path)
        l1.data["GREEN_CCD"].flat[:5] = -10.0
        assert QCL1(l1).negative_snr_fraction() is False

    def test_negative_snr_fraction_fail_red_chip(self, tmp_path):
        l1 = _make_kpf1(tmp_path)
        l1.data["RED_CCD"].flat[:5] = -10.0
        assert QCL1(l1).negative_snr_fraction() is False

    def test_negative_snr_fraction_pass_at_five_sigma(self, tmp_path):
        # Strictly below -5 counts; a pixel exactly at -5 sigma does not.
        l1 = _make_kpf1(tmp_path)
        l1.data["GREEN_CCD"].flat[:10] = -5.0
        assert QCL1(l1).negative_snr_fraction() is True

    def test_zero_variance_writes_snr_flag_zero(self, tmp_path, caplog):
        # No errstate: a degenerate variance surfaces rather than being silenced,
        # and QC.run records it as a failed flag.
        l1 = _make_kpf1(tmp_path)
        l1.data["GREEN_VAR"][:] = 0.0
        with caplog.at_level(logging.ERROR):
            QCL1(l1).run()
        assert l1.headers["QUALITY_CONTROL"]["L1SNROK"] == 0
        assert "negative_snr_fraction" in caplog.text

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
            "variance_positive": "L1VAROK",
            "negative_snr_fraction": "L1SNROK",
        }
        for method_name, key in expected.items():
            fn = QCL1.__dict__[method_name]
            assert fn._qc_key == key, (
                f"{method_name}: expected {key!r}, got {fn._qc_key!r}"
            )


# ---------------------------------------------------------------------------
# Full QCL1 run on an all-good synthetic L1
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mini_detector")
class TestQCL1Run:
    def test_all_good_flags_1(self, tmp_path):
        l1 = _make_kpf1(tmp_path)
        results = QCL1(l1).run()

        # BIASOK/DARKOK/FLATOK read the RECEIPT *SUB flags and the *AGE values
        # but are themselves QUALITY_CONTROL keywords; the applied-step flags
        # (OSCANSUB/BIASSUB/DARKSUB/FLATDIV) stay RECEIPT-only provenance.
        # KWRDPRL1 is absent on purpose: its check is stubbed and writes no flag.
        qc_keys = [
            "DATAPRL1",
            "RNOK",
            "RNNGOK",
            "BIASOK",
            "DARKOK",
            "FLATOK",
            "L1NANOK",
            "L1VAROK",
            "L1SNROK",
        ]
        for k in qc_keys:
            v = l1.headers["QUALITY_CONTROL"].get(k)
            assert v == 1, f"{k} should be 1 but is {v}"
            assert k in results

    def test_one_bad_check_writes_0(self, tmp_path):
        l1 = _make_kpf1(tmp_path, biassub=False)
        QCL1(l1).run()

        # QC writes the BIASOK flag and never touches RECEIPT's BIASSUB.
        assert l1.headers["QUALITY_CONTROL"].get("BIASOK") == 0


# ---------------------------------------------------------------------------
# QCL2 checks
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mini_detector")
class TestQCL2:
    def test_extraction_present_pass(self):
        kpf2 = _make_kpf2_nan_headers()
        assert QCL2(kpf2).extraction_present() is True

    def test_required_keywords_present_is_stubbed(self):
        # KWRDPRL2 is pending a KPF-owned definition of "required"; until then
        # it raises NotImplementedError, which QC.run treats as "write no flag".
        with pytest.raises(NotImplementedError, match="KWRDPRL2"):
            QCL2(_make_kpf2_nan_headers()).required_keywords_present()

    def test_extraction_present_fail_empty_kpf2(self):
        kpf2 = KPF2()
        for k in ["NANSCI1", "NANSCI2", "NANSCI3", "NANSKY", "NANCAL"]:
            kpf2.headers["QUALITY_CONTROL"][k] = (0, k)
        for k in ["ZEROSCI1", "ZEROSCI2", "ZEROSCI3", "ZEROSKY", "ZEROCAL"]:
            kpf2.headers["QUALITY_CONTROL"][k] = (0, k)
        assert QCL2(kpf2).extraction_present() is False

    def test_extraction_present_fail_variance_missing(self):
        kpf2 = _make_kpf2_nan_headers()
        kpf2.set_data("SCI2_VAR", np.array([], dtype=np.float32))
        assert QCL2(kpf2).extraction_present() is False

    def test_extraction_present_fail_wavelength_missing(self):
        # WavelengthCalibration runs before CheckpointL2, so an unattached WLS is
        # an incomplete product.
        kpf2 = _make_kpf2_nan_headers()
        kpf2.set_data("SCI2_WAVE", np.array([], dtype=np.float64))
        assert QCL2(kpf2).extraction_present() is False

    def test_extraction_present_fail_barycentric_missing(self):
        kpf2 = _make_kpf2_nan_headers()
        kpf2.set_data("BARYCORR_Z", np.array([], dtype=np.float64))
        assert QCL2(kpf2).extraction_present() is False

    def test_extraction_present_fail_wrong_shape(self):
        kpf2 = _make_kpf2_nan_headers()
        kpf2.set_data(
            "SCI2_FLUX", np.ones((NORDER_TOTAL, _NCOLS - 1), dtype=np.float32)
        )
        assert QCL2(kpf2).extraction_present() is False

    def test_extraction_present_fail_all_nan_orderlet(self):
        # An orderlet that never reached the detector is NaN-filled by extraction.
        kpf2 = _make_kpf2_nan_headers()
        kpf2.set_data(
            "SCI2_FLUX", np.full((NORDER_TOTAL, _NCOLS), np.nan, dtype=np.float32)
        )
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

    def test_flux_finite_fraction_missing_header_raises(self):
        kpf2 = _make_kpf2_nan_headers(nan_frac=0.0)
        del kpf2.headers["QUALITY_CONTROL"]["NANSCI1"]
        with pytest.raises(KeyError, match="NANSCI1"):
            QCL2(kpf2).flux_finite_fraction()

    def test_flux_finite_fraction_no_extensions_raises(self):
        # No flux arrays means zero total pixels, so no fraction can be formed.
        kpf2 = KPF2()
        for k in ["NANSCI1", "NANSCI2", "NANSCI3", "NANSKY", "NANCAL"]:
            kpf2.headers["QUALITY_CONTROL"][k] = (0, k)
        with pytest.raises(ZeroDivisionError):
            QCL2(kpf2).flux_finite_fraction()

    def test_nonzero_flux_pass(self):
        kpf2 = _make_kpf2_nan_headers(zero_frac=0.1)
        assert QCL2(kpf2).nonzero_flux() is True

    def test_nonzero_flux_fail_high_frac(self):
        kpf2 = _make_kpf2_nan_headers(zero_frac=0.75)
        assert QCL2(kpf2).nonzero_flux() is False

    def test_nonzero_flux_missing_raises(self):
        kpf2 = _make_kpf2_nan_headers()
        del kpf2.headers["QUALITY_CONTROL"]["ZEROSCI1"]
        with pytest.raises(KeyError, match="ZEROSCI1"):
            QCL2(kpf2).nonzero_flux()

    def test_nonzero_flux_exactly_half(self):
        # The check is strictly < 0.5.
        kpf2 = _make_kpf2_nan_headers(zero_frac=0.5)
        assert QCL2(kpf2).nonzero_flux() is False

    def test_nonzero_flux_no_extensions_raises(self):
        kpf2 = KPF2()
        for k in ["ZEROSCI1", "ZEROSCI2", "ZEROSCI3", "ZEROSKY", "ZEROCAL"]:
            kpf2.headers["QUALITY_CONTROL"][k] = (0, k)
        with pytest.raises(ZeroDivisionError):
            QCL2(kpf2).nonzero_flux()

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
        # FLUX populated with VAR emptied back to its default (0,) shape is a
        # malformed product, so the shape mismatch raises rather than skipping.
        kpf2 = _make_kpf2_nan_headers()
        kpf2.set_data("SKY_VAR", np.array([], dtype=np.float32))
        with pytest.raises(ValueError, match="could not be broadcast"):
            QCL2(kpf2).variance_positive()

    # --- science_snr (L2SNROK) ---

    def test_science_snr_pass(self):
        kpf2 = _make_kpf2_nan_headers()
        for wavelength in (452, 548, 652, 747, 852):
            kpf2.headers["QUALITY_CONTROL"][f"SNRSC{wavelength}"] = (20.0, "snr")
        assert QCL2(kpf2).science_snr() is True

    def test_science_snr_missing_raises(self):
        kpf2 = _make_kpf2_nan_headers()  # no SNR* headers
        with pytest.raises(KeyError, match="SNRSC452"):
            QCL2(kpf2).science_snr()

    def test_science_snr_fail_below_floor(self):
        kpf2 = _make_kpf2_nan_headers()
        for wavelength in (452, 548, 652, 747, 852):
            kpf2.headers["QUALITY_CONTROL"][f"SNRSC{wavelength}"] = (20.0, "snr")
        kpf2.headers["QUALITY_CONTROL"]["SNRSC852"] = (0.5, "snr")  # below floor
        assert QCL2(kpf2).science_snr() is False

    def test_science_snr_ignores_sky_and_cal(self):
        # Neither carries starlight, so neither has an SNR floor.
        kpf2 = _make_kpf2_nan_headers()
        for wavelength in (452, 548, 652, 747, 852):
            kpf2.headers["QUALITY_CONTROL"][f"SNRSC{wavelength}"] = (20.0, "snr")
            for code in ("SK", "CL"):
                kpf2.headers["QUALITY_CONTROL"][f"SNR{code}{wavelength}"] = (0.0, "snr")
        assert QCL2(kpf2).science_snr() is True

    def test_qc_keys_correct(self):
        expected = {
            "extraction_present": "DATAPRL2",
            "required_keywords_present": "KWRDPRL2",
            "flux_finite_fraction": "L2NANOK",
            "nonzero_flux": "L2FLXOK",
            "variance_positive": "L2VAROK",
            "science_snr": "L2SNROK",
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

    def test_ccf_rv_present_fail_when_ccf_all_nan(self):
        # A cube of the right shape carrying no finite value is present but not
        # populated.
        l4 = make_l4()
        l4.set_data("SCI2_CCF", np.full((NORDER_TOTAL, 5), np.nan))
        assert QCL4(l4).ccf_rv_present() is False

    def test_ccf_rv_present_fail_when_ccf_var_missing(self):
        # CCF_VAR is written 1:1 with the CCF and carries the RV photon error.
        l4 = make_l4()
        l4.set_data("SCI2_CCF_VAR", np.array([], dtype=np.float64))
        assert QCL4(l4).ccf_rv_present() is False

    def test_ccf_rv_present_fail_when_orders_missing(self):
        # Both the cube and the table run over every order of both chips.
        l4 = make_l4()
        l4.set_data("SCI2_RV", Table(l4.data["SCI2_RV"])[: NORDER_TOTAL - 1])
        assert QCL4(l4).ccf_rv_present() is False

    def test_ccf_rv_present_fail_when_columns_missing(self):
        # The EPRV-required columns, and the per-order BJD_TDB/BERV/WEIGHT the
        # DiagL4 dispersion metrics consume, are absent: the product is incomplete.
        l4 = KPF4()
        for fiber in ("SCI1", "SCI2", "SCI3"):
            l4.set_data(f"{fiber}_CCF", np.ones((NORDER_TOTAL, 5)))
            l4.set_data(f"{fiber}_CCF_VAR", np.ones((NORDER_TOTAL, 5)))
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
        l4 = make_l4(bervrng=0.02)
        assert QCL4(l4).berv_within_tolerance() is True

    def test_berv_within_tolerance_fail(self):
        l4 = make_l4(bervrng=0.5)
        assert QCL4(l4).berv_within_tolerance() is False

    def test_berv_within_tolerance_metric_absent_raises(self):
        # BERVRNG is required; DiagL4 either emits it or raises.
        with pytest.raises(KeyError, match="BERVRNG"):
            QCL4(make_l4()).berv_within_tolerance()

    def test_bjd_within_tolerance_pass(self):
        l4 = make_l4(bjdrng=0.5)
        assert QCL4(l4).bjd_within_tolerance() is True

    def test_bjd_within_tolerance_fail(self):
        l4 = make_l4(bjdrng=2.0)
        assert QCL4(l4).bjd_within_tolerance() is False

    def test_bjd_within_tolerance_metric_absent_raises(self):
        with pytest.raises(KeyError, match="BJDRNG"):
            QCL4(make_l4()).bjd_within_tolerance()

    def test_required_keywords_present_is_stubbed(self):
        # KWRDPRL4 is pending a KPF-owned definition of "required"; until then
        # it raises NotImplementedError, which QC.run treats as "write no flag".
        with pytest.raises(NotImplementedError, match="KWRDPRL4"):
            QCL4(make_l4()).required_keywords_present()

    def test_run_all_good(self):
        l4 = make_l4(bervrng=0.02, bjdrng=0.5)
        results = QCL4(l4).run()
        assert set(results) >= {"DATAPRL4", "BERVOK", "BJDOK"}
        # The stubbed KWRDPRL4 writes no flag at all.
        assert "KWRDPRL4" not in results
        qc = l4.headers["QUALITY_CONTROL"]
        assert qc["DATAPRL4"] == 1 and qc["BERVOK"] == 1 and qc["BJDOK"] == 1
        assert "KWRDPRL4" not in qc

    def test_run_flags_failure(self):
        # no CCF/RV, and out-of-tolerance BERV/BJD ranges
        l4 = make_l4(sci=False, bervrng=0.5, bjdrng=2.0)
        QCL4(l4).run()
        qc = l4.headers["QUALITY_CONTROL"]
        assert qc["DATAPRL4"] == 0 and qc["BERVOK"] == 0 and qc["BJDOK"] == 0


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
                f"{qc_cls.__name__} writes {key}, which no config/*-keywords.csv "
                "row registers"
            )
            assert key in registry.qc_flag_keywords_by_level[level], (
                f"{key} is not tagged as a {level} QC flag in the registry, so "
                "Checkpoint.qc_flags would never scan it"
            )
