"""
Tests for the KPF QC framework (Tasks 2-7).

Covers:
  - QC base class runner (TestQCBase)
  - QCL0 checks (TestQCL0)
  - QCL1 checks (TestQCL1)
  - QCL1 full run on an all-good synthetic L1 (TestQCL1Run)
  - QCL2 checks (TestQCL2)
  - CLI smoke tests (TestQCScript)

All tests use synthetic in-memory data — no real KPF files required.
"""

import os
import subprocess
import sys
import types

import numpy as np
import pytest
from astropy.io import fits

from kpfpipe import DETECTOR
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.quality_control.qc_flags.base import QC
from kpfpipe.quality_control.qc_flags.level0 import QCL0
from kpfpipe.quality_control.qc_flags.level1 import QCL1
from kpfpipe.quality_control.qc_flags.level2 import QCL2

_NORDER = {"GREEN": DETECTOR["norder"]["GREEN"], "RED": DETECTOR["norder"]["RED"]}
_NCOLS = 10  # small column count for fast tests


# ---------------------------------------------------------------------------
# Helpers: build minimal synthetic KPF objects in memory
# ---------------------------------------------------------------------------


def _make_kpf0(
    tmp_path, *, with_amps=True, exptime=60.0, obs_id="KP.20240405.00001.00"
):
    """Minimal 4-amp KPF0 object with required headers."""
    fn = str(tmp_path / f"{obs_id}.fits")
    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-04-05T01:00:37"
    primary.header["EXPTIME"] = exptime
    primary.header["OBJECT"] = "synthetic"
    primary.header["OFNAME"] = f"{obs_id}.fits"
    primary.header["IMTYPE"] = "Object"

    hdus = [primary]
    if with_amps:
        for chip in ["GREEN", "RED"]:
            for amp in range(1, 5):
                data = np.ones((10, 10), dtype=np.float32)
                hdus.append(fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}"))

    fits.HDUList(hdus).writeto(fn, overwrite=True)
    return KPF0.from_fits(fn)


def _seed_required_primary(kpf, qc_cls):
    """Seed every PRIMARY keyword ``qc_cls`` requires present, skipping any already
    set. The KWRDPR* check is presence-only, so placeholder sentinel values are
    fine; reusing the production ``_required_primary_keywords`` avoids drift.
    """
    for kw in qc_cls(kpf)._required_primary_keywords():
        if kw not in kpf.headers["PRIMARY"]:
            kpf.headers["PRIMARY"][kw] = ("UNKNOWN", "seeded for test")


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

    QC-relevant keywords now live on their registry-home extensions rather than
    PRIMARY: the applied-step flags (OSCANSUB/BIASSUB/DARKSUB/FLATDIV) on
    RECEIPT, and the read-noise and master-age keywords on QUALITY_CONTROL.
    They are seeded on the loaded object after ``from_fits`` returns.
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

    # Seed the full required-PRIMARY set so KWRDPRL1 (presence of all required
    # keywords) passes. KPF1.__init__ seeds the EPRV skeleton, but KPF1._read
    # replaces PRIMARY with the file's (sparse) header, so a synthetic from_fits L1
    # is otherwise missing them.
    _seed_required_primary(l1, QCL1)
    return l1


def _make_kpf2_with_flux(*, nan_frac=0.0, zero_frac=0.1, missing_ext=None):
    """Minimal KPF2 with all 10 {CHIP}_{FIBER}_FLUX extensions populated.

    Uses per-chip shapes matching NORDER_GREEN (35) and NORDER_RED (32)
    because KPF2's chip-prefix __setitem__ requires the correct row count.

    nan_frac: fraction of total pixels that are NaN (injected via NANSCI* headers).
    zero_frac: value written to ZEROFRAC header.
    missing_ext: optional chip_fiber key to leave empty (e.g. "GREEN_SKY_FLUX").
    """
    chips = ["GREEN", "RED"]
    fibers = ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]
    ncols = _NCOLS

    kpf2 = KPF2()

    total_pixels = 0
    for chip in chips:
        nrows = _NORDER[chip]
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


def _set_kpf2_var(kpf2, fill=1.0):
    """Populate all {CHIP}_{FIBER}_VAR extensions with a constant fill."""
    for chip in ["GREEN", "RED"]:
        nrows = _NORDER[chip]
        for fiber in ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]:
            kpf2.set_data(
                f"{chip}_{fiber}_VAR", np.full((nrows, _NCOLS), fill, dtype=np.float32)
            )


# ---------------------------------------------------------------------------
# Task 2: QC base class runner
# ---------------------------------------------------------------------------


class TestQCBase:
    """Runner behaviour: aggregation, failure, raises, empty."""

    def _make_obj(self):
        """Minimal object with a headers dict and a set_keyword router.

        QC.run() writes each result via ``set_keyword`` and aggregates ISGOOD; it
        does NOT validate headers (that moved to the checkpoints layer). It reads
        each check's comment off ``keyword_registry.routing`` (empty here -- these
        synthetic keys aren't registered, so comment "") and derives ISGOOD as the
        AND over ``keyword_registry.qc_flag_keywords`` present on QUALITY_CONTROL,
        so the stub declares the synthetic check keys as the QC-flag set. This fake
        stores every keyword on QUALITY_CONTROL (value only).
        """

        class _FakeObj:
            headers = {"PRIMARY": {}, "QUALITY_CONTROL": {}}
            keyword_registry = types.SimpleNamespace(
                routing={},
                qc_flag_keywords=frozenset(
                    {"CHECKA", "CHECKB", "CHKOK", "CHKFAIL", "FLAG", "ISGOOD"}
                ),
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

    def test_raising_check_propagates_runtime_error(self):
        obj = self._make_obj()

        class MyQC(QC):
            def check_boom(self):
                raise ValueError("boom!")

            check_boom._qc_key = "BOOM"

        with pytest.raises(RuntimeError, match="QC check 'check_boom' raised"):
            MyQC(obj).run()

    def test_empty_subclass_isgood_1(self):
        obj = self._make_obj()

        class EmptyQC(QC):
            pass

        results = EmptyQC(obj).run()
        assert results == {}
        assert obj.headers["QUALITY_CONTROL"]["ISGOOD"] == 1

    def test_repeated_run_resets_results(self):
        """Calling run() twice on the same instance should not accumulate state.

        First run with one failing check → ISGOOD=0. Mutate underlying state to make
        the check pass, run again → ISGOOD=1. Without the reset, the failed result
        from the first run would still be in self.results and ISGOOD would stay 0.
        """
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


# ---------------------------------------------------------------------------
# Task 3: QCL0 checks
# ---------------------------------------------------------------------------


class TestQCL0:
    def test_data_l0_red_green_pass(self, tmp_path):
        l0 = _make_kpf0(tmp_path, with_amps=True)
        assert QCL0(l0).data_l0_red_green() is True

    def test_data_l0_red_green_fail_missing(self, tmp_path):
        l0 = _make_kpf0(tmp_path, with_amps=False)
        assert QCL0(l0).data_l0_red_green() is False

    def test_data_l0_red_green_fail_empty(self, tmp_path):
        """An extension present with data=None (stored as array(None)) should fail."""
        fn = str(tmp_path / "KP.20240405.00002.00.fits")
        primary = fits.PrimaryHDU()
        primary.header["DATE-OBS"] = "2024-04-05T01:00:37"
        hdus = [primary]
        for chip in ["GREEN", "RED"]:
            for amp in range(1, 5):
                # data=None → KPF0 stores array(None, dtype=object) — treated as absent.
                hdus.append(fits.ImageHDU(data=None, name=f"{chip}_AMP{amp}"))
        fits.HDUList(hdus).writeto(fn, overwrite=True)
        l0 = KPF0.from_fits(fn)
        assert QCL0(l0).data_l0_red_green() is False

    def test_data_l0_red_green_pass_two_amp(self, tmp_path):
        """2-amp readout (only AMP1/AMP2 per chip; AMP3/4 absent) — the truth-frame
        layout — has data present and must pass."""
        fn = str(tmp_path / "KP.20240405.00003.00.fits")
        primary = fits.PrimaryHDU()
        primary.header["DATE-OBS"] = "2024-04-05T01:00:37"
        hdus = [primary]
        for chip in ["GREEN", "RED"]:
            for amp in (1, 2):
                data = np.ones((10, 10), dtype=np.float32)
                hdus.append(fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}"))
        fits.HDUList(hdus).writeto(fn, overwrite=True)
        l0 = KPF0.from_fits(fn)
        assert QCL0(l0).data_l0_red_green() is True

    def test_data_l0_red_green_fail_partial_amp(self, tmp_path):
        """A partial/invalid amp set (3 present) is not a supported readout mode
        (2 or 4), so the inferred count is rejected."""
        fn = str(tmp_path / "KP.20240405.00004.00.fits")
        primary = fits.PrimaryHDU()
        primary.header["DATE-OBS"] = "2024-04-05T01:00:37"
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
        """Remove OFNAME from PRIMARY so check fails."""
        l0 = _make_kpf0(tmp_path)
        del l0.headers["PRIMARY"]["OFNAME"]
        assert QCL0(l0).header_keywords_present() is False

    def test_exptime_sane_pass_positive(self, tmp_path):
        l0 = _make_kpf0(tmp_path, exptime=300.0)
        assert QCL0(l0).exptime_sane() is True

    def test_exptime_sane_pass_zero(self, tmp_path):
        """Bias frames legitimately have EXPTIME=0; should pass."""
        l0 = _make_kpf0(tmp_path, exptime=0.0)
        assert QCL0(l0).exptime_sane() is True

    def test_exptime_sane_fail_negative(self, tmp_path):
        l0 = _make_kpf0(tmp_path, exptime=-1.0)
        assert QCL0(l0).exptime_sane() is False

    def test_exptime_sane_fail_missing(self, tmp_path):
        l0 = _make_kpf0(tmp_path)
        del l0.headers["PRIMARY"]["EXPTIME"]
        assert QCL0(l0).exptime_sane() is False

    def test_not_junk_pass_no_file(self, tmp_path, monkeypatch):
        """No junk CSV → pass by default."""
        import kpfpipe.quality_control.qc_flags.level0 as mod

        monkeypatch.setattr(
            mod, "_JUNK_CSV", tmp_path / "reference" / "junk_observations.csv"
        )
        l0 = _make_kpf0(tmp_path)
        assert QCL0(l0).not_junk() is True

    def test_not_junk_pass_not_in_list(self, tmp_path, monkeypatch):
        import pandas as pd

        import kpfpipe.quality_control.qc_flags.level0 as mod

        csv_path = tmp_path / "junk_observations.csv"
        pd.DataFrame({"obs_id": ["KP.20240101.99999.00"]}).to_csv(csv_path, index=False)
        monkeypatch.setattr(mod, "_JUNK_CSV", csv_path)

        l0 = _make_kpf0(tmp_path)
        assert QCL0(l0).not_junk() is True

    def test_not_junk_fail_in_list(self, tmp_path, monkeypatch):
        import pandas as pd

        import kpfpipe.quality_control.qc_flags.level0 as mod

        obs_id = "KP.20240405.00001.00"
        csv_path = tmp_path / "junk_observations.csv"
        pd.DataFrame({"obs_id": [obs_id]}).to_csv(csv_path, index=False)
        monkeypatch.setattr(mod, "_JUNK_CSV", csv_path)

        l0 = _make_kpf0(tmp_path, obs_id=obs_id)
        assert QCL0(l0).not_junk() is False

    def test_not_junk_pass_none_obs_id(self, tmp_path, monkeypatch):
        """obs_id=None → passes (can't be in junk list)."""
        import pandas as pd

        import kpfpipe.quality_control.qc_flags.level0 as mod

        csv_path = tmp_path / "junk_observations.csv"
        pd.DataFrame({"obs_id": ["KP.20240405.00001.00"]}).to_csv(csv_path, index=False)
        monkeypatch.setattr(mod, "_JUNK_CSV", csv_path)

        l0 = _make_kpf0(tmp_path)
        l0.obs_id = None
        assert QCL0(l0).not_junk() is True

    def test_not_junk_malformed_csv_raises(self, tmp_path, monkeypatch):
        """CSV without 'obs_id' column → raises ValueError."""
        import pandas as pd

        import kpfpipe.quality_control.qc_flags.level0 as mod

        csv_path = tmp_path / "junk_observations.csv"
        pd.DataFrame({"wrong_col": ["whatever"]}).to_csv(csv_path, index=False)
        monkeypatch.setattr(mod, "_JUNK_CSV", csv_path)

        l0 = _make_kpf0(tmp_path)
        with pytest.raises(ValueError, match="obs_id"):
            QCL0(l0).not_junk()

    def test_not_junk_key_present(self):
        qc = QCL0.__dict__["not_junk"]
        assert qc._qc_key == "NOTJUNK"

    def test_dataprl0_key_and_comment(self):
        fn = QCL0.__dict__["data_l0_red_green"]
        assert fn._qc_key == "DATAPRL0"
        # The comment now lives in the registry Description (not on the method).
        assert "GREEN" in KPF0.keyword_registry.routing["DATAPRL0"][1]


# ---------------------------------------------------------------------------
# Task 4: QCL1 checks
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
        # Remove GREEN_CCD from data
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
            "ffi_finite": "FFIOK",
        }
        for method_name, key in expected.items():
            fn = QCL1.__dict__[method_name]
            assert fn._qc_key == key, (
                f"{method_name}: expected {key!r}, got {fn._qc_key!r}"
            )


# ---------------------------------------------------------------------------
# Task 4 integration: full QCL1 run on all-good synthetic L1
# ---------------------------------------------------------------------------


class TestQCL1Run:
    def test_all_good_isgood_1(self, tmp_path):
        l1 = _make_kpf1(tmp_path)
        results = QCL1(l1).run()

        isgood = l1.headers["QUALITY_CONTROL"].get("ISGOOD")
        assert isgood == 1

        # Every L1 QC flag lands on QUALITY_CONTROL. The per-calibration OK flags
        # (BIASOK/DARKOK/FLATOK) read the RECEIPT *SUB flags and DiagL1 *AGE
        # values but are themselves QUALITY_CONTROL keywords. The applied-step
        # flags (OSCANSUB/BIASSUB/DARKSUB/FLATDIV) stay RECEIPT-only provenance
        # and are not QC checks.
        qc_keys = [
            "DATAPRL1",
            "KWRDPRL1",
            "RNOK",
            "RNNGOK",
            "BIASOK",
            "DARKOK",
            "FLATOK",
            "FFIOK",
        ]
        for k in qc_keys:
            v = l1.headers["QUALITY_CONTROL"].get(k)
            assert v == 1, f"{k} should be 1 but is {v}"
            assert k in results

    def test_one_bad_check_isgood_0(self, tmp_path):
        l1 = _make_kpf1(tmp_path, biassub=False)
        QCL1(l1).run()
        isgood = l1.headers["QUALITY_CONTROL"].get("ISGOOD")
        assert isgood == 0

        # Bias not subtracted -> BIASOK fails (a QUALITY_CONTROL flag); the
        # RECEIPT BIASSUB provenance keyword is untouched by QC.
        assert l1.headers["QUALITY_CONTROL"].get("BIASOK") == 0

    def test_isgood_aggregates_propagated_flag(self, tmp_path):
        """ISGOOD is the running aggregate over EVERY QC flag on QUALITY_CONTROL,
        including ones propagated from a lower level -- not just this level's
        checks. Seed a failed L0 flag; all L1 checks still pass, yet ISGOOD=0."""
        l1 = _make_kpf1(tmp_path)
        l1.headers["QUALITY_CONTROL"]["DATAPRL0"] = (0, "L0 data present (propagated)")
        QCL1(l1).run()
        # This level's own checks all pass...
        assert l1.headers["QUALITY_CONTROL"].get("DATAPRL1") == 1
        # ...but the propagated L0 failure drags the aggregate down.
        assert l1.headers["QUALITY_CONTROL"].get("ISGOOD") == 0


# ---------------------------------------------------------------------------
# Task 5: QCL2 checks
# ---------------------------------------------------------------------------


class TestQCL2:
    def test_extraction_present_pass(self):
        kpf2 = _make_kpf2_with_flux()
        assert QCL2(kpf2).extraction_present() is True

    def test_required_keywords_present_pass(self):
        # Fresh KPF2 is rvdata-seeded with the EPRV-required PRIMARY keywords; seed
        # the KPF provenance cards too so all 44 are present.
        kpf2 = _make_kpf2_with_flux()
        _seed_required_primary(kpf2, QCL2)
        assert QCL2(kpf2).required_keywords_present() is True

    def test_required_keywords_present_fail_missing(self):
        kpf2 = _make_kpf2_with_flux()
        _seed_required_primary(kpf2, QCL2)
        del kpf2.headers["PRIMARY"]["INSTRUME"]
        assert QCL2(kpf2).required_keywords_present() is False

    def test_extraction_present_fail_empty_kpf2(self):
        """A freshly constructed KPF2 with no flux data → all chip-prefix sizes=0."""
        kpf2 = KPF2()
        for k in ["NANSCI1", "NANSCI2", "NANSCI3", "NANSKY", "NANCAL"]:
            kpf2.headers["QUALITY_CONTROL"][k] = (0, k)
        kpf2.headers["QUALITY_CONTROL"]["ZEROFRAC"] = (0.0, "z")
        assert QCL2(kpf2).extraction_present() is False

    def test_extraction_present_fail_one_trace_cleared(self):
        """Clearing a trace array (set to empty array) should fail the check."""
        kpf2 = _make_kpf2_with_flux()
        # Resolve alias SKY_FLUX → TRACE1_FLUX and set to empty so chip views
        # are size=0.
        kpf2.data["SKY_FLUX"] = np.array([], dtype=np.float32)
        assert QCL2(kpf2).extraction_present() is False

    def test_flux_finite_fraction_pass(self):
        """0 NaN entries → passes."""
        kpf2 = _make_kpf2_with_flux(nan_frac=0.0)
        assert QCL2(kpf2).flux_finite_fraction() is True

    def test_flux_finite_fraction_fail_too_many_nans(self):
        """Force nan_total/total_pixels > 1%."""
        kpf2 = _make_kpf2_with_flux()
        # total_pixels = (35+32) * 5 fibers * _NCOLS (10) = 3350
        # Set each NAN key to 200 → sum = 1000 → ~30% > 1%
        for k in ["NANSCI1", "NANSCI2", "NANSCI3", "NANSKY", "NANCAL"]:
            kpf2.headers["QUALITY_CONTROL"][k] = (200, k)
        assert QCL2(kpf2).flux_finite_fraction() is False

    def test_flux_finite_fraction_fail_missing_header(self):
        """Missing one NAN key → False."""
        kpf2 = _make_kpf2_with_flux(nan_frac=0.0)
        del kpf2.headers["QUALITY_CONTROL"]["NANSCI1"]
        assert QCL2(kpf2).flux_finite_fraction() is False

    def test_flux_finite_fraction_fail_no_extensions(self):
        """Empty KPF2 (no flux arrays) → False (zero total pixels)."""
        kpf2 = KPF2()
        for k in ["NANSCI1", "NANSCI2", "NANSCI3", "NANSKY", "NANCAL"]:
            kpf2.headers["QUALITY_CONTROL"][k] = (0, k)
        assert QCL2(kpf2).flux_finite_fraction() is False

    def test_nonzero_flux_pass(self):
        kpf2 = _make_kpf2_with_flux(zero_frac=0.1)
        assert QCL2(kpf2).nonzero_flux() is True

    def test_nonzero_flux_fail_high_frac(self):
        kpf2 = _make_kpf2_with_flux(zero_frac=0.75)
        assert QCL2(kpf2).nonzero_flux() is False

    def test_nonzero_flux_fail_missing(self):
        kpf2 = _make_kpf2_with_flux()
        del kpf2.headers["QUALITY_CONTROL"]["ZEROFRAC"]
        assert QCL2(kpf2).nonzero_flux() is False

    def test_nonzero_flux_exactly_half(self):
        """ZEROFRAC == 0.5 should fail (check is strictly < 0.5)."""
        kpf2 = _make_kpf2_with_flux(zero_frac=0.5)
        assert QCL2(kpf2).nonzero_flux() is False

    # --- variance_positive (L2VAROK) ---

    def test_variance_positive_pass(self):
        kpf2 = _make_kpf2_with_flux()
        _set_kpf2_var(kpf2, 1.0)
        assert QCL2(kpf2).variance_positive() is True

    def test_variance_positive_tolerates_zero(self):
        kpf2 = _make_kpf2_with_flux()
        _set_kpf2_var(kpf2, 0.0)  # zero variance is allowed
        assert QCL2(kpf2).variance_positive() is True

    def test_variance_positive_fail_negative(self):
        kpf2 = _make_kpf2_with_flux()
        _set_kpf2_var(kpf2, 1.0)
        var = np.full((_NORDER["GREEN"], _NCOLS), 1.0, dtype=np.float32)
        var[0, 0] = -1.0  # negative variance where flux is finite
        kpf2.set_data("GREEN_SCI1_VAR", var)
        assert QCL2(kpf2).variance_positive() is False

    def test_variance_positive_fail_no_var(self):
        kpf2 = _make_kpf2_with_flux()  # no VAR populated
        assert QCL2(kpf2).variance_positive() is False

    # --- science_snr (L2SNROK) ---

    def test_science_snr_pass(self):
        kpf2 = _make_kpf2_with_flux()
        kpf2.headers["QUALITY_CONTROL"]["GSNRSCI"] = (20.0, "g snr")
        kpf2.headers["QUALITY_CONTROL"]["RSNRSCI"] = (18.0, "r snr")
        assert QCL2(kpf2).science_snr() is True

    def test_science_snr_fail_missing(self):
        kpf2 = _make_kpf2_with_flux()  # no GSNRSCI/RSNRSCI headers
        assert QCL2(kpf2).science_snr() is False

    def test_science_snr_fail_below_floor(self):
        kpf2 = _make_kpf2_with_flux()
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
        }
        for method_name, key in expected.items():
            fn = QCL2.__dict__[method_name]
            assert fn._qc_key == key, (
                f"{method_name}: expected {key!r}, got {fn._qc_key!r}"
            )


# ---------------------------------------------------------------------------
# Task 6: CLI smoke tests
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _write_l0_fixture(path, *, passing=True):
    """Write a minimal L0 FITS fixture at path.

    passing=True  → all QCL0 checks pass (valid header keywords, finite EXPTIME,
                    amps present).
    passing=False → inject a failure (negative EXPTIME so EXPTIMOK fails).
    """
    primary = fits.PrimaryHDU()
    primary.header["DATE-OBS"] = "2024-04-05T01:00:37"
    primary.header["EXPTIME"] = 60.0 if passing else -1.0
    primary.header["OBJECT"] = "synthetic"
    primary.header["OFNAME"] = os.path.basename(path)
    primary.header["IMTYPE"] = "Object"

    hdus = [primary]
    for chip in ["GREEN", "RED"]:
        for amp in range(1, 5):
            data = np.ones((10, 10), dtype=np.float32)
            hdus.append(fits.ImageHDU(data=data, name=f"{chip}_AMP{amp}"))

    fits.HDUList(hdus).writeto(path, overwrite=True)


def _run_qc_script(fixture_path, level="L0", extra_args=None):
    """Run scripts/qc.py via subprocess and return the CompletedProcess."""
    cmd = [
        sys.executable,
        "scripts/qc.py",
        "--input",
        str(fixture_path),
        "--level",
        level,
    ]
    if extra_args:
        cmd.extend(extra_args)
    env = {**os.environ, "PYTHONPATH": _REPO_ROOT}
    return subprocess.run(cmd, cwd=_REPO_ROOT, env=env, capture_output=True, text=True)


class TestQCScript:
    """Smoke tests for scripts/qc.py via subprocess."""

    def test_all_passing_exit_0_isgood_pass(self, tmp_path):
        """All-good L0 → exit code 0, stdout contains 'ISGOOD: PASS'."""
        fixture = tmp_path / "KP.20240405.00001.00.fits"
        _write_l0_fixture(str(fixture), passing=True)

        result = _run_qc_script(fixture, level="L0")

        assert result.returncode == 0, (
            f"Expected exit 0, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "ISGOOD: PASS" in result.stdout, (
            f"Expected 'ISGOOD: PASS' in stdout:\n{result.stdout}"
        )

    def test_failure_injected_exit_1_isgood_fail(self, tmp_path):
        """L0 with negative EXPTIME → exit code 1, stdout contains 'ISGOOD: FAIL'."""
        fixture = tmp_path / "KP.20240405.00002.00.fits"
        _write_l0_fixture(str(fixture), passing=False)

        result = _run_qc_script(fixture, level="L0")

        assert result.returncode == 1, (
            f"Expected exit 1, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "ISGOOD: FAIL" in result.stdout, (
            f"Expected 'ISGOOD: FAIL' in stdout:\n{result.stdout}"
        )

    def test_missing_file_exit_2(self, tmp_path):
        """Non-existent file → exit code 2."""
        missing = tmp_path / "does_not_exist.fits"
        result = _run_qc_script(missing, level="L0")
        assert result.returncode == 2

    def test_no_args_exit_nonzero(self):
        """No args → argparse error → non-zero exit."""
        env = {**os.environ, "PYTHONPATH": _REPO_ROOT}
        result = subprocess.run(
            [sys.executable, "scripts/qc.py"],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
