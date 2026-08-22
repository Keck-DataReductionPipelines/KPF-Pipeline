"""Tests for the Checkpoint layer (quality_control/checkpoints).

Checkpoints are the third QC stage: they read the 0/1 QC flags and the product
headers, then warn or raise (never write). This pins:

  - ``unregistered_keywords`` -- raises on a card not registered for a governed
    extension (including a raw WMKO native leaked onto an EPRV PRIMARY); skips the
    raw WMKO L0 PRIMARY; passes a clean product.
  - ``qc_flags`` -- a failed (0) flag named in the level's ``RAISE_FLAGS`` raises;
    any other failed flag warns; all-pass is silent.

``run()`` additionally folds in the paired Diagnostics + QC stages before the
checkpoint methods; that orchestration is pinned in ``TestRunFoldsDiagnosticsAndQC``.
"""

import logging

import numpy as np
import pytest

from kpfpipe import DETECTOR
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.quality_control.checkpoints import (
    Checkpoint,
    CheckpointL0,
    CheckpointL1,
    CheckpointL2,
    CheckpointL4,
)

from ._data_models import (
    GOOD_DATES,
    make_l4,
    seed_required_primary,
    set_fiber_arrays,
    write_amp_l0,
)

_NORDER_TOTAL = DETECTOR["norder"]["GREEN"] + DETECTOR["norder"]["RED"]
_NCOL = 8  # DiagL2 metrics are pixel aggregates; detector width is moot here


class TestUnregisteredKeywords:
    def test_clean_product_passes(self, caplog):
        # A fresh KPF2 has no unregistered card and no QC flags, so both methods
        # are silent. run() is not used here: it folds in QC, which a bare product
        # cannot satisfy.
        l2 = KPF2()
        with caplog.at_level(logging.WARNING):
            chk = CheckpointL2(l2)
            chk.unregistered_keywords()
            chk.qc_flags()
        assert not caplog.records

    def test_unexpected_keyword_on_governed_extension_raises(self):
        l2 = KPF2()
        l2.headers["QUALITY_CONTROL"]["BOGUSKEY"] = (1, "not registered")
        with pytest.raises(ValueError, match="unregistered keyword 'BOGUSKEY'"):
            CheckpointL2(l2).unregistered_keywords()

    def test_native_wmko_leak_on_primary_raises(self):
        l2 = KPF2()
        # GAIAID is a raw WMKO native (kept in INSTRUMENT_HEADER, never on
        # PRIMARY), so the general unregistered-keyword check catches it -- no
        # dedicated WMKO-leak branch needed.
        l2.headers["PRIMARY"]["GAIAID"] = (12345, "leaked native")
        with pytest.raises(ValueError, match="unregistered keyword 'GAIAID'"):
            CheckpointL2(l2).unregistered_keywords()

    def test_registered_keyword_on_its_extension_passes(self):
        l2 = KPF2()
        l2.set_keyword("NANSCI1", 3)  # registered -> QUALITY_CONTROL
        CheckpointL2(l2).unregistered_keywords()  # no raise

    def test_raw_l0_primary_is_skipped(self):
        # The raw WMKO L0 PRIMARY is not registry-governed, so an unregistered
        # native on it must NOT raise at L0.
        l0 = KPF0()
        l0.headers["PRIMARY"]["GAIAID"] = (12345, "raw native")
        CheckpointL0(l0).unregistered_keywords()  # no raise


class TestQCFlags:
    def test_raise_flag_zero_raises(self):
        # DATAPRL2 is in CheckpointL2.RAISE_FLAGS, so a 0 is fatal.
        l2 = KPF2()
        l2.headers["QUALITY_CONTROL"]["DATAPRL2"] = (0, "data present")
        with pytest.raises(ValueError, match="DATAPRL2 = 0"):
            CheckpointL2(l2).qc_flags()

    def test_nonraise_flag_zero_warns(self, caplog):
        # L2VAROK is not a RAISE_FLAG, so a 0 lands in the ISGOOD summary rather
        # than raising (DATAPRL2/KWRDPRL2 = 1, so no fatal flag).
        l2 = KPF2()
        l2.headers["QUALITY_CONTROL"]["DATAPRL2"] = (1, "data present")
        l2.headers["QUALITY_CONTROL"]["KWRDPRL2"] = (1, "required present")
        l2.headers["QUALITY_CONTROL"]["L2VAROK"] = (0, "variance positive")
        with caplog.at_level(logging.WARNING):
            CheckpointL2(l2).qc_flags()
        assert "ISGOOD=0" in caplog.text
        assert "L2VAROK" in caplog.text

    def test_all_pass_silent(self, caplog):
        l2 = KPF2()
        l2.headers["QUALITY_CONTROL"]["DATAPRL2"] = (1, "data present")
        l2.headers["QUALITY_CONTROL"]["KWRDPRL2"] = (1, "required present")
        with caplog.at_level(logging.WARNING):
            CheckpointL2(l2).qc_flags()
        assert not caplog.records

    def test_lower_level_fatal_flag_warns_not_raises(self, caplog):
        # DATAPRL1 is fatal at L1 but is not an L2 flag, so an L2 checkpoint must
        # warn about it, not abort: the L1 checkpoint already had its chance to
        # raise. The fatal scan reads qc_flag_keywords_by_level[LEVEL] while the
        # warning summary reads the full cross-level set; collapsing the two
        # would turn every upstream warn into a downstream pipeline abort, and
        # every other test in this file would still pass.
        l2 = KPF2()
        l2.headers["QUALITY_CONTROL"]["DATAPRL2"] = (1, "data present")
        l2.headers["QUALITY_CONTROL"]["KWRDPRL2"] = (1, "required present")
        l2.headers["QUALITY_CONTROL"]["DATAPRL1"] = (0, "L1 data (propagated)")
        with caplog.at_level(logging.WARNING):
            CheckpointL2(l2).qc_flags()  # must not raise
        assert "DATAPRL1" in caplog.text

    def test_summary_lists_all_failing_flags_cross_level(self, caplog):
        # The ISGOOD summary names every failing flag on QUALITY_CONTROL,
        # including one propagated from a lower level (RNOK from L1).
        l2 = KPF2()
        l2.headers["QUALITY_CONTROL"]["DATAPRL2"] = (1, "data present")  # avoid raise
        l2.headers["QUALITY_CONTROL"]["KWRDPRL2"] = (
            1,
            "required present",
        )  # avoid raise
        l2.headers["QUALITY_CONTROL"]["RNOK"] = (0, "L1 read noise (propagated)")
        l2.headers["QUALITY_CONTROL"]["L2VAROK"] = (0, "L2 variance positive")
        with caplog.at_level(logging.WARNING):
            CheckpointL2(l2).qc_flags()
        assert "L2VAROK" in caplog.text
        assert "RNOK" in caplog.text


class TestRunFoldsDiagnosticsAndQC:
    """``run()`` runs the paired Diagnostics, then QC, then the checkpoint methods."""

    def test_run_order_and_qc_results_capture(self):
        calls = []

        class FakeDiag:
            def __init__(self, obj):
                pass

            def run(self):
                calls.append("diag")
                return {}

        class FakeQC:
            def __init__(self, obj):
                pass

            def run(self):
                calls.append("qc")
                return {"ISGOOD": (True, "")}

        class FakeCheckpoint(Checkpoint):
            LEVEL = "L2"
            DIAGNOSTICS = FakeDiag
            QC = FakeQC

            def probe(self):
                calls.append("checkpoint")

            probe._checkpoint_name = "probe"

        chk = FakeCheckpoint(KPF2())
        chk.run()

        # Diagnostics first, QC second, then the checkpoint method(s).
        assert calls[0] == "diag"
        assert calls[1] == "qc"
        assert "checkpoint" in calls[2:]
        # QC's result dict is captured for callers (e.g. scripts/quality_control/qc.py).
        assert chk.qc_results == {"ISGOOD": (True, "")}

    def test_missing_paired_classes_skip_those_stages(self, caplog):
        # A concrete-level checkpoint with DIAGNOSTICS = QC = None: run() does the
        # checkpoint methods only and leaves qc_results empty. (LEVEL must be a
        # recognized level -- qc_flags() looks it up directly, no silent default.)
        class NoStageCheckpoint(Checkpoint):
            LEVEL = "L2"

        chk = NoStageCheckpoint(KPF2())  # clean, empty -> checkpoint methods silent
        with caplog.at_level(logging.WARNING):
            chk.run()
        assert chk.qc_results == {}
        assert not caplog.records


# ---------------------------------------------------------------------------
# CheckpointL0 / CheckpointL2 -- the two run() seams the recipe drives
# ---------------------------------------------------------------------------
#
# recipes/kpf_drp_science.py calls CheckpointL0(l0).run() and CheckpointL2(l2).run()
# on the production path, but neither was exercised in process anywhere, so
# QCL0.run() and QCL2.run() never ran outside a full recipe or the CLI suite that
# `make test-fast` excludes. That left the DiagL2 -> QCL2 handshake unpinned:
# DiagL2 writes NANSCI*/ZEROFRAC and QCL2 reads them back by name, and renaming a
# key on one side (QCL2 returns False for a value it cannot find) kept the whole
# suite green while L2NANOK went to 0 on every real frame.


def _make_l2(*, populate=True):
    """KPF2 good enough for CheckpointL2.run(): clean FLUX/VAR on every fiber and
    the required PRIMARY keywords seeded so KWRDPRL2 passes."""
    l2 = KPF2()
    if populate:
        set_fiber_arrays(l2, "FLUX", 1.0, ncol=_NCOL)
        set_fiber_arrays(l2, "VAR", 0.25, ncol=_NCOL)
    seed_required_primary(l2, CheckpointL2.QC)
    return l2


class TestCheckpointL2:
    def test_run_composes_diagnostics_into_qc(self):
        l2 = _make_l2()
        CheckpointL2(l2).run()
        qc = l2.headers["QUALITY_CONTROL"]
        # The metrics DiagL2 measured...
        assert qc["NANSCI1"] == 0
        assert qc["ZEROFRAC"] == pytest.approx(0.0)
        # ...are the ones QCL2 read back, by name. This is the seam.
        assert qc["L2NANOK"] == 1
        assert qc["L2FLXOK"] == 1
        assert qc["ISGOOD"] == 1

    def test_run_detects_real_nan_pixels_through_the_seam(self):
        # Half of SCI1's pixels are NaN. DiagL2 must count them and QCL2 must
        # fail on the count it wrote -- neither half proves that alone.
        l2 = _make_l2()
        flux = l2.data["GREEN_SCI1_FLUX"]
        flux[:, : _NCOL // 2] = np.nan
        CheckpointL2(l2).run()
        qc = l2.headers["QUALITY_CONTROL"]
        assert qc["NANSCI1"] > 0
        assert qc["L2NANOK"] == 0
        assert qc["ISGOOD"] == 0

    def test_run_raises_when_extraction_missing(self):
        # DATAPRL2 is fatal (in RAISE_FLAGS): no extracted flux -> run() raises.
        with pytest.raises(ValueError, match="DATAPRL2 = 0"):
            CheckpointL2(_make_l2(populate=False)).run()


class TestCheckpointL0:
    def test_run_good_product_passes_and_writes_flags(self, tmp_path, caplog):
        # A calibration frame: no target to resolve, so no AstroQuery and no
        # network. This is the only in-process exercise of QCL0.run().
        fn = str(tmp_path / "KP.20240405.00001.00.fits")
        write_amp_l0(
            fn,
            namps=4,
            shape=(10, 10),
            primary_cards={
                "DATE-OBS": "2024-04-05T01:00:37",
                "MJD-OBS": 60405.04,
                "EXPTIME": 12.0,
                "OBJECT": "synthetic",
                "IMTYPE": "Bias",
                "PROGNAME": None,
                **GOOD_DATES,
            },
        )
        l0 = KPF0.from_fits(fn)
        with caplog.at_level(logging.WARNING):
            CheckpointL0(l0).run()
        qc = l0.headers["QUALITY_CONTROL"]
        assert qc["DATAPRL0"] == 1
        assert qc["KWRDPRL0"] == 1
        assert qc["ISGOOD"] == 1


# ---------------------------------------------------------------------------
# CheckpointL1 -- folds DiagL1 + QCL1, then validates the assembled FFI product
# ---------------------------------------------------------------------------


def _make_l1(*, ccd=True, shape=(8, 8)):
    """KPF1 good enough for CheckpointL1.run(): GREEN/RED CCD + VAR arrays, the
    applied-calibration flags on RECEIPT, and the read-noise and master-age
    keywords on QUALITY_CONTROL, all inside the ranges QCL1 accepts.

    Built in memory. Not the same as test_qc_flags.py's ``_make_kpf1``, which
    deliberately round-trips through from_fits to reproduce the sparse-PRIMARY
    case its KWRDPRL1 test needs; here the skeleton PRIMARY is what is wanted.
    """
    l1 = KPF1()
    l1.headers["PRIMARY"]["DATE-OBS"] = "2024-04-05T01:00:37"
    if ccd:
        for chip in ("GREEN", "RED"):
            l1.set_data(f"{chip}_CCD", np.ones(shape, dtype=np.float32))
            l1.set_data(f"{chip}_VAR", np.ones(shape, dtype=np.float32))

    receipt = l1.headers["RECEIPT"]
    for kw in ("OSCANSUB", "BIASSUB", "DARKSUB", "FLATDIV"):
        receipt[kw] = (True, "applied")

    qc = l1.headers["QUALITY_CONTROL"]
    qc["BIASAGE"] = (1.0, "Age of bias master [days]")
    qc["DARKAGE"] = (5.0, "Age of dark master [days]")
    qc["FLATAGE"] = (10.0, "Age of flat master [days]")
    for i in range(1, 5):
        qc[f"RNGREEN{i}"] = (3.5, "RN e-")
        qc[f"RNRED{i}"] = (4.0, "RN e-")
        qc[f"RNNGGR{i}"] = (1.0, "RNNG")
        qc[f"RNNGRD{i}"] = (1.0, "RNNG")

    for kw in CheckpointL1.QC(l1)._required_primary_keywords():
        if kw not in l1.headers["PRIMARY"]:
            l1.headers["PRIMARY"][kw] = ("UNKNOWN", "seeded for test")
    return l1


class TestCheckpointL1:
    def test_run_good_product_passes_and_writes_flags(self, caplog):
        l1 = _make_l1()
        with caplog.at_level(logging.WARNING):
            CheckpointL1(l1).run()
        assert not caplog.records
        qc = l1.headers["QUALITY_CONTROL"]
        assert qc["DATAPRL1"] == 1
        assert qc["KWRDPRL1"] == 1
        assert qc["ISGOOD"] == 1

    def test_run_raises_when_ccd_data_missing(self):
        # DATAPRL1 is fatal (in RAISE_FLAGS): no GREEN/RED CCD -> run() raises.
        l1 = _make_l1(ccd=False)
        with pytest.raises(ValueError, match="DATAPRL1 = 0"):
            CheckpointL1(l1).run()

    def test_run_raises_when_required_keyword_missing(self):
        # KWRDPRL1 is fatal (in RAISE_FLAGS): a missing required PRIMARY keyword
        # -> KWRDPRL1 = 0 -> run() raises.
        l1 = _make_l1()
        req = sorted(CheckpointL1.QC(l1)._required_primary_keywords())
        del l1.headers["PRIMARY"][req[0]]
        with pytest.raises(ValueError, match="KWRDPRL1 = 0"):
            CheckpointL1(l1).run()

    def test_run_warns_when_read_noise_out_of_range(self, caplog):
        # RNOK is not a RAISE_FLAG, so an out-of-range amp lands in the ISGOOD
        # summary rather than raising.
        l1 = _make_l1()
        l1.headers["QUALITY_CONTROL"]["RNGREEN1"] = (99.0, "RN e-")
        with caplog.at_level(logging.WARNING):
            CheckpointL1(l1).run()
        assert "ISGOOD=0" in caplog.text
        assert "RNOK" in caplog.text


# ---------------------------------------------------------------------------
# CheckpointL4 -- folds DiagL4 + QCL4, then validates the RV/CCF product
# ---------------------------------------------------------------------------


def _make_l4(*, sci=True):
    """KPF4 good enough for CheckpointL4.run(), from the shared builder.

    The arguments are load-bearing and must not be trimmed to the defaults:
    ``jitter=1e-7`` gives the per-order BJD/BERV scatter DiagL4 measures (about
    0.03 s and 3e-4 m/s, well inside the BJDOK/BERVOK gates), and ``sci_obj="target"``
    is what makes those gates apply at all. Without the jitter this stops being
    the suite's only composed DiagL4 -> QCL4 seam and becomes header-stuffing;
    passing ``bervrng=``/``bjdrng=`` instead would write the metrics directly and
    do the same damage.
    """
    l4 = make_l4(sci=sci, jitter=1e-7, berv=7.9, sci_obj="target", seed=3)
    seed_required_primary(l4, CheckpointL4.QC)
    return l4


class TestCheckpointL4:
    def test_run_good_product_passes_and_writes_flags(self, caplog):
        l4 = _make_l4()
        with caplog.at_level(logging.WARNING):
            CheckpointL4(l4).run()
        assert not caplog.records
        qc = l4.headers["QUALITY_CONTROL"]
        # The folded DiagL4 metrics and QCL4 flags both land on QUALITY_CONTROL.
        # Exact values, not presence: `is not None` accepts 0, nan or a string,
        # and this is the only assertion in the suite that the folded DiagL4
        # stage produced correct numbers. The fixture's per-order scatter is
        # 1e-7 about berv=7.9 and bjd=2460000.0, so the weighted means are known.
        assert qc["BERVMEAN"] == pytest.approx(7.9, abs=1e-5)
        assert qc["BJDMEAN"] == pytest.approx(2460000.0, abs=1e-5)
        # And the dispersion metrics are nonzero, which is what makes this the
        # composed seam: DiagL4 measured real per-order scatter and QCL4 gated on
        # what it measured. Dropping the fixture's jitter would leave every other
        # assertion here green while the seam quietly became a constant.
        assert qc["BERVRNG"] > 0.0
        assert qc["BJDRNG"] > 0.0
        assert qc["DATAPRL4"] == 1
        assert qc["ISGOOD"] == 1

    def test_run_raises_when_science_ccf_rv_missing(self):
        # DATAPRL4 is fatal (in RAISE_FLAGS): no science CCF/RV -> run() raises.
        l4 = _make_l4(sci=False)
        with pytest.raises(ValueError, match="DATAPRL4 = 0"):
            CheckpointL4(l4).run()

    def test_run_raises_when_required_keyword_missing(self):
        # KWRDPRL4 is fatal (in RAISE_FLAGS): a missing required PRIMARY keyword
        # -> KWRDPRL4 = 0 -> run() raises.
        l4 = _make_l4()
        req = sorted(CheckpointL4.QC(l4)._required_primary_keywords())
        del l4.headers["PRIMARY"][req[0]]
        with pytest.raises(ValueError, match="KWRDPRL4 = 0"):
            CheckpointL4(l4).run()
