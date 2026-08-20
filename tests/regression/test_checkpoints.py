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
from astropy.table import Table

from kpfpipe import DETECTOR
from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.quality_control.checkpoints import (
    Checkpoint,
    CheckpointL0,
    CheckpointL1,
    CheckpointL2,
    CheckpointL4,
)

_NORDER_TOTAL = DETECTOR["norder"]["GREEN"] + DETECTOR["norder"]["RED"]


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
    """KPF4 good enough for CheckpointL4.run(): science CCF + RV tables (with
    BJD_TDB/BERV/WEIGHT for DiagL4) and the required PRIMARY keywords seeded so
    KWRDPRL4 passes.

    SCI-OBJ is 'target' (star-illuminated) so the BJDOK/BERVOK gates apply, and the
    per-order BJD/BERV jitter is kept well inside those gates (1 s / 0.1 m/s) so a
    good product passes."""
    l4 = KPF4()
    l4.headers["INSTRUMENT_HEADER"]["SCI-OBJ"] = "target"
    if sci:
        rng = np.random.default_rng(3)
        for fiber in ("SCI1", "SCI2", "SCI3"):
            l4.set_data(f"{fiber}_CCF", np.ones((_NORDER_TOTAL, 5)))
            l4.set_data(
                f"{fiber}_RV",
                Table(
                    {
                        "ORDER_INDEX": np.arange(_NORDER_TOTAL),
                        "RV": rng.normal(0, 1e-3, _NORDER_TOTAL),
                        "BJD_TDB": 2460000.0 + rng.normal(0, 1e-7, _NORDER_TOTAL),
                        "BERV": 7.9 + rng.normal(0, 1e-7, _NORDER_TOTAL),
                        "WEIGHT": np.ones(_NORDER_TOTAL),
                    }
                ),
            )
    for kw in CheckpointL4.QC(l4)._required_primary_keywords():
        l4.headers["PRIMARY"][kw] = ("UNKNOWN", "seeded for test")
    return l4


class TestCheckpointL4:
    def test_run_good_product_passes_and_writes_flags(self, caplog):
        l4 = _make_l4()
        with caplog.at_level(logging.WARNING):
            CheckpointL4(l4).run()
        assert not caplog.records
        qc = l4.headers["QUALITY_CONTROL"]
        # The folded DiagL4 metrics and QCL4 flags both land on QUALITY_CONTROL.
        assert qc["BERVMEAN"] is not None and qc["BJDMEAN"] is not None
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
