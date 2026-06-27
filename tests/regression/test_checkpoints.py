"""Tests for the Checkpoint layer (quality_control/checkpoints).

Checkpoints are the third QC stage: they READ the 0/1 QC flags and the product
headers, then warn or raise (never write). This pins:

  - ``unregistered_keywords`` — raises on a card not registered for a governed
    extension (including a raw WMKO native leaked onto an EPRV PRIMARY); skips the
    raw WMKO L0 PRIMARY; passes a clean product. (Migrated from the old
    ``QC._validate_headers`` tests.)
  - ``qc_flags`` — a failed (0) flag named in the level's ``RAISE_FLAGS`` raises;
    any other failed flag warns; all-pass is silent.

``run()`` additionally folds in the paired Diagnostics + QC stages before the
checkpoint methods; that orchestration is pinned in ``TestRunFoldsDiagnosticsAndQC``.
"""

import warnings

import pytest

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.quality_control.checkpoints import Checkpoint, CheckpointL0, CheckpointL2


class TestUnregisteredKeywords:
    def test_clean_product_passes(self):
        # Fresh KPF2: EPRV-seeded PRIMARY (all registered) + empty governed
        # extensions -> no unregistered card, and no QC flags present -> qc_flags
        # is a no-op, so the checkpoint methods are silent. (run() is not used
        # here: it folds in QC, which a bare product can't satisfy -- the fold is
        # covered by TestRunFoldsDiagnosticsAndQC.)
        l2 = KPF2()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would fail this
            chk = CheckpointL2(l2)
            chk.unregistered_keywords()
            chk.qc_flags()

    def test_unexpected_keyword_on_governed_extension_raises(self):
        l2 = KPF2()
        l2.headers["QUALITY_CONTROL"]["BOGUSKEY"] = (1, "not registered")
        with pytest.raises(ValueError, match="unregistered keyword 'BOGUSKEY'"):
            CheckpointL2(l2).unregistered_keywords()

    def test_native_wmko_leak_on_primary_raises(self):
        l2 = KPF2()
        # GAIAID is a raw WMKO native (kept in INSTRUMENT_HEADER, never on PRIMARY).
        # It is not a registered PRIMARY keyword, so it fails the general
        # unregistered-keyword check (no dedicated WMKO-leak branch needed).
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

    def test_nonraise_flag_zero_warns(self):
        # KWRDPRL2 is not a RAISE_FLAG, so a 0 warns (data present so DATAPRL2 ok).
        l2 = KPF2()
        l2.headers["QUALITY_CONTROL"]["DATAPRL2"] = (1, "data present")
        l2.headers["QUALITY_CONTROL"]["KWRDPRL2"] = (0, "required present")
        with pytest.warns(UserWarning, match="KWRDPRL2 = 0"):
            CheckpointL2(l2).qc_flags()

    def test_all_pass_silent(self):
        l2 = KPF2()
        l2.headers["QUALITY_CONTROL"]["DATAPRL2"] = (1, "data present")
        l2.headers["QUALITY_CONTROL"]["KWRDPRL2"] = (1, "required present")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CheckpointL2(l2).qc_flags()

    def test_absent_flag_is_ignored(self):
        # A flag the QC stage never wrote is absent -> neither warns nor raises.
        l2 = KPF2()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CheckpointL2(l2).qc_flags()


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
        # QC's result dict is captured for callers (e.g. scripts/qc.py).
        assert chk.qc_results == {"ISGOOD": (True, "")}

    def test_missing_paired_classes_skip_those_stages(self):
        # Base Checkpoint has DIAGNOSTICS = QC = None (LEVEL None skips PRIMARY):
        # run() does the checkpoint methods only and leaves qc_results empty.
        chk = Checkpoint(KPF2())  # clean, empty -> checkpoint methods are silent
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chk.run()
        assert chk.qc_results == {}
