"""Tests for the Checkpoint layer (quality_control/checkpoints).

Checkpoints are the third QC stage: they READ the 0/1 QC flags and the product
headers, then warn or raise (never write). This pins:

  - ``unregistered_keywords`` — raises on a card not registered for a governed
    extension (including a raw WMKO native leaked onto an EPRV PRIMARY); skips the
    raw WMKO L0 PRIMARY; passes a clean product. (Migrated from the old
    ``QC._validate_headers`` tests.)
  - ``qc_flags`` — a failed (0) flag named in the level's ``RAISE_FLAGS`` raises;
    any other failed flag warns; all-pass is silent.
"""

import warnings

import pytest

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.quality_control.checkpoints import CheckpointL0, CheckpointL2


class TestUnregisteredKeywords:
    def test_clean_product_passes(self):
        # Fresh KPF2: EPRV-seeded PRIMARY (all registered) + empty governed
        # extensions -> no unregistered card, and no QC flags present -> qc_flags
        # is a no-op, so run() is fully silent.
        l2 = KPF2()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would fail this
            CheckpointL2(l2).run()

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
