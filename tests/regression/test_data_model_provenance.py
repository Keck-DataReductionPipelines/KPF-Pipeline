"""
Regression tests for WMKO FITS provenance / reduction-status keywords
(DRPVERNO, DRPTAG, PROGID, KOAID, DRPSTATU).

Synthetic data-model fixtures only; no real KPF files required.
"""

import importlib.metadata

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1


def _v(value):
    """Unwrap a PRIMARY (value, comment) tuple to its scalar value."""
    return value[0] if isinstance(value, tuple) else value


class TestProvenanceKeywords:
    """PROGID/KOAID/DRPVERNO/DRPTAG are stamped on the L1 EPRV PRIMARY by to_kpf1."""

    def test_to_kpf1_stamps_native_program_ids(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l0.headers["PRIMARY"]["PROGID"] = "U999"
        l0.headers["PRIMARY"]["KOAID"] = "KP.20201122.34567.89"
        prim = l0.to_kpf1().headers["PRIMARY"]
        assert _v(prim["PROGID"]) == "U999"
        assert _v(prim["KOAID"]) == "KP.20201122.34567.89"

    def test_to_kpf1_defaults_program_ids_to_unknown(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        for key in ("PROGID", "KOAID"):
            if key in l0.headers["PRIMARY"]:
                del l0.headers["PRIMARY"][key]
        prim = l0.to_kpf1().headers["PRIMARY"]
        assert _v(prim["PROGID"]) == "UNKNOWN"
        assert _v(prim["KOAID"]) == "UNKNOWN"

    def test_to_kpf1_stamps_drp_version(self, synthetic_l0_file):
        prim = KPF0.from_fits(synthetic_l0_file).to_kpf1().headers["PRIMARY"]
        version = importlib.metadata.version("kpfpipe")
        assert _v(prim["DRPTAG"]) == version
        assert _v(prim["DRPVERNO"]) == version

    def test_program_ids_survive_to_l4_and_validate(self, synthetic_l1_file):
        # to_kpf2/to_kpf4 call validate_eprv_primary; PROGID/KOAID are registered
        # KPF-pipeline keywords (L1-headers.csv) and must pass the union allowlist.
        l1 = KPF1.from_fits(synthetic_l1_file)
        l1.headers["PRIMARY"]["PROGID"] = "U999"
        l1.headers["PRIMARY"]["KOAID"] = "KP.20201122.34567.89"
        prim = l1.to_kpf2().to_kpf4().headers["PRIMARY"]
        assert _v(prim["PROGID"]) == "U999"
        assert _v(prim["KOAID"]) == "KP.20201122.34567.89"


class TestDrpStatus:
    """DRPSTATU default at L1, then per-module updates via the receipt override."""

    def test_default_after_to_kpf1(self, synthetic_l0_file):
        prim = KPF0.from_fits(synthetic_l0_file).to_kpf1().headers["PRIMARY"]
        assert _v(prim["DRPSTATU"]) == "File ingested into KPF-DRP"

    def test_module_receipt_updates_status(self, synthetic_l0_file):
        l1 = KPF0.from_fits(synthetic_l0_file).to_kpf1()
        l1.receipt_add_entry("image_assembly", "PASS")
        assert _v(l1.headers["PRIMARY"]["DRPSTATU"]) == "Image Assembly module complete"

    def test_master_receipt_updates_status(self, synthetic_l0_file):
        l1 = KPF0.from_fits(synthetic_l0_file).to_kpf1()
        l1.receipt_add_entry("master_bias", "PASS")
        assert _v(l1.headers["PRIMARY"]["DRPSTATU"]) == "Master Bias module complete"

    def test_internal_receipts_do_not_change_status(self, synthetic_l0_file):
        l1 = KPF0.from_fits(synthetic_l0_file).to_kpf1()
        l1.receipt_add_entry("radial_velocity", "PASS")
        for internal in ("to_kpf2", "to_kpf4", "to_fits", "from_fits"):
            l1.receipt_add_entry(internal, "PASS")
        assert (
            _v(l1.headers["PRIMARY"]["DRPSTATU"]) == "Radial Velocity module complete"
        )

    def test_l2_receipt_override(self, synthetic_l1_file):
        kpf2 = KPF1.from_fits(synthetic_l1_file).to_kpf2()
        kpf2.receipt_add_entry("barycentric_correction", "PASS")
        status = _v(kpf2.headers["PRIMARY"]["DRPSTATU"])
        assert status == "Barycentric Correction module complete"
