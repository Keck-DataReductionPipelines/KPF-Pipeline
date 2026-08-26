"""Tests for the KPF4 (RVs and CCFs / L4) data model, the L2->L4 transform, the
KPFMasterL4 stub, and top-level data-model package imports.

Synthetic FITS fixtures only -- no real KPF data needed.
"""

import numpy as np
import pandas as pd
import pytest

from kpfpipe import DETECTOR
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.data_models.masters import KPFMasterL4

from ._catalog import SOURCES, catalog_record_table
from ._dtype_policy import BJD, CCF, RV_FLOAT, assert_dtype, assert_roundtrip_dtype

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]
NORDER = NORDER_GREEN + NORDER_RED


class TestToKPF4:
    def test_to_kpf4_creates_kpf4(self):
        kpf2 = KPF2()
        kpf4 = kpf2.to_kpf4()
        assert isinstance(kpf4, KPF4)
        assert kpf4.level == 4

    def test_to_kpf4_forwards_primary_header(self):
        kpf2 = KPF2()
        kpf2.headers["PRIMARY"]["INSTRUME"] = "KPF"
        kpf2.headers["PRIMARY"]["OBJECT"] = "10700"
        kpf4 = kpf2.to_kpf4()
        assert kpf4.headers["PRIMARY"]["INSTRUME"] == "KPF"
        assert kpf4.headers["PRIMARY"]["OBJECT"] == "10700"

    def test_to_kpf4_sets_datalvl(self):
        kpf2 = KPF2()
        kpf4 = kpf2.to_kpf4()
        assert kpf4.headers["PRIMARY"].get("DATALVL") == "L4"

    def test_to_kpf4_carries_receipt(self):
        kpf2 = KPF2()
        kpf4 = kpf2.to_kpf4()
        assert "to_kpf4" in kpf4.receipt["FUNCTION"].values

    def test_to_kpf4_leaves_rv_empty(self):
        kpf2 = KPF2()
        kpf4 = kpf2.to_kpf4()
        assert "RV1" in kpf4.extensions
        assert len(kpf4.data["RV1"]) == 0

    def test_program_ids_survive_transform_and_validate(self, synthetic_l1_file):
        # PROGID/KOAID live on RECEIPT, not PRIMARY, and ride the RECEIPT forward.
        l1 = KPF1.from_fits(synthetic_l1_file)
        l1.headers["RECEIPT"]["PROGID"] = "U999"
        l1.headers["RECEIPT"]["KOAID"] = "KP.20201122.34567.89"
        l4 = l1.to_kpf2().to_kpf4()

        receipt = l4.headers["RECEIPT"]
        assert receipt.get("PROGID") == "U999"
        assert receipt.get("KOAID") == "KP.20201122.34567.89"
        assert "PROGID" not in l4.headers["PRIMARY"]

    def test_receipt_and_drpstatus_survive_roundtrip(self, tmp_path):
        # KPF4 reads through rvdata's RV4._read, a different path from KPF0/1,
        # so the L0/L1 round-trip twins cover none of this.
        l4 = KPF2().to_kpf4()
        l4.headers["PRIMARY"]["DATE-OBS"] = "2024-01-01T00:00:00"
        l4.receipt_add_entry("radial_velocity", "", "PASS")
        fn = str(tmp_path / "kpf_SL4_20240101T000000.fits")
        l4.to_fits(fn)

        back = KPF4.from_fits(fn)
        assert "radial_velocity" in back.receipt["FUNCTION"].values
        assert (
            back.headers["RECEIPT"].get("DRPSTATU") == "Radial Velocity module complete"
        )

    def test_kpf4_has_quality_control_extension(self):
        # KPF4 must create QUALITY_CONTROL (RV4 does not) so to_kpf4 has a
        # destination and the accumulated QC history reaches L4.
        assert "QUALITY_CONTROL" in KPF4().extensions

    def test_to_kpf4_forwards_quality_control_and_receipt_headers(self):
        # The accumulated L0/L1/L2 QC and provenance history must reach L4.
        kpf2 = KPF2()
        kpf2.set_keyword("NANSCI1", 7)  # DiagL2 metric -> QUALITY_CONTROL
        kpf2.headers["QUALITY_CONTROL"]["DATAPRL0"] = (1, "L0 flag (propagated)")
        kpf2.headers["RECEIPT"]["BIASSUB"] = (1, "bias subtracted")
        kpf4 = kpf2.to_kpf4()
        assert kpf4.headers["QUALITY_CONTROL"].get("NANSCI1") == 7
        assert kpf4.headers["QUALITY_CONTROL"].get("DATAPRL0") == 1
        assert kpf4.headers["RECEIPT"].get("BIASSUB") == 1

    def test_l4_quality_control_survives_round_trip(self, tmp_path):
        kpf2 = KPF2()
        kpf2.set_keyword("NANSCI1", 7)
        kpf4 = kpf2.to_kpf4()
        path = tmp_path / "kpf_SL4_20240101T000000.fits"
        kpf4.to_fits(str(path))
        back = KPF4.from_fits(str(path))
        assert "QUALITY_CONTROL" in back.extensions
        assert back.headers["QUALITY_CONTROL"].get("NANSCI1") == 7


class TestCatalogRecordPassthrough:
    """CATALOG_RECORD rides L2 -> L4 and survives KPF4's RV4 read path."""

    @staticmethod
    def _l2_with_catalog():
        kpf2 = KPF2()
        kpf2.set_data("CATALOG_RECORD", catalog_record_table())
        return kpf2

    def test_kpf4_has_catalog_record_extension(self):
        # Like QUALITY_CONTROL: RV4 does not create it, so KPF4 must, giving
        # to_kpf4's pass-through a destination.
        assert "CATALOG_RECORD" in KPF4().extensions

    def test_rows_reach_l4(self):
        # The rows ride along, so the astrometry stays with the RV it fed.
        kpf4 = self._l2_with_catalog().to_kpf4()
        assert [str(s) for s in kpf4.data["CATALOG_RECORD"]["source"]] == list(SOURCES)

    def test_catalog_record_roundtrip(self, tmp_path):
        path = tmp_path / "kpf_SL4_20240101T000001.fits"
        self._l2_with_catalog().to_kpf4().to_fits(str(path))
        back = KPF4.from_fits(str(path))
        assert [str(s) for s in back.data["CATALOG_RECORD"]["source"]] == list(SOURCES)


class TestKPF4:
    def test_kpf4_inherits_rv4(self):
        from rvdata.core.models.level4 import RV4

        kpf4 = KPF4()
        assert isinstance(kpf4, RV4)
        assert kpf4.level == 4

    def test_ccf_rv_extensions_per_orderlet(self):
        kpf4 = KPF4()
        for n in range(1, 6):
            assert f"CCF{n}" in kpf4.extensions
            assert f"CCF_VAR{n}" in kpf4.extensions
            assert f"RV{n}" in kpf4.extensions

    def test_trace_derived_aliases(self):
        # CCF{n}/CCF_VAR{n}/RV{n} <-> TRACE{n}: SCI2 is trace 3, SKY is 1, CAL is 5.
        kpf4 = KPF4()
        assert kpf4.data._resolve("SCI2_CCF") == "CCF3"
        assert kpf4.data._resolve("SCI2_CCF_VAR") == "CCF_VAR3"
        assert kpf4.data._resolve("SCI2_RV") == "RV3"
        assert kpf4.data._resolve("CAL_CCF") == "CCF5"
        assert kpf4.data._resolve("CAL_CCF_VAR") == "CCF_VAR5"
        assert kpf4.data._resolve("SKY_RV") == "RV1"
        # bare RV is not an alias (RV is trace-mapped, not a 1:1 alias)
        assert kpf4.data._resolve("RV") == "RV"

    @pytest.mark.parametrize("suffix", ["CCF", "CCF_VAR"])
    def test_ccf_chip_prefix_views(self, suffix):
        kpf4 = KPF4()
        green = np.ones((NORDER_GREEN, 5))
        red = 2 * np.ones((NORDER - NORDER_GREEN, 5))
        kpf4.set_data(f"GREEN_SCI2_{suffix}", green)
        kpf4.set_data(f"RED_SCI2_{suffix}", red)
        assert kpf4.data[f"SCI2_{suffix}"].shape == (NORDER, 5)
        np.testing.assert_array_equal(kpf4.data[f"GREEN_SCI2_{suffix}"], green)
        np.testing.assert_array_equal(kpf4.data[f"RED_SCI2_{suffix}"], red)

    def test_ccf_var_survives_round_trip(self, tmp_path):
        kpf4 = KPF2().to_kpf4()
        ccf = np.arange(NORDER * 5, dtype=float).reshape(NORDER, 5)
        var = ccf + 0.5
        kpf4.set_data("SCI2_CCF", ccf)
        kpf4.set_data("SCI2_CCF_VAR", var)
        path = tmp_path / "kpf_SL4_20240101T000002.fits"
        kpf4.to_fits(str(path))
        back = KPF4.from_fits(str(path))
        assert "CCF_VAR3" in back.extensions
        np.testing.assert_allclose(np.asarray(back.data["SCI2_CCF_VAR"]), var)
        np.testing.assert_allclose(np.asarray(back.data["SCI2_CCF"]), ccf)

    def test_rv_chip_prefix_views(self):
        # RV tables are row-sliced (green = rows 0:NORDER_GREEN, red the rest).
        kpf4 = KPF4()
        kpf4.set_data(
            "SCI2_RV",
            pd.DataFrame(
                {"ORDER_INDEX": np.arange(NORDER), "RV": np.arange(NORDER, dtype=float)}
            ),
        )
        green, red = kpf4.data["GREEN_SCI2_RV"], kpf4.data["RED_SCI2_RV"]
        assert len(green) == NORDER_GREEN and len(red) == NORDER - NORDER_GREEN
        np.testing.assert_array_equal(
            np.asarray(green["ORDER_INDEX"]), np.arange(NORDER_GREEN)
        )
        np.testing.assert_array_equal(
            np.asarray(red["ORDER_INDEX"]), np.arange(NORDER_GREEN, NORDER)
        )
        assert "GREEN_SCI2_RV" in kpf4.data

    def test_rv_chip_prefix_is_read_only(self):
        # RV tables are written whole; a chip-prefixed write must fail loud.
        kpf4 = KPF4()
        with pytest.raises(KeyError, match="read-only"):
            kpf4.set_data("GREEN_SCI2_RV", pd.DataFrame({"RV": np.zeros(NORDER_GREEN)}))

    def test_data_get_chip_prefix_returns_slice(self):
        # .get() honors chip-prefix slicing, like __getitem__.
        kpf4 = KPF4()
        kpf4.set_data("SCI2_CCF", np.ones((NORDER, 5)))
        green = kpf4.data.get("GREEN_SCI2_CCF")
        assert green.shape == (NORDER_GREEN, 5)

    def test_data_get_plain_key_and_default(self):
        kpf4 = KPF4()
        assert kpf4.data.get("CCF1") is not None  # canonical extension exists
        assert kpf4.data.get("NOT_A_KEY", "fallback") == "fallback"


class TestKPFMasterStubs:
    def test_master_l4_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            KPFMasterL4()

    def test_master_l4_inherits_kpf4(self):
        assert issubclass(KPFMasterL4, KPF4)


class TestDtypeProvenance:
    """EPRV mandates 64-bit for the L4 product: the CCF cubes and the RV table's
    WAVE_START/WAVE_END/BJD_TDB. A downscale anywhere here costs RV accuracy, so
    assert both the in-memory dtype and what survives the FITS round-trip."""

    @staticmethod
    def _populated():
        kpf4 = KPF2().to_kpf4()
        kpf4.set_data("SCI2_CCF", np.ones((NORDER, 5), dtype=np.float64))
        kpf4.set_data(
            "SCI2_RV",
            pd.DataFrame(
                {
                    "ORDER_INDEX": np.arange(NORDER),
                    "RV": np.zeros(NORDER, dtype=np.float64),
                    "RV_ERR": np.full(NORDER, 1e-3, dtype=np.float64),
                    "WAVE_START": np.full(NORDER, 4500.0, dtype=np.float64),
                    "WAVE_END": np.full(NORDER, 8700.0, dtype=np.float64),
                    "BJD_TDB": np.full(NORDER, 2460000.0, dtype=np.float64),
                }
            ),
        )
        return kpf4

    def test_populated_l4_is_born_float64(self):
        # One in-memory check, not one per column: it localizes a break to
        # set_data, which the round-trip tests below cannot distinguish from a
        # FITS-layer break.
        kpf4 = self._populated()
        assert_dtype(kpf4.data["SCI2_CCF"], CCF, "SCI2_CCF in-mem")
        table = kpf4.data["SCI2_RV"]
        for column in ("RV", "RV_ERR", "WAVE_START", "WAVE_END"):
            assert_dtype(table[column], RV_FLOAT, f"SCI2_RV {column} in-mem")
        assert_dtype(table["BJD_TDB"], BJD, "SCI2_RV BJD_TDB in-mem")

    def test_ccf_cube_survives_round_trip_as_float64(self, tmp_path):
        # assert_roundtrip_dtype reads BITPIX off the raw HDU, so it needs the
        # on-disk extension name (SCI2_CCF is an alias for CCF3).
        assert_roundtrip_dtype(
            KPF4,
            self._populated(),
            "CCF3",
            CCF,
            tmp_path,
            name="kpf_SL4_20240101T000003.fits",
        )

    def test_rv_table_survives_round_trip_as_float64(self, tmp_path):
        path = str(tmp_path / "kpf_SL4_20240101T000004.fits")
        self._populated().to_fits(path)
        table = KPF4.from_fits(path).data["SCI2_RV"]
        for column in ("RV", "RV_ERR", "WAVE_START", "WAVE_END", "BJD_TDB"):
            assert_dtype(table[column], RV_FLOAT, f"SCI2_RV {column} after round-trip")
