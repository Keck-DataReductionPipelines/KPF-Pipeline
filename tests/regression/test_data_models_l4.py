"""
Tests for the KPF4 (RVs and CCFs / L4) data model, the L2->L4 transform,
the KPFMasterL4 stub, and top-level data-model package imports.

Uses synthetic FITS fixtures — no real KPF data needed.
"""

import numpy as np
import pandas as pd
import pytest

from kpfpipe import DETECTOR
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.data_models.masters import KPFMasterL4

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
        kpf2.headers["PRIMARY"]["OBJECT"] = "HD_10700"
        kpf4 = kpf2.to_kpf4()
        assert kpf4.headers["PRIMARY"]["INSTRUME"] == "KPF"
        assert kpf4.headers["PRIMARY"]["OBJECT"] == "HD_10700"

    def test_to_kpf4_sets_datalvl(self):
        kpf2 = KPF2()
        kpf4 = kpf2.to_kpf4()
        datalvl = kpf4.headers["PRIMARY"]["DATALVL"]
        assert (datalvl[0] if isinstance(datalvl, tuple) else datalvl) == "L4"

    def test_to_kpf4_carries_receipt(self):
        kpf2 = KPF2()
        kpf4 = kpf2.to_kpf4()
        assert "to_kpf4" in kpf4.receipt["Module_Name"].values

    def test_to_kpf4_leaves_rv_empty(self):
        kpf2 = KPF2()
        kpf4 = kpf2.to_kpf4()
        assert "RV1" in kpf4.extensions
        assert len(kpf4.data["RV1"]) == 0

    def test_program_ids_survive_transform_and_validate(self, synthetic_l1_file):
        """PROGID/KOAID set on the L1 PRIMARY survive L1->L2->L4 and pass
        validate_eprv_primary (called by to_kpf2/to_kpf4); they are registered
        in L1-headers.csv and the validator allowlist is the union of levels."""
        l1 = KPF1.from_fits(synthetic_l1_file)
        l1.headers["PRIMARY"]["PROGID"] = "U999"
        l1.headers["PRIMARY"]["KOAID"] = "KP.20201122.34567.89"
        prim = l1.to_kpf2().to_kpf4().headers["PRIMARY"]

        def _scalar(v):
            return v[0] if isinstance(v, tuple) else v

        assert _scalar(prim["PROGID"]) == "U999"
        assert _scalar(prim["KOAID"]) == "KP.20201122.34567.89"


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
            assert f"RV{n}" in kpf4.extensions

    def test_trace_derived_aliases(self):
        # CCF{n}/RV{n} <-> TRACE{n}: SCI2 is trace 3, CAL is trace 1, SKY is 5.
        kpf4 = KPF4()
        assert kpf4.data._resolve("SCI2_CCF") == "CCF3"
        assert kpf4.data._resolve("SCI2_RV") == "RV3"
        assert kpf4.data._resolve("CAL_CCF") == "CCF1"
        assert kpf4.data._resolve("SKY_RV") == "RV5"
        # bare RV is not an alias (RV is trace-mapped, not a 1:1 alias)
        assert kpf4.data._resolve("RV") == "RV"

    def test_ccf_chip_prefix_views(self):
        kpf4 = KPF4()
        green = np.ones((NORDER_GREEN, 5))
        red = 2 * np.ones((NORDER - NORDER_GREEN, 5))
        kpf4.set_data("GREEN_SCI2_CCF", green)
        kpf4.set_data("RED_SCI2_CCF", red)
        assert kpf4.data["SCI2_CCF"].shape == (NORDER, 5)
        np.testing.assert_array_equal(kpf4.data["GREEN_SCI2_CCF"], green)
        np.testing.assert_array_equal(kpf4.data["RED_SCI2_CCF"], red)

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
        with pytest.raises(NotImplementedError):
            KPFMasterL4()

    def test_master_l4_inherits_kpf4(self):
        assert issubclass(KPFMasterL4, KPF4)


class TestImports:
    def test_data_models_import(self):
        from kpfpipe.data_models import KPF0, KPF1, KPF2, KPF4

        assert KPF0 is not None
        assert KPF1 is not None
        assert KPF2 is not None
        assert KPF4 is not None

    def test_masters_data_models_import(self):
        from kpfpipe.data_models.masters import KPFMasterL1, KPFMasterL2, KPFMasterL4

        assert KPFMasterL1 is not None
        assert KPFMasterL2 is not None
        assert KPFMasterL4 is not None

    def test_rvdata_import(self):
        from rvdata.core.models.level2 import RV2
        from rvdata.core.models.level4 import RV4

        assert RV2 is not None
        assert RV4 is not None
