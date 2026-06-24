"""
Regression tests for WMKO FITS provenance and reduction metadata.

Uses synthetic data-model fixtures only; no real KPF files are required.
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table

from kpfpipe import DETECTOR
from kpfpipe.data_models.base import kpf_drp_version
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2

NORDER = DETECTOR["norder"]["GREEN"] + DETECTOR["norder"]["RED"]


def test_l0_to_fits_writes_status_version_program_ids_and_receipt(
    synthetic_l0_file, tmp_path
):
    from kpfpipe.data_models.level0 import KPF0

    l0 = KPF0.from_fits(synthetic_l0_file)
    l0.headers["PRIMARY"]["PROGID"] = "U999"
    l0.headers["PRIMARY"]["KOAID"] = "KP.20201122.34567.89"

    out_fn = str(tmp_path / "provenance_l0.fits")
    l0.to_fits(out_fn)

    with fits.open(out_fn) as hdul:
        primary = hdul["PRIMARY"].header
        assert primary["DRPVERNO"] == kpf_drp_version()
        assert primary["DRPSTATU"] == "COMPLETE"
        assert primary["PROGID"] == "U999"
        assert primary["KOAID"] == "KP.20201122.34567.89"

        receipt = Table.read(hdul["RECEIPT"]).to_pandas()
        assert "to_fits" in receipt["Module_Name"].values


def test_l1_to_kpf2_propagates_program_ids_to_written_l2_primary(
    synthetic_l1_file, tmp_path
):
    l1 = KPF1.from_fits(synthetic_l1_file)
    l1.headers["PRIMARY"]["PROGID"] = "U999"
    l1.headers["PRIMARY"]["KOAID"] = "KP.20201122.34567.89"

    kpf2 = l1.to_kpf2()
    kpf2.set_data("TRACE3_FLUX", np.ones((NORDER, 8), dtype=np.float32))

    out_fn = str(tmp_path / "provenance_l2.fits")
    kpf2.to_fits(out_fn)

    with fits.open(out_fn) as hdul:
        primary = hdul["PRIMARY"].header
        assert primary["DRPVERNO"] == kpf_drp_version()
        assert primary["DRPSTATU"] == "COMPLETE"
        assert primary["PROGID"] == "U999"
        assert primary["KOAID"] == "KP.20201122.34567.89"

        receipt = Table.read(hdul["RECEIPT"]).to_pandas()
        assert "to_kpf2" in receipt["Module_Name"].values
        assert "to_fits" in receipt["Module_Name"].values


def test_l2_to_kpf4_carries_provenance_keywords_to_written_l4_primary(tmp_path):
    kpf2 = KPF2()
    kpf2.headers["PRIMARY"]["PROGID"] = "U999"
    kpf2.headers["PRIMARY"]["KOAID"] = "KP.20201122.34567.89"

    kpf4 = kpf2.to_kpf4()
    out_fn = str(tmp_path / "provenance_l4.fits")
    kpf4.to_fits(out_fn)

    with fits.open(out_fn) as hdul:
        primary = hdul["PRIMARY"].header
        assert primary["DRPVERNO"] == kpf_drp_version()
        assert primary["DRPSTATU"] == "COMPLETE"
        assert primary["PROGID"] == "U999"
        assert primary["KOAID"] == "KP.20201122.34567.89"
