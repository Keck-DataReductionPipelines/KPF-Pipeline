"""
KPF Masters Level 2 data model.

Extracted-spectrum-level masters calibration product (e.g., master
wavelength solutions, master flats).

Inherits from KPFMasterModel and KPF2. The full L2 schema and the KPF
alias system are inherited from KPF2 (see L2-aliases.csv and
L2-trace-map.csv for the mapping between KPF-internal fiber names like
SCI2_WAVE and the standard names like TRACE3_WAVE). Only masters-
specific additions (INPUT_FILES) are declared in
Masters-L2-extensions.csv.

Filename convention: masters products follow WMKO filename format
(KP.YYYYMMDD.NNNNN.NN.fits), not the EPRV per-level convention.
"""

import importlib.resources
import os
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.masters.base import KPFMasterModel

_config_path = importlib.resources.files("kpfpipe.data_models.config")
_MASTERS_L2_EXTENSIONS = pd.read_csv(_config_path / "Masters-L2-extensions.csv")


class KPFMasterL2(KPFMasterModel, KPF2):
    """
    KPF Masters Level 2 extracted-spectrum calibration product.

    Thin wrapper around KPF2 with masters-specific DATALVL, filename
    prefix, and INPUT_FILES extension. Inherits the full L2 schema and
    the KPF2 alias system (e.g., SCI2_WAVE -> TRACE3_WAVE, with chip-
    prefix access GREEN_SCI2_WAVE / RED_SCI2_WAVE).

    Usage:
        m2 = KPFMasterL2()
        m2 = KPFMasterL2.from_fits("/path/to/master_l2.fits")
    """

    _DATALVL = "ML2"
    _FILENAME_PREFIX = "kpf_ML2"
    _known_extensions = set(_MASTERS_L2_EXTENSIONS["Name"]) | set(KPF2().extensions.keys())

    def __init__(self):
        KPF2.__init__(self)
        self.level = 2

        for _, row in _MASTERS_L2_EXTENSIONS.iterrows():
            if row["Required"] and row["Name"] not in self.extensions:
                self.create_extension(row["Name"], row["DataType"])

    @classmethod
    def from_fits(cls, fn, **kwargs):
        """
        Create a KPFMasterL2 instance from a FITS file.

        Overrides RVDataModel.from_fits() to bypass the base class read()
        dispatch, which hardcodes RV2._read for level==2 and does not know
        about masters-specific extensions (e.g., INPUT_FILES).
        """
        if not os.path.isfile(fn):
            raise IOError(f"{fn} does not exist.")
        if not fn.endswith(".fits") and not fn.endswith(".fit"):
            raise IOError("Input file must be a FITS file.")

        this_data = cls()

        with fits.open(fn) as hdul:
            this_data.filename = os.path.basename(fn)
            this_data.dirname = os.path.dirname(fn)
            this_data._read(hdul)

        this_data.receipt_add_entry("from_fits", "PASS")
        return this_data

    def _read(self, hdul):
        """
        Read all extensions from a Masters L2 FITS HDUList.

        Handles known extensions and accepts unknown extensions (with a
        warning). Mirrors KPF1._read in style and structure.
        """
        for hdu in hdul:
            ext_name = hdu.name

            if isinstance(hdu, fits.PrimaryHDU):
                fits_type = "PrimaryHDU"
            elif isinstance(hdu, (fits.ImageHDU, fits.CompImageHDU)):
                fits_type = "ImageHDU"
            elif isinstance(hdu, fits.BinTableHDU):
                fits_type = "BinTableHDU"
            else:
                continue

            if ext_name not in self.extensions:
                if ext_name != "PRIMARY":
                    if ext_name not in self._known_extensions:
                        warnings.warn(
                            f"Non-standard extension '{ext_name}' found in L2 file.",
                            UserWarning,
                        )
                    self.create_extension(ext_name, fits_type)

            if ext_name == "PRIMARY":
                pass
            elif ext_name == "RECEIPT":
                t = Table.read(hdu)
                df = t.to_pandas()
                receipt_columns = [
                    "Time", "Code_Release", "Commit_Hash",
                    "Branch_Name", "Module_Name", "Status"
                ]
                if df.empty:
                    df = pd.DataFrame(columns=receipt_columns)
                else:
                    all_cols = df.columns.union(receipt_columns, sort=False)
                    df = df.reindex(columns=all_cols).fillna("")
                self.receipt = df
            elif fits_type == "ImageHDU":
                self.set_data(ext_name, np.array(hdu.data))
            elif fits_type == "BinTableHDU":
                self.set_data(ext_name, Table.read(hdu))

            self.set_header(ext_name, hdu.header)
