"""
KPF Level 0 (raw CCD) data model.

Reads raw FITS files from the KPF instrument at Keck Observatory.
L0 files contain amplifier readouts, exposure meter tables, guide camera,
telemetry, and telescope metadata. Extensions vary between observations.

Subclasses RVDataModel (via KPFDataModel) to reuse its extension/header/data
infrastructure and receipt system.
"""

import os
import re
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from kpfpipe.data_models.base import KPFDataModel

import importlib.resources

_config_path = importlib.resources.files("kpfpipe.data_models.config")
L0_EXTENSIONS = pd.read_csv(_config_path / "L0-extensions.csv")
_KNOWN_L0_EXTENSIONS = set(L0_EXTENSIONS["Name"].tolist())


class KPF0(KPFDataModel):
    """
    KPF Level 0 raw data model.

    Represents a raw CCD readout from the KPF instrument. Extensions
    vary between observations; the reader accepts whatever is present
    in the FITS file.

    Usage:
        l0 = KPF0.from_fits("/path/to/KP.20240113.23249.10.fits")
        l0.data["GREEN_AMP1"]   # numpy array
        l0.headers["PRIMARY"]   # header dict
    """

    def __init__(self):
        super().__init__()
        self.level = 0
        self.create_extension("PRIMARY", "PrimaryHDU")

    @classmethod
    def from_fits(cls, fn, **kwargs):
        """
        Create a KPF0 instance from a FITS file.

        Overrides RVDataModel.from_fits() to bypass the base class read()
        dispatch, which only handles levels 2-4.
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

        obs_id_match = re.match(r"(KP\.\d{8}\.\d{5}\.\d{2})", os.path.basename(fn))
        if obs_id_match:
            this_data.obs_id = obs_id_match.group(1)

        this_data.receipt_add_entry("from_fits", "PASS")
        return this_data

    def _read(self, hdul):
        """
        Read all extensions from an L0 FITS HDUList.

        Iterates through all HDUs and creates extensions dynamically
        based on what is present. CompImageHDU is transparently
        decompressed by astropy.
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
                    if ext_name not in _KNOWN_L0_EXTENSIONS:
                        warnings.warn(
                            f"Non-standard extension '{ext_name}' found in L0 file.",
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

    def generate_standard_filename(self):
        """KPF L0 filenames follow the KP.YYYYMMDD.NNNNN.NN.fits pattern."""
        if self.obs_id is not None:
            return f"{self.obs_id}.fits"
        raise ValueError("Cannot generate filename: obs_id not set")

    def to_fits(self, fn=None):
        """Write L0 data to a FITS file (plain ImageHDU, no compression)."""
        if fn is None:
            fn = self.generate_standard_filename()
        if not fn.endswith(".fits"):
            raise NameError("Filename must end with .fits")

        self.receipt_add_entry("to_fits", "PASS")

        if "PRIMARY" in self.headers:
            self.headers["PRIMARY"]["FILENAME"] = (
                os.path.basename(fn), "Name of the FITS file"
            )

        hdu_list = self._create_hdul()
        hdul = fits.HDUList(hdu_list)
        dirname = os.path.dirname(fn)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname, exist_ok=True)
        hdul.writeto(fn, overwrite=True, output_verify="silentfix")
        hdul.close()
        return fn

    def info(self):
        """Print summary of L0 data model contents."""
        if self.filename:
            print(f"KPF L0: {self.filename}")
        else:
            print("Empty KPF0 data product")
        if self.obs_id:
            print(f"Obs ID: {self.obs_id}")

        print(f"\n{'Extension':<20s} {'Type':<15s} {'Shape/Size':<20s}")
        print("=" * 55)
        for name, ext_type in self.extensions.items():
            if name == "PRIMARY":
                n_cards = len(self.headers.get(name, {}))
                print(f"{'PRIMARY':<20s} {'header':<15s} {n_cards} cards")
                continue
            ext = self.data.get(name)
            if isinstance(ext, np.ndarray):
                print(f"{name:<20s} {'array':<15s} {str(ext.shape):<20s}")
            elif isinstance(ext, Table):
                print(f"{name:<20s} {'table':<15s} {len(ext)} rows")
            else:
                print(f"{name:<20s} {ext_type:<15s} {'(empty)':<20s}")
