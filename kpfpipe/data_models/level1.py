"""
KPF Level 1 (assembled 2D frame) data model.

Represents an assembled CCD frame after combining amplifier readouts.
Also used for master calibration products (bias, dark, flat) which
share the same structure (mean + variance frames for GREEN and RED).

Subclasses RVDataModel (via KPFDataModel) to reuse its extension/header/data
infrastructure and receipt system.
"""

import datetime
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
L1_EXTENSIONS = pd.read_csv(_config_path / "L1-extensions.csv")
_KNOWN_L1_EXTENSIONS = set(L1_EXTENSIONS["Name"].tolist())

_kpf_config = importlib.resources.files("rvdata.instruments.kpf.config")
_HEADER_MAP = pd.read_csv(_kpf_config / "header_map.csv")


class KPF1(KPFDataModel):
    """
    KPF Level 1 assembled 2D frame data model.

    After image assembly, the L1 product contains assembled GREEN_CCD
    and RED_CCD frames with corresponding variance frames, plus
    pass-through extensions from L0 (CA_HK, exposure meter, telemetry).

    Also used for master calibration products (bias, dark, flat).

    Usage:
        l1 = KPF1.from_fits("/path/to/kpf_L1_20240113T102656.fits")
        l1.data["GREEN_CCD"]  # numpy array, 4080x4080
        l1.data["RED_CCD"]    # numpy array, 4080x4080
    """

    def __init__(self):
        super().__init__()
        self.level = 1

        for _, row in L1_EXTENSIONS.iterrows():
            if row["Required"] and row["Name"] not in self.extensions:
                self.create_extension(row["Name"], row["DataType"])

    @classmethod
    def from_fits(cls, fn, **kwargs):
        """
        Create a KPF1 instance from a FITS file.

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
        Read all extensions from an L1 FITS HDUList.

        Handles known extensions from the CSV definition and also
        accepts unknown extensions (with a warning).
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
                    if ext_name not in _KNOWN_L1_EXTENSIONS:
                        warnings.warn(
                            f"Non-standard extension '{ext_name}' found in L1 file.",
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
        """
        KPF L1 filenames follow kpf_L1_YYYYMMDDThhmmss.fits convention.
        Uses DATE-OBS from PRIMARY header.
        """
        if "PRIMARY" in self.headers:
            date_obs = self.headers["PRIMARY"].get("DATE-OBS")
            if date_obs is not None:
                if isinstance(date_obs, tuple):
                    date_obs = date_obs[0]
                date_str = str(date_obs).split(".")[0]
                try:
                    dt = datetime.datetime.fromisoformat(date_str)
                    datetime_str = dt.strftime("%Y%m%dT%H%M%S")
                    return f"kpf_L1_{datetime_str}.fits"
                except ValueError:
                    pass
        raise ValueError("Cannot generate filename: DATE-OBS not available")

    def to_fits(self, fn=None):
        """Write L1 data to a FITS file."""
        if fn is None:
            fn = self.generate_standard_filename()
        if not fn.endswith(".fits"):
            raise NameError("Filename must end with .fits")

        self.receipt_add_entry("to_fits", "PASS")

        if "PRIMARY" in self.headers:
            self.headers["PRIMARY"]["FILENAME"] = (
                os.path.basename(fn), "Name of the FITS file"
            )
            self.headers["PRIMARY"]["DATALVL"] = ("L1", "Data product level")

        hdu_list = self._create_hdul()
        hdul = fits.HDUList(hdu_list)
        dirname = os.path.dirname(fn)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname, exist_ok=True)
        hdul.writeto(fn, overwrite=True, output_verify="silentfix")
        hdul.close()
        return fn

    # Mapping of L1 extension names → KPF2/RV2 extension names for pass-through
    _L1_TO_KPF2_PASSTHROUGH = {
        "TELEMETRY": "TELEMETRY",
        "EXPMETER_SCI": "EXPMETER",
        "CA_HK": "ANCILLARY_SPECTRUM",
    }

    def to_kpf2(self):
        """Create a KPF2 scaffold from this L1, carrying over headers and pass-through extensions.

        Returns a KPF2 with PRIMARY header keywords mapped from KPF-native
        to EPRV standard (using rvdata's header_map.csv), the full L1 PRIMARY
        stored in INSTRUMENT_HEADER, and pass-through extensions (TELEMETRY,
        EXPMETER_SCI→EXPMETER, CA_HK→ANCILLARY_SPECTRUM). KPF-friendly
        aliases are registered automatically (e.g., SCI2_FLUX → TRACE3_FLUX,
        CA_HK → ANCILLARY_SPECTRUM). Trace data arrays are created but
        empty — the caller (spectral extraction) fills those in.
        """
        from kpfpipe.data_models.level2 import KPF2

        kpf2 = KPF2()

        # Map KPF-native header keywords to EPRV standard using header_map.csv
        if "PRIMARY" in self.headers:
            l1_header = self.headers["PRIMARY"]
            for _, row in _HEADER_MAP.iterrows():
                standard_key = str(row["STANDARD"]).strip()
                instrument_key = str(row["INSTRUMENT"]).strip() if pd.notna(row["INSTRUMENT"]) else ""
                default_val = row["DEFAULT"] if pd.notna(row["DEFAULT"]) else None

                if instrument_key and instrument_key in l1_header:
                    value = l1_header[instrument_key]
                    kpf2.headers["PRIMARY"][standard_key] = value
                elif default_val is not None and str(default_val).strip():
                    kpf2.headers["PRIMARY"][standard_key] = default_val

            # Store full L1 PRIMARY header in INSTRUMENT_HEADER
            for key, value in l1_header.items():
                kpf2.headers["INSTRUMENT_HEADER"][key] = value

        # Pass-through extensions with renaming
        for l1_ext, kpf2_ext in self._L1_TO_KPF2_PASSTHROUGH.items():
            if l1_ext in self.extensions:
                l1_type = self.extensions[l1_ext]
                if kpf2_ext not in kpf2.extensions:
                    kpf2.create_extension(kpf2_ext, l1_type)
                elif kpf2.extensions[kpf2_ext] != l1_type:
                    # Update type to match actual L1 data
                    kpf2.extensions[kpf2_ext] = l1_type
                if l1_ext in self.data and self.data[l1_ext] is not None:
                    kpf2.set_data(kpf2_ext, self.data[l1_ext])
                if l1_ext in self.headers:
                    kpf2.set_header(kpf2_ext, self.headers[l1_ext])

        # Carry forward receipt
        if self.receipt is not None and not self.receipt.empty:
            kpf2.receipt = self.receipt.copy()

        # Store obs_id for traceability
        if self.obs_id is not None:
            kpf2.headers["PRIMARY"]["ORIGID"] = (self.obs_id, "Original L0 observation ID")

        kpf2.headers["PRIMARY"]["DATALVL"] = ("L2", "Data product level")
        kpf2.receipt_add_entry("to_kpf2", "PASS")
        return kpf2

    def info(self):
        """Print summary of L1 data model contents."""
        if self.filename:
            print(f"KPF L1: {self.filename}")
        else:
            print("Empty KPF1 data product")
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
