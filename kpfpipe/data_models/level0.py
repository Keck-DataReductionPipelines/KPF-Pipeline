"""
KPF Level 0 (raw CCD) data model.

Reads raw FITS files from the KPF instrument at Keck Observatory.
L0 files contain amplifier readouts, exposure meter tables, guide camera,
telemetry, and telescope metadata. Extensions vary between observations.

Subclasses RVDataModel (via KPFDataModel) to reuse its extension/header/data
infrastructure and receipt system.
"""

import importlib.resources
import os
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from kpfpipe.data_models.base import KPFDataModel
from kpfpipe.utils.kpf import get_obs_id

_config_path = importlib.resources.files("kpfpipe.data_models.config")
L0_EXTENSIONS = pd.read_csv(_config_path / "L0-extensions.csv")
_KNOWN_L0_EXTENSIONS = set(L0_EXTENSIONS["Name"].tolist())


class KPF0(KPFDataModel):
    """
    KPF Level 0 raw data model.

    Represents a raw CCD readout from the KPF instrument. Extensions
    vary between observations; the reader accepts whatever is present
    in the FITS file. Construct from a FITS file with
    `KPF0.from_fits(path)`, then read amplifier arrays from `data`
    (e.g. `data["GREEN_AMP1"]`) and header dicts from `headers`
    (e.g. `headers["PRIMARY"]`).
    """

    def __init__(self):
        super().__init__()
        self.level = 0

        for _, row in L0_EXTENSIONS.iterrows():
            if row["Required"] and row["Name"] not in self.extensions:
                self.create_extension(row["Name"], row["DataType"])

    def read(self, hdul, instrument=None, overwrite=False, **kwargs):
        """
        Route L0 FITS reads to `KPF0._read`.

        `RVDataModel.read` has no lvl==0 dispatch branch, so the inherited
        `from_fits` would never call into `_read` without this override.
        """
        self._read(hdul)
        if self.filename is not None:
            try:
                self.obs_id = get_obs_id(self.filename)
            except ValueError:
                pass

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
                            stacklevel=2,
                        )
                    self.create_extension(ext_name, fits_type)

            if ext_name == "PRIMARY":
                pass
            elif ext_name == "RECEIPT":
                t = Table.read(hdu)
                df = t.to_pandas()
                receipt_columns = [
                    "Time",
                    "Code_Release",
                    "Commit_Hash",
                    "Branch_Name",
                    "Module_Name",
                    "Status",
                ]
                if df.empty:
                    df = pd.DataFrame(columns=receipt_columns)
                else:
                    all_cols = df.columns.union(receipt_columns, sort=False)
                    df = df.reindex(columns=all_cols).fillna("")
                self.receipt = df
            elif fits_type == "ImageHDU":
                # np.array (not asarray) materializes the memmapped HDU into RAM
                # before from_fits closes the file; a view would dangle afterward.
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
                os.path.basename(fn),
                "Name of the FITS file",
            )

        hdu_list = self._create_hdul()
        hdul = fits.HDUList(hdu_list)
        dirname = os.path.dirname(fn)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname, exist_ok=True)
        hdul.writeto(fn, overwrite=True, output_verify="silentfix")
        hdul.close()
        return fn

    _L0_TO_L1_PASSTHROUGH = [
        "CA_HK",
        "EXPMETER_SCI",
        "EXPMETER_SKY",
        "TELEMETRY",
        "CONFIG",
    ]

    def to_kpf1(self):
        """
        Create a KPF1 scaffold from this L0, carrying over headers and
        pass-through extensions.

        The raw WMKO PRIMARY header is converted to EPRV-standard keyword names
        and values here (the single conversion site; see
        `kpfpipe.data_models.header_standard`), so the L1 PRIMARY is already
        EPRV-standard. The verbatim raw L0 PRIMARY is preserved in the
        immutable INSTRUMENT_HEADER extension. Downstream stages read raw
        instrument keywords from INSTRUMENT_HEADER and write EPRV/registered
        KPF keywords to PRIMARY.

        Returns a KPF1 with EPRV PRIMARY header, INSTRUMENT_HEADER, pass-through
        extensions (CA_HK, EXPMETER_SCI/SKY, TELEMETRY, CONFIG), receipt, and
        obs_id copied over. GREEN_CCD, GREEN_VAR, RED_CCD, RED_VAR are created
        but empty — the caller (image assembly) fills those in.
        """
        from kpfpipe.data_models.header_standard import (
            build_instrument_header,
            convert_native_to_eprv,
        )
        from kpfpipe.data_models.level1 import KPF1  # deferred: avoids circular import

        l1 = KPF1()

        # Convert the raw WMKO PRIMARY to EPRV-standard names/values, and
        # preserve the verbatim raw L0 PRIMARY in INSTRUMENT_HEADER (immutable
        # pure pass-through — nothing else ever writes to it).
        if "PRIMARY" in self.headers:
            native_primary = self.headers["PRIMARY"]
            for key, value in convert_native_to_eprv(native_primary).items():
                l1.headers["PRIMARY"][key] = value

            if "INSTRUMENT_HEADER" not in l1.extensions:
                l1.create_extension("INSTRUMENT_HEADER", "ImageHDU")
            instrument_header = l1.headers["INSTRUMENT_HEADER"]
            for key, value in build_instrument_header(native_primary).items():
                instrument_header[key] = value

        # Copy pass-through extensions (data + header)
        for ext_name in self._L0_TO_L1_PASSTHROUGH:
            if ext_name in self.extensions:
                ext_type = self.extensions[ext_name]
                if ext_name not in l1.extensions:
                    l1.create_extension(ext_name, ext_type)
                if ext_name in self.data and self.data[ext_name] is not None:
                    l1.set_data(ext_name, self.data[ext_name])
                if ext_name in self.headers:
                    l1.set_header(ext_name, self.headers[ext_name])

        # Carry forward receipt
        if self.receipt is not None and not self.receipt.empty:
            l1.receipt = self.receipt.copy()

        # Copy obs_id
        l1.obs_id = self.obs_id

        l1.headers["PRIMARY"]["DATALVL"] = ("L1", "Data product level")
        l1.receipt_add_entry("to_l1", "PASS")
        return l1

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
