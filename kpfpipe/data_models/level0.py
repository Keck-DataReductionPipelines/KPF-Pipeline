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
import re
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from kpfpipe import DETECTOR, __version__
from kpfpipe.data_models.base import HEADER_MAP, KPFDataModel
from kpfpipe.utils.kpf import get_obs_id

_config_path = importlib.resources.files("kpfpipe.data_models.config")
L0_EXTENSIONS = pd.read_csv(_config_path / "L0-extensions.csv")
_KNOWN_L0_EXTENSIONS = set(L0_EXTENSIONS["Name"].tolist())

# WMKO-native L0 filename: KP.YYYYMMDD.NNNNN.NN.fits (the obs_id plus .fits).
_L0_FILENAME_PATTERN = re.compile(r"KP\.\d{8}\.\d{5}\.\d{2}\.fits")

# --- L0→L1 conversion constants (consumed only by wmko_to_eprv) --------------
# Detector truth: KPF has 35 green + 32 red echelle orders (rvdata's header_map
# default of 65 is wrong — see notes/header_audit.md A2).
_NUMORDER = int(DETECTOR["norder"]["GREEN"]) + int(DETECTOR["norder"]["RED"])

# Initial DRPSTATU value on the L1 EPRV PRIMARY, before any pipeline module runs.
# Each module overwrites it via the receipt_add_entry override (see base.py).
_DRPSTATU_DEFAULT = "File ingested into KPF-DRP"


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
        Route L0 FITS reads to `KPF0._read`, then stamp DRP provenance.

        `RVDataModel.read` has no lvl==0 dispatch branch, so the inherited
        `from_fits` would never call into `_read` without this override.
        """
        self._read(hdul)
        if "PRIMARY" in self.headers:
            self._stamp_provenance()
        if self.filename is not None:
            try:
                self.obs_id = get_obs_id(self.filename)
            except ValueError:
                pass

    def _stamp_provenance(self):
        """Stamp WMKO DRP-RUN provenance onto the L0 PRIMARY at read time.

        This is the single population site for the four provenance cards
        (their `Populated by` in config/L0-headers.csv is `KPF0.from_fits`):
        DRPVERNO (DRP-RUN-11), PROGID/KOAID (DRP-RUN-19), DRPSTATU (DRP-RUN-20).
        `to_kpf1` passes them through unchanged onto the EPRV L1 PRIMARY.

        DRPVERNO and DRPSTATU are always (re)stamped. PROGID/KOAID must come from
        the WMKO-native file; an absent (or empty) value is defaulted to UNKNOWN
        with a warning, so the cards are always present downstream.
        """
        primary = self.headers["PRIMARY"]
        primary["DRPVERNO"] = (
            __version__,
            "Pipeline version (WMKO DRP-RUN-11; EPRV equivalent is DRPTAG)",
        )
        primary["DRPSTATU"] = (_DRPSTATU_DEFAULT, "DRP reduction status (DRP-RUN-20)")
        for key, label in (
            ("PROGID", "WMKO program ID (DRP-RUN-19)"),
            ("KOAID", "KOA archive ID (DRP-RUN-19)"),
        ):
            if not primary.get(key):
                warnings.warn(
                    f"{key} absent from L0 PRIMARY; defaulting to 'UNKNOWN'",
                    UserWarning,
                    stacklevel=2,
                )
                primary[key] = ("UNKNOWN", label)

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

    def check_filename_convention(self, filename):
        """KPF L0 uses the WMKO-native KP.YYYYMMDD.NNNNN.NN.fits name."""
        basename = os.path.basename(filename)
        if not _L0_FILENAME_PATTERN.fullmatch(basename):
            warnings.warn(
                f"Filename '{basename}' does not follow the KPF L0 naming "
                "convention (KP.YYYYMMDD.NNNNN.NN.fits)",
                stacklevel=2,
            )
            return False
        return True

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

    def wmko_to_eprv(self):
        """Map this L0's raw WMKO PRIMARY to an EPRV-standard PRIMARY dict.

        For each header_map row, take the instrument (WMKO) value when present,
        else the row default. Then apply the value corrections the installed
        header_map gets wrong (NUMORDER, JD_UTC), stamp the EPRV DRP version
        (DRPTAG), and forward the DRP-RUN provenance cards
        (DRPVERNO/PROGID/KOAID/DRPSTATU) already on the L0 PRIMARY (stamped at
        read by ``_stamp_provenance``) onto the EPRV L1 PRIMARY.

        Returns
        -------
        dict
            EPRV-standard PRIMARY keyword -> value (a few as ``(value, comment)``).
        """
        wmko_primary = self.headers["PRIMARY"]
        out = {}
        for _, row in HEADER_MAP.iterrows():
            standard_key = str(row["STANDARD"]).strip()
            instrument_key = (
                str(row["INSTRUMENT"]).strip() if pd.notna(row["INSTRUMENT"]) else ""
            )
            default_val = row["DEFAULT"] if pd.notna(row["DEFAULT"]) else None

            if instrument_key and instrument_key in wmko_primary:
                out[standard_key] = wmko_primary.get(instrument_key)
            elif default_val is not None and str(default_val).strip():
                out[standard_key] = default_val

        # --- Value corrections (see notes/header_audit.md A1/A2/A3) ---
        out["NUMORDER"] = (_NUMORDER, "Number of echelle orders (green+red)")
        # header_map maps JD_UTC <- MJD-OBS but drops the epoch offset, leaving a
        # raw MJD (A1); KPF's canonical exposure time is MJD-OBS, so add 2400000.5.
        mjd = wmko_primary.get("MJD-OBS")
        if mjd not in (None, "", "UNKNOWN"):
            out["JD_UTC"] = (
                float(mjd) + 2400000.5,
                "[day] Julian date of exposure start",
            )
        out["DRPTAG"] = (__version__, "DRP version")

        # DRP-RUN provenance (DRPVERNO/PROGID/KOAID/DRPSTATU) is stamped onto the
        # L0 PRIMARY at read (see _stamp_provenance); forward it verbatim, value
        # and comment, onto the EPRV L1 PRIMARY.
        for key in ("DRPVERNO", "PROGID", "KOAID", "DRPSTATU"):
            if key in wmko_primary:
                out[key] = (wmko_primary[key], wmko_primary.comments[key])
        return out

    def build_instrument_header(self):
        """Comment-preserving verbatim copy of the L0 PRIMARY as ingested.

        INSTRUMENT_HEADER is an immutable, pure pass-through of the L0 PRIMARY as
        read from disk -- the raw instrument cards plus the four DRP-RUN
        provenance cards stamped at read (see ``_stamp_provenance``); nothing
        writes to it after ``to_kpf1``. Returning a ``fits.Header`` copy
        preserves values *and* comments (and commentary cards), unlike a scalar
        dict.
        """
        return self.as_fits_header(self.headers["PRIMARY"])

    def to_kpf1(self):
        """
        Create a KPF1 scaffold from this L0, carrying over headers and
        pass-through extensions.

        The raw WMKO PRIMARY header is converted to EPRV-standard keyword names
        and values here (the single conversion site; see `wmko_to_eprv`), so the
        L1 PRIMARY is already EPRV-standard. The DRP-RUN provenance cards
        (DRPVERNO/PROGID/KOAID/DRPSTATU), stamped onto the L0 PRIMARY at read,
        are forwarded unchanged. The L0 PRIMARY as ingested is preserved in the
        immutable INSTRUMENT_HEADER extension. Downstream stages read raw
        instrument keywords from INSTRUMENT_HEADER and write EPRV/registered
        KPF keywords to PRIMARY.

        Returns a KPF1 with EPRV PRIMARY header, INSTRUMENT_HEADER, pass-through
        extensions (CA_HK, EXPMETER_SCI/SKY, TELEMETRY, CONFIG), receipt, and
        obs_id copied over. GREEN_CCD, GREEN_VAR, RED_CCD, RED_VAR are created
        but empty — the caller (image assembly) fills those in.
        """
        from kpfpipe.data_models.level1 import KPF1  # deferred: avoids circular import

        l1 = KPF1()

        # Convert the raw WMKO PRIMARY to EPRV-standard names/values, and
        # preserve the verbatim raw L0 PRIMARY (values + comments) in
        # INSTRUMENT_HEADER (immutable pass-through — nothing else writes to it).
        if "PRIMARY" in self.headers:
            for key, value in self.wmko_to_eprv().items():
                l1.headers["PRIMARY"][key] = value

            if "INSTRUMENT_HEADER" not in l1.extensions:
                l1.create_extension("INSTRUMENT_HEADER", "ImageHDU")
            l1.set_header("INSTRUMENT_HEADER", self.build_instrument_header())

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
