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
from rvdata.core.models.definitions import BASE_RECEIPT_COLUMNS
from rvdata.core.tools.headers import parse_value_to_datatype

from kpfpipe import __version__
from kpfpipe.data_models.base import KPFDataModel
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.utils.kpf import get_obs_id

_config_path = importlib.resources.files("kpfpipe.data_models.config")
_L0_EXTENSIONS = pd.read_csv(_config_path / "L0-extensions.csv")
_KNOWN_L0_EXTENSIONS = set(_L0_EXTENSIONS["Name"].tolist())

# WMKO-native L0 filename: KP.YYYYMMDD.NNNNN.NN.fits (the obs_id plus .fits).
_L0_FILENAME_PATTERN = re.compile(r"KP\.\d{8}\.\d{5}\.\d{2}\.fits")

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

        for _, row in _L0_EXTENSIONS.iterrows():
            if row["Required"] and row["Name"] not in self.extensions:
                self.create_extension(row["Name"], row["DataType"])

    def read(self, hdul, instrument=None, overwrite=False, **kwargs):
        """
        Route L0 FITS reads to `KPF0._read`, then stamp DRP provenance.

        `RVDataModel.read` has no lvl==0 dispatch branch, so the inherited
        `from_fits` would never call into `_read` without this override.
        """
        self._read(hdul)
        # Derive obs_id before stamping: _stamp_wmko_tracking writes it as the
        # ORIGID provenance card, so it must be known by then.
        if self.filename is not None:
            try:
                self.obs_id = get_obs_id(self.filename)
            except ValueError:
                pass
        if "PRIMARY" in self.headers:
            self._stamp_wmko_tracking()

    def _stamp_wmko_tracking(self):
        """Stamp WMKO DRP-RUN provenance onto the L0 RECEIPT at read time.

        This is the single population site for the provenance cards (their
        `PopulatedBy` in config/L0-headers.csv is `KPF0.from_fits`):
        DRPVERNO (DRP-RUN-11), PROGID/KOAID (DRP-RUN-19), DRPSTATU (DRP-RUN-20),
        and ORIGID (the original L0 obs_id). They are written to their registry
        home, RECEIPT, via `set_keyword`, and ride the RECEIPT-header forward onto
        L1/L2/L4 (see `to_kpf1`). The raw PRIMARY and its INSTRUMENT_HEADER
        snapshot are left untouched.

        DRPVERNO and DRPSTATU are always (re)stamped. PROGID/KOAID are read from
        the WMKO-native PRIMARY; an absent (or empty) value is defaulted to UNKNOWN
        with a warning, so the cards are always present downstream. ORIGID is
        stamped only when the obs_id was resolved (e.g. from the filename).
        """
        self.set_keyword("DRPVERNO", __version__)
        self.set_keyword("DRPSTATU", _DRPSTATU_DEFAULT)
        if self.obs_id is not None:
            self.set_keyword("ORIGID", self.obs_id)
        primary = self.headers["PRIMARY"]
        for key in ("PROGID", "KOAID"):
            value = primary.get(key)
            if not value:
                warnings.warn(
                    f"{key} absent from L0 PRIMARY; defaulting to 'UNKNOWN'",
                    UserWarning,
                    stacklevel=2,
                )
                value = "UNKNOWN"
            self.set_keyword(key, value)

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
                receipt_columns = BASE_RECEIPT_COLUMNS["Name"].tolist()
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

        self.receipt_add_entry("to_fits", f"out_filepath={fn}", "PASS")

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
        "DRP_CONFIG",
    ]

    def _map_header(self):
        """Map this L0's raw WMKO PRIMARY to an EPRV-standard PRIMARY dict.

        A pure tabular application of the registry's (sanitized) header_map: for
        each row, take the instrument (WMKO) value when present, else the row
        default, and type it to the EPRV DataType. The registry already dropped
        non-mapping rows (unregistered targets like PARANG, and keywords sourced
        elsewhere -- NUMORDER/DRPTAG seeded, DATALVL the model level), so there is
        no filter or per-keyword correction here.

        The sole exception is ``JD_UTC``: it is a per-frame value transform of the
        native ``MJD-OBS`` (+ epoch offset), not a static default, so header_map
        cannot carry it; it is computed below. DRP-RUN provenance
        (DRPVERNO/PROGID/KOAID/DRPSTATU) lives on RECEIPT (see
        ``_stamp_wmko_tracking``), not here.

        Returns
        -------
        dict
            EPRV-standard PRIMARY keyword -> typed value.
        """
        wmko_primary = self.headers["PRIMARY"]
        out = {}
        for _, row in self.keyword_registry.header_map.iterrows():
            eprv_key = str(row["STANDARD"]).strip()
            instrument_key = (
                str(row["INSTRUMENT"]).strip() if pd.notna(row["INSTRUMENT"]) else ""
            )
            default_val = row["DEFAULT"] if pd.notna(row["DEFAULT"]) else None

            if instrument_key and instrument_key in wmko_primary:
                raw_value = wmko_primary.get(instrument_key)
            elif default_val is not None and str(default_val).strip():
                raw_value = default_val
            else:
                continue
            # Type the value to its EPRV DataType (rvdata-vocab) so the L1 overlay
            # matches L2's typing (e.g. NUMTRACE '5' -> 5), for both native values
            # and CSV-string defaults. Emit the bare value so to_kpf1's assignment
            # preserves the comment KPF1.__init__ seeded onto the PRIMARY card.
            dt = self.keyword_registry.eprv_primary_datatypes.get(eprv_key)
            out[eprv_key] = (
                parse_value_to_datatype(eprv_key, dt, raw_value)[0] if dt else raw_value
            )

        # JD_UTC: KPF's canonical exposure time is the native MJD-OBS; convert it to
        # a full Julian date (+ 2400000.5). The one value transform header_map can't
        # express, so it lives here rather than as a static map default.
        mjd = wmko_primary.get("MJD-OBS")
        if mjd not in (None, "", "UNKNOWN"):
            out["JD_UTC"] = (
                float(mjd) + 2400000.5,
                "[day] Julian date of exposure start",
            )
        return out

    def to_kpf1(self):
        """
        Create a KPF1 scaffold from this L0, carrying over headers and
        pass-through extensions.

        The raw WMKO PRIMARY header is converted to EPRV-standard keyword names
        and values here (the single conversion site; see `_map_header`), so the
        L1 PRIMARY is already EPRV-standard (EPRV-registered keywords only). The
        DRP-RUN provenance cards (DRPVERNO/PROGID/KOAID/DRPSTATU) live on RECEIPT
        (stamped at read) and reach L1 via the RECEIPT-header forward below. The
        L0 PRIMARY as ingested is preserved in the immutable INSTRUMENT_HEADER
        extension. Downstream stages read raw instrument keywords from
        INSTRUMENT_HEADER and write EPRV/registered KPF keywords to PRIMARY.

        Returns a KPF1 with EPRV PRIMARY header, INSTRUMENT_HEADER, pass-through
        extensions (CA_HK, EXPMETER_SCI/SKY, TELEMETRY, DRP_CONFIG), receipt, and
        obs_id copied over. GREEN_CCD, GREEN_VAR, RED_CCD, RED_VAR are created
        but empty — the caller (image assembly) fills those in.
        """
        kpf1 = KPF1()

        # Convert the raw WMKO PRIMARY to EPRV-standard names/values, and preserve
        # the raw L0 PRIMARY verbatim (values + comments, via as_fits_header) in the
        # immutable INSTRUMENT_HEADER -- only the raw instrument cards (DRP-RUN
        # provenance lives on RECEIPT), and nothing else ever writes to it.
        if "PRIMARY" in self.headers:
            for key, value in self._map_header().items():
                kpf1.headers["PRIMARY"][key] = value

            if "INSTRUMENT_HEADER" not in kpf1.extensions:
                kpf1.create_extension("INSTRUMENT_HEADER", "ImageHDU")
            kpf1.set_header(
                "INSTRUMENT_HEADER", self.as_fits_header(self.headers["PRIMARY"])
            )

        # Copy pass-through extensions (data + header)
        for ext_name in self._L0_TO_L1_PASSTHROUGH:
            if ext_name in self.extensions:
                ext_type = self.extensions[ext_name]
                if ext_name not in kpf1.extensions:
                    kpf1.create_extension(ext_name, ext_type)
                if ext_name in self.data and self.data[ext_name] is not None:
                    kpf1.set_data(ext_name, self.data[ext_name])
                if ext_name in self.headers:
                    kpf1.set_header(ext_name, self.headers[ext_name])

        # Forward the L0 QUALITY_CONTROL and RECEIPT *headers* onto L1 (value +
        # comment), mirroring to_kpf2/to_kpf4 so all three conversions share one
        # invariant. QUALITY_CONTROL carries the QCL0 booleans + ISGOOD; RECEIPT
        # carries the four DRP-RUN provenance cards stamped at read. Downstream L1
        # stages append to both. (PRIMARY/INSTRUMENT_HEADER are not forwarded
        # here -- L0 PRIMARY is converted via _map_header above, not copied.)
        self._forward_headers(kpf1, ("QUALITY_CONTROL", "RECEIPT"))

        # Carry forward receipt
        if self.receipt is not None and not self.receipt.empty:
            kpf1.receipt = self.receipt.copy()

        # Copy obs_id
        kpf1.obs_id = self.obs_id

        # DATALVL is set by KPF1.__init__ (= KPF1._DATALVL) and _map_header no
        # longer emits it (dropped from header_map), so no fixup is needed here.
        kpf1.receipt_add_entry("to_kpf1", "", "PASS")
        return kpf1

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
