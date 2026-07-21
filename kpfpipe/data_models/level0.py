"""
KPF Level 0 (raw CCD) data model.

Raw FITS readout from the KPF instrument: amplifier arrays plus exposure
meter, guide camera, telemetry, and telescope metadata.
"""

import importlib.resources
import logging
import os
import re

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import Table
from rvdata.core.models.definitions import BASE_RECEIPT_COLUMNS
from rvdata.core.tools.headers import parse_value_to_datatype

from kpfpipe import __version__
from kpfpipe.data_models.base import KPFDataModel
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.utils.io import kpf_filename
from kpfpipe.utils.kpf import get_obs_id

logger = logging.getLogger(__name__)

_config_path = importlib.resources.files("kpfpipe.data_models.config")
_L0_EXTENSIONS = pd.read_csv(_config_path / "L0-extensions.csv")
_KNOWN_L0_EXTENSIONS = set(_L0_EXTENSIONS["Name"].tolist())

# WMKO-native L0 filename: KP.YYYYMMDD.NNNNN.NN.fits (the obs_id plus .fits).
_L0_FILENAME_PATTERN = re.compile(r"KP\.\d{8}\.\d{5}\.\d{2}\.fits")

# Initial DRPSTATU value on the L1 EPRV PRIMARY, before any pipeline module runs.
# Each module overwrites it via the receipt_add_entry override (see base.py).
_DRPSTATU_DEFAULT = "File ingested into KPF-DRP"

# Schema of the CATALOG_RECORD BinTable extension: one row per resolved source
# (wmko/gaia/simbad) plus the merged 'kpf-drp' canonical row, carrying the record
# fields plus a leading 'source' label and 'astr_src' (the source label whose
# astrometry supplied this row's position block -- itself for a source row, the
# merge base for the canonical row). Float columns hold NaN where a value is
# missing; string columns hold "". Units document the canonical schema (deg, mas/yr
# incl. cos Dec, mas, km/s, jyear).
_CATALOG_COLUMNS = (
    "source",
    "source_id",
    "astr_src",
    "ra",
    "dec",
    "pmra",
    "pmdec",
    "parallax",
    "rv",
    "frame",
    "epoch",
    "equinox",
)
_CATALOG_STR_COLUMNS = frozenset({"source", "source_id", "astr_src", "frame"})
_CATALOG_UNITS = {
    "ra": u.deg,
    "dec": u.deg,
    "pmra": u.mas / u.yr,
    "pmdec": u.mas / u.yr,
    "parallax": u.mas,
    "rv": u.km / u.s,
    "epoch": u.yr,
    "equinox": u.yr,
}
# Presence flag written to the CATALOG_RECORD header per source (int 0/1).
_CATALOG_FLAGS = {"wmko": "WMKOCR", "gaia": "GAIACR", "simbad": "SIMBADCR"}


class KPF0(KPFDataModel):
    """
    KPF Level 0 raw data model.

    Represents a raw CCD readout from the KPF instrument. Construct from a
    FITS file with ``KPF0.from_fits(path)``, then read amplifier arrays from
    ``data`` (e.g. ``data["GREEN_AMP1"]``) and header dicts from ``headers``
    (e.g. ``headers["PRIMARY"]``).
    """

    def __init__(self):
        super().__init__()
        self.level = 0

        for _, row in _L0_EXTENSIONS.iterrows():
            if row["Required"] and row["Name"] not in self.extensions:
                self.create_extension(row["Name"], row["DataType"])

    def read(self, hdul, instrument=None, overwrite=False, **kwargs):
        """Route L0 FITS reads to ``KPF0._read``, then stamp DRP provenance.

        ``RVDataModel.read`` has no lvl==0 dispatch branch, so the inherited
        ``from_fits`` would never call into ``_read`` without this override.
        (This is the canonical statement of the read-override rationale; the
        L1 and Masters-L2 overrides refer back here.)
        """
        self._read(hdul)
        # Derive obs_id before stamping: _stamp_wmko_tracking writes it as the
        # ORIGID provenance card, so it must be known by then.
        self.obs_id = get_obs_id(self.filename)
        if "PRIMARY" in self.headers:
            self._stamp_wmko_tracking()
            # Populate the native (telescope-side) wmko row of CATALOG_RECORD from
            # the raw TARG* astrometry.
            self.set_catalog_record("wmko", self._wmko_catalog_record())

    def _stamp_wmko_tracking(self):
        """Stamp WMKO DRP-RUN provenance onto the L0 RECEIPT at read time.

        The single population site for DRPVERNO, PROGID/KOAID, DRPSTATU, and ORIGID
        (the original L0 obs_id), written to their registry home (RECEIPT) via
        ``set_keyword`` and ridden forward onto L1/L2/L4 (see ``to_kpf1``).
        DRPVERNO/DRPSTATU/ORIGID are always (re)stamped. KOAID and PROGID are not
        on the raw WMKO PRIMARY -- they map from OFNAME and PROGNAME respectively.
        A missing PROGNAME defaults PROGID to UNKNOWN with a warning; a missing
        OFNAME (the archive obs_id) raises.
        """
        self.set_keyword("DRPVERNO", __version__)
        self.set_keyword("DRPSTATU", _DRPSTATU_DEFAULT)
        self.set_keyword("ORIGID", self.obs_id)
        primary = self.headers["PRIMARY"]

        koaid = primary.get("OFNAME")
        if not koaid:
            raise ValueError("OFNAME absent from L0 PRIMARY; cannot set KOAID")
        self.set_keyword("KOAID", koaid)

        progname = primary.get("PROGNAME")
        if not progname:
            logger.warning(
                "PROGNAME absent from L0 PRIMARY; defaulting PROGID to 'UNKNOWN'"
            )
            progname = "UNKNOWN"
        self.set_keyword("PROGID", progname)

    def _wmko_catalog_record(self):
        """Build the native WMKO/DCS target record from L0 PRIMARY TARG*.

        Telescope-side astrometry (no external query), converted to the canonical
        schema: TARGRA/TARGDEC sexagesimal (hourangle / deg) -> deg; TARGPMRA s/yr
        and TARGPMDC arcsec/yr -> mas/yr; TARGFRAM FK5 J2000 relabeled ICRS (the
        ~23 mas frame tie is negligible). Returns None -- so the frame gets
        WMKOCR=0 -- when there is no target pointing (TARGRA absent, e.g. a
        calibration) or the TARG* astrometry cannot be parsed (warned, never
        raised, so a malformed file still loads). Well-formedness is gated
        downstream by QCL0 (RADECOK), not here.
        """
        primary = self.headers["PRIMARY"]
        if primary.get("TARGRA") is None:
            return None
        try:
            dec = Angle(primary["TARGDEC"], unit=u.deg).deg
            pmra, pmdec = primary.get("TARGPMRA"), primary.get("TARGPMDC")
            return {
                "source_id": primary.get("OBJECT"),
                "ra": Angle(primary["TARGRA"], unit=u.hourangle).to(u.deg).value,
                "dec": dec,
                "pmra": None
                if pmra is None
                else pmra * 15.0 * np.cos(np.radians(dec)) * 1e3,
                "pmdec": None if pmdec is None else pmdec * 1e3,
                "parallax": primary.get("TARGPLAX"),
                "rv": primary.get("TARGRADV"),
                "frame": "icrs",
                "epoch": primary.get("TARGEPOC"),
                "equinox": primary.get("TARGEQUI"),
            }
        except Exception as exc:
            logger.warning(
                "could not build wmko CATALOG_RECORD from L0 PRIMARY TARG* "
                "(%s: %s); left empty",
                type(exc).__name__,
                exc,
            )
            return None

    def set_catalog_record(self, source, record):
        """Upsert one source's row into the CATALOG_RECORD extension + set its flag.

        The single writer for CATALOG_RECORD, shared by the read path (``wmko``),
        AstroQuery (``gaia``/``simbad`` and the merged ``kpf-drp`` row). ``record`` is a
        canonical record dict (the _CATALOG_COLUMNS fields minus ``source``) or None.
        ``astr_src`` defaults to ``source`` when the record omits it (a source row's
        position is its own); the merged row supplies its base's label explicitly.
        Existing rows for other sources are preserved (upsert), so callers add their row
        independently. A None record clears the source's flag and writes no row;
        otherwise the row is (re)written and the flag set to 1. A source without a
        registered presence flag (e.g. ``kpf-drp``, which must always be present) writes
        no flag keyword. Missing floats become NaN, missing strings "".
        """
        table = self.data["CATALOG_RECORD"]
        rows = {}
        if table.colnames:
            for row in table:
                rows[str(row["source"])] = {
                    name: row[name] for name in _CATALOG_COLUMNS
                }
        if record is None:
            rows.pop(source, None)
        else:
            rows[source] = {"source": source, "astr_src": source, **record}

        ordered = list(rows.values())
        new_table = Table()
        for name in _CATALOG_COLUMNS:
            if name in _CATALOG_STR_COLUMNS:
                new_table[name] = np.array(
                    ["" if r[name] is None else r[name] for r in ordered], dtype=str
                )
            else:
                new_table[name] = np.array(
                    [np.nan if r[name] is None else r[name] for r in ordered],
                    dtype=float,
                )
                new_table[name].unit = _CATALOG_UNITS[name]
        self.set_data("CATALOG_RECORD", new_table)
        flag = _CATALOG_FLAGS.get(source)
        if flag is not None:
            self.set_keyword(flag, 1 if record is not None else 0)

    def _read(self, hdul):
        """Read all extensions from an L0 FITS HDUList.

        Iterates through all HDUs and creates extensions dynamically
        based on what is present. ``CompImageHDU`` is transparently
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
                        raise ValueError(
                            f"Non-standard extension {ext_name!r} in L0 file"
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
            elif ext_name == "CATALOG_RECORD":
                # astropy reads NaN float cells back as masked; fill to NaN so
                # consumers see one missing-value sentinel regardless of whether the
                # table was just built by AstroQuery or round-tripped through FITS.
                self.set_data(ext_name, Table.read(hdu).filled(np.nan))
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
            logger.warning(
                "Filename '%s' does not follow the KPF L0 naming "
                "convention (KP.YYYYMMDD.NNNNN.NN.fits)",
                basename,
            )
            return False
        return True

    def generate_standard_filename(self):
        """KPF L0 filenames follow the KP.YYYYMMDD.NNNNN.NN.fits pattern.

        Raises
        ------
        ValueError
            If ``obs_id`` is unset or invalid.
        """
        return kpf_filename(self.obs_id, "L0")

    def to_fits(self, fn=None):
        """Write L0 data to a FITS file (plain ImageHDU, no compression)."""
        if fn is None:
            fn = self.generate_standard_filename()
        if not fn.endswith(".fits"):
            raise NameError("Filename must end with .fits")

        self.receipt_add_entry("to_fits", f"out_filepath={fn}", "PASS")
        # Warn-only advisory (match rvdata); the write still proceeds.
        self.check_filename_convention(fn)

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
        logger.info("wrote %s to %s", type(self).__name__, fn)
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
        each row take the instrument value if present, else the row default, and
        type it to the EPRV DataType. ``JD_UTC`` is the sole exception -- a
        per-frame transform of the native ``MJD-OBS``, computed below.

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
                # Verbatim: some cards legitimately read "None" (e.g. CAL-OBJ when
                # the cal fiber is dark -> TRACE1/CLSRC1); real data, not a sentinel.
                raw_value = wmko_primary.get(instrument_key)
            elif default_val is not None and str(default_val).strip():
                raw_value = default_val
            else:
                continue
            # Type to the EPRV DataType so the L1 overlay matches L2's typing (e.g.
            # NUMTRACE '5' -> 5). Emit the bare value so to_kpf1's assignment keeps
            # the comment KPF1.__init__ seeded onto the PRIMARY card.
            dt = self.keyword_registry.eprv_primary_datatypes.get(eprv_key)
            out[eprv_key] = (
                parse_value_to_datatype(eprv_key, dt, raw_value)[0] if dt else raw_value
            )

        # JD_UTC: convert the native MJD-OBS to a full Julian date. The one value
        # transform header_map can't express, so it lives here, not as a default.
        mjd = wmko_primary.get("MJD-OBS")
        if mjd not in (None, "", "UNKNOWN"):
            out["JD_UTC"] = (
                float(mjd) + 2400000.5,
                "[day] Julian date of exposure start",
            )
        return out

    def to_kpf1(self):
        """Create a KPF1 scaffold from this L0, carrying over headers and
        pass-through extensions.

        The raw WMKO PRIMARY is converted to EPRV-standard names/values here (the
        single conversion site; see ``_map_header``), and preserved verbatim in the
        immutable INSTRUMENT_HEADER. DRP-RUN provenance lives on RECEIPT (stamped
        at read) and reaches L1 via the header forward below.

        Returns a KPF1 with EPRV PRIMARY, INSTRUMENT_HEADER, pass-through
        extensions (CA_HK, EXPMETER_SCI/SKY, TELEMETRY, DRP_CONFIG), receipt, and
        obs_id copied over. GREEN_CCD, GREEN_VAR, RED_CCD, RED_VAR are created
        empty -- the caller (image assembly) fills those in.
        """
        kpf1 = KPF1()

        # Convert the raw WMKO PRIMARY to EPRV names/values, and snapshot the raw
        # L0 PRIMARY verbatim into the immutable INSTRUMENT_HEADER.
        if "PRIMARY" in self.headers:
            for key, value in self._map_header().items():
                kpf1.headers["PRIMARY"][key] = value

            if "INSTRUMENT_HEADER" not in kpf1.extensions:
                kpf1.create_extension("INSTRUMENT_HEADER", "ImageHDU")
            kpf1.set_header(
                "INSTRUMENT_HEADER", self.as_fits_header(self.headers["PRIMARY"])
            )

        for ext_name in self._L0_TO_L1_PASSTHROUGH:
            if ext_name in self.extensions:
                ext_type = self.extensions[ext_name]
                if ext_name not in kpf1.extensions:
                    kpf1.create_extension(ext_name, ext_type)
                if ext_name in self.data and self.data[ext_name] is not None:
                    kpf1.set_data(ext_name, self.data[ext_name])
                if ext_name in self.headers:
                    kpf1.set_header(ext_name, self.headers[ext_name])

        # Forward the L0 QUALITY_CONTROL and RECEIPT headers onto L1, mirroring
        # to_kpf2/to_kpf4. (PRIMARY is converted via _map_header above, not copied.)
        self._forward_headers(kpf1, ("QUALITY_CONTROL", "RECEIPT"))

        if self.receipt is not None and not self.receipt.empty:
            kpf1.receipt = self.receipt.copy()
        kpf1.obs_id = self.obs_id

        # DATALVL is set by KPF1.__init__; _map_header no longer emits it, so no
        # fixup is needed here.
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
