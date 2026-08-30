"""
KPF Level 0 (raw CCD) data model.

Raw FITS readout from the KPF instrument: amplifier arrays plus exposure
meter, guide camera, telemetry, and telescope metadata.
"""

import importlib.metadata
import logging
import os

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from kpfpipe import REPO_ROOT, __githash__, __version__
from kpfpipe.data_models.base import KPFDataModel
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.utils.kpf import get_obs_id

logger = logging.getLogger(__name__)

# EPRV-standard compliance is pinned to the installed rv-data-standard release
# (environment.yml pins it exactly): EPRVTAG is its version ("v0.4.0"), VOCLASS
# the release month ("EPRVSTANDARD2026.06"). The release date is not in package
# metadata (PyPI-only), so map it from the exact pin here; bump both together.
_RVDATA_VERSION = importlib.metadata.version("rv-data-standard")
_RVDATA_RELEASE_MONTHS = {"0.4.0": "2026.06"}

# The calibration half of KPF's IMTYPE vocabulary; 'Object' is the other half.
_CAL_OBSTYPES = frozenset({"Bias", "Dark", "Flatlamp", "Arclamp", "Etalon"})


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

        # No PRIMARY seed here: ``_read`` replaces the stored PRIMARY header
        # wholesale, so anything stamped at construction would be discarded --
        # and a KPF0 read from disk must reflect the unaltered file.
        # ``standardize_header_format`` seeds it once, after the read.
        self._create_manifest_extensions()
        self._fill_typed_empty_tables()
        self._set_ext_descript()

    @classmethod
    def from_fits(cls, fn, instrument=None, standardize=False, **kwargs):
        """Read an L0 from FITS, standardizing its header on the way in if asked.

        ``standardize`` defaults to False so a naive read reflects the file on
        disk: PRIMARY as WMKO wrote it, INSTRUMENT_HEADER empty. The pipeline
        always passes True -- every stage after the load reads the EPRV PRIMARY.
        """
        obj = super().from_fits(fn, instrument=instrument, **kwargs)
        if standardize:
            obj.standardize_header_format()
        return obj

    @property
    def standardized(self):
        """True once ``standardize_header_format`` has run on this L0.

        The receipt is the pipeline's existing, explicit, persisted answer to
        "which steps have run", so it -- not the header or extension structure --
        is the discriminator. A fresh ``KPF0()`` has a zero-row receipt, hence
        the ``not r.empty`` guard.
        """
        r = self.receipt
        return (
            r is not None
            and not r.empty
            and "standardize_header_format" in r["FUNCTION"].values
        )

    def read(self, hdul, instrument=None, overwrite=False, **kwargs):
        """Read an L0 FITS HDUList, then stamp DRP provenance onto RECEIPT."""
        super().read(hdul, instrument=instrument, overwrite=overwrite, **kwargs)
        # Derive obs_id before stamping: _stamp_wmko_tracking writes it as the
        # ORIGID provenance card, so it must be known by then.
        self.obs_id = get_obs_id(self.filename)
        if "PRIMARY" in self.headers:
            self._stamp_wmko_tracking()

    @property
    def _native_header(self):
        """The raw instrument header, wherever this L0 currently keeps it.

        Before standardization that is PRIMARY; after it, INSTRUMENT_HEADER --
        including on a re-read of a standardized L0 written back to disk, whose
        PRIMARY is the EPRV header and carries no native card at all.

        Keyed on ``standardized``, the semantic fact, not on whether
        INSTRUMENT_HEADER holds anything: the manifest creates that extension on
        every L0, so a raw one round-trips with a header of structural cards --
        empty of content but not falsy.
        """
        if self.standardized:
            return self.headers["INSTRUMENT_HEADER"]
        return self.headers["PRIMARY"]

    def _stamp_wmko_tracking(self):
        """Stamp WMKO DRP-RUN provenance onto the L0 RECEIPT at read time.

        The single population site for DRPVERNO, PROGID/KOAID, DRPSTATU, and ORIGID
        (the original L0 obs_id), written to their registry home (RECEIPT) via
        ``set_keyword`` and ridden forward onto L1/L2/L4 (see ``to_kpf1``).
        DRPVERNO/DRPSTATU/ORIGID are always (re)stamped. KOAID and PROGID are not
        EPRV keywords -- they map from the native OFNAME and PROGNAME, so they are
        read off ``_native_header``. A missing PROGNAME defaults PROGID to UNKNOWN
        with a warning; a missing OFNAME (the archive obs_id) raises.
        """
        self.set_keyword("DRPVERNO", __version__)
        self.set_keyword("DRPSTATU", "File ingested into KPF-DRP")
        self.set_keyword("ORIGID", self.obs_id)
        primary = self._native_header

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

    def standardize_header_format(self):
        """Convert this L0's PRIMARY from WMKO-native to EPRV-standard, in place.

        The single conversion site, run at load by ``from_fits(standardize=True)``
        and before any other stage, so everything downstream reads one PRIMARY:
        the EPRV-standard one. The raw instrument header is preserved verbatim in
        INSTRUMENT_HEADER, which is where a module that genuinely needs a native
        card reads it from.

        Snapshots that native header, replaces PRIMARY with the registry's typed
        EPRV skeleton, fills it from the header map, then stamps the values that
        are computed rather than mapped (INSTERA, and the DRP/EPRV provenance
        tags). A second call is a no-op.

        Returns
        -------
        KPF0
            This object, standardized.
        """
        if self.standardized:
            return self

        native = self.as_fits_header(self.headers["PRIMARY"])
        self.set_header("INSTRUMENT_HEADER", native)

        self.headers["PRIMARY"].clear()
        self._seed_primary()
        self._fill_from_native(native)
        self._stamp_observing_mode(native)
        self._stamp_instrument_era()

        self.set_keyword("DATALVL", "L0")
        self.set_keyword("DRPTAG", __version__)
        self.set_keyword("DRPHASH", __githash__)
        self.set_keyword("EPRVTAG", f"v{_RVDATA_VERSION}")
        self.set_keyword(
            "VOCLASS",
            f"EPRVSTANDARD{_RVDATA_RELEASE_MONTHS[_RVDATA_VERSION]}",
        )

        self.receipt_add_entry("standardize_header_format", "", "PASS")
        return self

    def _fill_from_native(self, native):
        """Apply ``EPRV-header-map.csv`` to the native header, card by card.

        A pure tabular pass over the PRIMARY-targeted map rows: take the
        instrument value if the native header carries it, else the row default,
        and type it to the registry DataType. A row with neither is skipped, so
        the seeded blank card stands. ``JD_UTC`` is the one non-tabular value --
        a per-frame transform of the native ``MJD-OBS`` -- and is applied last.

        The ``KPF_EXT=QUALITY_CONTROL`` rows are not filled here: their source
        extension is still empty at this point in the pipeline, so ``DiagL0``
        stamps them onto PRIMARY when it computes them.
        """
        registry = self.keyword_registry
        for row in registry.header_map.itertuples(index=False):
            kpf_ext = "" if pd.isna(row.KPF_EXT) else str(row.KPF_EXT).strip()
            if kpf_ext not in ("", "PRIMARY"):
                continue
            eprv_key = str(row.EPRV_KEY).strip()
            kpf_key = "" if pd.isna(row.KPF_KEY) else str(row.KPF_KEY).strip()
            default = None if pd.isna(row.DEFAULT) else row.DEFAULT

            if kpf_key and kpf_key in native:
                # Verbatim: some cards legitimately read "None" (e.g. CAL-OBJ when
                # the cal fiber is dark -> TRACE5/CLSRC5); real data, not a sentinel.
                raw_value = native.get(kpf_key)
            elif default is not None and str(default).strip():
                raw_value = default
            else:
                continue

            self.set_keyword(
                eprv_key,
                registry._parse_value(
                    eprv_key, registry.datatype_for(eprv_key, "PRIMARY"), raw_value
                ),
            )

        # JD_UTC: convert the native MJD-OBS to a full Julian date. The one value
        # transform the header map can't express, so it lives here, not as a default.
        mjd = native.get("MJD-OBS")
        if mjd not in (None, "", "UNKNOWN"):
            self.set_keyword("JD_UTC", float(mjd) + 2400000.5)

    def _stamp_observing_mode(self, native):
        """Stamp ISSOLAR and OBSMODE from the OBSTYPE the tabular fill just mapped.

        OBSMODE is redundant for KPF, which has one optical configuration: the
        EPRV standard defines it for instruments with several (hi-res/low-res),
        so here it only restates OBSTYPE and ISSOLAR as sci/cal/solar. An IMTYPE
        outside the vocabulary is a frame this DRP cannot classify, so it raises.
        """
        obstype = str(self.headers["PRIMARY"]["OBSTYPE"]).strip()
        is_solar = any(
            str(native.get(key, "")).strip().lower() == "socal"
            for key in ("OBJECT", "TARGNAME")
        )
        if obstype == "Object":
            mode = "solar" if is_solar else "sci"
        elif obstype in _CAL_OBSTYPES:
            mode = "cal"
        else:
            raise ValueError(
                f"{self.obs_id} has IMTYPE {obstype!r}, which is not one of KPF's "
                f"observation types ('Object', {', '.join(sorted(_CAL_OBSTYPES))})"
            )
        self.set_keyword("ISSOLAR", is_solar)
        self.set_keyword("OBSMODE", mode)

    def _stamp_instrument_era(self):
        """Stamp INSTERA from JD_UTC against ``reference/instrument_eras.csv``.

        Runs after the tabular fill, which is what supplies JD_UTC. A frame that
        cannot be dated, or that no era covers, has no reference calibrations, so
        this raises rather than shipping an unattributable product.
        """
        obs_time = pd.to_datetime(
            self.headers["PRIMARY"]["JD_UTC"], unit="D", origin="julian"
        )
        if pd.isna(obs_time):
            raise ValueError(
                f"Cannot infer the instrument era of {self.obs_id}: its JD_UTC is "
                f"{self.headers['PRIMARY']['JD_UTC']!r}"
            )
        eras = pd.read_csv(
            f"{REPO_ROOT}/reference/instrument_eras.csv",
            parse_dates=["UT_start_date", "UT_end_date"],
        )
        in_era = eras[
            (eras["UT_start_date"] <= obs_time) & (obs_time <= eras["UT_end_date"])
        ]
        if in_era.empty:
            raise ValueError(
                f"No KPF instrument era covers {obs_time}; the eras of "
                f"reference/instrument_eras.csv do not span it"
            )
        self.set_keyword("INSTERA", str(in_era.iloc[0]["INSTERA"]))

    def to_fits(self, fn=None):
        """Write L0 data to a FITS file (plain ImageHDU, no compression)."""
        if fn is None:
            fn = self.generate_standard_filename()
        if not fn.endswith(".fits"):
            raise NameError("Filename must end with .fits")

        self.receipt_add_entry("to_fits", f"out_filepath={fn}", "PASS")
        # Warn-only advisory (match rvdata); the write still proceeds.
        self.check_filename_convention(fn)

        # FILENAME is an EPRV PRIMARY keyword, so it is only correct to stamp it
        # on a standardized L0; a raw one must reflect the file it came from.
        if self.standardized:
            self.set_keyword("FILENAME", os.path.basename(fn))

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
        "CATALOG_RECORD",
        "INSTRUMENT_HEADER",
    ]

    def to_kpf1(self):
        """Create a KPF1 scaffold from this L0, carrying over headers and
        pass-through extensions.

        The PRIMARY is already EPRV-standard (converted once by
        ``standardize_header_format``, which also snapshotted the raw instrument
        header into INSTRUMENT_HEADER), so this is a pure forward: PRIMARY,
        QUALITY_CONTROL and RECEIPT overlay card by card onto the L1 skeleton,
        and the pass-through extensions carry their data across.

        Returns a KPF1 with EPRV PRIMARY, INSTRUMENT_HEADER, pass-through
        extensions (CA_HK, EXPMETER_SCI/SKY, TELEMETRY, DRP_CONFIG,
        CATALOG_RECORD), receipt, and obs_id copied over. GREEN_CCD, GREEN_VAR,
        RED_CCD, RED_VAR are created empty -- the caller (image assembly) fills
        those in.

        Raises
        ------
        ValueError
            If this L0 has not been standardized. Forwarding a raw WMKO PRIMARY
            onto an EPRV L1 PRIMARY would be silent corruption, so a mis-ordered
            call fails loud and names the step that is missing.
        """
        if not self.standardized:
            raise ValueError(
                f"{self.obs_id} has not been standardized; call "
                "standardize_header_format before to_kpf1"
            )

        kpf1 = KPF1()

        for ext_name in self._L0_TO_L1_PASSTHROUGH:
            if ext_name in self.extensions:
                ext_type = self.extensions[ext_name]
                if ext_name not in kpf1.extensions:
                    kpf1.create_extension(ext_name, ext_type)
                if ext_name in self.data and self.data[ext_name] is not None:
                    kpf1.set_data(ext_name, self.data[ext_name])
                if ext_name in self.headers:
                    kpf1.set_header(ext_name, self.headers[ext_name])

        self._forward_headers(kpf1, ("PRIMARY", "QUALITY_CONTROL", "RECEIPT"))

        if self.receipt is not None and not self.receipt.empty:
            kpf1.receipt = self.receipt.copy()
        kpf1.obs_id = self.obs_id

        kpf1.set_keyword("DATALVL", "L1")
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
