"""
KPF Level 0 (raw CCD) data model.

Raw FITS readout from the KPF instrument: amplifier arrays plus exposure
meter, guide camera, telemetry, and telescope metadata.
"""

import importlib.metadata
import logging
import math

import astropy.units as u
import pandas as pd
from astropy.coordinates import Angle

from kpfpipe import DETECTOR, REPO_ROOT, __githash__, __version__
from kpfpipe.data_models.base import KPFDataModel
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.utils.astro import KECK_LOCATION
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

# KPF's fiber-source names -> the EPRV calibration-source vocabulary of CLSRC#.
_CAL_SOURCES = {
    "target": "Target",
    "sky": "Sky",
    "none": "None",
    "th_gold": "ThAr",
    "th_daily": "ThAr",
    "lfcfiber": "LFC",
    "etalonfiber": "Etalon",
}


class KPF0(KPFDataModel):
    """
    KPF Level 0 raw data model.

    Represents a raw CCD readout from the KPF instrument. Construct from a
    FITS file with ``KPF0.from_fits(path)``, then read amplifier arrays from
    ``data`` (e.g. ``data["GREEN_AMP1"]``) and header dicts from ``headers``
    (e.g. ``headers["PRIMARY"]``).
    """

    # A raw L0 carries the native header it was read with; see _SEEDS_PRIMARY.
    _SEEDS_PRIMARY = False

    def __init__(self):
        super().__init__()
        self.level = 0
        self._build()

    @classmethod
    def from_fits(cls, fn, standardize=False):
        """Read an L0 from FITS, standardizing its header on the way in if asked.

        ``standardize`` defaults to False so a naive read reflects the file on
        disk: PRIMARY as WMKO wrote it, INSTRUMENT_HEADER empty. The pipeline
        always passes True -- every stage after the load reads the EPRV PRIMARY.
        """
        obj = super().from_fits(fn)
        if standardize:
            obj.standardize_headers()
        return obj

    @property
    def standardized(self):
        """True once ``standardize_headers`` has run on this L0.

        The receipt is the pipeline's existing, explicit, persisted answer to
        "which steps have run", so it -- not the header or extension structure --
        is the discriminator. A fresh ``KPF0()`` has a zero-row receipt, hence
        the ``not r.empty`` guard.
        """
        r = self.receipt
        return (
            r is not None
            and not r.empty
            and "standardize_headers" in r["FUNCTION"].values
        )

    def read(self, hdul, instrument=None):
        """Read an L0 FITS HDUList, resolving ``obs_id`` from its filename.

        The obs_id must be known before ``standardize_headers``, which
        writes it as the ORIGID provenance card.
        """
        super().read(hdul, instrument=instrument)
        self.obs_id = get_obs_id(self.filename)

    def standardize_headers(self):
        """Convert this L0's PRIMARY from WMKO-native to EPRV-standard, in place.

        The single conversion site, run at load by ``from_fits(standardize=True)``
        and before any other stage, so everything downstream reads one PRIMARY:
        the EPRV-standard one. The raw instrument header is preserved verbatim in
        INSTRUMENT_HEADER, which is where a module that genuinely needs a native
        card reads it from.

        Snapshots that native header, replaces PRIMARY with the registry's typed
        EPRV skeleton, fills it from the header map, then stamps the values that
        are computed rather than mapped (INSTERA, the site coordinates, and the
        WMKO/DRP/EPRV provenance cards). A second call is a no-op.

        Returns
        -------
        KPF0
            This object, standardized.
        """
        if self.standardized:
            return self

        self.set_header(
            "INSTRUMENT_HEADER", self.as_fits_header(self.headers["PRIMARY"])
        )

        # map keywords from INSTRUMENT_HEADER -> PRIMARY and compute the rest
        self.headers["PRIMARY"].clear()
        self._seed_primary()
        self._fill_from_native()
        self._program_identification()
        self._instrument_era()
        self._observing_mode()
        self._site_coordinates()
        self._tcs_pointing()
        self._drp_metadata()

        # file identification metadata
        koaid = self.headers["INSTRUMENT_HEADER"].get("OFNAME")
        if not koaid and self.filename:
            raise ValueError("OFNAME absent from L0 PRIMARY; cannot set KOAID")
        self.set_keyword("KOAID", koaid)
        self.set_keyword("ORIGID", self.obs_id)
        self.set_keyword("DATALVL", "L0")

        self.receipt_add_entry("standardize_headers", "", "PASS")
        return self

    def _fill_from_native(self):
        """Apply ``header-map.csv`` to the native header, card by card.

        A pure tabular pass over the PRIMARY-targeted map rows: take the
        instrument value if the native header carries it, else the row default,
        and type it to the registry DataType. A row with neither is skipped, so
        the seeded blank card stands. ``JD_UTC`` is the one non-tabular value --
        a per-frame transform of the native ``MJD-OBS`` -- and is applied last.

        The ``KPF_EXT=QUALITY_CONTROL`` rows are not filled here: their source
        extension is still empty at this point in the pipeline, so ``DiagL0``
        stamps them onto PRIMARY when it computes them.
        """
        native = self.headers["INSTRUMENT_HEADER"]
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

    def _program_identification(self):
        """Stamp the observing program's identity onto PRIMARY.

        These map from native cards, so they are read off INSTRUMENT_HEADER
        rather than the EPRV skeleton this is filling. A card the frame does not
        carry defaults to UNKNOWN with a warning. ORGANIZA is the data owner, one
        organization for every KPF frame. PROGID and PROGRAM are the WMKO and
        EPRV spellings of one value.
        """
        self.set_keyword("ORGANIZA", "WMKO")

        program = self._resolve_native("PROGNAME", ("GRPROGNA", "RDPROGNA"))
        self.set_keyword("PROGID", program)
        self.set_keyword("PROGRAM", program)
        self.set_keyword("PINAME", self._resolve_native("PROGPI"))
        self.set_keyword(
            "OBSERVER", self._resolve_native("OBSERVER", ("GROBSERV", "RDOBSERV"))
        )

    def _resolve_native(self, key, fallbacks=()):
        """``key``'s native value, or 'UNKNOWN' when the frame names none.

        A missing card falls back to the guider and readout copies KPF writes
        alongside it, which stand in only when they agree -- two that disagree
        name no one value.
        """
        native = self.headers["INSTRUMENT_HEADER"]
        value = native.get(key)
        if value:
            return value
        copies = {native.get(k) for k in fallbacks} - {None, ""}
        if len(copies) == 1:
            return copies.pop()
        if copies:
            logger.warning(
                "%s absent from L0 PRIMARY and %s disagree; defaulting to 'UNKNOWN'",
                key,
                "/".join(fallbacks),
            )
        else:
            logger.warning("%s absent from L0 PRIMARY; defaulting to 'UNKNOWN'", key)
        return "UNKNOWN"

    def _instrument_era(self):
        """Stamp INSTERA from JD_UTC against ``reference/instrument_eras.csv``.

        Runs after the tabular fill, which is what supplies JD_UTC. A frame no
        era covers -- an undated one included, since its JD_UTC parses to NaT,
        which no interval contains -- has no reference calibrations, so this
        raises rather than shipping an unattributable product.
        """
        obs_time = pd.to_datetime(
            self.headers["PRIMARY"]["JD_UTC"], unit="D", origin="julian"
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

    def _observing_mode(self):
        """Stamp ISSOLAR, OBSMODE and CLSRC# from what the tabular fill mapped.

        OBSMODE is redundant for KPF, which has one optical configuration: the
        EPRV standard defines it for instruments with several (hi-res/low-res),
        so here it only restates OBSTYPE and ISSOLAR as sci/cal/solar. An IMTYPE
        outside the vocabulary is a frame this DRP cannot classify, so it raises.

        CLSRC# names each trace's illumination source in the EPRV vocabulary,
        normalized from the KPF fiber-source name TRACE# carries. A source the
        standard does not name (a broadband flat, say) is stamped verbatim, and
        the modules that dispatch on it reject it there.
        """
        native = self.headers["INSTRUMENT_HEADER"]
        prim = self.headers["PRIMARY"]
        obstype = str(prim["OBSTYPE"]).strip()
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

        for trace in range(1, DETECTOR["numtrace"] + 1):
            source = prim.get(f"TRACE{trace}")
            if not source:
                continue
            name = str(source).strip().lower()
            self.set_keyword(f"CLSRC{trace}", _CAL_SOURCES.get(name, source))

    def _site_coordinates(self):
        """Stamp the observatory location onto PRIMARY from ``KECK_LOCATION``.

        The site is a property of the telescope, not of the frame, so it is
        computed here rather than mapped from a native card. 1e-5 deg is ~1 m,
        against the ~140 m a 1 cm/s barycentric correction needs
        (dv = omega * dx); rounding also absorbs astropy's geodetic noise.
        """
        self.set_keyword("GEOSYS", KECK_LOCATION.ellipsoid)
        self.set_keyword("OBSLON", round(KECK_LOCATION.lon.deg, 5))
        self.set_keyword("OBSLAT", round(KECK_LOCATION.lat.deg, 5))
        self.set_keyword("OBSALT", round(KECK_LOCATION.height.to_value("m"), 3))
        self.set_keyword("OBSGEO-X", round(KECK_LOCATION.x.to_value("m"), 3))
        self.set_keyword("OBSGEO-Y", round(KECK_LOCATION.y.to_value("m"), 3))
        self.set_keyword("OBSGEO-Z", round(KECK_LOCATION.z.to_value("m"), 3))

    def _tcs_pointing(self):
        """Stamp the pointing cards derived from the TCS cards the map filled.

        TZA1 is the complement of the TEL1 elevation. PARST1/PAREND1 are the
        parallactic angle -- the angle at the target between the celestial pole
        and the zenith, positive east of north::

            q = atan2(sin H, tan(lat) cos(dec) - sin(dec) cos H)

        (Meeus 1998, eq. 14.1), evaluated half an exposure either side of THA1,
        the hour angle the TCS read at TTIME, mid-exposure. Neglecting refraction
        puts it ~0.1 deg from the native PARANG of a frame at airmass 1.3.

        A frame the TCS never pointed -- a bias, a lamp -- carries no TCS cards
        to derive from, and these stay blank with them.
        """
        _SIDEREAL_RATE = 1.0027379  # hours per hour of UT

        prim = self.headers["PRIMARY"]
        if prim.get("TEL1") is not None:
            self.set_keyword("TZA1", round(90.0 - float(prim["TEL1"]), 2))
        if not prim.get("THA1") or not prim.get("TDEC1"):
            return

        lat = KECK_LOCATION.lat.rad
        dec = Angle(prim["TDEC1"], unit=u.deg).rad
        mid = Angle(prim["THA1"], unit=u.hourangle)
        half = (
            0.5 * float(prim.get("EXPTIME") or 0.0) / 3600.0 * _SIDEREAL_RATE
        ) * u.hourangle
        for keyword, ha in (("PARST1", mid - half), ("PAREND1", mid + half)):
            q = math.atan2(
                math.sin(ha.rad),
                math.tan(lat) * math.cos(dec) - math.sin(dec) * math.cos(ha.rad),
            )
            self.set_keyword(keyword, round(math.degrees(q), 2))

    def _drp_metadata(self):
        """Stamp the pipeline and standard version cards onto PRIMARY.

        DRPVERNO is the WMKO spelling of the DRP version, DRPTAG the EPRV one;
        both carry ``kpfpipe.__version__``.
        """
        self.set_keyword("DRPVERNO", __version__)
        self.set_keyword("DRPTAG", __version__)
        self.set_keyword("DRPHASH", __githash__)
        self.set_keyword("EPRVTAG", f"v{_RVDATA_VERSION}")
        self.set_keyword(
            "VOCLASS",
            f"EPRVSTANDARD{_RVDATA_RELEASE_MONTHS[_RVDATA_VERSION]}",
        )

    _L0_TO_L1_PASSTHROUGH = [
        "CA_HK",
        "EXPMETER_SCI",
        "EXPMETER_SKY",
        "TELEMETRY",
        "INSTRUMENT_HEADER",
        "CATALOG_RECORD",
        "DRP_CONFIG",
    ]

    def to_kpf1(self):
        """Create a KPF1 scaffold from this L0, carrying over headers and
        pass-through extensions.

        The PRIMARY is already EPRV-standard (from ``standardize_headers``),
        so this is a pure forward. GREEN/RED CCD and VAR are created empty -- the
        caller (image assembly) fills those in.

        Raises
        ------
        ValueError
            If this L0 has not been standardized -- forwarding a raw WMKO PRIMARY
            onto an EPRV L1 PRIMARY would be silent corruption.
        """
        if not self.standardized:
            raise ValueError(
                f"{self.obs_id} has not been standardized; call "
                "standardize_headers before to_kpf1"
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
