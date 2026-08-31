"""
KPF Level 0 (raw CCD) data model.

Raw FITS readout from the KPF instrument: amplifier arrays plus exposure
meter, guide camera, telemetry, and telescope metadata.
"""

import importlib.metadata
import logging

import pandas as pd

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

    # A raw L0 carries the native header it was read with; see _SEEDS_PRIMARY.
    _SEEDS_PRIMARY = False

    def __init__(self):
        super().__init__()
        self.level = 0
        self._build()

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
        """Read an L0 FITS HDUList, resolving ``obs_id`` from its filename.

        The obs_id must be known before ``standardize_header_format``, which
        writes it as the ORIGID provenance card.
        """
        super().read(hdul, instrument=instrument, overwrite=overwrite, **kwargs)
        self.obs_id = get_obs_id(self.filename)

    def _stamp_wmko_tracking(self, native):
        """Stamp WMKO DRP-RUN provenance onto PRIMARY from ``native``.

        The single population site for DRPVERNO, ORIGID (the original L0 obs_id)
        and PROGID/KOAID, which are not EPRV keywords -- they map from the native
        OFNAME and PROGNAME, so they are read off the snapshot rather than the
        EPRV skeleton this is filling. A missing PROGNAME defaults PROGID to
        UNKNOWN with a warning; a missing OFNAME (the archive obs_id) raises on
        a frame read from disk. DRPSTATU is left to ``receipt_add_entry``, which
        advances it per module.
        """
        self.set_keyword("DRPVERNO", __version__)
        self.set_keyword("ORIGID", self.obs_id)

        koaid = native.get("OFNAME")
        # Only a frame read from disk must name itself; an in-memory L0 has no
        # archive identity, and blanks KOAID as it blanks ORIGID above.
        if not koaid and self.filename:
            raise ValueError("OFNAME absent from L0 PRIMARY; cannot set KOAID")
        self.set_keyword("KOAID", koaid)

        progname = native.get("PROGNAME")
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
        are computed rather than mapped (INSTERA, and the WMKO/DRP/EPRV
        provenance cards). A second call is a no-op.

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
        self._stamp_wmko_tracking(native)
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

    @property
    def _stamps_filename(self):
        """FILENAME is an EPRV PRIMARY keyword, so it is only correct to stamp it
        on a standardized L0; a raw one must reflect the file it came from."""
        return self.standardized

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

        The PRIMARY is already EPRV-standard (from ``standardize_header_format``),
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
