"""WMKO-native -> EPRV-standard L0 header conversion.

The single conversion site. It runs immediately after ``KPF0.from_fits``, before
any other module, so every stage downstream of the load reads one PRIMARY: the
EPRV-standard one. The raw instrument header is preserved verbatim in
INSTRUMENT_HEADER, which is where a module that genuinely needs a native card
reads it from.

``KPF0`` itself does not standardize on read -- a KPF0 loaded from disk must
reflect the unaltered file -- so ``KPF0.to_kpf1`` raises on an L0 that has not
been through here rather than silently forwarding native cards onto an EPRV L1.
"""

import importlib.metadata
import logging

import pandas as pd

from kpfpipe import REPO_ROOT, __githash__, __version__

logger = logging.getLogger(__name__)

# EPRV-standard compliance is pinned to the installed rv-data-standard release
# (environment.yml pins it exactly): EPRVTAG is its version ("v0.4.0"), VOCLASS
# the release month ("EPRVSTANDARD2026.06"). The release date is not in package
# metadata (PyPI-only), so map it from the exact pin here; bump both together.
_RVDATA_VERSION = importlib.metadata.version("rv-data-standard")
_RVDATA_RELEASE_MONTHS = {"0.4.0": "2026.06"}

# The calibration half of KPF's IMTYPE vocabulary; 'Object' is the other half.
_CAL_OBSTYPES = frozenset({"Bias", "Dark", "Flatlamp", "Arclamp", "Etalon"})


class StandardizeDataFormat:
    """Convert a raw L0's PRIMARY header to the EPRV standard, in place.

    Parameters
    ----------
    l0_obj : KPF0
        The freshly loaded L0. ``perform`` mutates and returns it.
    """

    def __init__(self, l0_obj):
        self.l0_obj = l0_obj

    def perform(self):
        """Standardize ``l0_obj``'s PRIMARY header; a no-op if already done.

        Snapshots the native header into INSTRUMENT_HEADER, replaces PRIMARY with
        the registry's typed EPRV skeleton, fills it from the header map, then
        stamps the values that are computed rather than mapped (INSTERA, and the
        DRP/EPRV provenance tags).

        Returns
        -------
        KPF0
            The same object, standardized.
        """
        l0 = self.l0_obj
        if "PRIMARY" not in l0.headers:
            raise ValueError(
                f"{type(l0).__name__} has no PRIMARY header to standardize"
            )
        if l0.standardized:
            return l0

        native = l0.as_fits_header(l0.headers["PRIMARY"])
        l0.set_header("INSTRUMENT_HEADER", native)

        l0.headers["PRIMARY"].clear()
        l0._seed_primary()
        self._fill_from_native(l0, native)
        self._stamp_observing_mode(l0, native)
        self._stamp_instrument_era(l0)

        l0.set_keyword("DATALVL", "L0")
        l0.set_keyword("DRPTAG", __version__)
        l0.set_keyword("DRPHASH", __githash__)
        l0.set_keyword("EPRVTAG", f"v{_RVDATA_VERSION}")
        l0.set_keyword(
            "VOCLASS",
            f"EPRVSTANDARD{_RVDATA_RELEASE_MONTHS[_RVDATA_VERSION]}",
        )

        l0.receipt_add_entry("standardize_data_format", "", "PASS")
        return l0

    @staticmethod
    def _fill_from_native(l0, native):
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
        registry = l0.keyword_registry
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

            l0.set_keyword(
                eprv_key,
                registry._parse_value(
                    eprv_key, registry.datatype_for(eprv_key, "PRIMARY"), raw_value
                ),
            )

        # JD_UTC: convert the native MJD-OBS to a full Julian date. The one value
        # transform the header map can't express, so it lives here, not as a default.
        mjd = native.get("MJD-OBS")
        if mjd not in (None, "", "UNKNOWN"):
            l0.set_keyword("JD_UTC", float(mjd) + 2400000.5)

    @staticmethod
    def _stamp_observing_mode(l0, native):
        """Stamp ISSOLAR and OBSMODE from the OBSTYPE the tabular fill just mapped.

        OBSMODE is redundant for KPF, which has one optical configuration: the
        EPRV standard defines it for instruments with several (hi-res/low-res),
        so here it only restates OBSTYPE and ISSOLAR as sci/cal/solar. An IMTYPE
        outside the vocabulary is a frame this DRP cannot classify, so it raises.
        """
        obstype = str(l0.headers["PRIMARY"]["OBSTYPE"]).strip()
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
                f"{l0.obs_id} has IMTYPE {obstype!r}, which is not one of KPF's "
                f"observation types ('Object', {', '.join(sorted(_CAL_OBSTYPES))})"
            )
        l0.set_keyword("ISSOLAR", is_solar)
        l0.set_keyword("OBSMODE", mode)

    @staticmethod
    def _stamp_instrument_era(l0):
        """Stamp INSTERA from JD_UTC against ``reference/instrument_eras.csv``.

        Runs after the tabular fill, which is what supplies JD_UTC. A frame that
        cannot be dated, or that no era covers, has no reference calibrations, so
        this raises rather than shipping an unattributable product.
        """
        obs_time = pd.to_datetime(
            l0.headers["PRIMARY"]["JD_UTC"], unit="D", origin="julian"
        )
        if pd.isna(obs_time):
            raise ValueError(
                f"Cannot infer the instrument era of {l0.obs_id}: its JD_UTC is "
                f"{l0.headers['PRIMARY']['JD_UTC']!r}"
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
        l0.set_keyword("INSTERA", str(in_era.iloc[0]["INSTERA"]))
