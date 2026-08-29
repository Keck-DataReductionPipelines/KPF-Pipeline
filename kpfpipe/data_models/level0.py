"""
KPF Level 0 (raw CCD) data model.

Raw FITS readout from the KPF instrument: amplifier arrays plus exposure
meter, guide camera, telemetry, and telescope metadata.
"""

import logging
import os
import re

import numpy as np
from astropy.io import fits
from astropy.table import Table

from kpfpipe import __version__
from kpfpipe.data_models.base import KPFDataModel
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.utils.io import kpf_filename
from kpfpipe.utils.kpf import get_obs_id

logger = logging.getLogger(__name__)

# WMKO-native L0 filename: KP.YYYYMMDD.NNNNN.NN.fits (the obs_id plus .fits).
_L0_FILENAME_PATTERN = re.compile(r"KP\.\d{8}\.\d{5}\.\d{2}\.fits")

# Initial DRPSTATU value on the L1 EPRV PRIMARY, before any pipeline module runs.
# Each module overwrites it via the receipt_add_entry override (see base.py).
_DRPSTATU_DEFAULT = "File ingested into KPF-DRP"


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
        # StandardizeDataFormat seeds it once, after the read.
        self._create_manifest_extensions()
        self._set_ext_descript()

    @property
    def standardized(self):
        """True once ``StandardizeDataFormat`` has run on this L0.

        The receipt is the pipeline's existing, explicit, persisted answer to
        "which steps have run", so it -- not the header or extension structure --
        is the discriminator. A fresh ``KPF0()`` has a zero-column receipt, hence
        the ``not r.empty`` guard.
        """
        r = self.receipt
        return (
            r is not None
            and not r.empty
            and "standardize_data_format" in r["FUNCTION"].values
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
        self.set_keyword("DRPSTATU", _DRPSTATU_DEFAULT)
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
        ``StandardizeDataFormat``, which also snapshotted the raw instrument
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
            call fails loud and names the module that is missing.
        """
        if not self.standardized:
            raise ValueError(
                f"{self.obs_id} has not been standardized; run "
                "StandardizeDataFormat before to_kpf1"
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
