"""
KPF Level 1 (assembled FFI) data model.

Assembled full frame images (FFIs) built by combining the L0 amplifier
readouts. GREEN and RED CCDs are stored in separate extensions. Also
used for FFI masters calibrations (bias, dark, flat).
"""

import logging
import os

import numpy as np
from astropy.io import fits
from astropy.table import Table

from kpfpipe.data_models.base import KPFDataModel
from kpfpipe.data_models.level2 import KPF2

logger = logging.getLogger(__name__)


class KPF1(KPFDataModel):
    """
    KPF Level 1 assembled FFI data model.

    Contains the assembled GREEN_CCD and RED_CCD frames with their variance
    frames, plus pass-through extensions from L0 (CA_HK, exposure meter,
    telemetry). Construct from a FITS file with ``KPF1.from_fits(path)``, then
    read the assembled frames from ``data`` (e.g. ``data["GREEN_CCD"]``,
    ``data["RED_CCD"]``).
    """

    def __init__(self):
        super().__init__()
        self.level = 1

        self._create_manifest_extensions()
        self._fill_typed_empty_tables()
        # Seed PRIMARY with the registry's typed L1 skeleton (defaults +
        # comments). L1 is not an EPRV level, so the skeleton is stamped here;
        # the values standardize_header_format wrote at L0 are forwarded over it by
        # KPF0.to_kpf1.
        self._seed_primary()
        # DATALVL's seeded value is the L0 default; restamp it for this level.
        self.set_keyword("DATALVL", "L1")
        self._set_ext_descript()

    def to_fits(self, fn=None):
        """Write L1 data to a FITS file."""
        if fn is None:
            fn = self.generate_standard_filename()
        if not fn.endswith(".fits"):
            raise NameError("Filename must end with .fits")

        self.receipt_add_entry("to_fits", f"out_filepath={fn}", "PASS")
        # Warn-only advisory (match rvdata); the write still proceeds.
        self.check_filename_convention(fn)

        if "PRIMARY" in self.headers:
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

    # Mapping of L1 extension names → KPF2 extension names for pass-through.
    # CA_HK is excluded: it is a raw 2D CCD image, not an extracted spectrum.
    # ANCILLARY_SPECTRUM (BinTableHDU) should be populated after Ca HK extraction.
    _L1_TO_L2_PASSTHROUGH = {
        "TELEMETRY": "TELEMETRY",
        "EXPMETER_SCI": "EXPMETER",
        "CATALOG_RECORD": "CATALOG_RECORD",
    }

    def to_kpf2(self):
        """Create a KPF2 scaffold from this L1, carrying over headers and
        pass-through extensions.

        The L1 PRIMARY is already EPRV-standard (converted in KPF0.to_kpf1), so
        the PRIMARY and INSTRUMENT_HEADER headers are a pure pass-through. Pass-
        through extensions (TELEMETRY, EXPMETER_SCI->EXPMETER, CATALOG_RECORD) and
        KPF-friendly aliases (e.g. SCI2_FLUX -> TRACE3_FLUX) are handled below.
        Trace arrays are created empty -- the caller (spectral extraction) fills
        those in.
        """
        kpf2 = KPF2()

        # Forward PRIMARY + INSTRUMENT_HEADER card-by-card (the native->EPRV
        # conversion happened once in KPF0.to_kpf1); PRIMARY overlays onto kpf2's
        # EPRV seed (native wins), INSTRUMENT_HEADER stays a verbatim copy.
        self._forward_headers(kpf2, ("PRIMARY", "INSTRUMENT_HEADER"))

        for kpf1_ext, kpf2_ext in self._L1_TO_L2_PASSTHROUGH.items():
            if kpf1_ext in self.extensions:
                kpf1_type = self.extensions[kpf1_ext]
                if kpf2_ext not in kpf2.extensions:
                    kpf2.create_extension(kpf2_ext, kpf1_type)
                elif kpf2.extensions[kpf2_ext] != kpf1_type:
                    # Update type to match actual L1 data
                    kpf2.extensions[kpf2_ext] = kpf1_type
                if kpf1_ext in self.data and self.data[kpf1_ext] is not None:
                    kpf2.set_data(kpf2_ext, self.data[kpf1_ext])
                if kpf1_ext in self.headers:
                    kpf2.set_header(kpf2_ext, self.headers[kpf1_ext])

        # Forward the L1 QUALITY_CONTROL and RECEIPT headers onto L2. The receipt
        # *table* propagates via the copy below, but the header cards are separate
        # and must be carried explicitly; downstream L2 stages append to them.
        self._forward_headers(kpf2, ("QUALITY_CONTROL", "RECEIPT"))

        if self.receipt is not None and not self.receipt.empty:
            kpf2.receipt = self.receipt.copy()

        # Carry the obs_id through; ORIGID (stamped at L0) rides forward on the
        # RECEIPT header above, so it is not rewritten here.
        kpf2.obs_id = self.obs_id

        kpf2.set_keyword("DATALVL", "L2")
        kpf2.receipt_add_entry("to_kpf2", "", "PASS")
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
