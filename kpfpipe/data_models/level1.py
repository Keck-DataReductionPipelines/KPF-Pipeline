"""
KPF Level 1 (assembled FFI) data model.

Assembled full frame images (FFIs) built by combining the L0 amplifier
readouts. GREEN and RED CCDs are stored in separate extensions. Also
used for FFI masters calibrations (bias, dark, flat).
"""

import logging

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
        # L1 is not an EPRV level, but the PRIMARY skeleton is still stamped at
        # construction; the values standardize_header_format wrote at L0 are
        # forwarded over it by KPF0.to_kpf1.
        super().__init__()
        self.level = 1
        self._build()

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

        KPF-friendly aliases (e.g. SCI2_FLUX -> TRACE3_FLUX) apply automatically
        via the L2 alias system. Trace arrays are created empty -- the caller
        (spectral extraction) fills those in.
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
        # PRIMARY header above, so it is not rewritten here.
        kpf2.obs_id = self.obs_id

        kpf2.set_keyword("DATALVL", "L2")
        kpf2.receipt_add_entry("to_kpf2", "", "PASS")
        return kpf2
