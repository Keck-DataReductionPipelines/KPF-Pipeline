"""
KPF Level 1 (assembled FFI) data model.

Assembled full frame images (FFIs) built by combining the L0 amplifier
readouts. GREEN and RED CCDs are stored in separate extensions. Also
used for FFI masters calibrations (bias, dark, flat).
"""

import importlib.resources
import logging
import os
import re

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from rvdata.core.models.definitions import BASE_RECEIPT_COLUMNS

from kpfpipe.data_models.base import KPFDataModel
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.utils.io import kpf_filename

logger = logging.getLogger(__name__)

_config_path = importlib.resources.files("kpfpipe.data_models.config")
_L1_EXTENSIONS = pd.read_csv(_config_path / "L1-extensions.csv")
_KNOWN_L1_EXTENSIONS = set(_L1_EXTENSIONS["Name"].tolist())

# EPRV-like L1 filename, but with L1 instead of the EPRV SL#: the EPRV regex
# only accepts SL2/SL3/SL4, so KPF L1 uses kpf_L1_YYYYMMDDThhmmss.fits.
_L1_FILENAME_PATTERN = re.compile(r"kpf_L1_\d{8}T\d{6}\.fits")


class KPF1(KPFDataModel):
    """
    KPF Level 1 assembled FFI data model.

    Contains the assembled GREEN_CCD and RED_CCD frames with their variance
    frames, plus pass-through extensions from L0 (CA_HK, exposure meter,
    telemetry). Construct from a FITS file with ``KPF1.from_fits(path)``, then
    read the assembled frames from ``data`` (e.g. ``data["GREEN_CCD"]``,
    ``data["RED_CCD"]``).
    """

    _DATALVL = "L1"
    _known_extensions = _KNOWN_L1_EXTENSIONS

    def __init__(self):
        super().__init__()
        self.level = 1

        for _, row in _L1_EXTENSIONS.iterrows():
            if row["Required"] and row["Name"] not in self.extensions:
                self.create_extension(row["Name"], row["DataType"])

        # Seed PRIMARY with the EPRV Required keyword skeleton (typed defaults +
        # comments). L1 is not an EPRV level, so KPF1 has no RV1 to inherit this
        # from; stamp it from the registry so KWRDPRL1 is meaningful and native
        # values (overlaid in KPF0.to_kpf1) win over defaults.
        for kw, value in self.keyword_registry.eprv_primary_seed.items():
            self.headers["PRIMARY"][kw] = value
        # DATALVL is EPRV-Required, so the seed defaults it to "UNKNOWN"; correct it
        # in-memory (to_kpf1 / to_fits set it too, but a fresh KPF1 should read L1).
        self.set_keyword("DATALVL", self._DATALVL)

    def read(self, hdul, instrument=None, overwrite=False, **kwargs):
        """Route L1 FITS reads to ``KPF1._read``.

        Needed for the same reason as the L0 override; see ``KPF0.read`` for the
        canonical rationale (``RVDataModel.read`` has no lvl==1 dispatch branch,
        so the inherited ``from_fits`` would never call into ``_read``).
        """
        self._read(hdul)

    def _read(self, hdul):
        """Read all extensions from an L1 FITS HDUList.

        Handles known extensions from the CSV definition and also
        accepts unknown extensions (with a warning).
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
                    if ext_name not in self._known_extensions:
                        logger.warning(
                            "Non-standard extension '%s' found in L1 file.", ext_name
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
                # Materialize the memmap before from_fits closes the file
                # (np.array, not asarray); see KPF0._read for the full rationale.
                self.set_data(ext_name, np.array(hdu.data))
            elif fits_type == "BinTableHDU":
                self.set_data(ext_name, Table.read(hdu))

            self.set_header(ext_name, hdu.header)

    def check_filename_convention(self, filename):
        """KPF L1 uses an EPRV-like name with L1 (not SL#): kpf_L1_YYYYMMDDThhmmss.fits.

        The EPRV regex only accepts SL2/SL3/SL4, so L1 has its own convention.
        """
        basename = os.path.basename(filename)
        if not _L1_FILENAME_PATTERN.fullmatch(basename):
            logger.warning(
                "Filename '%s' does not follow the KPF L1 naming "
                "convention (kpf_L1_YYYYMMDDThhmmss.fits)",
                basename,
            )
            return False
        return True

    def generate_standard_filename(self):
        """KPF L1 filenames follow the kpf_L1_YYYYMMDDThhmmss.fits convention.

        Raises
        ------
        ValueError
            If ``obs_id`` is unset or invalid.
        """
        return kpf_filename(self.obs_id, "L1")

    def to_fits(self, fn=None):
        """Write L1 data to a FITS file."""
        if fn is None:
            fn = self.generate_standard_filename()
        if not fn.endswith(".fits"):
            raise NameError("Filename must end with .fits")

        self.receipt_add_entry("to_fits", f"out_filepath={fn}", "PASS")

        if "PRIMARY" in self.headers:
            self.set_keyword("FILENAME", os.path.basename(fn))
            self.set_keyword("DATALVL", self._DATALVL)

        hdu_list = self._create_hdul()
        hdul = fits.HDUList(hdu_list)
        dirname = os.path.dirname(fn)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname, exist_ok=True)
        hdul.writeto(fn, overwrite=True, output_verify="silentfix")
        hdul.close()
        logger.info("wrote %s to %s", type(self).__name__, fn)
        return fn

    # Mapping of L1 extension names → KPF2/RV2 extension names for pass-through.
    # CA_HK is excluded: it is a raw 2D CCD image, not an extracted spectrum.
    # ANCILLARY_SPECTRUM (BinTableHDU) should be populated after Ca HK extraction.
    _L1_TO_L2_PASSTHROUGH = {
        "TELEMETRY": "TELEMETRY",
        "EXPMETER_SCI": "EXPMETER",
    }

    def to_kpf2(self):
        """Create a KPF2 scaffold from this L1, carrying over headers and
        pass-through extensions.

        The L1 PRIMARY is already EPRV-standard (converted in KPF0.to_kpf1), so
        the PRIMARY and INSTRUMENT_HEADER headers are a pure pass-through. Pass-
        through extensions (TELEMETRY, EXPMETER_SCI->EXPMETER) and KPF-friendly
        aliases (e.g. SCI2_FLUX -> TRACE3_FLUX) are handled below. Trace arrays
        are created empty -- the caller (spectral extraction) fills those in.
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
