"""
KPF Level 1 (assembled 2D frame) data model.

Represents an assembled CCD frame after combining amplifier readouts.
Also used for master calibration products (bias, dark, flat) which
share the same structure (mean + variance frames for GREEN and RED).

Subclasses RVDataModel (via KPFDataModel) to reuse its extension/header/data
infrastructure and receipt system.
"""

import datetime
import importlib.resources
import os
import re
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from kpfpipe.data_models.base import KPFDataModel
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.utils.kpf import get_obs_id

_config_path = importlib.resources.files("kpfpipe.data_models.config")
_L1_EXTENSIONS = pd.read_csv(_config_path / "L1-extensions.csv")
_KNOWN_L1_EXTENSIONS = set(_L1_EXTENSIONS["Name"].tolist())

# EPRV-like L1 filename, but with L1 instead of the standard SL#: the EPRV regex
# only accepts SL2/SL3/SL4, so KPF L1 uses kpf_L1_YYYYMMDDThhmmss.fits.
_L1_FILENAME_PATTERN = re.compile(r"kpf_L1_\d{8}T\d{6}\.fits")


class KPF1(KPFDataModel):
    """
    KPF Level 1 assembled 2D frame data model.

    After image assembly, the L1 product contains assembled GREEN_CCD
    and RED_CCD frames with corresponding variance frames, plus
    pass-through extensions from L0 (CA_HK, exposure meter, telemetry).

    Also used for master calibration products (bias, dark, flat).
    Construct from a FITS file with `KPF1.from_fits(path)`, then read the
    assembled 4080x4080 frames from `data` (e.g. `data["GREEN_CCD"]`,
    `data["RED_CCD"]`).
    """

    _DATALVL = "L1"
    _FILENAME_PREFIX = "kpf_L1"
    _known_extensions = _KNOWN_L1_EXTENSIONS

    def __init__(self):
        super().__init__()
        self.level = 1

        for _, row in _L1_EXTENSIONS.iterrows():
            if row["Required"] and row["Name"] not in self.extensions:
                self.create_extension(row["Name"], row["DataType"])

        # Seed PRIMARY with the EPRV Required keyword skeleton (typed defaults +
        # comments), mirroring how rvdata's RV2.__init__ seeds KPF2. L1 is not an
        # EPRV level, so KPF1 has no RV1 to inherit this from; we stamp it from the
        # registry's eprv_primary_seed (the single source of truth) so KWRDPRL1 is
        # meaningful and native values (overlaid in KPF0.to_kpf1) win over defaults.
        for kw, value in self.keyword_registry.eprv_primary_seed.items():
            self.headers["PRIMARY"][kw] = value
        # DATALVL is EPRV-Required, so the seed defaults it to "UNKNOWN"; correct it
        # in-memory (to_kpf1 / to_fits set it too, but a fresh KPF1 should read L1).
        self.set_keyword("DATALVL", self._DATALVL)

    def read(self, hdul, instrument=None, overwrite=False, **kwargs):
        """
        Route L1 FITS reads to `KPF1._read`.

        `RVDataModel.read` has no lvl==1 dispatch branch, so the inherited
        `from_fits` would never call into `_read` without this override.
        """
        self._read(hdul)
        if self.filename is not None:
            try:
                self.obs_id = get_obs_id(self.filename)
            except ValueError:
                pass

    def _read(self, hdul):
        """
        Read all extensions from an L1 FITS HDUList.

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
                        warnings.warn(
                            f"Non-standard extension '{ext_name}' found in L1 file.",
                            UserWarning,
                            stacklevel=2,
                        )
                    self.create_extension(ext_name, fits_type)

            if ext_name == "PRIMARY":
                pass
            elif ext_name == "RECEIPT":
                t = Table.read(hdu)
                df = t.to_pandas()
                receipt_columns = [
                    "Time",
                    "Code_Release",
                    "Commit_Hash",
                    "Branch_Name",
                    "Module_Name",
                    "Status",
                ]
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
        """KPF L1 uses an EPRV-like name with L1 (not SL#): kpf_L1_YYYYMMDDThhmmss.fits.

        The EPRV regex only accepts SL2/SL3/SL4, so L1 has its own convention.
        """
        basename = os.path.basename(filename)
        if not _L1_FILENAME_PATTERN.fullmatch(basename):
            warnings.warn(
                f"Filename '{basename}' does not follow the KPF L1 naming "
                "convention (kpf_L1_YYYYMMDDThhmmss.fits)",
                stacklevel=2,
            )
            return False
        return True

    def generate_standard_filename(self):
        """
        KPF L1 filenames follow kpf_L1_YYYYMMDDThhmmss.fits convention.

        Uses DATE-OBS from the PRIMARY header.
        """
        if "PRIMARY" in self.headers:
            date_obs = self.headers["PRIMARY"].get("DATE-OBS")
            if date_obs is not None:
                date_str = str(date_obs).split(".")[0]
                try:
                    dt = datetime.datetime.fromisoformat(date_str)
                    datetime_str = dt.strftime("%Y%m%dT%H%M%S")
                    return f"{self._FILENAME_PREFIX}_{datetime_str}.fits"
                except ValueError:
                    pass
        raise ValueError("Cannot generate filename: DATE-OBS not available")

    def to_fits(self, fn=None):
        """Write L1 data to a FITS file."""
        if fn is None:
            fn = self.generate_standard_filename()
        if not fn.endswith(".fits"):
            raise NameError("Filename must end with .fits")

        self.receipt_add_entry("to_fits", "PASS")

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
        return fn

    # Mapping of L1 extension names → KPF2/RV2 extension names for pass-through.
    # CA_HK is excluded: it is a raw 2D CCD image, not an extracted spectrum.
    # ANCILLARY_SPECTRUM (BinTableHDU) should be populated after Ca HK extraction.
    _L1_TO_KPF2_PASSTHROUGH = {
        "TELEMETRY": "TELEMETRY",
        "EXPMETER_SCI": "EXPMETER",
    }

    def to_kpf2(self):
        """
        Create a KPF2 scaffold from this L1, carrying over headers and
        pass-through extensions.

        The L1 PRIMARY is already EPRV-standard (converted upstream in
        KPF0.to_kpf1), so headers are a pure pass-through: the EPRV PRIMARY and
        the immutable INSTRUMENT_HEADER are forwarded unchanged (value + comment).
        Header validation no longer runs here — it moved to the checkpoints
        layer (quality_control/checkpoints, Checkpoint.unregistered_keywords).
        Pass-through extensions (TELEMETRY,
        EXPMETER_SCI→EXPMETER, CA_HK→ANCILLARY_SPECTRUM) and KPF-friendly
        aliases (e.g., SCI2_FLUX → TRACE3_FLUX) are handled below. Trace data
        arrays are created but empty — the caller (spectral extraction) fills
        those in.
        """
        kpf2 = KPF2()

        # Headers are a pure pass-through; the native→EPRV conversion and the
        # INSTRUMENT_HEADER snapshot were done once in KPF0.to_kpf1. Forward
        # PRIMARY + INSTRUMENT_HEADER (and QC/RECEIPT below) card-by-card,
        # preserving comments; PRIMARY overlays onto kpf2's EPRV seed (native
        # wins), INSTRUMENT_HEADER stays a verbatim copy.
        self._forward_headers(kpf2, ("PRIMARY", "INSTRUMENT_HEADER"))

        # Pass-through extensions with renaming
        for l1_ext, kpf2_ext in self._L1_TO_KPF2_PASSTHROUGH.items():
            if l1_ext in self.extensions:
                l1_type = self.extensions[l1_ext]
                if kpf2_ext not in kpf2.extensions:
                    kpf2.create_extension(kpf2_ext, l1_type)
                elif kpf2.extensions[kpf2_ext] != l1_type:
                    # Update type to match actual L1 data
                    kpf2.extensions[kpf2_ext] = l1_type
                if l1_ext in self.data and self.data[l1_ext] is not None:
                    kpf2.set_data(kpf2_ext, self.data[l1_ext])
                if l1_ext in self.headers:
                    kpf2.set_header(kpf2_ext, self.headers[l1_ext])

        # Forward the L1 QUALITY_CONTROL and RECEIPT *headers* onto L2, with
        # comments. The receipt *table* propagates via the receipt copy below, but
        # the RECEIPT header cards (OSCANSUB / *FILE applied by ImageAssembly /
        # CalibrationAssociation) and the QUALITY_CONTROL cards (RN*, *AGE, QC
        # booleans) are separate and must be carried explicitly; downstream L2
        # stages (BarycentricCorrection, DiagL2, QCL2) append to them.
        self._forward_headers(kpf2, ("QUALITY_CONTROL", "RECEIPT"))

        # Carry forward receipt
        if self.receipt is not None and not self.receipt.empty:
            kpf2.receipt = self.receipt.copy()

        # Carry obs_id through, both as the model attribute and (for traceability
        # on the product itself) the ORIGID keyword, registered to RECEIPT.
        kpf2.obs_id = self.obs_id
        if self.obs_id is not None:
            kpf2.set_keyword("ORIGID", self.obs_id)

        kpf2.set_keyword("DATALVL", "L2")
        kpf2.receipt_add_entry("to_kpf2", "PASS")
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
