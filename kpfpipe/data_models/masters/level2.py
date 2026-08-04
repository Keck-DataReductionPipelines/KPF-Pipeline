"""
KPF Masters Level 2 data model.

Extracted-spectrum-level calibration (wavelength solution, flat). Extends
KPF2 with a per-type extension set: a WLS master (kind="wls") holds wavelength
solutions, while a flat master (kind="flat") holds extracted spectra.
"""

import importlib.resources
import logging
import os

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from rvdata.core.models.definitions import BASE_RECEIPT_COLUMNS

from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.masters.base import KPFMasterModel

logger = logging.getLogger(__name__)

_config_path = importlib.resources.files("kpfpipe.data_models.config")
_ML2_WLS_EXTENSIONS = pd.read_csv(_config_path / "ML2-wls-extensions.csv")
_ML2_FLAT_EXTENSIONS = pd.read_csv(_config_path / "ML2-flat-extensions.csv")
# Per-type authoritative extension manifests, keyed by the ``kind`` selector.
_ML2_MANIFESTS = {"wls": _ML2_WLS_EXTENSIONS, "flat": _ML2_FLAT_EXTENSIONS}


class KPFMasterL2(KPFMasterModel, KPF2):
    """
    KPF Masters Level 2 extracted-spectrum calibration.

    Thin wrapper around KPF2 with masters-specific extension names.
    Inherits the full L2 schema and the KPF2 alias system (e.g.
    SCI2_WAVE -> TRACE3_WAVE, with chip-prefix access GREEN_SCI2_WAVE /
    RED_SCI2_WAVE).

    Construct with an explicit type, e.g. ``KPFMasterL2(kind="wls")``, or
    load a product from disk with ``KPFMasterL2.from_fits(path)``, which
    infers the kind from header keyword MASTYPE.
    """

    _DATALVL = "ML2"
    # MASTYPE (WMKO token) -> schema kind, for from_fits. L2 masters are the WLS
    # (thar) master and the flat master.
    _MASTYPE_TO_KIND = {"thar": "wls", "flat": "flat"}
    # Allowed on read regardless of kind: the union of both per-type manifests plus
    # the inherited KPF2 schema, so _read never warns about a "non-standard"
    # extension when loading either master type.
    _known_extensions = (
        set(_ML2_WLS_EXTENSIONS["Name"])
        | set(_ML2_FLAT_EXTENSIONS["Name"])
        | set(KPF2().extensions.keys())
    )

    def __init__(self, kind):
        if kind not in _ML2_MANIFESTS:
            raise ValueError(
                f"KPFMasterL2 kind must be one of {sorted(_ML2_MANIFESTS)}, got "
                f"{kind!r}"
            )
        self.kind = kind
        KPF2.__init__(self)
        # KPF2.__init__ seeds the EPRV L2 science PRIMARY skeleton. Masters are out
        # of EPRV scope and carry their own minimal PRIMARY, so drop it and stamp
        # DATALVL ("ML2") ourselves (rvdata's to_fits never re-stamps DATALVL).
        self.headers["PRIMARY"].clear()
        self.level = 2

        # ML2-{kind}-extensions.csv is the authoritative manifest for this master
        # type. KPF2.__init__ builds the full science L2 schema (needed for the
        # alias system); drop anything the manifest omits, then create any Required
        # row not present. *_WLS_COEFFS are Required=False, created per-chip on
        # demand in make_master_l2.
        manifest_df = _ML2_MANIFESTS[kind]
        manifest = set(manifest_df["Name"])
        for ext in list(self.extensions):
            if ext not in manifest:
                self.del_extension(ext)
        for _, row in manifest_df.iterrows():
            if row["Required"] and row["Name"] not in self.extensions:
                self.create_extension(row["Name"], row["DataType"])

        self.set_keyword("DATALVL", self._DATALVL)

    @classmethod
    def from_fits(cls, fn, instrument=None, **kwargs):
        """Load an L2 master, inferring ``kind`` from the file's PRIMARY MASTYPE.

        rvdata's RVDataModel.from_fits builds the instance via ``cls()`` with no
        args, which a required ``kind`` rejects. This override reads MASTYPE
        (always set by ``set_input_files`` on a written master), maps it to a
        schema kind, and constructs explicitly. Mirrors the base from_fits flow
        (the base also computes an unused MD5 digest, which is omitted here).

        Raises
        ------
        OSError
            If ``fn`` does not exist or is not a ``.fits``/``.fit`` file.
        ValueError
            If the PRIMARY MASTYPE is missing or maps to no known schema kind.
        """
        if not os.path.isfile(fn):
            raise OSError(f"{fn} does not exist.")
        if not fn.endswith(".fits") and not fn.endswith(".fit"):
            raise OSError("input files must be FITS files")
        # This override does not go through KPFDataModel.from_fits (it builds
        # the instance itself), so it logs its own read record (DRP-RUN-08).
        logger.info("reading %s from %s", cls.__name__, fn)
        with fits.open(fn) as hdul:
            mastype = hdul["PRIMARY"].header.get("MASTYPE")
            kind = cls._MASTYPE_TO_KIND.get(str(mastype).lower()) if mastype else None
            if kind is None:
                raise ValueError(
                    f"cannot infer KPFMasterL2 kind: PRIMARY MASTYPE {mastype!r} is "
                    f"not one of {sorted(cls._MASTYPE_TO_KIND)}"
                )
            obj = cls(kind)
            obj.filename = os.path.basename(fn)
            obj.dirname = os.path.dirname(fn)
            obj.read(hdul, instrument, **kwargs)
        obj.receipt_add_entry("from_fits", f"fn={fn}, instrument={instrument}", "PASS")
        return obj

    def read(self, hdul, instrument=None, overwrite=False, **kwargs):
        """Route Masters L2 FITS reads to ``KPFMasterL2._read``.

        Same override pattern as ``KPF0.read`` (see there for the canonical
        rationale), but for a distinct reason: ``RVDataModel.read`` *does*
        dispatch lvl==2, to ``RV2._read``, which only knows the EPRV standard L2
        extensions and would KeyError on masters-specific ones (INPUT_FILES,
        ``*_WLS_COEFFS``).
        """
        self._read(hdul)

    def _read(self, hdul):
        """Read all extensions from a Masters L2 FITS HDUList.

        Handles known extensions and accepts unknown extensions (with a
        warning). Mirrors ``KPF1._read`` in style and structure.
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
                            "Non-standard extension '%s' found in L2 file.", ext_name
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
