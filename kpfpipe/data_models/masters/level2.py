"""
KPF Masters Level 2 data model.

Extracted-spectrum-level calibration (wavelength solution, flat). Extends
KPF2 with a per-type extension set: a WLS master (kind="wls") holds wavelength
solutions, while a flat master (kind="flat") holds extracted spectra.
"""

import logging
import os

from astropy.io import fits

from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.masters.base import KPFMasterModel

logger = logging.getLogger(__name__)


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

    # MASTYPE (WMKO token) -> schema kind, for from_fits. L2 masters are the WLS
    # (thar) master and the flat master.
    _MASTYPE_TO_KIND = {"thar": "wls", "flat": "flat"}
    # The manifest suffixes that select an L2 master's extension set.
    _KINDS = ("flat", "wls")
    # A master read is lenient: an unknown extension warns rather than raises.
    _strict_read = False

    def __init__(self, kind):
        if kind not in self._KINDS:
            raise ValueError(
                f"KPFMasterL2 kind must be one of {list(self._KINDS)}, got {kind!r}"
            )
        # Set before KPF2.__init__, which reaches _manifest: the data model that
        # picks this master's manifest and PRIMARY seed is ML2-{kind}.
        self.kind = kind
        super().__init__()
        # rvdata's to_fits never re-stamps DATALVL, so stamp the master value
        # over the "L2" KPF2.__init__ wrote. A header value, not a data model.
        self.set_keyword("DATALVL", "ML2")

    @property
    def _data_model(self):
        """``ML2-flat`` or ``ML2-wls``: an L2 master's shape depends on its kind."""
        return f"ML{self.level}-{self.kind}"

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
