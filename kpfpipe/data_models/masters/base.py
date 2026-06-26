"""
KPF masters base data model.

Base class for all masters calibration data models. Inherits from
KPFDataModel and initializes the data model infrastructure without
creating any science-level extensions.

Level-specific masters classes use double inheritance (e.g. KPFMasterL1
subclasses both KPFMasterModel and KPF1). This gives the level-specific
class access to science model methods (from_fits, to_fits, _read, info,
etc.) while the extension setup is controlled entirely by KPFMasterModel
and its subclasses.

Masters products differ from science products in extension naming to
avoid confusion: units and normalization conventions differ by
calibration type (bias, dark, flat) and are not the same as raw
science counts (e.g., GREEN_CCD).

Filename convention (WMKO DRP-RUN-05): masters are written as
``{KOAID-of-first-input}_master_{type}_L{level}.fits`` (e.g.
``KP.20240405.49597.71_master_bias_L1.fits``), built by
``generate_standard_filename()``.
"""

import os
import re
import warnings

import pandas as pd

from kpfpipe.data_models.base import KPFDataModel
from kpfpipe.utils.io import build_filepath
from kpfpipe.utils.kpf import get_obs_id

# WMKO DRP-RUN-05 master name: {KOAID}_master_{type}_L{N}.fits, where KOAID is a
# KP.YYYYMMDD.NNNNN.NN obs_id, type is bias/dark/flat/thar, and N is 1/2/4.
_MASTER_FILENAME_PATTERN = re.compile(
    r"KP\.\d{8}\.\d{5}\.\d{2}_master_(bias|dark|flat|thar)_L[124]\.fits"
)


class KPFMasterModel(KPFDataModel):
    """
    Base class for KPF masters calibration data models.

    Inherits from KPFDataModel and initializes only the base data model
    infrastructure. Science-level extension setup is intentionally skipped
    so that level-specific subclasses can install masters extensions instead.
    Normalization conventions differ by calibration type (bias, dark, flat).
    """

    def __init__(self):
        KPFDataModel.__init__(self)

    def check_filename_convention(self, filename):
        """Masters use the WMKO DRP-RUN-05 name: {KOAID}_master_{type}_L{N}.fits.

        Defined on KPFMasterModel (which precedes KPF1/KPF2/KPF4 in every masters
        MRO) so the master convention wins over the per-level science checks.
        """
        basename = os.path.basename(filename)
        if not _MASTER_FILENAME_PATTERN.fullmatch(basename):
            warnings.warn(
                f"Filename '{basename}' does not follow the KPF masters naming "
                "convention ({KOAID}_master_{bias,dark,flat,thar}_L{1,2,4}.fits)",
                stacklevel=2,
            )
            return False
        return True

    def set_input_files(self, file_list, master_type):
        """
        Record the stacked input L0 files and the master calibration type.

        `master_type` is the WMKO filename token ('bias', 'dark', 'flat', or
        'thar'); it is stored in the PRIMARY ``MASTYPE`` header so that
        `generate_standard_filename()` can always build a DRP-RUN-05-compliant
        name, including after a `from_fits()` round-trip.
        """
        self.set_data("INPUT_FILES", pd.DataFrame({"FILENAME": file_list}))
        # Written directly (not via set_keyword): masters keep MASTYPE on their own
        # PRIMARY, out of EPRV/registry scope. The header is a fits.Header, so a
        # bare string reads back as a scalar via .get() in
        # generate_standard_filename() (a value/comment tuple would too, but the
        # bare form keeps this out-of-scope card simple).
        self.headers["PRIMARY"]["MASTYPE"] = master_type

    def generate_standard_filename(self):
        """
        Return the WMKO DRP-RUN-05 master filename for this product:
        ``{KOAID-of-first-input}_master_{type}_L{level}.fits``.

        The KOAID is read from the first recorded input file and the type from
        the PRIMARY ``MASTYPE`` header (both set by `set_input_files()`). Raises
        if either is missing, so a non-compliant master name can never be
        produced; `build_filepath` validates the type token and level.
        """
        master_type = self.headers["PRIMARY"].get("MASTYPE")
        if not master_type:
            raise ValueError(
                "MASTYPE header is unset; call set_input_files() before "
                "generating a master filename"
            )
        input_files = self.data.get("INPUT_FILES")
        if input_files is None or len(input_files) == 0:
            raise ValueError(
                "INPUT_FILES is empty; cannot determine the first-input KOAID"
            )
        koaid = get_obs_id(str(input_files["FILENAME"][0]))
        return build_filepath(koaid, f"L{self.level}", master=master_type)
