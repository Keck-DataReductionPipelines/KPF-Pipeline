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

import pandas as pd

from kpfpipe.data_models.base import KPFDataModel
from kpfpipe.utils.io import build_filepath
from kpfpipe.utils.kpf import get_obs_id


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
        """Masters use the WMKO DRP-RUN-05 name, not the EPRV SL# pattern.

        Defined explicitly (not merely inherited from KPFDataModel) so it wins
        over the KPF2/KPF4 SL# restore: KPFMasterModel precedes KPF2/KPF4 in every
        masters MRO (e.g. KPFMasterL2 -> KPFMasterModel -> KPF2 -> ...).
        """
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
        # Plain string (not a value/comment tuple): the in-memory PRIMARY header
        # is a dict, so a tuple would be returned verbatim by .get() and break
        # generate_standard_filename() before the FITS round-trip.
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
