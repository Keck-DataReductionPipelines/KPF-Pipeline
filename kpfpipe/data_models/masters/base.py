"""
KPF Masters base data model.

Shared base for the masters calibration products. Extends KPFDataModel with
the masters' own keyword/extension tables (ML1, ML2-flat, ML2-wls), so a
master builds its extensions and seeds its PRIMARY by exactly the same
mechanism every science level uses -- off its own manifest.

Masters products differ from science products in extension naming to
avoid confusion (see KPFMasterL1/L2/L4)
"""

import logging
import os
import re

import pandas as pd

from kpfpipe.data_models.base import KPFDataModel
from kpfpipe.utils.io import kpf_filename
from kpfpipe.utils.kpf import get_obs_id

logger = logging.getLogger(__name__)

# WMKO DRP-RUN-05 master name: {KOAID}_master_{type}_L{N}.fits, where KOAID is a
# KP.YYYYMMDD.NNNNN.NN obs_id, type is bias/dark/flat/thar, and N is 1/2/4.
_MASTER_FILENAME_PATTERN = re.compile(
    r"KP\.\d{8}\.\d{5}\.\d{2}_master_(bias|dark|flat|thar)_L[124]\.fits"
)


class KPFMasterModel(KPFDataModel):
    """
    Base class for KPF masters calibration data models.

    Inherits from KPFDataModel and chains through the science level's
    ``__init__`` like any other model; what makes it a master is the data model,
    which redirects the manifest and the PRIMARY seed to the ML tables.
    """

    @property
    def _data_model(self):
        """This master's keyword/extension data model: ``ML{level}``.

        Masters share levels 1 and 2 with the science chain, so the level alone
        does not name their tables; the ML prefix is what separates them.
        """
        return f"ML{self.level}"

    def check_filename_convention(self, filename):
        """Masters use the WMKO DRP-RUN-05 name: {KOAID}_master_{type}_L{N}.fits.

        Defined on KPFMasterModel (which precedes KPF1/KPF2/KPF4 in every masters
        MRO) so the master convention wins over the per-level science checks.
        """
        basename = os.path.basename(filename)
        if not _MASTER_FILENAME_PATTERN.fullmatch(basename):
            logger.warning(
                "Filename '%s' does not follow the KPF masters naming "
                "convention ({KOAID}_master_{bias,dark,flat,thar}_L{1,2,4}.fits)",
                basename,
            )
            return False
        return True

    def set_input_files(self, file_list, master_type):
        """
        Record the stacked input L0 files and the master calibration type.

        ``master_type`` is the WMKO filename token ('bias', 'dark', 'flat', or
        'thar'); it is stored in the PRIMARY ``MASTYPE`` header so that
        ``generate_standard_filename()`` can always build a DRP-RUN-05-compliant
        name, including after a ``from_fits()`` round-trip.
        """
        self.set_data("INPUT_FILES", pd.DataFrame({"FILENAME": file_list}))
        # MASTYPE is out of EPRV scope but registered so it routes through
        # set_keyword (-> PRIMARY) like every other KPF keyword.
        self.set_keyword("MASTYPE", master_type)

    def generate_standard_filename(self):
        """
        Return the WMKO DRP-RUN-05 master filename for this product:
        ``{KOAID-of-first-input}_master_{type}_L{level}.fits``.

        The KOAID is read from the first recorded input file and the type from
        the PRIMARY ``MASTYPE`` header (both set by ``set_input_files()``). Raises
        if either is missing, so a non-compliant master name can never be
        produced; ``kpf_filename`` validates the type token and level.
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
        return kpf_filename(koaid, f"L{self.level}", master=master_type)
