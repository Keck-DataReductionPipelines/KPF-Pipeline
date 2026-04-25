"""
KPF Master Wavelength Solution construction module.
"""
from kpfpipe.modules.masters.base import BaseMasterModule
from kpfpipe.modules.calibration_association import CalibrationAssociation
from kpfpipe.modules.image_processing import ImageProcessing
from kpfpipe.modules.spectral_extraction import SpectralExtraction
from kpfpipe.utils.config import ConfigHandler


class WLS(BaseMasterModule):
    """
    Construct a master wavelength solution from a stack of ThAr L0 exposures.

    Each frame is individually processed through the full L0→L2 pipeline
    (image assembly, bias subtraction, spectral extraction) before the
    extracted spectra are combined into a wavelength solution.

    Parameters
    ----------
    l0_file_list : list of str
        Sorted list of L0 FITS file paths to process.
    config : None | dict | ConfigHandler
        Module configuration.
    """
    def __init__(self, l0_file_list, config=None):
        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["DATA_DIRS", "KPFPIPE", "WLS"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")
        super().__init__(l0_file_list, params)
        self._data_root = params.get('KPF_DATA_INPUT')

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_frame(self, fn, ncache=0, exptime_tolerance=None):
        # load and assemble a single raw image frame from L0 --> L1
        return super()._load_frame(fn, ncache=ncache, exptime_tolerance=exptime_tolerance)

    def _extract_frame(self, l1_obj):
        # process and extract single image frame from L1 --> L2
        calibration_association = CalibrationAssociation(l1_obj, {'KPF_DATA_INPUT': self._data_root})
        l1_obj = calibration_association.perform(['bias'])

        image_processing = ImageProcessing(l1_obj)
        l1_obj = image_processing.perform()

        spectral_extraction = SpectralExtraction(l1_obj)
        l2_obj = spectral_extraction.perform()

        if not hasattr(self, 'thar_l2_list'):
            self.thar_l2_list = []
        self.thar_l2_list.append(l2_obj)

        return l2_obj

    # ------------------------------------------------------------------
    # Algorithm steps
    # ------------------------------------------------------------------

    def process_stack_l0_to_l2(self, l0_file_list=None):
        """
        Process each ThAr L0 frame through the full L0→L2 pipeline.

        Parameters
        ----------
        l0_file_list : list of str, optional
            L0 files to process. Defaults to self.l0_file_list.

        Returns
        -------
        list of KPF2
            Extracted L2 objects for all successfully processed frames.

        Raises
        ------
        ValueError
            If more than 20% of frames fail to load.
        """
        if l0_file_list is None:
            l0_file_list = self.l0_file_list

        failure = 0

        for fn in l0_file_list:
            l1_obj, success = self._load_frame(fn, ncache=0)

            if not success:
                failure += 1
                if failure / len(l0_file_list) > 0.2:
                    raise ValueError(f"more than 20% of frames in stack failed to load")
                continue

            l2_obj = self._extract_frame(l1_obj)

        return self.thar_l2_list if hasattr(self, 'thar_l2_list') else []
