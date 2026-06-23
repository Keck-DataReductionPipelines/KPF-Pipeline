"""KPF Master Flat construction module."""

from kpfpipe.modules.masters.base import BaseMasterModule
from kpfpipe.utils.config import ConfigHandler


class Flat(BaseMasterModule):
    """
    Construct a master flat frame from a stack of L0 flat exposures.

    Standard reduction: a flat is bias- and dark-subtracted
    (`_STANDARD_CALIBRATIONS = ("bias", "dark")`). Not yet implemented.

    Parameters
    ----------
    l0_file_list : list of str
        Sorted list of L0 FITS file paths to stack.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: stack_sigma,
        exptime_tolerance, chips.
    """

    _STANDARD_CALIBRATIONS = ("bias", "dark")

    def __init__(self, l0_file_list, config=None):
        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(
                [
                    "DATA_DIRS",
                    "KPFPIPE",
                    "FLAT",
                    "MODULE_CALIBRATION_ASSOCIATION",
                    "MODULE_IMAGE_PROCESSING",
                ]
            )
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")
        super().__init__(l0_file_list, params)

    def info(self):
        """Print a summary of the module configuration (not yet implemented)."""
        print("Flat")
        print("  l0_file_list:")
        for fn in self.l0_file_list:
            print(f"    {fn}")
        print(f"  chips:  {self.chips}")
        print("  make_master_l1() is not yet implemented for Flat")
