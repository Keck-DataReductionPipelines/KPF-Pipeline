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

    def _info_text(self):
        """Build the info() report text."""
        lines = []
        lines.append("Flat")
        lines.append("  l0_file_list:")
        for fn in self.l0_file_list:
            lines.append(f"    {fn}")
        lines.append(f"  chips:  {self.chips}")
        lines.append("  make_master_l1() is not yet implemented for Flat")
        return "\n".join(lines)

    def info(self):
        """Print a summary of the module configuration (not yet implemented)."""
        print(self._info_text())
