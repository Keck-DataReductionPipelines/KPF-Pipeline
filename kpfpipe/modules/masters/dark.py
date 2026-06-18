"""KPF Master Dark construction module."""

from kpfpipe.modules.masters.base import BaseMasterModule
from kpfpipe.utils.config import ConfigHandler


class Dark(BaseMasterModule):
    """
    Construct a master dark frame from a stack of L0 dark exposures.

    Parameters
    ----------
    l0_file_list : list of str
        Sorted list of L0 FITS file paths to stack.
    config : None | dict | ConfigHandler
        Module configuration. Recognized keys: nframe_stream, stack_sigma,
        exptime_tolerance, chips.
    """

    def __init__(self, l0_file_list, config=None):
        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["DATA_DIRS", "KPFPIPE", "DARK"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")
        super().__init__(l0_file_list, params)

    def info(self):
        """Print a summary of the module configuration (not yet implemented)."""
        print("Dark")
        print("  l0_file_list:")
        for fn in self.l0_file_list:
            print(f"    {fn}")
        print(f"  chips:  {self.chips}")
        print("  make_master_l1() is not yet implemented for Dark")
