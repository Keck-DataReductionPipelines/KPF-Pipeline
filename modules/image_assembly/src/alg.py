import pandas as pd

from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger
from modules.Utils.config_parser import ConfigHandler

class ImageAssemblyAlg:
    """
    Docstring
    """
    def __init__(self, 
                 data_type, 
                 target_L0,
                 default_config_path,
                 logger=None,
                 ):
        # config inputs
        self.config = ConfigClass(default_config_path)
        self.cfg_params = ConfigHandler(self.config, 'PARAM')

        if logger == None:
            self.log = start_logger('ImageAssembly', default_config_path)
        else:
            self.log = logger

        # required data inputs
        self.data_type = data_type
        self.target_L0 = target_L0

        # create self.orientation dictionary
        self._read_orientation_reference('GREEN')
        self._read_orientation_reference('RED')

    def _read_orientation_reference(self, chip):
        if not hasattr(self, 'orientation'):
            self.orientation = {}

        filepath = str(self.cfg_params.get_config_value('channel_orientation_ref_path_green'))
        with open(filepath, 'r') as f:
            self.orientation[chip] = pd.read_csv(f, delimiter=' ')