import pandas as pd

class ImageAssemblyAlg:
    """
    Docstring
    """
    def __init__(self, 
                 data_type, 
                 target_L0
                 ):
        # config inputs
        self.config = ConfigClass(default_config_path)
        if logger == None:
            self.log = start_logger('ImageAssembly', default_config_path)
        else:
            self.log = logger

        # required data inputs
        self.data_type = data_type
        self.target_L0 = target_L0

        # parse config
        cfg_params = ConfigHandler(self.config, 'PARAM')

        self.orientation = {}
        self._read_orientation_reference('GREEN')
        self._read_orientation_reference('RED')

        
        def _read_orientation_reference(self, chip):
            if not hasattr(self, 'orientation'):
                self.orientation = {}

            filepath = str(cfg_params.get_config_value('channel_orientation_ref_path_green'))
            with open(filepath, 'r') as f:
                self.orientation[chip] = pd.read_csv(f, delimiter=' ')