import warnings

from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger

from modules.Utils.config_parser import ConfigHandler

class ActvityAlg:
    """
    Add docstring
    """
    def __init__(self, input_l1, default_config_path, logger=None):
        # Input arguments
        self.config = ConfigClass(default_config_path)
        if logger == None:
            self.log = start_logger('Activity', default_config_path)
        else:
            self.log = logger
            
        cfg_params = ConfigHandler(self.config, 'PARAM')
        
        self.L1 = input_l1
                
        self.add_extensions()
        
    def add_keywords(self):
        header = self.target_l1.header['PRIMARY']
        # Example of adding a keyword
        #header['BLAZCORR'] = 1
        
    # Example of adding an extension; this should be modified based on a new 
    # data model for the ACTIVITY extension
    def add_extensions(self):
        self.target_l1.create_extension('GREEN_SCI_BLAZE1', np.array)
        
    # Modify this method
    def compute_activity(self):
        try:
            activity_method = self.__getattribute__(self.method)
        except AttributeError:
            self.log.error(f'Activity method {self.method} not implemented.')
            raise(AttributeError)
            
        out_l1 = activity_method()        
        self.add_keywords()
        
        return out_l1
    
    
    ##### Add methods here to compute activity indices and store them as 
    ##### attributes, perhaps as a Pandas dataframe self.activity_df.
    ##### This dataframe can then later be converted to a FITS table 
    ##### by the data model.