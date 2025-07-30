import configparser
import traceback

from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.models.level0 import KPF0
from modules.spectral_extraction.src.alg import SpectralExtractionAlg
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/spectral_extraction/configs/default.cfg'

class SpectralExtraction(KPF0_Primitive):
    """
    Primitive for spectral extraction module
    """
    def __init__(self, action, context):
        
        # Initialize parent class
        KPF0_Primitive.__init__(self, action, context)
        
        # Input arguments
        self.target_2D = self.action.args[0]
        self.master_flat_2D = self.action.args[1]
        self.order_trace_green = self.action.args[2]
        self.order_trace_red = self.action.args[3]
        self.start_order_green = self.action.args[4]
        self.start_order_red = self.action.args[5]
        
        # Handle master_flat_2D conversion if needed
        # CalibrationLookup returns file paths, but algorithm expects KPF0 objects
        if isinstance(self.master_flat_2D, str):
            # It's a file path, load it as KPF0 object
            self.master_flat_2D = KPF0.from_fits(self.master_flat_2D, data_type='KPF')
        
        # Input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['spectral_extraction']
        except:
            self.config_path = DEFAULT_CFG_PATH

        print(self.config_path)
            
    def _perform(self):
        exit_code = 0
        try:
            dummy = self.target_2D['GREEN_CCD'].shape

            spectralextraction = SpectralExtractionAlg(self.target_2D,
                                                       self.master_flat_2D,
                                                       self.order_trace_green,
                                                       self.order_trace_red,
                                                       self.start_order_green,
                                                       self.start_order_red,
                                                       self.config_path
                                                      )
            
            for chip in ['GREEN', 'RED']:
                spectralextraction.target_l1 = spectralextraction.extract_ccd(chip)
            exit_code = 1
            return Arguments([exit_code, spectralextraction.target_l1])
        except Exception as e:
            self.logger.error(f"SpectralExtraction algorithm failed: {e}\n{traceback.format_exc()}")
            return Arguments([exit_code, None])

    def _pre(self):
        pass
    
    def _post(self):
        pass