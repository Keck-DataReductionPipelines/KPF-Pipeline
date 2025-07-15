import configparser
import traceback

from kpfpipe.primitives.level0 import KPF0_Primitive
from modules.spectral_extraction_gjg.src.alg import SpectralExtractionAlg
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/spectral_extraction_gjg/configs/default.cfg'

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
        self.stray_light_image = self.action.args[2]
        self.order_trace_green = self.action.args[3]
        self.order_trace_red = self.action.args[4]
        
        # Input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['spectral_extraction_gjg']
        except:
            self.config_path = DEFAULT_CFG_PATH
            
    def _perform(self):
        exit_code = 0
        try:
            spectralextraction = SpectralExtractionAlg(self.target_2D,
                                                       self.master_flat_2D,
                                                       self.stray_light_image,
                                                       self.order_trace_green,
                                                       self.order_trace_red,
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