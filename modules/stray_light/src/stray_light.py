import configparser
import traceback

from kpfpipe.primitives.level0 import KPF0_Primitive
from modules.stray_light.src.alg import StrayLightAlg
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/stray_light/configs/default.cfg'

class EstimateStrayLight(KPF0Primitive):
    """
    Docstring
    """
    def __init__(self, action, context):
        
        # Initialize parent class
        KPF0_Primitive.__init__(self, action, context)
        
        # Input arguments
        self.target_2D = self.action.args[0]           # KPF 2D object
        self.masters_order_mask = self.action.args[1]
        
        # Input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['stray_light']
        except:
            self.config_path = DEFAULT_CFG_PATH
            
    def _perform(self):
        exit_code = 0
        try:
            straylight = StrayLightAlg(self.target_2D, 
                                       self.masters_order_mask,
                                       self.config_path
                                      )
            
            stray_light_image, inter_order_mask = straylight.estimate_stray_light()
            exit_code = 1
            return Arguments([exit_code, stray_light_image, inter_order_mask])
        except Exception as e:
            self.logger.error(f"StrayLight algorithm failed: {e}\n{traceback.format_exc()}")
            return Arguments([exit_code, None, None])

    def _pre(self):
        pass
    
    def _post(self):
        pass