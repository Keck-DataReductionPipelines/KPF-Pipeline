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
        self.order_trace_green = self.action.args[1]   # TODO: read in .csv order trace
        self.order_trace_red = self.action.args[2]
        
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
                                       self.order_trace_green, 
                                       self.order_trace_red,
                                       self.config_path
                                      )
            
            out_2D = straylight.estimate_stray_light()
            exit_code = 1
            return Arguments([exit_code, out_2D])
        except Exception as e:
            self.logger.error(f"StrayLight algorithm failed: {e}\n{traceback.format_exc()}")
            return Arguments([exit_code, None])        

    def _pre(self):
        pass
    
    def _post(self):
        pass