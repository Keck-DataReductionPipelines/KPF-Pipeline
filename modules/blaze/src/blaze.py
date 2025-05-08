import configparser
import traceback

from kpfpipe.primitives.level1 import KPF1_Primitive
from modules.blaze.src.alg import BlazeAlg
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/blaze/configs/default.cfg'

class AddBlaze(KPF1_Primitive):
    """Docstring
    
    """
    def __init__(self, action, context):
        
        # Initialize parent class
        KPF1_Primitive.__init__(self, action, context)
        
        # Input arguments
        self.target_l1 = self.action.args[0]           # KPF L1 object
        self.smooth_lamp_l1 = self.action.args[1]      # KPF L1 object
        #self.method = self.action.args[2]              # string of method to use
        
        # Input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['blaze']
        except:
            self.config_path = DEFAULT_CFG_PATH
            
    def _perform(self):
        exit_code = 0
        try:
            blaze = BlazeAlg(self.target_l1, self.smooth_lamp_l1, self.config_path)
            out_l1 = blaze.apply_blaze_correction()
            exit_code = 1
        except Exception as e:
            self.logger.error(f"Blaze algorithm failed: {e}\n{traceback.format_exc()}")
        
        return Arguments(out_l1)
    
    def _pre(self):
        pass
    
    def _post(self):
        pass