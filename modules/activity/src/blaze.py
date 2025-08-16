import configparser
import traceback

from kpfpipe.primitives.level1 import KPF1_Primitive
from modules.activity.src.alg import ActivityAlg
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/activity/configs/default.cfg'

class ComputeActivityIndices(KPF1_Primitive):
    """
    Add docstring
    """
    def __init__(self, action, context):
        
        # Initialize parent class
        KPF1_Primitive.__init__(self, action, context)
        
        # Input arguments
        self.target_l1 = self.action.args[0]           # KPF L1 object
        #self.method = self.action.args[2]             # string of method to use
        
        # Input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['activity']
        except:
            self.config_path = DEFAULT_CFG_PATH
            
    def _perform(self):
        exit_code = 0
        try:
            activity = ActivityAlg(self.target_l1, self.config_path)
            ## Add code here
            
            exit_code = 1
            return Arguments([exit_code, out_l1])
        except Exception as e:
            self.logger.error(f"Activity algorithm failed: {e}\n{traceback.format_exc()}")
            return Arguments([exit_code, None])        

    def _pre(self):
        pass
    
    def _post(self):
        pass