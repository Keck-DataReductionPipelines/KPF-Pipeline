import configparser
import traceback

from kpfpipe.primitives.level0 import KPF0_Primitive
from modules.image_assembly.src.alg import ImageAssemblyAlg
from keckdrpframework.models.arguments import Arguments

# Global read-only variables 
DEFAULT_CFG_PATH = 'modules/image_assembly/configs/default.cfg'

class ImageAssembly(KPF0_Primitive):
    """
    Primitive for image assembly module
    """
    def __init__(self, action, context):
        
        # Initialize parent class
        KPF0_Primitive.__init__(self, action, context)
        
        # Input arguments
        self.target_l0 = self.action.args[0]
        
        # Input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['image_assembly']
        except:
            self.config_path = DEFAULT_CFG_PATH

        print(self.config_path)

    def _perform(self):
        exit_code = 0
        try:
            imageassembly = ImageAssemblyAlg(self.target_l0,
                                             self.config_path
                                            )
            
            for chip in ['GREEN', 'RED']:
                self.logger.info("Image Assembly: Processing chip: " + chip)
                imageassembly.target_l0 = imageassembly.assemble_image(chip)
            exit_code = 1
            return Arguments([exit_code, imageassembly.target_l0])
        except Exception as e:
            self.logger.error(f"ImageAssembly algorithm failed: {e}\n{traceback.format_exc()}")
            return Arguments([exit_code, None])

    def _pre(self):
        pass
    
    def _post(self):
        pass