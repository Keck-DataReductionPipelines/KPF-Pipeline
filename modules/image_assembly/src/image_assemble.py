
# Standard dependencies
import configparser
import numpy as np
from astropy.io import fits

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.models.level0 import KPF0

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.image_assembly.src.alg import ImageAssemblyAlg

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/image_assembly/configs/default.cfg'

class ImageAssembly(KPF0_Primitive):
    """This module defines class `ImageAssembly,` which inherits from `KPF0_Primitive` and provides methods
    to perform the event `image assembly` in the recipe.

    Args:
        KPF0_Primitive: Parent class
        action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `ImageAssembly` event issued in recipe.
        context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `image_assembly` module in master config file associated with recipe.

    Attributes:
        rawfile (kpfpipe.models.level0.KPF0): Instance of `KPF0`,  assigned by `actions.args[0]`
        prl_overscan_reg (kpfpipe.models.level0.KPF0): Instance of `KPF0`,  assigned by `actions.args[1]`
        srl_overscan_reg (kpfpipe.models.level0.KPF0): Instance of `KPF0`,  assigned by `actions.args[2]`
        mode (kpfpipe.models.level0.KPF0): Instance of `KPF0`,  assigned by `actions.args[3]`
        order (kpfpipe.models.level0.KPF0): Instance of `KPF0`,  assigned by `actions.args[4]`
        oscan_clip_no (kpfpipe.models.level0.KPF0): Instance of `KPF0`,  assigned by `actions.args[5]`
        ref_output (kpfpipe.models.level0.KPF0): Instance of `KPF0`,  assigned by `actions.args[6]`
        ffi_exts (kpfpipe.models.level0.KPF0): Instance of `KPF0`,  assigned by `actions.args[7]`
        data_type (kpfpipe.models.level0.KPF0): Instance of `KPF0`,  assigned by `actions.args[8]`

        config_path (str): Path of config file for image assembly.
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger
        alg (modules.image_assembly.src.alg.ImageAssemblyAlg): Instance of `ImageAssembly,` which has operation codes for image assembly.
    """
    
    def __init__(self, 
                action:Action, 
                context:ProcessingContext) -> None:
        
        #Initialize parent class
        KPF0_Primitive.__init__(self,action,context)
        
        #Input arguments

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['image_assembly']
        except:
            self.config_path = DEFAULT_CFG_PATH

        self.config.read(self.config_path)

        #Start logger
        self.logger=None
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))
        
        #Image assembly algorithm setup
        self.alg = ImageAssemblyAlg(self.rawfile,self.prl_overscan_reg,
                                    self.srl_overscan_reg,self.mode,
                                    self.order,self.oscan_clip_no,self.ref_output,
                                    self.ffi_exts,self.data_type)
    
    def _perform(self) -> None:
        """Primitive action - 
        Performs image processing by calling method 'image_assembly' from ImageAssemble.
        Returns the assembled L0 object.
        
        Returns:
            HDUList: FITS file with extensions populated with full frame images 
        """
        l0_obj = self.alg.run_setup()
        return Arguments(l0_obj)