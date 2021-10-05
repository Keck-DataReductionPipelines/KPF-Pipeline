
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
from modules.flat_fielding.src.alg import FlatFieldingAlg
#from modules.utils.frame_combine import frame_combine

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/flat_fielding/configs/default.cfg'

class FlatFielding(KPF0_Primitive):
    """
    This module defines class `FlatFielding,` which inherits from `KPF0_Primitive` and provides methods 
    to perform the event `flat fielding` in the recipe.

    Args:
        KPF0_Primitive: Parent class
        action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `FlatFielding` event issued in recipe.
        context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `flat_fielding` module in master config file associated with recipe.

    Attributes:
        rawdata (kpfpipe.models.level0.KPF0): Instance of `KPF0`,  assigned by `actions.args[0]`
        masterflat (kpfpipe.models.level0.KPF0): Instance of `KPF0`,  assigned by `actions.args[1]`
        data_type (kpfpipe.models.level0.KPF0): Instance of `KPF0`,  assigned by `actions.args[2]`
        config_path (str): Path of config file for the computation of flat fielding.
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger
        alg (modules.flat_fielding.src.alg.FlatFielding): Instance of `FlatFielding,` which has operation codes for flat fielding.


    """
    def __init__(self, action:Action, context:ProcessingContext) -> None:
         
        """
        FlatFielding constructor.

        Args:
            action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `FlatFielding` event issued in recipe:

                `action.args[0]`(kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing raw image data
                `action.args[1]`(kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing master flat data
                `action.args[2]`(kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing the instrument/data type

            context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `flat_fielding` module in master config file associated with recipe.

       """ 
        #Initialize parent class
        KPF0_Primitive.__init__(self,action,context)
        
              #Input arguments
        self.raw_file=self.action.args[0]
        self.masterflat=self.action.args[1]
        self.ffi_exts=self.action.args[2]
        self.data_type=self.action.args[3]
        
        #Input configuration
        self.config=configparser.ConfigParser()
        try:
            config_path=context.config_path['flat_fielding']
        except:
            config_path = DEFAULT_CFG_PATH
        self.config.read(config_path)

        #Start logger
        self.logger=start_logger(self.__class__.__name__,config_path)
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(config_path))

        #Flat Fielding algorithm setup

        #Option 1
        self.alg=FlatFieldingAlg(self.raw_file,self.ffi_exts,config=self.config,logger=self.logger)

        #Preconditions
        
        #Postconditions
        
        #Perform - primitive`s action
    def _perform(self) -> None:
        """
        Primitive action - perform flat division by calling method `flat_fielding` from FlatFielding.
        Returns the flat-corrected raw data, L0 object.

        Returns:
            Arguments object(np.ndarray): Level 0, flat-corrected, raw observation data
        """
        self.masterflat = fits.open(self.masterflat)

        #Option 1:
        if self.logger:
            self.logger.info("Flat-fielding: dividing raw image by master flat")
        flat_result=self.alg(self.masterflat)
        return Arguments(self.alg.get())
