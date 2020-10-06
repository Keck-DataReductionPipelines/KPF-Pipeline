import configparser
import numpy as np

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.models.level0 import KPF0

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.bias_subtraction.src.alg import BiasSubtraction
from modules.utils.frame_combine import frame_combine

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/flat_fielding/configs/default.cfg'

class FlatFielding(KPF0_Primitive):
    def __init__(self, action:Action, context:ProcessingContext) -> None:

        #Initialize parent class
        KPF0_Primitive.__init__(self,action,context)
        self.logger=start_logger(self.__class__.__name__, config_path)

        #Input argument
        #self.input=action.args[0]
        self.rawdata=self.input.data
        
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

        #Bias subtraction algorithm setup
        self.alg=FlatFielding(self.rawimage,self.masterflat,config=self.config,logger=self.logger)

        #Preconditions
        """
        Potential preconditions:
        """
        #Postconditions
        
        #Perform - primitive's action
    def _perform(self) -> None:

        # 1) stack flat files using util fxn, creates master flat (prelim, this function will become more nuanced)
        if self.logger:
            self.logger.info("Flat Division: creating master flat...")
        masterflat_data=frame_combine(flats_data)

        # 2) divide raw by master flat
        if self.logger:
            self.logger.info("Flat Division: dividing raw image by master flat...")
        flat_corrected_raw=self.alg.flat_fielding(rawdata,masterflat_data)

        