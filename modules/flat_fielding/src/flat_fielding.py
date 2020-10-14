
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
from modules.flat_fielding.src.alg import FlatFielding
#from modules.utils.frame_combine import frame_combine

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/flat_fielding/configs/default.cfg'

class FlatFielding(KPF0_Primitive):
    def __init__(self, action:Action, context:ProcessingContext) -> None:
       """
        FlatFielding constructer
       """ 
        #Initialize parent class
        KPF0_Primitive.__init__(self,action,context)

        #Input argument
        #self.input=action.args[0]
        self.rawdata=self.action.args[0]
        self.masterflat=self.action.args[1]
        self.data_type=self.action.args[2]
        
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
        self.alg=FlatFielding(self.rawdata,config=self.config,logger=self.logger)

        #Preconditions
        """
        Check for some necessary pre conditions
        """
        #Postconditions
        """
        Check for some necessary post conditions
        """
        #Perform - primitive`s action
    def _perform(self) -> None:
        """Primitive action - perform flat division by calling method `flat_fielding` from FlatFielding

        Returns:
            Level 0, flat-corrected, raw observation data
        """

        #Option 1:
        if self.logger:
            self.logger.info("Flat-fielding: dividing raw image by master flat")
        flat_result=self.alg(self.masterflat)
        return Arguments(self.alg.get())
