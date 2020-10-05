# Standard dependencies
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
#from modules.utils.frame_combine import frame_combine

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/bias_subtraction/configs/default.cfg'

class BiasSubtraction(KPF0_Primitive):

    def __init__(self, action:Action, context:ProcessingContext) -> None:

        #Initialize parent class
        KPF0_Primitive.__init__(self,action,context)
        self.logger=start_logger(self.__class__.__name__, config_path)

        #Input arguments
        self.rawdata=self.action.args[0]
        self.masterbias=self.action.args[1]
        self.data_type=self.action.args[2]

        #Input configuration
        self.config=configparser.ConfigParser()
        try:
            config_path=context.config_path['bias_subtraction']
        except:
            config_path = DEFAULT_CFG_PATH
        self.config.read(config_path)

        #Start logger
        self.logger=start_logger(self.__class__.__name__,config_path)
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(config_path))

        #Bias subtraction algorithm setup
        self.alg=BiasSubtraction(self.rawimage,self.masterbias,config=self.config,logger=self.logger)

        #Preconditions

        #Postconditions
        
        #Perform - primitive's action
    def _perform(self) -> None:

        # 1) get raw data from file

        rawdata=KPF0.from_fits(self.rawdata,self.data_type)
        self.logger.info(f'file: {rawdata}, obj.data_type is {type(obj.data)}')

        # 2) get bias data from file
        
        masterbias=KPF0.from_fits(self.masterbias,self.data_type)
        self.logger.info(f'file: {masterbias}, obj.data_type is {type(obj.data)}')

        # 3) subtract master bias from raw
        if self.logger:
            self.logger.info("Bias Subtraction: subtracting master bias from raw image...")
        bias_corrected_raw=self.alg.bias_subtraction(rawdata,masterbias)

