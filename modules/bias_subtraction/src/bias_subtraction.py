"""
This module defines class 'BiasSubtraction,' which inherits from KPF0_Primitive and provides methods
    to perform the event 'bias subtraction' in the recipe.

    Attributes:
        BiasSubtraction
    
    Description:
        * Method '__init__':
            BiasSubtraction constructor
            The following arguments are passed to '__init__':

                - 'action (keckdrpframework.models.action.Action)': 'action.args' contains
                positional arguments and keyword arguments passed by the 'BiasSubtraction' event issued in the recipe:

                    - 'action.args[0] (kpfpipe.models.level0.KPF0)': Instance of 'KPF0' containing raw image data
                    - 'action.args[1] (kpfpipe.models.level0.KPF0)': Instance of 'KPF0' containing master bias data
                    - 'action.args[2] (kpfpipe.models.level0.KPF0)': Instance of 'KPF0' containing the instrument/data type

                - 'context (keckdrpframework.models.processing_context.ProcessingContext)': 'context.config_path'
                contains the path of the config file defined for the 'bias_subtraction' module in the master config
                file associated with the recipe.

            The following attributes are defined to initialize the object:
                
                - 'rawdata (kpfpipe.models.level0.KPF0)': Instance of 'KPF0',  assigned by 'actions.args[0]'
                - 'masterbias (kpfpipe.models.level0.KPF0)': Instance of 'KPF0',  assigned by 'actions.args[1]'
                - 'data_type (kpfpipe.models.level0.KPF0)': Instance of 'KPF0',  assigned by 'actions.args[2]'
                - 'config_path (str)': Path of config file for the computation of bias subtraction.
                - 'config (configparser.ConfigParser)': Config context.
                - 'logger (logging.Logger)': Instance of logging.Logger
                - 'alg (modules.bias_subtraction.src.alg.BiasSubtraction)': Instance of 'BiasSubtraction,' which has operation
                codes for bias subtraction.

        * Method '_perform':

                -   BiasSubtraction returns the bias-corrected raw data, L0 object
    Usage:
        For the recipe, the bias subtraction event is issued like the following:

            :
            raw_file=find_files('input location')
            raw_min_bias=BiasSubtraction(raw_file, master_result_data, 'NEID')
            :

        where 'raw_file' is a L0 raw data file.

"""



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
from modules.bias_subtraction.src.alg import BiasSubtractionAlg

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/bias_subtraction/configs/default.cfg'

class BiasSubtraction(KPF0_Primitive):

    def __init__(self, 
                action:Action, 
                context:ProcessingContext) -> None:

        #Initialize parent class
        KPF0_Primitive.__init__(self,action,context)

        #Input arguments
        self.rawdata=self.action.args[0]
        self.masterbias=self.action.args[1]
        self.data_type=self.action.args[2]

        #Input configuration
        self.config=configparser.ConfigParser()
        try:
            self.config_path=context.config_path['bias_subtraction']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)


        #Start logger
        self.logger=None
        #self.logger=start_logger(self.__class__.__name__,config_path)
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        #Bias subtraction algorithm setup
        # """
        # option 1
        self.alg=BiasSubtractionAlg(self.rawdata,self.config,self.logger)
        # """
        #Preconditions

        #Postconditions
        
        #Perform - primitive's action
    def _perform(self) -> None:

        # 1) subtract master bias from raw
        if self.logger:
            self.logger.info("Bias Subtraction: subtracting master bias from raw image...")
        # option 1
        self.alg.bias_subtraction(self.masterbias)
        return Arguments(self.alg.get())

        """
        #option 2 (no alg file)
        self.rawdata.data=self.rawdata.data-self.masterbias.data
        return Arguments(self.rawdata)
        """