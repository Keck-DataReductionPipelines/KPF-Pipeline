"""
    This module defines class `FlatFielding,` which inherits from KPF0_Primitive and provides methods
    to perform the event `flat fielding` in the recipe.

    Attributes:
        FlatFielding
    
    Description:
        * Method `__init__`:
            FlatFielding constructor
            The following arguments are passed to `__init__`:

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains
                positional arguments and keyword arguments passed by the `FlatFielding` event issued in the recipe:

                    - `action.args[0] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing raw image data
                    - `action.args[1] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing master flat data
                    - `action.args[2] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing the instrument/data type

                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                contains the path of the config file defined for the `flat_fielding` module in the master config
                file associated with the recipe.

            The following attributes are defined to initialize the object:
                
                - `rawdata (kpfpipe.models.level0.KPF0)`: Instance of `KPF0`,  assigned by `actions.args[0]`
                - `masterflat (kpfpipe.models.level0.KPF0)`: Instance of `KPF0`,  assigned by `actions.args[1]`
                - `data_type (kpfpipe.models.level0.KPF0)`: Instance of `KPF0`,  assigned by `actions.args[2]`
                - `config_path (str)`: Path of config file for the computation of flat fielding.
                - `config (configparser.ConfigParser)`: Config context.
                - `logger (logging.Logger)`: Instance of logging.Logger
                - `alg (modules.flat_fielding.src.alg.FlatFielding)`: Instance of `FlatFielding,` which has operation
                codes for flat fielding.

        * Method `_perform`:

                -   FlatFielding returns the flat-corrected raw data, L0 object
    Usage:
        For the recipe, the flat fielding event is issued like the following:

            :
            raw_file_name=find_files(`input location`)
            raw_file=kpf0_from_fits(raw_file_name)
            raw_div_flat=FlatFielding(raw_file, master_result_data, `NEID`)
            :

        where `raw_file_name` is the string to a L0 raw data file, `raw_file` is the raw FITS data, and `raw_div_flat` is what is now flat-corrected raw data.

"""


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
        """Primitive action - 
        perform flat division by calling method `flat_fielding` from FlatFielding

        Returns:
            Level 0, flat-corrected, raw observation data
        """

        #Option 1:
        if self.logger:
            self.logger.info("Flat-fielding: dividing raw image by master flat")
        flat_result=self.alg(self.masterflat)
        return Arguments(self.alg.get())


