# Standard dependencies
import configparser
import numpy as np
import pandas as pd

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.models.level1 import KPF1

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.cont_norm.src.alg import ContNormAlg

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/cont_norm/configs/default.cfg'

class ContinuumNorm(KPF1_Primitive):
    """This module defines class `ContNorm` which inherits from KPF1_Primitive and provides methods
    to perform the event `Continuum Normalization` in the recipe.

    Args:
        KPF1_Primitive: Parent class.
        action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `ContinuumNormalization` event issued in recipe.
        context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `continuum_normalization` module in master config file associated with recipe.

    Attributes:
        l1_obj (kpfpipe.models.level1.KPF1): Instance of `KPF1`, assigned by `actions.args[0]`
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger
        alg (modules.wavelength_cal.src.alg.LFCWaveCalibration): Instance of `LFCWaveCalibration,` which has operation codes for LFC Wavelength Calibration.

    """
    def __init__(self, action:Action, context:ProcessingContext) -> None:
        """
        ContNorm constructor.

        Args:
            action (Action): Contains positional arguments and keyword arguments passed by the `ContinuumNormalization` event issued in recipe:
              
                `action.args[0] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing level 1 spectrum
                `action.args[1] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing data type.

            context (ProcessingContext): Contains path of config file defined for `cont_norm` module in master config file associated with recipe.

        """
        #Initialize parent class
        KPF1_Primitive.__init__(self,action,context)

        #input recipe arguments
        self.l1_obj=self.action.args[0]
        self.data_type=self.action.args[1]

        #Input configuration
        self.config=configparser.ConfigParser()
        try:
            self.config_path=context.config_path['cont_norm']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        #Start logger
        self.logger=None
        #self.logger=start_logger(self.__class__.__name__,config_path)
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        #Continuum normalization algorithm setup
        self.alg=ContNormAlg(self.config,self.logger)

    #Perform
    def _perform(self) -> None:
        """
        Primitive action - 
        Performs continuum normalization by calling on ContNormAlg in alg.

        Returns:
            norm: Normalized spectrum.

        """
        #extract extensions (for NEID: sciwave and sciflux)
        if self.logger:
            self.logger.info("Continuum Normalization: Extracting SCIWAVE & SCIFLUX extensions")
        sciflux = self.l1_obj.data['SCI'][0,:,:]#0 referring to 'flux'
        sciwave = self.l1_obj.data['SCI'][2,:,:]#2 referring to 'wave'

        #run continuum normalization
        if self.logger:
            self.logger.info("Continuum Normalization: Extracting wavelength and flux data")
        norm = self.alg.run_cont_norm(sciwave,sciflux)

        #write to fits file
        return Arguments(norm)

