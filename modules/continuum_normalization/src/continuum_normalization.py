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
from modules.continuum_normalization.src.alg import ContinuumNorm

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/continuum_normalization/configs/default.cfg'

class ContNorm(KPF0_Primitive):
    """This module defines class `ContNorm` which inherits from KPF0_Primitive and provides methods
    to perform the event `Continuum Normalization` in the recipe.

    Args:
        KPF0_Primitive: Parent class.
        action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `ContinuumNormalization` event issued in recipe.
        context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `continuum_normalization` module in master config file associated with recipe.

    Attributes:
        l0_obj (kpfpipe.models.level1.KPF1): Instance of `KPF0`, assigned by `actions.args[0]`
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger
        alg (modules.wavelength_cal.src.alg.LFCWaveCalibration): Instance of `LFCWaveCalibration,` which has operation codes for LFC Wavelength Calibration.

    """

    def __init__(self, 
                action:Action, 
                context:ProcessingContext) -> None:
        """
        ContNorm constructor.

        Args:
            action (Action): Contains positional arguments and keyword arguments passed by the `ContinuumNormalization` event issued in recipe:
              
                `action.args[0] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing level 0 spectrum

            context (ProcessingContext): Contains path of config file defined for `continuum_normalization` module in master config file associated with recipe.

        """
        #Initialize parent class
        KPF0_Primitive.__init__(self,action,context)

        #input recipe arguments
        self.l0_obj=self.action.args[0]

        #Input configuration
        self.config=configparser.ConfigParser()
        try:
            self.config_path=context.config_path['continuum_normalization']
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
        self.alg=ContinuumNorm(self.config,self.logger)

        #Preconditions

        #Postconditions

    #Perform
    def _perform(self) -> None:
        """
        Primitive action - 
        Performs continuum normalization by calling on ContinuumNorm in alg.

        Returns:

        """
        if self.logger:
            self.logger.info("Continuum Normalization: Extracting spectrum data")
        raw_spectrum=self.l0_obj.data['SCI'][0,:,:] #correct extension?

        if self.logger:
            self.logger.info("Continuum Normalization: Polynomial fitting")
        polyfit_cn= self.alg.flatspec(raw_spectrum,)


