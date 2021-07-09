# Standard dependencies
import configparser
import numpy as np
import pandas as pd

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.models.level1 import KPF1

# External dependencies
from modules.Utils.config_parser import ConfigHandler
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.continuum_normalization.src.alg import ContinuumNorm

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/continuum_normalization/configs/default.cfg'

class ContNorm(KPF1_Primitive):
    """This module defines class `ContNorm` which inherits from KPF0_Primitive and provides methods
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

    def __init__(self, 
                action:Action, 
                context:ProcessingContext) -> None:
        """
        ContNorm constructor.

        Args:
            action (Action): Contains positional arguments and keyword arguments passed by the `ContinuumNormalization` event issued in recipe:
              
                `action.args[0] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing level 1 spectrum

            context (ProcessingContext): Contains path of config file defined for `continuum_normalization` module in master config file associated with recipe.

        """
        #Initialize parent class
        KPF1_Primitive.__init__(self,action,context)

        #input recipe arguments
        self.l1_obj=self.action.args[0]

        #Input configuration
        self.config=configparser.ConfigParser()
        try:
            self.config_path=context.config_path['continuum_normalization']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        ### check formatting
        configpull=ConfigHandler(config,'PARAM')
        self.run_cont_norm = configpull.get_config_value('run_cont_norm', True)
        self.cont_norm_poly = configpull.get_config_value('cont_norm_poly', True)
        self.cont_norm_alpha = configpull.get_config_value('cont_norm_alpha', True)
        ###

        #Start logger
        self.logger=None
        #self.logger=start_logger(self.__class__.__name__,config_path)
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        #Continuum normalization algorithm setup
        self.alg=ContinuumNorm(self.config,self.logger)


    #Perform
    def _perform(self) -> None:
        """
        Primitive action - 
        Performs continuum normalization by calling on ContinuumNorm in alg.

        Returns:
            poly_norm_spec: Polynomial method normalized spectrum
            poly_yfit: Y-values of fitted polynomial from polynomial method 
            afs_norm_spec: Alphashape method normalized spectrum
            afs_yfit: Y-values of fitted curve from alphashape method

        """
        if self.logger:
            self.logger.info("Continuum Normalization: Extracting wavelength and flux data")
        norm = alg.run_cont_norm(l1_obj)

        return Arguments(norm)

