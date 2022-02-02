# Standard dependencies, # Everything is based on the line activity Measurements module so far.
import configparser
import numpy as np
import pandas as pd
from astropy.io import fits

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.models.level1 import KPF1

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.continuum_normalization.src.alg import ContNormAlgg
from modules.line_activity.src.alg import CalcHalpha, LineActivityAlg # new to line activity

# Global read-only variables
# DEFAULT_CFG_PATH = 'modules/continuum_normalization/configs/default.cfg'
DEFAULT_CFG_PATH = 'modules/line_activity/configs/default.cfg'

class Line_Activity(KPF1_Primitive):
    """This module defines class 'LineActivity' which inherits from KPF1_Primitive and provides methods
    to perform the event `Calculate Line Activity` in the recipe.  
    Args:
        KPF1_Primitive: Parent class.
        action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `LineActivity` event issued in recipe.
        context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `LineActivity` module in master config file associated with recipe.
    Attributes:
        l1_obj (kpfpipe.models.level1.KPF1): Instance of `KPF1`, assigned by `actions.args[0]`
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger
        alg (modules.wavelength_cal.src.alg.LFCWaveCalibration): Instance of `LFCWaveCalibration,` which has operation codes for LFC Wavelength Calibration.
    """
    def __init__(self, action:Action, context:ProcessingContext) -> None:
        """
        LineActivity constructor.
        Args:
            action (Action): Contains positional arguments and keyword arguments passed by the `LineActivity` event issued in recipe:
              
                `action.args[0] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing level 1 spectrum
                `action.args[1] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing data type.
            context (ProcessingContext): Contains path of config file defined for `Line Activity` module in master config file associated with recipe.
        """
        print('[{}] Inside! '.format(self.__class__.__name__))
        #Initialize parent class
        KPF1_Primitive.__init__(self,action,context)

        #input recipe arguments
        self.l1_obj=self.action.args[0]
        # self.data_type=self.action.args[1]

        #Input configuration
        self.config=configparser.ConfigParser()
#        try:
#            self.config_path=context.config_path['LineActiivty'] # Not sure what should be in quotes here.
#        except:
#            self.config_path = DEFAULT_CFG_PATH
#        self.config.read(self.config_path)
        self.config_path = DEFAULT_CFG_PATH # HTI not sure aobut his one.
        self.config.read(self.config_path)


        #Start logger
        self.logger=None
        #self.logger=start_logger(self.__class__.__name__,config_path)
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        #Line Activity algorithm setup
        self.alg=LineActivityAlg(self.config,self.logger)

    #Perform
    def _perform(self) -> None:
        """
        Primitive action - 
        Performs line activity Measurements by calling on LineActivityAlg in alg.
        Returns:
            EW: equivalent width of H-alpha
        """
        print('[{}] Inside! '.format(self.__class__.__name__))
        #extract extensions (for NEID: sciwave and sciflux)
        if self.logger:
            self.logger.info("line activity Measurements: Extracting SCIWAVE & SCIFLUX extensions")
        sciflux = self.l1_obj.data['SCI1'][0,:,:]#0 referring to 'flux'
        sciwave = self.l1_obj.data['SCI1'][1,:,:]#1 referring to 'wave'
        #run line activity Measurements
        if self.logger:
            self.logger.info("line activity Measurements: Extracting wavelength and flux data")
        EW = self.alg.CalcHalpha(sciwave,sciflux)
        

#        #new fits creation  # For line_activity, we want to update a .fits file eventually.
#        if self.logger:
#            self.logger.info("line activity Measurements: Creating FITS for line activity Measurements output storage")
            #in progress
            #write to fits file
            
        #return Arguments(hdul)