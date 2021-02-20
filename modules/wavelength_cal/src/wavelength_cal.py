# Standard dependencies
import configparser
import numpy as np
import pandas as pd

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.models.level1 import KPF1

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.wavelength_cal.src.alg import LFCWaveCalibration

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/wavelength_cal/configs/default.cfg'

class WaveCalibrate(KPF0_Primitive):
    """
    This module defines class `WaveCalibrate,` which inherits from KPF0_Primitive and provides methods
    to perform the event `LFC wavelength calibration` in the recipe.

    Args:
        KPF0_Primitive: Parent class
        action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `LFCWaveCalibration` event issued in recipe.
        context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `wavelength_cal` module in master config file associated with recipe.

    Attributes:
        l1_obj (kpfpipe.models.level0.KPF0): Instance of `KPF0`, assigned by `actions.args[0]`
        master_wavelength (kpfpipe.models.level0.KPF0): Instance of `KPF0`, assigned by `actions.args[1]`
        data_type (kpfpipe.models.level0.KPF0): Instance of `KPF0`,  assigned by `actions.args[2]`
        config_path (str): Path of config file for LFC wavelength calibration.
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger
        alg (modules.wavelength_cal.src.alg.LFCWaveCalibration): Instance of `LFCWaveCalibration,` which has operation codes for LFC Wavelength Calibration.
    """
    def __init__(self, 
                action:Action,
                context:ProcessingContext) -> None:
        """
        WaveCalibrate constructor.

        Args:
            action (Action): Contains positional arguments and keyword arguments passed by the `LFCWaveCal` event issued in recipe:
              
                `action.args[0] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing Laser Frequency Comb (LFC)
                `action.args[1] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing master file
                `action.args[2] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing data type

            context (ProcessingContext): Contains path of config file defined for `wavelength_cal` module in master config file associated with recipe.
        """
        #Initialize parent class
        KPF0_Primitive.__init__(self,action,context)

        #Input arguments
        self.l1_obj=self.action.args[0]
        self.master_wavelength=self.action.args[1]
        self.data_type=self.action.args[2]

        #Input configuration
        self.config=configparser.ConfigParser()
        try:
            self.config_path=context.config_path['wavelength_cal']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        #Start logger
        self.logger=None
        #self.logger=start_logger(self.__class__.__name__,config_path)
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))


        #Wavelength calibration algorithm setup
        self.alg=LFCWaveCalibration(self.config,self.logger)

        #Preconditions
       
        #Postconditions
        
    #Perform - primitive's action
    def _perform(self) -> None:
        """Primitive action - 
        Performs wavelength calibration by calling method 'run_wave_cal' from alg.py.

        Returns:
            wave_per_pix (np.ndarray): Wavelengths per pixel 
        """
        # 1. extract extensions (calflux and sciwave) 
        if self.logger:
            self.logger.info("Wavelength Calibration: Extracting CALFLUX and master calibration data")
        calflux=self.l1_obj['CAL'][0].data #0 referring to 'flux'
        master_data=self.master_wavelength['MASTER'].data 

        # 2. run wavecal
        if self.logger:
            self.logger.info("Wavelength Calibration: Running wavelength calibration")
        wave_per_pix=self.alg.run_wave_cal(calflux,master_data)

        # 3. overwrite calwave with wavelength calibration output (wavelength per pixel)
        self.l1_obj['CAL'][1].data=wave_per_pix

        #return Arguments(wave_per_pix)
