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
        LFCData (kpfpipe.models.level0.KPF0): Instance of `KPF0`, assigned by `actions.args[0]`
        f0 (np.float): Instance of `KPF0`, assigned by `actions.args[1]`
        f_rep (np.float): Instance of `KPF0`, assigned by `actions.args[2]`
        min_wave (np.float): Instance of `KPF0`, assigned by `actions.args[3]`
        max_wave (np.float): Instance of `KPF0`, assigned by `actions.args[4]`
        fit_order(np.int): Instance of `KPF0`, assigned by `actions.args[5]`
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
              
                `action.args[0]`(kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing Laser Frequency Comb (LFC) data
                `action.args[1]`(kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing initial frequency
                `action.args[2]`(kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing repetition frequency
                `action.args[3]`(kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing minimum wavelength of wavelength range
                `action.args[4]`(kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing maximum wavelength of wavelength range
                `action.args[5]`(kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing polynomial-fit order

            context (ProcessingContext): Contains path of config file defined for `wavelength_cal` module in master config file associated with recipe.
        """
        #Initialize parent class
        KPF0_Primitive.__init__(self,action,context)

        #Input arguments
        self.LFCdata=self.action.args[0]
        #self.data_type=self.action.args[1]

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

        self.alg=LFCWaveCalibration(self.f0,self.f_rep,self.config,self.logger)

        #Preconditions
       
        #Postconditions
        
    #Perform - primitive's action
    def _perform(self) -> None:
        """Primitive action - 
        Performs wavelength calibration by calling methods 'peak_detect' and 'poly_fit' from LFCWaveCalibration.

        Returns:
            wave_soln (np.Polynomial): Wavelength solution 
        """
        #return will be to_fits in recipe
    
        #1 mode_nos
        if self.logger:
            self.logger.info("Wavelength Calibration: Generating comb lines")
        clines_ang= self.alg.mode_nos(self.min_wave,self.max_wave,self.f0,self.f_rep)

        #output comb_lines_ang to steps 3,4,5

        #2 peak detection & gaussian fit
        if self.logger:
            self.logger.info("Wavelength Calibration: Detecting LFC peaks")
        peaks,props=self.alg.peak_detect(self.LFCData)
        #output peaks to steps 4,5,6, and props

        #3 mode match
        if self.logger:
            self.logger.info("Wavelength Calibration: Fitting mode numbers to corresponding peaks")
        idx=self.alg.mode_match(clines_ang,peaks)
        #output idx for step 5,6

        #4 fit polynomial 
        if self.logger:
            self.logger.info("Wavelength Calibration: Fitting order-by-order polynomial sol'n")
        wave_soln=self.alg.poly_fit(self.fit_order,clines_ang,peaks,idx)
        #output wave_soln_leg or wave_soln_poly to step 6

        #5 residuals 
        if self.logger:
            self.logger.info("Wavelength Calibration: Calculating residual standard deviation")
        self.alg.residuals(clines_ang,idx,wave_soln,peaks)

        #need to add Return Arguments(wave_soln)