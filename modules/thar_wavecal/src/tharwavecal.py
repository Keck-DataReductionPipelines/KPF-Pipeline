# Standard dependencies
import configparser
import numpy as np

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.models.level1 import KPF1

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.thar_wavecal.src.alg import ThArCalibrationAlg

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/thar_wavecal/configs/default.cfg'

class ThArCalibrate():
    """This module defines class `EtalonWaveCal,` which inherits from KPF1_Primitive and provides methods
    to perform the event `Etalon wavelength calibration` in the recipe.

    Args:
        KPF1_Primitive: Parent class
        action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `EtalonWaveCal` event issued in recipe.
        context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `etalon_wavecal` module in master config file associated with recipe.

    Attributes:
        l1_obj (kpfpipe.models.level1.KPF1): Instance of `KPF1`, assigned by `actions.args[0]`
        linelist_path (kpfpipe.models.level1.KPF1): Instance of `KPF1`,  assigned by `actions.args[1]`
        linelist_sub_path (kpfpipe.models.level1.KPF1): Instance of `KPF1`,  assigned by `actions.args[2]`
        data_type (kpfpipe.models.level1.KPF1): Instance of `KPF1`,  assigned by `actions.args[3]`
        config_path (str): Path of config file for Etalon wavelength calibration.
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger
        alg (modules.wavelength_cal.src.alg.EtalonWaveCalAlg): Instance of `EtalonWaveCal,` which has operation codes for Etalon Wavelength Calibration.
    """
    def __init__(self, 
                action:Action,
                context:ProcessingContext) -> None:
        """ThAr Wave Calibration constructor.

        Args:
            action (Action): Contains positional arguments and keyword arguments passed by the `TharWaveCal` event issued in recipe:
                
                `action.args[0] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing level 1 file
                `action.args[1] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing path to line list
                `action.args[2] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing path to line list subset
                `action.args[3] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing whether or not to create plots
                
            context (ProcessingContext): Contains path of config file defined for `wavelength_cal` module in master config file associated with recipe.
        """

        KPF1_Primitive.__init__(self,action,context)

        self.l1_obj=self.action.args[0]
        self.linelist_path=self.action.args[1]
        self.linelist_sub_path=self.action.args[2]
        self.data_type=self.action.args[3]

        #Input configuration
        self.config=configparser.ConfigParser()
        try:
            self.config_path=context.config_path['thar_wavecal']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        #Start logger
        self.logger=None
        #self.logger=start_logger(self.__class__.__name__,config_path)
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        self.alg = ThArCalibrationAlg(self.config,self.logger)

    def _perform(self): -> None:
        """Performs ThAr wavelength calibration algorithm
        """
        if self.logger:
            self.logger.info("ThAr Wavelength Calibration: Loading linelist and linelist subset paths")
        linelist = np.load(self.linelist_path)
        linelist_sub = np.load(self.linelist_sub_path, allow_pickle=True).tolist()

        if self.logger:
            self.logger.info("ThAr Wavelength Calibration: Loading Redman line lists")
        redman_w = np.array(linelist['redman_w'], dtype=float)
        redman_i = np.array(linelist['redman_i'], dtype=float)

        if self.logger:
            self.logger.info("ThAr Wavelength Calibration: Loading flux and wavelengths")
        assert self.l1_obj.header['CAL-OBJ'].startswith('ThAr') #check this through line 75
        flux = self.l1_obj.data['CAL'][0,:,:]
        flux = np.nan_to_num(flux)
        flux[flux < 0] = np.min(flux[flux > 0])
        wls = self.l1_obj.data['SCI'][1,:,:]

        if self.logger:
            self.logger.info("ThAr Wavelength Calibration: Running calibration algorithm") 
        thar_wavesoln = self.alg.run_on_all_orders(flux,redman_w,redman_i,wls,self.plot_toggle)

        self.l1_obj.data['SCI'][1,:,:] = thar_wavesoln

        return Arguments(self.l1_obj)