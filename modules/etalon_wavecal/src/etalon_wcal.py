# Pipeline dependencies
import configparser
import numpy as np

from kpfpipe.logger import start_logger
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.models.level1 import KPF1

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.etalon_wavecal.src.alg import EtalonWaveCalAlg

DEFAULT_CFG_PATH = 'modules/etalon_wavecal/configs/default.cfg'

class EtalonWaveCal(KPF1_Primitive):
    """This module defines class `EtalonWaveCal,` which inherits from KPF1_Primitive and provides methods
    to perform the event `Etalon wavelength calibration` in the recipe.

    Args:
        KPF1_Primitive: Parent class
        action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `EtalonWaveCal` event issued in recipe.
        context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `etalon_wavecal` module in master config file associated with recipe.

    Attributes:
        l1_obj (kpfpipe.models.level1.KPF1): Instance of `KPF1`, assigned by `actions.args[0]`
        data_type (kpfpipe.models.level1.KPF1): Instance of `KPF1`,  assigned by `actions.args[1]`
        config_path (str): Path of config file for Etalon wavelength calibration.
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger
        alg (modules.wavelength_cal.src.alg.EtalonWaveCalAlg): Instance of `EtalonWaveCal,` which has operation codes for Etalon Wavelength Calibration.
    """
    def __init__(self, 
                action:Action,
                context:ProcessingContext) -> None:
        """Etalon Wavelength Calibration constructor.

        Args:
            action (Action): Contains positional arguments and keyword arguments passed by the `EtalonWaveCal` event issued in recipe:
                    
                    `action.args[0] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing level 1 file
                    `action.args[1] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing level 1 file data type

            context (ProcessingContext): Contains path of config file defined for `etalon_wavecal` module in master config file associated with recipe.

        """

        KPF1_Primitive.__init__(self,action,context)

        self.l1_obj=self.action.args[0]
        #self.data_type=self.action.args[1]

        #Input configuration
        self.config=configparser.ConfigParser()
        try:
            self.config_path=context.config_path['etalon_wavecal']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        #Start logger
        self.logger=None
        #self.logger=start_logger(self.__class__.__name__,config_path)
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        #Etalon wavelength calibration algorithm setup
        self.alg=EtalonWaveCalAlg(self.config,self.logger)

    def _perform(self) -> None:
        if self.logger:
            self.logger.info("Etalon Wavelength Calibration: Loading flux and wavelengths")
        assert self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('Etalon') #check this through line 75
        flux = self.l1_obj['CALFLUX']
        flux = np.nan_to_num(flux)
        #neid masking
        flux[:,425:450] = 0

        if self.logger:
            self.logger.info("Etalon Wavelength Calibration: Running calibration algorithm")
        self.alg.run_on_all_orders(flux)

        #return Arguments(self.l1_obj)
