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
from modules.thar_wavecal.src.alg import TharWaveCalAlg

DEFAULT_CFG_PATH = 'modules/thar_wavecal/configs/default.cfg'

class TharWaveCal(KPF1_Primitive):
    """This module defines class `ThArWaveCal,` which inherits from KPF1_Primitive and provides methods
    to perform the event `ThAr wavelength calibration` in the recipe.

    Args:
        KPF1_Primitive: Parent class
        action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `ThArWaveCal` event issued in recipe.
        context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `thar_wavecal` module in master config file associated with recipe.

    Attributes:
        l1_obj (kpfpipe.models.level1.KPF1): Instance of `KPF1`, assigned by `actions.args[0]`
        data_type (kpfpipe.models.level1.KPF1): Instance of `KPF1`,  assigned by `actions.args[1]`
        config_path (str): Path of config file for ThAr wavelength calibration.
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger
        alg (modules.wavelength_cal.src.alg.ThArWaveCalAlg): Instance of `ThArWaveCal,` which has operation codes for ThAr Wavelength Calibration.
    """
    def __init__(self, 
                action:Action,
                context:ProcessingContext) -> None:
        """ThAr Wavelength Calibration constructor.

        Args:
            action (Action): Contains positional arguments and keyword arguments passed by the `ThArWaveCal` event issued in recipe:
                    
                    `action.args[0] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing level 1 file
                    `action.args[1] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing level 1 file data type

            context (ProcessingContext): Contains path of config file defined for `thar_wavecal` module in master config file associated with recipe.
        """

        KPF1_Primitive.__init__(self,action,context)

        self.l1_obj=self.action.args[0]
        self.linelist=self.action.args[1]
        self.linelist_sub=self.action.args[2]

        self.config=configparser.ConfigParser()
        try:
            self.config_path=context.config_path['thar_wavecal']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        #Start logger
        self.logger=None
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        #ThAr wavelength calibration algorithm setup
        self.alg=TharWaveCalAlg(self.config,self.logger)

    def _perform(self) -> None:
        if self.logger:
            self.logger.info("ThAr Wavelength Calibration: Loading flux and wavelengths")
        assert self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('ThAr') #check this through line 75
        flux = self.l1_obj['CALFLUX']
        flux = np.nan_to_num(flux)
        flux[flux < 0] = np.min(flux[flux > 0])

        linelist = np.load(self.linelist)
        linelist_sub = np.load(self.linelist_sub,allow_pickle=True).tolist()
        redman_w = np.array(linelist['redman_w'],dtype=float)
        redman_i = np.array(linelist['redman_i'],dtype=float)
        #neid wls
        wls = self.l1_obj['SCIWAVE']
    
        if self.logger:
            self.logger.info("ThAr Wavelength Calibration: Running calibration algorithm")
        wl_soln = self.alg.run_on_all_orders(flux,redman_w,redman_i,linelist_sub,wls)
        
        self.l1_obj['SCIWAVE'] = wl_soln

        return Arguments(self.l1_obj)
        
