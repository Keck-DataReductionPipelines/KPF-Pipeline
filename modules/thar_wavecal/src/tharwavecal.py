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

    def __init__(self, 
                action:Action,
                context:ProcessingContext) -> None:
        """[summary]

        Args:
            action (Action): [description]
            context (ProcessingContext): [description]
        """

        KPF1_Primitive.__init__(self,action,context)


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
        """[summary]
        """

