# Standard dependencies
import configparser
import numpy as np
import traceback

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.models.level0 import KPF0

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.quicklook.src.alg import QuicklookAlg

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/quicklook/configs/default.cfg'

class Quicklook(KPF0_Primitive):

    def __init__(self,
                    action:Action,
                    context:ProcessingContext) -> None:

        KPF0_Primitive.__init__(self,action,context)

        #Input arguments
        self.input_file = self.action.args[0]
        self.output_dir = self.action.args[1]
        self.qlp_level  = self.action.args[2]

        #Input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['quicklook']
        except:
            self.config_path = DEFAULT_CFG_PATH

        self.config.read(self.config_path)

        #Start logger
        self.logger=None
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        #Algorithm set up
        self.alg=QuicklookAlg(config=self.config,logger=self.logger)

    def _perform(self) -> None:
        try:
            if self.qlp_level == 'L0':
                self.alg.qlp_L0(self.input_file, self.output_dir)
            elif self.qlp_level == '2D':
                self.alg.qlp_2D(self.input_file, self.output_dir)
            elif self.qlp_level == 'L1':
                self.alg.qlp_L1(self.input_file, self.output_dir)
        except Exception as e:
            # Allow recipe to continue if QLP fails
            self.logger.error(f"Failure in L0 quicklook pipeline: {e}\n{traceback.format_exc()}")
            pass
