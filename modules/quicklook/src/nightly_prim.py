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
from modules.quicklook.src.alg_nightly import Nightly_summaryAlg

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/quicklook/configs/default.cfg'

class Nightly_summary(KPF0_Primitive):

    def __init__(self,
                    action:Action,
                    context:ProcessingContext) -> None:

        KPF0_Primitive.__init__(self,action,context)

        #Input arguments
        self.night=self.action.args[0]
        #self.file_name=self.action.args[1]
        #self.output_dir=self.action.args[1]
        #self.end_of_night_summary=self.action.args[2]
        # input configuration
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
        self.alg=Nightly_summaryAlg(config=self.config,logger=self.logger)

    def _perform(self) -> None:
        self.alg.nightly_procedures(self.night)
