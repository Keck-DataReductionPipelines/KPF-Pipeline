# Standard dependencies
import configparser

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.core import KPF_Primitive

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from kpfpipe.modules.order_trace.alg import OptimalExtractionAlg

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/optimal_extraction/configs/default.cfg'


class RadialVelocityInit(KPF_Primitive):
    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        """
        Example KPF module
        """
        # Initialize parent class
        KPF_Primitive.__init__(self, action, context)
        # start a logger
        self.logger = start_logger(self.__class__.__name__, None)

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            config_path = action.args.config
            self.config.read(config_path)
        except:
            self.config.read(DEFAULT_CFG_PATH)

        # Order trace algorithm setup
        self.alg = RadialVelocityInit(self.config, self.logger)

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # input argument must be KPF0

        return True

    def _perform(self):
        init_status = self.alg.start()
        assert(init_status['status'])

        return Arguments(init_status['data'])


