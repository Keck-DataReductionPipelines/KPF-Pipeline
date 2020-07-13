# Standard dependencies
import configparser

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.models.level0 import KPF0

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from kpfpipe.modules.order_trace.alg import OptimalExtractionAlg

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/optimal_extraction/configs/default.cfg'


class OptimalExtraction(KPF0_Primitive):
    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:
        """
        Example KPF module
        """
        # Initialize parent class
        KPF0_Primitive.__init__(self, action, context)
        # start a logger
        self.logger = start_logger(self.__class__.__name__, None)
        if not self.logger:
            self.logger = self.context.logger

        # input argument
        # action.args[0] is for level 0 fits
        # action.args[1] is for level 0 flat with order trace result extension

        self.input_spectrum = action.args[0]
        self.input_flat = action.args[1]
        # input configuration
        self.config = configparser.ConfigParser()
        try:
            config_path = action.args.config
            self.config.read(config_path)
        except:
            self.config.read(DEFAULT_CFG_PATH)

        # Order trace algorithm setup
        self.alg = OrderTraceAlg(self.input_flat, self.input_spectrum, config)

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # input argument must be KPF0
        success = isinstance(self.input_flat, KPF0) and isinstance(self.input_spectrum, KPF0)

        return success

    def _perform(self):
        # rectification_method: OptimalExtractAlg.NoRECT(fastest) OptimalExtractAlg.VERTICAL, OptimalExtractAlg.NORMAL
        # extraction_method: 'optimal' (default), 'sum'
        opt_ext_result = self.alg.extract_spectrum(rectification_method=OptimalExtractAlg.VERTICAL)
        # result in dataframe
        optimal_extraction_result = opt_ext_result['optimal_extraction_result']
        # convert to level 1 data?

        return self.action.args


