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

        # input argument
        # assume action.args.arg.flat_data includes data, order_trace_result and header['ORDERTRACE']['POLY_ORD']
        # and action.args.args.arg.spectrum_data includes data, header['PRIMARY']
        self.flat_data = action.args.arg.flat_data
        self.spectral_data = action.args.arg.spectral_data
        # input configuration
        self.config = configparser.ConfigParser()
        try:
            config_path = action.args.config
            self.config.read(config_path)
        except:
            self.config.read(DEFAULT_CFG_PATH)

        # Order trace algorithm setup
        self.alg = OrderTraceAlg(self.flat_data.data, self.spectral_data, config)

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # input argument must be KPF0
        success = isinstance(self.flat_data, KPF0) and isinstance(self.spectral_data, KPF0)

        return success

    def _perform(self):
        opt_ext_result = self.alg.extract_spectrum(rectification_method=OptimalExtractAlg.VERTICAL, print_progress='no')
        self.action.args.optimal_extraction_result = opt_ext_result['optimal_extraction_result']

        return self.action.args


