# Standard dependencies
import configparser

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.models.level1 import KPF1

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.radial_velocity.src.alg import RadialVelocityAlg

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/optimal_extraction/configs/default.cfg'


class RadialVelocityComputeStats(KPF_Primitive):
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

        # input argument
        # assume action.args.arg.flat_data includes data, order_trace_result and header['ORDERTRACE']['POLY_ORD']
        # and action.args.args.arg.spectrum_data includes data, header['PRIMARY']
        self.input = action.args[0]
        # input configuration
        self.config = configparser.ConfigParser()
        try:
            config_path = action.args.config
            self.config.read(config_path)
        except:
            self.config.read(DEFAULT_CFG_PATH)

        self.spectrum_data = self.input.flux['SCI1'] if hasattr(self.input, 'flux') else None
        self.wave_cal = self.input.wave['SCI1'] if hasattr(self.input, 'wave') else None
        self.header = self.input.header['PRIMARY'] if hasattr(self.input, 'header') else None
        # Order trace algorithm setup
        self.alg = RadialVelocityAlg(self.spectrum_data, self.wave_cal,  self.header, self.init_data, config)

    def _pre_condition(self) -> bool:
        """
        Check for some necessary pre conditions
        """
        # input argument must be KPF0
        success = isinstance(self.flat_data, KPF0) and isinstance(self.spectral_data, KPF0)

        return success

    def _perform(self):
        rv_results = self.alg.compute_rv_by_cc(self.ref_ccf)
        assert(rv_results['ccf_df'])
        self.input.create_extension('CCF')
        self.input.extension['CCF'] = rv_results['ccf_df']

        if self.logger:
            self.logger.info("RadialVelocity: Done!")

        return Arguments(self.input)



