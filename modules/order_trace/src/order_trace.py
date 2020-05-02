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
from kpfpipe.modules.order_trace.alg import OrderTraceAlg

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/order_trace/default.cfg'

class OrderTrace(KPF0_Primitive):

    def __init__(self, 
                 action: Action,
                 context: ProcessingContext) -> None:
        '''
        Example KPF module
        '''
        # Initialize parent class
        KPF0_Primitive.__init__(self, action, context)
        # start a logger
        self.logger = start_logger(self.__class__.__name__, None)

        # input argument 
        self.flat_data = action.args.arg
        # input configuration
        self.config = configparser.ConfigParser()
        try: 
            config_path = action.args.config
            self.config.read(config_path)
        except:
            self.config.read(DEFAULT_CFG_PATH)

        # Order trace algorithm setup 
        self.alg = OrderTraceAlg(self.flat_data, self.config['PARAM'], self.config['DEBUG'])

    
    def _pre_condition():
        '''
        Check for some necessary pre conditions
        '''
        # input argument must be KPF0
        success = isinstance(self.flat_data, KPF0)

        return success

    def _perform(self) -> None:
        """
        This primitive's action
        """

        # 1) Locate cluster
        cluster_xy = self.alg.locate_clusters()

        # 2) assign cluster id and do basic cleaning
        x, y, index = self.alg.form_clusters(cluster_xy['x'], cluster_xy['y'])

        power = self.alg.get_poly_degree()
        # 3) advanced cleaning
        index, all_status = self.alg.advanced_cluster_cleaning_handler(index, x, y)
        x, y, index = self.alg.reorganize_index(index, x, y)

        # 4) clean clusters along bottom and top border
        x, y, index_b = self.alg.clean_clusters_on_border(x, y, index, 0)
        new_x, new_y, new_index = self.alg.clean_clusters_on_border(x, y, index_b, ny-1)

        # 5) Merge cluster
        c_x, c_y, c_index, cluster_coeffs, cluster_points, errors = \
            self.alg.merge_clusters_and_clean(new_index, new_x, new_y)

        # 6) Find width
        all_widths = self.alg.find_all_cluster_widths(c_index, cluster_coeffs,  cluster_points,
                                                      power_for_width_estimation=3)

        df = write_cluster_into_dataframe(all_widths, cluster_coeffs)

        self.action.args.order_trace_result = df
        return self.action.args
