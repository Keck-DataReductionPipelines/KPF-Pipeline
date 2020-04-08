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
        self.alg = OrderTraceAlg(self.flat_data, self.config['PARAM'])

    
    def _pre_condition():
        '''
        Check for some necessary pre conditions
        '''
        # input argument must be KPF0
        success = isinstance(self.flat_data, KPF0)

        return success

    def _perform(self) -> None:
        '''
        This primitive's action
        '''
        # 1) Locate cluster
        cluster_xy = self.alg.locate_cluster()

        # 2) assign cluster id and do basic cleaning
        cluster_info = self.collect_clusters(cluster_xy)
        clean_info = remove_cluster_noise(cluster_info, cluster_xy)
        x, y, index = reorganize_index(clean_info, cluster_xy)

        # 3) advanced cleaning
        index, all_status = advanced_cluster_cleaning_handler(index, x, y, self.config['power'])
        x, y, index = reorganize_index(index, x, y)

        # 4) clean clusters along bottom and top border
        index_b = clean_clusters_on_border(x, y, index, 0)
        index_t = clean_clusters_on_border(x, y, index_b, ny-1)
        x_border, y_border, index_border =  self.reorganize_index(index_t, x, y)

        # 5) Merge cluster
        merge_x, merge_y, merge_index, merge_coeffs = self.merge_clusters(index_border, x_border, y_border, power)
        c_x, c_y, c_index = self.remove_broken_cluster(merge_index, merge_x, merge_y, merge_coeffs)
        cluster_coeffs, errors = self.curve_fitting_on_all_clusters(c_index, c_x, c_y, power)
        cluster_points = self.get_cluster_points(cluster_coeffs, power)

        # 6) Find width
        all_widths = self.find_all_cluster_widths(c_index, c_x, c_y, cluster_coeffs,  cluster_points, power)
        return {'cluster_index': c_index, 'cluster_x': c_x, 'cluster_y': c_y, 'widths': all_widths,
                'coeffs': cluster_coeffs, 'errors': errors}