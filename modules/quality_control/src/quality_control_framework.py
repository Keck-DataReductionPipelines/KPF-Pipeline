import ast
import configparser as cp
from modules.Utils.kpf_parse import HeaderParse
import modules.quality_control.src.quality_control as qc
from modules.quality_control.src.quality_control import execute_all_QCs

# Pipeline dependencies
from kpfpipe.logger import *
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/quality_control/configs/default.cfg'

class QualityControlFramework(KPF0_Primitive):

    """
    Description:
        Performs quality control on a FITS file.  Includes logic for automatically determining the data level.

    Arguments:
        data_type (str): Type of data (e.g., KPF).
        data_level_str (str): L0, 2D, L1, L2 are possible choices.
        fits_object (KPF object): L0/2D/L1/L2 KPF object
    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.data_level_str = self.action.args[1]
        self.kpf_object = self.action.args[2]
        self.qc_list_flag = self.action.args[3]

        try:
            self.module_config_path = context.config_path['quality_control']
            print("--->",self.__class__.__name__,": self.module_config_path =",self.module_config_path)
        except:
            self.module_config_path = DEFAULT_CFG_PATH

        print("{} class: self.module_config_path = {}".format(self.__class__.__name__,self.module_config_path))

        print("Starting logger...")
        self.logger = start_logger(self.__class__.__name__, self.module_config_path)

        if self.logger is not None:
            print("--->self.logger is not None...")
        else:
            print("--->self.logger is None...")

        self.logger.info('Started {}'.format(self.__class__.__name__))
        self.logger.debug('module_config_path = {}'.format(self.module_config_path))

        module_config_obj = cp.ConfigParser()
        res = module_config_obj.read(self.module_config_path)
        if res == []:
            raise IOError('failed to read {}'.format(self.module_config_path))

        module_param_cfg = module_config_obj['PARAM']
        debug_level_cfg_str = module_param_cfg.get('debug_level')
        self.debug_level_cfg = ast.literal_eval(debug_level_cfg_str)
        self.logger.info('Type of self.debug_level_cfg = {}'.format(type(self.debug_level_cfg)))


    def _perform(self):

        """
        Returns exitcode:
            0 = Normal
        """
 
        quality_control_exit_code = 0

        # Execute appropriate QC tests
        self.kpf_object = execute_all_QCs(self.kpf_object, self.data_level_str, logger=self.logger)
        
        # Optionally list QC metrics.
        if self.qc_list_flag == 1:
            qc_obj.qcdefinitions.list_qc_metrics()

        # Add RECEIPT entry
        self.kpf_object.receipt_add_entry('QualityControl', self.__module__, f'data_level_str={self.data_level_str}', 'PASS')
        
        # Finish.
        self.logger.info('Finished {}'.format(self.__class__.__name__))

        return Arguments([quality_control_exit_code, self.kpf_object])
