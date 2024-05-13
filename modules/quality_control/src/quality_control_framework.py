import ast
import configparser as cp
import modules.quality_control.src.quality_control as qc
from modules.Utils.kpf_parse import HeaderParse

# temporarily:
#import inspect

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
        actual_dir (str): Prefix of actual directory outside container that maps to /data (e.g., /data/kpf)
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
        #self.logger.info('self.debug_level_cfg = {}'.format(self.debug_level_cfg))
        self.logger.info('Type of self.debug_level_cfg = {}'.format(type(self.debug_level_cfg)))


    def _perform(self):

        """
        Returns exitcode:
            0 = Normal
        """
 
        quality_control_exit_code = 0

        # Define QC object
        if 'L0' in self.data_level_str:
            qc_obj = qc.QCL0(self.kpf_object)
        elif '2D' in self.data_level_str:
            qc_obj = qc.QC2D(self.kpf_object)
        elif 'L1' in self.data_level_str:
            qc_obj = qc.QCL1(self.kpf_object)
        elif 'L2' in self.data_level_str:
            qc_obj = qc.QCL2(self.kpf_object)
    
        # Get a list of QC method names appropriate for the data level
        qc_names = []
        for qc_name in qc_obj.qcdefinitions.names:
            if self.data_level_str in qc_obj.qcdefinitions.kpf_data_levels[qc_name]:
                qc_names.append(qc_name)

        # Run the QC tests and add result to keyword to header
        for qc_name in qc_names:
            try:
                primary_header = HeaderParse(self.kpf_object, 'PRIMARY')
                this_spectrum_type = primary_header.get_name(use_star_names=False)    
                spectrum_types = qc_obj.qcdefinitions.spectrum_types[qc_name]
                if (this_spectrum_type in spectrum_types) or ('all' in spectrum_types):
                    self.logger.info(f'Running QC: {qc_name} ({qc_obj.qcdefinitions.descriptions[qc_name]})')
                    method = getattr(qc_obj, qc_name) # get method with the name 'qc_name'
                    qc_value = method() # evaluate method
                    self.logger.info(f'QC result: {qc_value} (True = pass)')
                    qc_obj.add_qc_keyword_to_header(qc_name, qc_value)
                else:
                    self.logger.info(f'Not running QC: {qc_name} ({qc_obj.qcdefinitions.descriptions[qc_name]}) because spectrum type {this_spectrum_type} not in list of spectrum types: {spectrum_types}')
            except AttributeError as e:
                self.logger.info(f'Method {qc_name} does not exist in qc_obj or another AttributeError occurred: {e}')
                pass
            except Exception as e:
                self.logger.info(f'An error occurred when executing {qc_name}:', str(e))
                pass

        # Optionally list QC metrics.
        if self.qc_list_flag == 1:
            qc_obj.qcdefinitions.list_qc_metrics()

        # Finish.
        self.logger.info('Finished {}'.format(self.__class__.__name__))

        return Arguments([quality_control_exit_code, self.kpf_object])
