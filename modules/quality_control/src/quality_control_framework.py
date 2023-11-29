import ast
import configparser as cp
import modules.quality_control.src.quality_control as qc

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
        fits_filename (str): Input FITS filename.
        fits_object (str):
        actual_dir (str): Prefix of actual directory outside container that maps to /data (e.g., /data/kpf)
    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.data_level_str = self.action.args[1]
        self.fits_filename = self.action.args[2]
        self.fits_object = self.action.args[3]
        self.qc_list_flag = self.action.args[4]

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

        self.logger.info('self.data_type = {}'.format(self.data_type))
        self.logger.info('self.fits_filename = {}'.format(self.fits_filename))

        module_config_obj = cp.ConfigParser()
        res = module_config_obj.read(self.module_config_path)
        if res == []:
            raise IOError('failed to read {}'.format(self.module_config_path))

        module_param_cfg = module_config_obj['PARAM']

        debug_level_cfg_str = module_param_cfg.get('debug_level')
        self.debug_level_cfg = ast.literal_eval(debug_level_cfg_str)

        self.logger.info('self.debug_level_cfg = {}'.format(self.debug_level_cfg))

        self.logger.info('Type of self.debug_level_cfg = {}'.format(type(self.debug_level_cfg)))


    def _perform(self):

        """
        Returns exitcode:
            0 = Normal
        """

        quality_control_exit_code = 0

        # Perform quality control.

        self.logger.info('Performing quality control on {}'.format(self.fits_filename))

        if 'L0' in self.data_level_str:
            qc_obj = qc.QCL0(self.fits_object)
            name = 'jarque_bera_test_red_amp1'
            value = 3.14159256
            qc_obj.add_qc_keyword_to_header(name,value)
        elif '2D' in self.data_level_str:
            qc_obj = qc.QC2D(self.fits_object)
            name = 'jarque_bera_test_red_amp1'
            value = 3.14159256
            qc_obj.add_qc_keyword_to_header(name,value)
        elif 'L1' in self.data_level_str:

            self.logger.info('self.data_level_str = {}'.format(self.data_level_str))

            qc_obj = qc.QCL1(self.fits_object)

            name = 'monotonic_wavelength_solution_check'
            try:
                qc_obj.add_qc_keyword_to_header_for_monotonic_wls(name)
            except:
                pass

        elif 'L2' in self.data_level_str:
            qc_obj = qc.QCL2(self.fits_object)
            name = 'jarque_bera_test_red_amp1'
            value = 3.14159256
            qc_obj.add_qc_keyword_to_header(name,value)

        # Optionally list QC metrics.

        if self.qc_list_flag == 1:
            qc_obj.qcdefinitions.list_qc_metrics()

        # Finish.

        self.logger.info('Finished {}'.format(self.__class__.__name__))

        return Arguments([quality_control_exit_code, self.fits_object])
