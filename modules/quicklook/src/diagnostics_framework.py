import ast
import configparser as cp
import modules.quicklook.src.diagnostics as diagnostics

# Pipeline dependencies
from kpfpipe.logger import *
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/quality_control/configs/default.cfg'

class DiagnosticsFramework(KPF0_Primitive):

    """
    Description:
        Adds diagnostics information to FITS headers of KPF files.

    Arguments:
        kpf_object (obj):
        data_level_str (str): L0, 2D, L1, L2 are possible choices.
        diagnostics_name (str): name of diagnostics to add to headers
    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_level_str   = self.action.args[0]
        self.kpf_object       = self.action.args[1]
        self.diagnostics_name = self.action.args[2]

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
        self.logger.info('self.diagnostics_name = {}'.format(self.diagnostics_name))

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
            1 = Normal
            0 = Don't save file
        """

        exit_code = 0

        # Measure Diagnostics.
        if 'L0' in self.data_level_str:
            pass
            
        elif '2D' in self.data_level_str:
            if self.diagnostics_name == 'add_headers_dark_current_2D':
                self.logger.info('Measuring diagnostics: {}'.format(self.diagnostics_name))
                self.kpf_object = diagnostics.add_headers_dark_current_2D(self.kpf_object, logger=None)
                exit_code = 1
            
        elif 'L1' in self.data_level_str:
            pass
            
        elif 'L2' in self.data_level_str:
            pass

        # Finish.
        self.logger.info('Finished {}'.format(self.__class__.__name__))

        return Arguments(exit_code, self.kpf_object)
