# Standard dependencies
import ast
import traceback
import configparser as cp

# Pipeline dependencies
from kpfpipe.logger import *
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments

# Local dependencies
#import modules.quicklook.src.diagnostics as diagnostics
from modules.quicklook.src.diagnostics import execute_all_diagnostics
#from modules.Utils.utils import styled_text
from modules.Utils.kpf_parse import HeaderParse
from modules.Utils.kpf_parse import get_datecode
from modules.Utils.kpf_parse import get_data_products_2D
from modules.Utils.kpf_parse import get_data_products_L1
from modules.Utils.kpf_parse import get_data_products_L2

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/quicklook/configs/default.cfg'

class DiagnosticsFramework(KPF0_Primitive):
    """
    Description:
        Adds diagnostics information to FITS headers of KPF files.

    Arguments:
        kpf_object (obj):
        data_level_str (str): L0, 2D, L1, L2 are possible choices.
        diagnostics_name (str): 'all' or name of diagnostics to add to headers; 
                                if 'all', then all diagnostics associated with 
                                data level are computed
    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        # Input arguments
        self.data_level_str   = self.action.args[0]
        self.kpf_object       = self.action.args[1]
        self.diagnostics_name = self.action.args[2]

        # Input configuration
        self.config = cp.ConfigParser()
        try:
            self.config_path = context.config_path['quicklook']
        except:
            self.config_path = DEFAULT_CFG_PATH

        self.config.read(self.config_path)

        # Start logger - check if already available
        self.logger = None
        
        # First check if context has a valid logger
        if hasattr(self.context, 'logger') and self.context.logger is not None:
            try:
                # Test if the logger is functional
                self.context.logger.handlers
                self.logger = self.context.logger
            except (AttributeError, Exception):
                # Context logger is not valid, will create new one below
                pass
        
        # If no valid context logger, check for existing class logger or create new one
        if self.logger is None:
            logger_name = self.__class__.__name__
            existing_logger = logging.getLogger(logger_name)
            
            if existing_logger.handlers and hasattr(self.__class__, '_class_logger'):
                # Reuse existing class logger
                self.logger = self.__class__._class_logger
            else:
                # Create new logger and cache it
                self.logger = start_logger(logger_name, self.config_path)
                self.__class__._class_logger = self.logger
        
        self.logger.info('Started {} instance'.format(self.__class__.__name__))
        self.logger.info('self.diagnostics_name = {}'.format(self.diagnostics_name))


    def _perform(self):
        """
        Returns exitcode:
            1 = Normal
            0 = Don't save file
        """

        exit_code = 0
        
        # Execute diagnostics
        self.kpf_object = execute_all_diagnostics(self.kpf_object, 
                                                  self.data_level_str, 
                                                  self.diagnostics_name, 
                                                  logger=self.logger, 
                                                  log_timing=True)
                    
        # Add RECEIPT entry
        self.kpf_object.receipt_add_entry('Diagnostics', self.__module__, f'data_level_str={self.data_level_str}, diagnostics_name={self.diagnostics_name}', 'PASS')
            
        # Finish
        self.logger.info('Finished {}'.format(self.__class__.__name__))
        return Arguments([exit_code, self.kpf_object])
