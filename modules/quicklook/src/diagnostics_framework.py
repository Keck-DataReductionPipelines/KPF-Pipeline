import ast
import configparser as cp
import modules.quicklook.src.diagnostics as diagnostics
from modules.Utils.kpf_parse import HeaderParse

# Pipeline dependencies
from kpfpipe.logger import *
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/quicklook/configs/default.cfg'

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

        #Input arguments
        self.data_level_str   = self.action.args[0]
        self.kpf_object       = self.action.args[1]
        self.diagnostics_name = self.action.args[2]

        #Input configuration
        self.config = cp.ConfigParser()
        try:
            self.config_path = context.config_path['quicklook']
        except:
            self.config_path = DEFAULT_CFG_PATH

        self.config.read(self.config_path)

        #Start logger
        self.logger=None
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Started {}'.format(self.__class__.__name__))
        self.logger.info('self.diagnostics_name = {}'.format(self.diagnostics_name))


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
                primary_header = HeaderParse(self.kpf_object, 'PRIMARY')
                name = primary_header.get_name()
                if name == 'Dark':
                    self.logger.info('Measuring diagnostics: {}'.format(self.diagnostics_name))
                    self.kpf_object = diagnostics.add_headers_dark_current_2D(self.kpf_object, logger=None)
                    exit_code = 1
                else: 
                    self.logger.info("Observation type {} != 'Dark'.  Dark current not computed.".format(name))
            
        elif 'L1' in self.data_level_str:
            pass
            
        elif 'L2' in self.data_level_str:
            pass

        # Finish.
        self.logger.info('Finished {}'.format(self.__class__.__name__))

        return Arguments(exit_code, self.kpf_object)
