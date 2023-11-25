# Standard dependencies
import ast
import traceback
import configparser as cp

# Pipeline dependencies
from kpfpipe.logger import *
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments

# Local dependencies
import modules.quicklook.src.diagnostics as diagnostics
from modules.Utils.kpf_parse import HeaderParse
from modules.Utils.kpf_parse import get_data_products_L1

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
            # Dark Current
            if self.diagnostics_name == 'add_headers_dark_current_2D':
                try:
                    primary_header = HeaderParse(self.kpf_object, 'PRIMARY')
                    name = primary_header.get_name()
                    if name == 'Dark':
                        self.logger.info('Measuring diagnostics: {}'.format(self.diagnostics_name))
                        self.kpf_object = diagnostics.add_headers_dark_current_2D(self.kpf_object, logger=self.logger)
                        exit_code = 1
                    else: 
                        self.logger.info("Observation type {} != 'Dark'.  Dark current not computed.".format(name))
                except Exception as e:
                    self.logger.error(f"Measuring dark current failed: {e}\n{traceback.format_exc()}")

            # Guider
            if self.diagnostics_name == 'add_headers_guider':
                try:
                    self.logger.info('Measuring diagnostics: {}'.format(self.diagnostics_name))
                    self.kpf_object = diagnostics.add_headers_guider(self.kpf_object, logger=self.logger)
                    exit_code = 1
                except Exception as e:
                    self.logger.error(f"Measuring guider diagnostics failed: {e}\n{traceback.format_exc()}")

            # Exposure Meter
            if self.diagnostics_name == 'add_headers_exposure_meter':
                try:
                    self.logger.info('Measuring diagnostics: {}'.format(self.diagnostics_name))
                    self.kpf_object = diagnostics.add_headers_exposure_meter(self.kpf_object, logger=self.logger)
                    exit_code = 1
                except Exception as e:
                    self.logger.error(f"Measuring exposure meter diagnostics failed: {e}\n{traceback.format_exc()}")
                        
        elif 'L1' in self.data_level_str:
            # L1 SNR
            if self.diagnostics_name == 'add_headers_L1_SNR':
                try:
                    data_products = get_data_products_L1(self.kpf_object )
                    if ('Green' in data_products) or ('Red' in data_products): 
                        if True:
                            self.logger.info('Measuring diagnostics: {}'.format(self.diagnostics_name))
                            self.kpf_object = diagnostics.add_headers_L1_SNR(self.kpf_object, logger=self.logger)
                            exit_code = 1
                            print('exit_code = ' + str(exit_code))
                        else: 
                            self.logger.info("L1 SNR diagnostics not computed.")
                    else: 
                        self.logger.info("Green/Red not in L1 file. SNR diagnostics not computed.")
                except Exception as e:
                    self.logger.error(f"Measuring L1 SNR failed: {e}\n{traceback.format_exc()}")
            
            # Order Flux Ratios
            if self.diagnostics_name == 'add_headers_order_flux_ratios':
                try:
                    data_products = get_data_products_L1(self.kpf_object )
                    if ('Green' in data_products) or ('Red' in data_products): 
                        if True:
                            self.logger.info('Measuring diagnostics: {}'.format(self.diagnostics_name))
                            self.kpf_object = diagnostics.add_headers_order_flux_ratios(self.kpf_object, logger=self.logger)
                            exit_code = 1
                            print('exit_code = ' + str(exit_code))
                        else: 
                            self.logger.info("L1 SNR diagnostics not computed.")
                    else: 
                        self.logger.info("Green/Red not in L1 file. Flux ratio diagnostics not computed.")
                except Exception as e:
                    self.logger.error(f"Measuring orderlet flux ratios failed: {e}\n{traceback.format_exc()}")

            # Orderlet Flux Ratios
            if self.diagnostics_name == 'add_headers_orderlet_flux_ratios':
                try:
                    data_products = get_data_products_L1(self.kpf_object )
                    if ('Green' in data_products) or ('Red' in data_products): 
                        if True:
                            self.logger.info('Measuring diagnostics: {}'.format(self.diagnostics_name))
                            self.kpf_object = diagnostics.add_headers_orderlet_flux_ratios(self.kpf_object, logger=self.logger)
                            exit_code = 1
                            print('exit_code = ' + str(exit_code))
                        else: 
                            self.logger.info("L1 SNR diagnostics not computed.")
                    else: 
                        self.logger.info("Green/Red not in L1 file. Flux ratio diagnostics not computed.")
                except Exception as e:
                    self.logger.error(f"Measuring orderlet flux ratios failed: {e}\n{traceback.format_exc()}")

        elif 'L2' in self.data_level_str:
            pass

        # Finish.
        self.logger.info('Finished {}'.format(self.__class__.__name__))

        return Arguments(exit_code, self.kpf_object)
