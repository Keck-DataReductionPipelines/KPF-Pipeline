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
from modules.Utils.utils import styled_text
from modules.Utils.kpf_parse import HeaderParse
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
                                if 'all', then all diagnostics associated with data level are computed
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

        # Start logger
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
            # 2D flux
            if (self.diagnostics_name == 'all') or \
               (self.diagnostics_name == 'add_headers_2D_flux'):
                try:
                    self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_2D_flux", style="Bold", color="Blue")}')
                    self.kpf_object = diagnostics.add_headers_2D_flux(self.kpf_object, logger=self.logger)
                    exit_code = 1
                except Exception as e:
                    self.logger.error(f"Measuring 2D flux failed: {e}\n{traceback.format_exc()}")

            # Dark Current
            if (self.diagnostics_name == 'all') or \
               (self.diagnostics_name == 'add_headers_dark_current_2D'):
                try:
                    primary_header = HeaderParse(self.kpf_object, 'PRIMARY')
                    name = primary_header.get_name()
                    if name == 'Dark':
                        #data_products = get_data_products_2D(self.kpf_object)
                        #if (('Green' in data_products) or ('Red' in data_products)) and ('Telemetry' in data_products): 
                        if True:
                            self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_dark_current_2D", style="Bold", color="Blue")}')
                            self.kpf_object = diagnostics.add_headers_dark_current_2D(self.kpf_object, logger=self.logger)
                            exit_code = 1
                    else: 
                        self.logger.info("Observation type {} != 'Dark'.  Dark current not computed.".format(name))
                except Exception as e:
                    self.logger.error(f"Measuring dark current failed: {e}\n{traceback.format_exc()}")

            # Guider
            if (self.diagnostics_name == 'all') or \
               (self.diagnostics_name == 'add_headers_guider'):
                try:
                    self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_guider", style="Bold", color="Blue")}')
                    self.kpf_object = diagnostics.add_headers_guider(self.kpf_object, logger=self.logger)
                    exit_code = 1
                except Exception as e:
                    self.logger.error(f"Measuring guider diagnostics failed: {e}\n{traceback.format_exc()}")

            # HK
            if (self.diagnostics_name == 'all') or \
               (self.diagnostics_name == 'add_headers_hk'):
                try:
                    self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_hk", style="Bold", color="Blue")}')
                    self.kpf_object = diagnostics.add_headers_hk(self.kpf_object, logger=self.logger)
                    exit_code = 1
                except Exception as e:
                    self.logger.error(f"Measuring HK diagnostics failed: {e}\n{traceback.format_exc()}")

            # Exposure Meter
            if (self.diagnostics_name == 'all') or \
               (self.diagnostics_name == 'add_headers_exposure_meter'):
                try:
                    self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_exposure_meter", style="Bold", color="Blue")}')
                    self.kpf_object = diagnostics.add_headers_exposure_meter(self.kpf_object, logger=self.logger)
                    exit_code = 1
                except Exception as e:
                    self.logger.error(f"Measuring exposure meter diagnostics failed: {e}\n{traceback.format_exc()}")
                        
            # Masters Age - Bias, Dark, Flat
            if (self.diagnostics_name == 'all') or \
               (self.diagnostics_name == 'add_headers_masters_age_2D'):
                try:
                    self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_masters_age_2D", style="Bold", color="Blue")}')
                    self.kpf_object = diagnostics.add_headers_masters_age_2D(self.kpf_object, logger=self.logger)
                    exit_code = 1
                except Exception as e:
                    self.logger.error(f"Age of masters for Bias/Dark/Flat not computed: {e}\n{traceback.format_exc()}")

            # Cross-dispersion offset
            if (self.diagnostics_name == 'all') or \
               (self.diagnostics_name == 'add_headers_2D_xdisp_offset'):
                try:
                    primary_header = HeaderParse(self.kpf_object, 'PRIMARY')
                    name = primary_header.get_name()
                    if name == 'Flat':
                        self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_2D_xdisp_offset", style="Bold", color="Blue")}')
                        self.kpf_object = diagnostics.add_headers_2D_xdisp_offset(self.kpf_object, logger=self.logger)
                        exit_code = 1
                    else: 
                        self.logger.info("Observation type {} != 'Flat'.  Cross-disperion offset not computed.".format(name))
                except Exception as e:
                    self.logger.error(f"Measuring Cross-disperion offset failed: {e}\n{traceback.format_exc()}")

                        
        elif 'L1' in self.data_level_str:
            # WLS Age
            if (self.diagnostics_name == 'all') or \
               (self.diagnostics_name == 'add_headers_masters_age_L1'):
                try:
                    data_products = get_data_products_L1(self.kpf_object )
                    if ('Green' in data_products) or ('Red' in data_products): 
                        if True:
                            self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_masters_age_L1", style="Bold", color="Blue")}')
                            self.kpf_object = diagnostics.add_headers_masters_age_L1(self.kpf_object, logger=self.logger)
                            exit_code = 1
                        else: 
                            self.logger.info("Age of masters for wavelength solution not computed.")
                    else: 
                        self.logger.info("Green/Red not in L1 file. Age of masters for wavelength solution not computed.")
                except Exception as e:
                    self.logger.error(f"Measuring L1 master age failed: {e}\n{traceback.format_exc()}")

            # Order Trace and Smooth Lamp Age
            if (self.diagnostics_name == 'all') or \
               (self.diagnostics_name == 'add_headers_trace_lamp_age_L1'):
                try:
                    data_products = get_data_products_L1(self.kpf_object )
                    if ('Green' in data_products) or ('Red' in data_products): 
                        if True:
                            self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_trace_lamp_age_L1", style="Bold", color="Blue")}')
                            self.kpf_object = diagnostics.add_headers_trace_lamp_age_L1(self.kpf_object, logger=self.logger)
                            exit_code = 1
                        else: 
                            self.logger.info("Age of smooth lamp and order trace not computed.")
                    else: 
                        self.logger.info("Green/Red not in L1 file. Age of smooth lamp and order trace not computed.")
                except Exception as e:
                    self.logger.error(f"Measuring L1 order trace/smooth lamp failed: {e}\n{traceback.format_exc()}")

            # L1 SNR
            if (self.diagnostics_name == 'all') or \
               (self.diagnostics_name == 'add_headers_L1_SNR'):
                try:
                    data_products = get_data_products_L1(self.kpf_object )
                    if ('Green' in data_products) or ('Red' in data_products): 
                        if True:
                            self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_L1_SNR", style="Bold", color="Blue")}')
                            self.kpf_object = diagnostics.add_headers_L1_SNR(self.kpf_object, logger=self.logger)
                            exit_code = 1
                        else: 
                            self.logger.info("L1 SNR diagnostics not computed.")
                    else: 
                        self.logger.info("Green/Red not in L1 file. SNR diagnostics not computed.")
                except Exception as e:
                    self.logger.error(f"Measuring L1 SNR failed: {e}\n{traceback.format_exc()}")
            
            # L1 Order Flux Ratios
            if (self.diagnostics_name == 'all') or \
               (self.diagnostics_name == 'add_headers_L1_order_flux_ratios'):
                try:
                    data_products = get_data_products_L1(self.kpf_object )
                    if ('Green' in data_products) or ('Red' in data_products): 
                        if True:
                            self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_L1_order_flux_ratios", style="Bold", color="Blue")}')
                            self.kpf_object = diagnostics.add_headers_L1_order_flux_ratios(self.kpf_object, logger=self.logger)
                            exit_code = 1
                        else: 
                            self.logger.info("L1 SNR diagnostics not computed.")
                    else: 
                        self.logger.info("Green/Red not in L1 file. Flux ratio diagnostics not computed.")
                except Exception as e:
                    self.logger.error(f"Measuring orderlet flux ratios failed: {e}\n{traceback.format_exc()}")

            # L1 Orderlet Flux Ratios
            if (self.diagnostics_name == 'all') or \
               (self.diagnostics_name == 'add_headers_L1_orderlet_flux_ratios'):
                try:
                    data_products = get_data_products_L1(self.kpf_object )
                    if ('Green' in data_products) or ('Red' in data_products): 
                        if True:
                            self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_orderlet_flux_ratios", style="Bold", color="Blue")}')
                            self.kpf_object = diagnostics.add_headers_L1_orderlet_flux_ratios(self.kpf_object, logger=self.logger)
                            exit_code = 1
                        else: 
                            self.logger.info("L1 SNR diagnostics not computed.")
                    else: 
                        self.logger.info("Green/Red not in L1 file. Flux ratio diagnostics not computed.")
                except Exception as e:
                    self.logger.error(f"Measuring orderlet flux ratios failed: {e}\n{traceback.format_exc()}")

            # L1 LFC and first/last spectral orders with good lines
            if (self.diagnostics_name == 'all') or \
               (self.diagnostics_name == 'add_headers_L1_cal_line_quality'):
                try:
                    data_products = get_data_products_L1(self.kpf_object )
                    if ('Green' in data_products) or ('Red' in data_products): 
                        primary_header = HeaderParse(self.kpf_object, 'PRIMARY')
                        name = primary_header.get_name()
                        if name == 'LFC':
                            self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_L1_cal_line_quality", style="Bold", color="Blue")}')
                            self.kpf_object = diagnostics.add_headers_L1_cal_line_quality(self.kpf_object, cal='LFC', logger=self.logger)
                            exit_code = 1
                        elif name == 'Etalon':
                            self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_L1_cal_line_quality", style="Bold", color="Blue")}')
                            self.kpf_object = diagnostics.add_headers_L1_cal_line_quality(self.kpf_object, cal='Etalon', logger=self.logger)
                            exit_code = 1
                        else: 
                            self.logger.info("Observation type {} != 'LFC' or 'Etalon'.  Line diagnostics not computed.".format(name))
                    else: 
                        self.logger.info("Green/Red not in L1 file. LFC/Etalon line diagnostics not computed.")

                except Exception as e:
                    self.logger.error(f"Measuring LFC/Etalon line diagnostics failed: {e}\n{traceback.format_exc()}")

            # L1 count saturated lines for Cal lamps
            if (self.diagnostics_name == 'all') or \
               (self.diagnostics_name == 'add_headers_L1_saturated_lines'):
                try:
                    data_products = get_data_products_L1(self.kpf_object )
                    if ('Green' in data_products) or ('Red' in data_products): 
                        primary_header = HeaderParse(self.kpf_object, 'PRIMARY')
                        name = primary_header.get_name()
                        if name in ['LFC', 'Etalon', 'ThAr', 'UNe']:
                            self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_L1_saturated_lines", style="Bold", color="Blue")}')
                            self.kpf_object = diagnostics.add_headers_L1_saturated_lines(self.kpf_object, logger=self.logger)
                            exit_code = 1
                        else: 
                            self.logger.info("Observation type {} not in ['LFC', 'Etalon', 'ThAr', 'UNe'].  Saturated lines not counted.".format(name))
                    else: 
                        self.logger.info("Green/Red not in L1 file. Saturated lines not counted.")

                except Exception as e:
                    self.logger.error(f"Counting saturated lines failed: {e}\n{traceback.format_exc()}")

            # L1 standard deviation of WLS compared to WLS_ref
            if (self.diagnostics_name == 'all') or \
               (self.diagnostics_name == 'add_headers_L1_std_wls'):
                try:
                    data_products = get_data_products_L1(self.kpf_object )
                    if ('Green' in data_products) or ('Red' in data_products): 
                        primary_header = HeaderParse(self.kpf_object, 'PRIMARY')
                        self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_L1_std_wls", style="Bold", color="Blue")}')
                        self.kpf_object = diagnostics.add_headers_L1_std_wls(self.kpf_object, logger=self.logger)
                        exit_code = 1
                    else: 
                        self.logger.info("Green/Red not in L1 file. Stdev of WLS - WLS_ref diagnostics not computed.")

                except Exception as e:
                    self.logger.error(f"Measuring stdev WLS diagnostics failed: {e}\n{traceback.format_exc()}")



        elif 'L2' in self.data_level_str:
            # L2 - Barycentric correction
            if (self.diagnostics_name == 'all') or \
               (self.diagnostics_name == 'add_headers_L2_barycentric'):
                try:
                    data_products = get_data_products_L2(self.kpf_object )
                    if ('Green' in data_products) or ('Red' in data_products): 
                        if True:
                            self.logger.info(f'{styled_text("Diagnostics:", style="Bold", color="Magenta")} {styled_text("add_headers_L2_barycentric", style="Bold", color="Blue")}')
                            self.kpf_object = diagnostics.add_headers_L2_barycentric(self.kpf_object, logger=self.logger)
                            exit_code = 1
                        else: 
                            self.logger.info("L2 BCV and BJD diagnostics not computed.")
                    else: 
                        self.logger.info("Green/Red not in L2 file. BCV and BJD diagnostics not computed.")
                except Exception as e:
                    self.logger.error(f"Measuring L2 BCV/BJD failed: {e}\n{traceback.format_exc()}")
            
        # Add RECEIPT entry
        self.kpf_object.receipt_add_entry('Diagnostics', self.__module__, f'data_level_str={self.data_level_str}, diagnostics_name={self.diagnostics_name}', 'PASS')
            
        # Finish
        self.logger.info('Finished {}'.format(self.__class__.__name__))
        return Arguments([exit_code, self.kpf_object])
