# standard dependencies
import configparser
import numpy as np

# pipeline dependencies
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.logger import start_logger

# external dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# local dependencies
from modules.wavelength_cal.src.alg import WaveCalibration

# global read-only variables
DEFAULT_CFG_PATH = 'modules/wavelength_cal/configs/default.cfg'

class WaveCalibrate(KPF1_Primitive):
    
    def __init__(self, action:Action, context:ProcessingContext) -> None:
        KPF1_Primitive.__init__(self, action, context)
        
        self.l1_obj = self.action.args[0]
        self.cal_type = self.action.args[1]
        self.cal_orderlette_names = self.action.args[2]
        self.linelist_path = self.action.args[3]
        self.save_wl_pixel_toggle = self.action.args[4]
        self.quicklook = self.action.args[5]
        self.data_type =self.action.args[6]

        #Input configuration
        self.config=configparser.ConfigParser()
        try:
            config_path=context.config_path['wavelength_cal']
        except:
            config_path = DEFAULT_CFG_PATH
        self.config.read(config_path)

        #Start logger
        self.logger=start_logger(self.__class__.__name__,config_path)
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(config_path))

        self.alg = WaveCalibration(self.save_wl_pixel_toggle,self.quicklook,self.self.config, self.logger)

    def _perform(self) -> None: 
        
        if self.cal_type == 'LFC' or 'ThAr' or 'Etalon':
            for prefix in self.cal_orderlette_names:
                calflux = self.l1_obj[prefix]
                calflux = np.nan_to_num(calflux)
            
                rough_wls = self.master_wavelength['SCIWAVE'] ### from fits in recipe, check this
            
                if self.linelist_path is not None:
                    peak_wavelengths_ang = np.load(
                        self.linelist_path, allow_pickle=True
                    ).tolist()
                else:
                    peak_wavelengths_ang = None
            
                #### lfc ####
                if self.cal_type == 'LFC':
                    if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('LFC'):
                        raise ValueError('Not an LFC file!')
                    
                    if self.logger:
                        self.logger.info("Wavelength Calibration: Getting comb frequency values.")

                    if self.f0_key is not None:
                        if type(self.f0_key) == str:
                            comb_f0 = float(self.l1_obj.header['PRIMARY'][self.f0_key])
                        if type(self.f0_key) == float:
                            comb_f0 = self.f0_key
                    else:
                        raise ValueError('f_0 value not found.')
                    
                    if self.frep_key is not None:
                        if type(self.frep_key) == str:
                            comb_fr = float(self.l1_obj.header['PRIMARY'][self.frep_key])
                        if type(self.frep_key) == float:
                            comb_fr = self.frep_key
                    else:
                        raise ValueError('f_rep value not found')
                    
                    lfc_allowed_wls = self.alg.comb_gen(comb_f0, comb_fr)
                    
                    wl_soln, wls_and_pixels = self.alg.run_wavelength_cal(
                        calflux,self.cal_type,rough_wls,peak_wavelengths_ang,lfc_allowed_wls)
                
                #### thar ####    
                elif self.cal_type == 'ThAr':
                    if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('ThAr'):
                        raise ValueError('Not a ThAr file!')
                    
                    wl_soln, wls_and_pixels = self.alg.run_wavelength_cal(
                        calflux,self.cal_type,rough_wls,peak_wavelengths_ang)
                    
                #### etalon ####    
                elif self.cal_type == 'Etalon':
                    if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('Etalon'):
                        raise ValueError('Not an Etalon file!')

                    wl_soln,wls_and_pixels = self.alg.run_wavelength_cal(
                        calflux,self.cal_type,rough_wls,peak_wavelengths_ang)
                
                
                elif self.cal_type == 'Drift':
                    self.alg.plot_drift(wl_file1,wl_file2)
                    
                else:
                    raise ValueError(
                        'cal_type {} not recognized. Available options are LFC, ThAr, & Etalon'.format(
                            self.cal_type))
                
        