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
        self.save_wl_pixel_toggle = self.action.args[3]
        self.quicklook = self.action.args[4]
        self.data_type =self.action.args[5]
        self.output_ext = self.action.args[6]
        
        args_keys = [item for item in action.args.iter_kw() if item != "name"]
        self.filename = action.args['filename'] if 'filename' in args_keys else None
        #getting filename so as to steal its date suffix 
        self.rough_wls = action.args['rough_wls'] if 'rough_wls' in args_keys else None
        self.linelist_path = action.args['linelist_path'] if 'linelist_path' in args_keys else None
        self.f0_key = action.args['f0_key'] if 'f0_key' in args_keys else None
        self.clip_peaks_toggle = action.args['clip_peaks_toggle'] if 'clip_peaks_toggle' in args_keys else None
        self.frep_key = action.args['frep_key'] if 'frep_key' in args_keys else None
        self.prev_wl_pixel_ref = action.args['prev_wl_pixel_ref'] if 'prev_wl_pixel_ref' in args_keys else None
        ## ^ how will we automate cycling through the most recent files?
    
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

        self.alg = WaveCalibration(self.cal_type,self.clip_peaks_toggle,self.quicklook,self.config,self.logger)

    def _perform(self) -> None: 
        
        if self.cal_type == 'LFC' or 'ThAr' or 'Etalon':
            file_name_split = self.filename.split('_')
            datetime_suffix = file_name_split[-1].split('.')[0]
            for prefix in self.cal_orderlette_names:
                calflux = self.l1_obj[prefix]
                calflux = np.nan_to_num(calflux)
            
                #PUT BACK: rough_wls = self.master_wavelength['SCIWAVE'] ### from fits in recipe, check this
                rough_wls = self.l1_obj['SCIWAVE']   
                        
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
                    
                    if self.linelist_path is not None:
                        peak_wavelengths_ang = np.load(
                            self.linelist_path, allow_pickle=True
                        ).tolist()
                    else:
                        peak_wavelengths_ang = None
                    
                    lfc_allowed_wls = self.alg.comb_gen(comb_f0, comb_fr)
                    
                    #rough_wls = self.master_wavelength['SCIWAVE'] #TODO ### check this
                    
                    wl_soln, wls_and_pixels = self.alg.run_wavelength_cal(
                        calflux,peak_wavelengths_ang=peak_wavelengths_ang,rough_wls=rough_wls,lfc_allowed_wls=lfc_allowed_wls)
                    
                    if self.save_wl_pixel_toggle == True:
                        file_suffix = self.cal_type + '_' + datetime_suffix + '.npy'
                        wl_pixel_filename = self.alg.save_wl_pixel_info(file_suffix,wls_and_pixels)
                        
                #TODO: should peak wavelengths ang be in all of them?
                #### thar ####    
                elif self.cal_type == 'ThAr':
                    if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('ThAr'):
                        raise ValueError('Not a ThAr file!')
                    
                    if self.linelist_path is not None:
                        peak_wavelengths_ang = np.load(
                            self.linelist_path, allow_pickle=True
                        ).tolist()
                    else:
                        peak_wavelengths_ang = None
                    
                    wl_soln, wls_and_pixels = self.alg.run_wavelength_cal(
                        calflux,peak_wavelengths_ang=peak_wavelengths_ang)
                    
                    if self.save_wl_pixel_toggle == True:
                        file_suffix = self.cal_type + '_' + datetime_suffix + '.npy'
                        wl_pixel_filename = self.alg.save_wl_pixel_info(file_suffix,wls_and_pixels)
                    
                #### etalon ####    
                elif self.cal_type == 'Etalon':
                    if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('Etalon'):
                        raise ValueError('Not an Etalon file!')
                    
                    #rough_wls = self.master_wavelength['SCIWAVE'] ### TODO: from fits in recipe, check this

                    if self.linelist_path is not None:
                        peak_wavelengths_ang = np.load(
                            self.linelist_path, allow_pickle=True
                        ).tolist()
                    else:
                        peak_wavelengths_ang = None

                    wl_soln,wls_and_pixels = self.alg.run_wavelength_cal(
                        calflux,rough_wls,peak_wavelengths_ang)

                    if self.save_wl_pixel_toggle == True:
                        file_suffix = self.cal_type + '_' + datetime_suffix + '.npy'
                        wl_pixel_filename = self.alg.save_wl_pixel_info(file_suffix,wls_and_pixels)
                else:
                    raise ValueError(
                        'cal_type {} not recognized. Available options are LFC, ThAr, & Etalon'.format(
                            self.cal_type))
                
            ## need to save data into correct extension
            
                if self.prev_wl_pixel_ref is not None:
                    self.alg.plot_drift(self.prev_wl_pixel_ref, wl_pixel_filename)
            
            ## where to save final polynomial solution
            
                
        