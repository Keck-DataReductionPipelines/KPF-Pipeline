# standard dependencies
import configparser
import numpy as np
from astropy import constants as cst, units as u

# pipeline dependencies
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.logger import start_logger

# external dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# local dependencies
from modules.wavelength_cal.src.alg import WaveCalibration, calcdrift_polysolution

# global read-only variables
DEFAULT_CFG_PATH = 'modules/wavelength_cal/configs/default.cfg'

class WaveCalibrate(KPF1_Primitive):
    """
    This module defines class `WaveCalibrate,` which inherits from `KPF1_Primitive` and provides methods 
    to perform the event `wavelength calibration` in the recipe.
    
    Args:
        KPF1_Primitive: Parent class
        action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `WaveCalibrate` event issued in recipe.
        context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `wavelength_cal` module in master config file associated with recipe.
    
    Attributes:
        l1_obj (kpfpipe.models.level1.KPF1): Instance of `KPF1`,  assigned by `actions.args[0]`
        cal_type (kpfpipe.models.level1.KPF1): Instance of `KPF1`,  assigned by `actions.args[1]`
        cal_orderlette_names (kpfpipe.models.level1.KPF1): Instance of `KPF1`,  assigned by `actions.args[2]`
        save_wl_pixel_toggle (kpfpipe.models.level1.KPF1): Instance of `KPF1`,  assigned by `actions.args[3]`
        quicklook (kpfpipe.models.level1.KPF1): Instance of `KPF1`,  assigned by `actions.args[4]`
        data_type (kpfpipe.models.level1.KPF1): Instance of `KPF1`,  assigned by `actions.args[5]`
        output_ext (kpfpipe.models.level1.KPF1): Instance of `KPF1`,  assigned by `actions.args[6]`
        config_path (str): Path of config file for the computation of wavelength_calibration.
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger
        alg (modules.wavelength_cal.src.alg.WaveCalibrate): Instance of `WaveCalibrate,` which has operation codes for wavelength calibration.
    """
    def __init__(self, action:Action, context:ProcessingContext) -> None:
        """
        WaveCalibrate constructor.
        Args:
            action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `WaveCalibrate` event issued in recipe:
                `action.args[0]`(kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing level 1 data
                `action.args[1]`(kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing calibration type
                `action.args[2]`(kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing names of calibration extensions
                `action.args[3]`(kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing bool regarding saving of wavelength-pixel solution
                `action.args[4]`(kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing bool regarding running of quicklook algorithms
                `action.args[5]`(kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing data/instrument type
                `action.args[6]`(kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing name of FITS extension to output result to
            context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `wavelength_cal` module in master config file associated with recipe.
        """ 
        KPF1_Primitive.__init__(self, action, context)
        
        self.l1_obj = self.action.args[0]
        self.cal_type = self.action.args[1]
        self.cal_orderlette_names = self.action.args[2]
        self.save_wl_pixel_toggle = self.action.args[3]
        self.quicklook = self.action.args[4]
        self.data_type =self.action.args[5]
        self.output_ext = self.action.args[6]
        
        args_keys = [item for item in action.args.iter_kw() if item != "name"]
        self.filename = action.args['filename'] if \
            'filename' in args_keys else None
        self.save_diagnostics = action.args['save_diagnostics'] if \
            'save_diagnostics' in args_keys else None
        #getting filename so as to steal its date suffix 
        self.rough_wls = action.args['rough_wls'] if \
            'rough_wls' in args_keys else None
        self.linelist_path = action.args['linelist_path'] if \
            'linelist_path' in args_keys else None
        self.output_dir = action.args['output_dir'] if \
            'output_dir' in args_keys else None
        ## ^ TODO: how will we automate cycling through the most recent files?

        self.f0_key = action.args['f0_key'] if 'f0_key' in args_keys else None
        self.clip_peaks_toggle = action.args['clip_peaks_toggle'] if \
            'clip_peaks_toggle' in args_keys else None
        self.frep_key = action.args['frep_key'] if 'frep_key' in args_keys \
            else None
    
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

        self.alg = WaveCalibration(
            self.cal_type, self.clip_peaks_toggle, self.quicklook,
            self.save_diagnostics, self.config, self.logger
        )

    def _perform(self) -> None: 
        """
        Primitive action - perform wavelength calibration by calling method `wavelength_cal` from WaveCalibrate.
        Depending on the type of calibration, will save/compute some combination of wavelength solution, 
        wavelength-pixel map, instrument drift.
        
        Returns:
            Level 1 Data Object
        """
        if self.cal_type == 'LFC' or 'ThAr' or 'Etalon':
            file_name_split = self.l1_obj.filename.split('_')
            datetime_suffix = file_name_split[-1].split('.')[0]
            for prefix in self.cal_orderlette_names:
                calflux = self.l1_obj[prefix]
                calflux = np.nan_to_num(calflux)
                        
                #### lfc ####
                if self.cal_type == 'LFC':
                    if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith(
                        'LFC'
                    ):
                        pass # TODO: fix
                        # raise ValueError(
                        #     'Not an LFC file! CAL-OBJ is {}.'.format(
                        #         self.l1_obj.header['PRIMARY']['CAL-OBJ']
                        #     )
                        # )
                    
                    if self.logger:
                        self.logger.info(
                            "Wavelength Calibration: Getting comb frequency \
                                values."
                        )

                    if self.f0_key is not None:
                        if type(self.f0_key) == str:
                            comb_f0 = float(
                                self.l1_obj.header['PRIMARY'][self.f0_key]
                            )
                        if type(self.f0_key) == float:
                            comb_f0 = self.f0_key

                    else:
                        raise ValueError('f_0 value not found.')
                    
                    if self.frep_key is not None:
                        if type(self.frep_key) == str:
                            comb_fr = float(
                                self.l1_obj.header['PRIMARY'][self.frep_key]
                            )
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
                                        
                    wl_soln, wls_and_pixels = self.alg.run_wavelength_cal(
                        calflux, peak_wavelengths_ang=peak_wavelengths_ang,
                        rough_wls=self.rough_wls, 
                        lfc_allowed_wls=lfc_allowed_wls
                    )
                    
                    if self.save_wl_pixel_toggle == True:
                        file_name = self.output_dir + self.cal_type + '_' + \
                            datetime_suffix + '.npy'
                        self.alg.save_wl_pixel_info(file_name,wls_and_pixels)
                        
                    self.l1_obj[self.output_ext] = wl_soln
                
                #### thar ####    
                elif self.cal_type == 'ThAr':
                    if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith(
                        'ThAr'
                    ):
                        pass # TODO: fix
                        # raise ValueError('Not a ThAr file!')
                    
                    if self.linelist_path is not None:
                        peak_wavelengths_ang = np.load(
                            self.linelist_path, allow_pickle=True
                        ).tolist()
                    else:
                        raise ValueError('ThAr run requires linelist_path')
                    
                    wl_soln, wls_and_pixels = self.alg.run_wavelength_cal(
                        calflux,peak_wavelengths_ang=peak_wavelengths_ang, 
                        rough_wls=self.rough_wls
                    )
                    
                    if self.save_wl_pixel_toggle == True:
                        file_name = self.output_dir + self.cal_type + '_' + \
                            datetime_suffix + '.npy'
                        wl_pixel_filename = self.alg.save_wl_pixel_info(
                            file_name, wls_and_pixels
                        )

                    self.l1_obj[self.output_ext] = wl_soln

                #### etalon ####    
                elif self.cal_type == 'Etalon':
                    if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith(
                        'Etalon'
                    ):
                        raise ValueError('Not an Etalon file!')
                    
                    if self.linelist_path is not None:
                        peak_wavelengths_ang = np.load(
                            self.linelist_path, allow_pickle=True
                        ).tolist()
                    else:
                        peak_wavelengths_ang = None

                    _, wls_and_pixels = self.alg.run_wavelength_cal(
                        calflux, self.rough_wls, 
                        peak_wavelengths_ang=peak_wavelengths_ang
                    )

                    if self.save_wl_pixel_toggle == True:
                        file_name = self.output_dir + self.cal_type + '_' + \
                            datetime_suffix + '.npy'
                        wl_pixel_filename = self.alg.save_wl_pixel_info(
                            file_name, wls_and_pixels
                        )
                
                    # if we've just got one etalon frame, the wl solution
                    # that should be assigned to the file is the master (usually 
                    # LFC) solution
                    wl_soln = self.rough_wls
                    if peak_wavelengths_ang is not None:

                        # calculate drift [cm/s]
                        drift_all_orders = calcdrift_polysolution(
                            self.linelist_path, file_name
                        )
                        avg_drift = np.mean(drift_all_orders[:,1])

                        # convert drift to angstroms
                        beta = avg_drift / cst.c.to(u.cm/u.s).value
                        delta_lambda_over_lambda = -1 + np.sqrt(
                            (1 + beta)/ (1 - beta)
                        )
                        delta_lambda = delta_lambda_over_lambda * wl_soln

                        # update wls using calculated average drift
                        wl_soln = wl_soln + delta_lambda

                    self.l1_obj[self.output_ext] = wl_soln

        else:
            raise ValueError(
                'cal_type {} not recognized. Available options are LFC, ThAr, \
                & Etalon'.format(self.cal_type)
            )
                            
        return Arguments(self.l1_obj)
            ## where to save final polynomial solution
            
                
        
