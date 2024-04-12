# standard dependencies
import os
import fnmatch
import configparser
import numpy as np
import pandas as pd
from astropy import constants as cst, units as u
import datetime
from modules.quicklook.src.analyze_wls import write_wls_json

# pipeline dependencies
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.logger import start_logger
from kpfpipe.models.level1 import KPF1

# external dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# local dependencies
from modules.wavelength_cal.src.alg import WaveCalibration, WaveInterpolation, calcdrift_polysolution

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
        cal_orderlet_names (kpfpipe.models.level1.KPF1): Instance of `KPF1`,  assigned by `actions.args[2]`
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
                `action.args[5]`(kpfpipe.models.level1.KPF1)`: minimum order to fit
                `action.args[6]`(kpfpipe.models.level1.KPF1)`: maximum order to fit
                `action.args[7]`(kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing data/instrument type
                `action.args[8]`(kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing name of FITS extension to output result to
            context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `wavelength_cal` module in master config file associated with recipe.
        """ 
        KPF1_Primitive.__init__(self, action, context)
        
        self.l1_obj = self.action.args[0]
        self.cal_type = self.action.args[1]
        self.cal_orderlet_names = self.action.args[2]
        self.save_wl_pixel_toggle = self.action.args[3]
        self.quicklook = self.action.args[4]
        self.min_order = self.action.args[5]
        self.max_order = self.action.args[6]
        self.data_type =self.action.args[7]
        self.output_ext = self.action.args[8]
        args_keys = [item for item in action.args.iter_kw() if item != "name"]
        self.filename = action.args['filename'] if \
            'filename' in args_keys else None
        self.save_diagnostics = action.args['save_diagnostics'] if \
            'save_diagnostics' in args_keys else None
        self.json_filename = action.args['json_filename'] if \
            'json_filename' in args_keys else None
        #getting filename so as to steal its date suffix 
        self.rough_wls = action.args['rough_wls'] if \
            'rough_wls' in args_keys else None
        self.linelist_path = action.args['linelist_path'] if \
            'linelist_path' in args_keys else None
        self.output_dir = action.args['output_dir'] if \
            'output_dir' in args_keys else None
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

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
            self.min_order, self.max_order, self.save_diagnostics, self.config, self.logger
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
            self.file_name_split = self.l1_obj.filename.split('_')[0]
            self.file_name = self.l1_obj.filename.split('.')[0]

            # Create dictionary for storing information about the WLS, fits of each order, fits of each line, etc.
            self.wls_dict = {
                #"name" = 'KP.20230829.12345.67_WLS', # unique string to identify WLS
                'wls_processing_date' : str(datetime.datetime.now()), # data and time that this dictionary was created
                'cal_type' : self.cal_type, # LFC, etalon, ThAr, UNe
                'orderlets' : {} #one orderlet_dict for each combination of chip and orderlet, e.g. 'RED_SCI1'
            }

            for i, prefix in enumerate(self.cal_orderlet_names):
                print('\nCalibrating orderlet {}.'.format(prefix))
                
                # Create a dictionary for each orderlet that will be filled in later
                full_name = prefix.replace('_FLUX', '') # like GREEN_SCI1
                orderlet_name = full_name.split('_')[1]
                chip_name = prefix.split('_')[0]
                self.wls_dict['chip'] = chip_name
                self.wls_dict['orderlets'][orderlet_name] = {
                    'full_name' : full_name, # e.g., RED_SCI1
                    'orderlet' : orderlet_name, # SCI1, SCI2, SCI3, SKY, CAL
                    'chip' : chip_name, # GREEN or RED
                }

                if self.save_diagnostics is not None:
                    self.alg.save_diagnostics_dir = '{}/{}/'.format(self.save_diagnostics, prefix)

                output_ext = self.output_ext[i]
                calflux = self.l1_obj[prefix]
                calflux = np.nan_to_num(calflux)
                        
                #### lfc ####
                if self.cal_type == 'LFC':
                    line_list, wl_soln, orderlet_dict = self.calibrate_lfc(calflux, output_ext=output_ext)
                    # self.drift_correction(prefix, line_list, wl_soln)
                    self.wls_dict['orderlets'][orderlet_name]['norders'] = self.max_order-self.min_order+1
                    self.wls_dict['orderlets'][orderlet_name]['orders'] = orderlet_dict
 
                #### thar ####    
                elif self.cal_type == 'ThAr':
                    if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('ThAr'):
                        pass # TODO: fix
                        # raise ValueError('Not a ThAr file!')
                    
                    if self.linelist_path is not None:
                        peak_wavelengths_ang = pd.read_csv(self.linelist_path,
                                                           header=None,
                                                           names=['wave', 'weight'],
                                                           delim_whitespace=True)
                        peak_wavelengths_ang = peak_wavelengths_ang.query('weight == 1')
                    else:
                        raise ValueError('ThAr run requires linelist_path')

                    wl_soln, wls_and_pixels, orderlet_dict = self.alg.run_wavelength_cal(
                        calflux,peak_wavelengths_ang=peak_wavelengths_ang, 
                        rough_wls=self.rough_wls
                    )
                    
                    if self.save_wl_pixel_toggle == True:
                        wlpixelwavedir = self.output_dir + '/wlpixelfiles/'
                        if not os.path.exists(wlpixelwavedir):
                            os.mkdir(wlpixelwavedir)
                        file_name = wlpixelwavedir + self.cal_type + 'lines_' + self.file_name + "_" + '{}.npy'.format(prefix)
                        wl_pixel_filename = self.alg.save_wl_pixel_info(file_name, wls_and_pixels)

                    self.l1_obj[output_ext] = wl_soln
                    self.wls_dict['orderlets'][orderlet_name]['norders'] = self.max_order-self.min_order+1
                    self.wls_dict['orderlets'][orderlet_name]['orders'] = orderlet_dict

                #### etalon ####    
                elif self.cal_type == 'Etalon':
                    if 'Etalon' not in self.l1_obj.header['PRIMARY']['CAL-OBJ']:
                        raise ValueError('Not an Etalon file!')
                    
                    peak_wavelengths_ang = None
                    _, wls_and_pixels, orderlet_dict = self.alg.run_wavelength_cal(
                        calflux, self.rough_wls, 
                        peak_wavelengths_ang=peak_wavelengths_ang,
                        input_filename=self.l1_obj.filename
                    )

                    if self.save_wl_pixel_toggle == True:
                        wlpixelwavedir = self.output_dir + '/wlpixelfiles/'
                        if not os.path.exists(wlpixelwavedir):
                            os.mkdir(wlpixelwavedir)
                        file_name = wlpixelwavedir + self.cal_type + 'lines_' + \
                            self.file_name + "_" + '{}.npy'.format(prefix)
                        wl_pixel_filename = self.alg.save_wl_pixel_info(
                            file_name, wls_and_pixels
                        )
                        # Save updated mask positions, start with testdir, same as above
                        maskdir = self.output_dir+ '/wlpixelfiles/'
                        filename = maskdir + self.cal_type + 'mask_'+self.cal_orderlet_names[0]+'_' + self.file_name + ".csv"
                        self.alg.save_etalon_mask_update(filename,wls_and_pixels)

                
                    # if we've just got one etalon frame, the wl solution that should be
                    # assigned to the file is the master (usually LFC) solution
                    wl_soln = self.rough_wls
                    if peak_wavelengths_ang is not None:

                        # calculate drift [cm/s]
                        drift_all_orders = calcdrift_polysolution(self.linelist_path, file_name)
                        avg_drift = np.mean(drift_all_orders[:,1])

                        # convert drift to angstroms
                        beta = avg_drift / cst.c.to(u.cm/u.s).value
                        delta_lambda_over_lambda = -1 + np.sqrt((1 + beta)/ (1 - beta))
                        delta_lambda = delta_lambda_over_lambda * wl_soln

                        # update wls using calculated average drift
                        wl_soln = wl_soln + delta_lambda

                    self.l1_obj[output_ext] = wl_soln
                    # The two lines haven't been tested yet - AWH oct 19 2023
                    try:
                        self.wls_dict['orderlets'][orderlet_name]['norders'] = self.max_order-self.min_order+1
                        self.wls_dict['orderlets'][orderlet_name]['orders'] = orderlet_dict
                    except Exception as e:
                        print('wls_dist re: etalon did not work.  It is untested.')
                        print(e)
  
            # Save WLS dictionary as a JSON file 
            if self.json_filename != None:
                print('*******************************************')
                print('Saving JSON file with WLS fit information: ' +  self.json_filename)
                json_dir = os.path.dirname(self.json_filename)
                if not os.path.isdir(json_dir):
                    os.makedirs(json_dir)
                write_wls_json(self.wls_dict, self.json_filename)

        else:
            raise ValueError('cal_type {} not recognized. Available options are LFC, ThAr, & Etalon'.format(self.cal_type))
                            
        return Arguments(self.l1_obj)
            ## where to save final polynomial solution
            
        
    def calibrate_lfc(self, calflux, output_ext=None):
        if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('LFC'):
            pass # TODO: fix
            # raise ValueError('Not an LFC file! CAL-OBJ is {}.'.format(self.l1_obj.header['PRIMARY']['CAL-OBJ']))
        
        if self.logger:
            self.logger.info("Wavelength Calibration: Getting comb frequency values.")

        if self.f0_key is not None:
            if type(self.f0_key) == str:
                comb_f0 = float(self.l1_obj.header['PRIMARY'][self.f0_key])
            if type(self.f0_key) == float or type(self.f0_key) == int:
                comb_f0 = self.f0_key

        else:
            raise ValueError('f_0 value not found.')
        
        if self.frep_key is not None:
            if type(self.frep_key) == str:
                comb_fr = float(self.l1_obj.header['PRIMARY'][self.frep_key])
            if type(self.frep_key) == float or type(self.frep_key) == int:
                comb_fr = self.frep_key
        else:
            raise ValueError('f_rep value not found')
    
        peak_wavelengths_ang = None

        lfc_allowed_wls = self.alg.comb_gen(comb_f0, comb_fr)

        wl_soln, wls_and_pixels, orderlet_dict = self.alg.run_wavelength_cal(
            calflux, peak_wavelengths_ang=peak_wavelengths_ang,
            rough_wls=self.rough_wls, 
            lfc_allowed_wls=lfc_allowed_wls
        )
        
        if self.save_wl_pixel_toggle == True:
            wlpixelwavedir = self.output_dir + '/wlpixelfiles/'
            if not os.path.exists(wlpixelwavedir):
                os.mkdir(wlpixelwavedir)
            file_name = wlpixelwavedir + self.cal_type + 'lines_' + \
                self.file_name + "_" + '{}.npy'.format(output_ext)
            self.alg.save_wl_pixel_info(file_name,wls_and_pixels)
        else:
            file_name = None
            
        if output_ext != None:
            self.l1_obj[output_ext] = wl_soln

        return (file_name, wl_soln, orderlet_dict)
    
    def drift_correction(self, orderlet, line_list, wl_soln):

        calflux = self.l1_obj[orderlet]
        output_ext = orderlet.replace('FLUX', 'WAVE')
        file_name, _ = self.calibrate_lfc(calflux)

        drift_all_orders = calcdrift_polysolution(line_list, file_name)
        avg_drift = np.nanmedian(drift_all_orders[:,1])
        print("Average drift for {} = {:.2f} cm/s".format(orderlet, avg_drift))

        # convert drift to angstroms
        beta = avg_drift / cst.c.to(u.cm/u.s).value
        delta_lambda_over_lambda = -1 + np.sqrt((1 + beta)/ (1 - beta))
        delta_lambda = delta_lambda_over_lambda * wl_soln

        # update wls using calculated average drift
        wl_soln = wl_soln - delta_lambda

        self.l1_obj[output_ext] = wl_soln


class WaveInterpolate(KPF1_Primitive):
    """
    This module defines class `WaveInterpolate,` which inherits from `KPF1_Primitive` and provides methods 
    to interpolate a wavelength solution between two bracketing calibrations in the recipe.
    
    Args:
        KPF1_Primitive: Parent class
        action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `WaveInterpolate` event issued in recipe.
        context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `wavelength_cal` module in master config file associated with recipe.
    
    Attributes:
        l1_obj (kpfpipe.models.level1.KPF1): Instance of `KPF1`. Interpolated WLS will be injected into this object assigned by `actions.args[0]`
        wls1_filename (string): Input WLS prior to the observation, assigned by `actions.args[1]`
        wls2_filename (string): Input WLS after the observation, assigned by `actions.args[2]`
        wls_extensions (list): List of the WLS extensions, assigned by `actions.args[3]`
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger
        alg (modules.wavelength_cal.src.alg.WaveInterpolate): Instance of `WaveInterpolate,` which has operation codes for wavelength interpolation.
    """
    def __init__(self, action:Action, context:ProcessingContext) -> None:
        """
        WaveInterpolate constructor.
        """ 
        KPF1_Primitive.__init__(self, action, context)
        
        self.l1_wls1 = self.action.args[0]
        self.l1_wls2 = self.action.args[1]
        self.l1_interp = self.action.args[2]
        self.config=configparser.ConfigParser()

        try:
            config_path=context.config_path['wavelength_interpolate']
        except:
            config_path = DEFAULT_CFG_PATH
        self.config.read(config_path)

        #Start logger
        self.logger=start_logger(self.__class__.__name__,config_path)
        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(config_path))

        l1_timestamp = self.l1_interp.header['PRIMARY']['DATE-BEG']
        wls1_timestamp = self.l1_wls1.header['PRIMARY']['DATE-BEG']
        wls2_timestamp = self.l1_wls2.header['PRIMARY']['DATE-BEG']
        wls_timestamps = [wls1_timestamp, wls2_timestamp]

        wls_extensions = []
        for name in self.l1_interp.extensions.keys():
            if fnmatch.fnmatch(name, 'GREEN*WAVE*') or fnmatch.fnmatch(name, 'RED*WAVE*'):
                wls_extensions.append(name)
        wls1_arrays = {}
        wls2_arrays = {}
        for ext in wls_extensions:
            wls1_arrays[ext] = self.l1_wls1[ext]
            wls2_arrays[ext] = self.l1_wls2[ext]

        self.alg = WaveInterpolation(l1_timestamp, wls_timestamps, wls1_arrays, wls2_arrays)

    def _perform(self) -> None: 
        """
        Primitive action - perform wavelength interpolation by calling method `wavelength_interpolate` from WaveInterpolate.
        This method will update the input L1 object with the interpolated wavelength solition
        
        Returns:
            Level 1 Data Object
        """

        new_wls_arrays = self.alg.wave_interpolation(method='linear') 
        for ext, wls in new_wls_arrays.items():
            self.l1_interp[ext] = new_wls_arrays[ext]
                            
        return Arguments(self.l1_interp)
            
