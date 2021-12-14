# Standard dependencies
import configparser
import numpy as np

# Pipeline dependencies
from kpfpipe.primitives.level1 import KPF1_Primitive

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from modules.wavelength_cal.src.alg import LFCWaveCalibration

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/wavelength_cal/configs/LFC_NEID.cfg'

class WaveCalibrate(KPF1_Primitive):
    """
    This module defines class `WaveCalibrate,` which inherits from KPF1_Primitive and provides methods
    to perform the event `LFC wavelength calibration` in the recipe.

    Args:
        KPF1_Primitive: Parent class
        action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `LFCWaveCalibration` event issued in recipe.
        context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `wavelength_cal` module in master config file associated with recipe.

    Attributes:
        l1_obj (kpfpipe.models.level1.KPF1): Instance of `KPF1`, assigned by `actions.args[0]`
        master_wavelength (kpfpipe.models.level1.KPF1): Instance of `KPF1`, assigned by `actions.args[1]`
        data_type (kpfpipe.models.level1.KPF1): Instance of `KPF1`,  assigned by `actions.args[2]`
        config_path (str): Path of config file for LFC wavelength calibration.
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger
        alg (modules.wavelength_cal.src.alg.LFCWaveCalibration): Instance of `LFCWaveCalibration,` which has operation codes for LFC Wavelength Calibration.
    """

    default_args_val = {
            'data_type': 'KPF'
        }

    def __init__(self, 
                action:Action,
                context:ProcessingContext) -> None:
        """
        WaveCalibrate constructor.

        Args:
            action (Action): Contains positional arguments and keyword arguments passed by the `LFCWaveCal` event issued in recipe:
              
                `action.args[0] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing level 1 file
                `action.args[1] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing master file
                `action.args[2] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing data type

            context (ProcessingContext): Contains path of config file defined for `wavelength_cal` module in master config file associated with recipe.
        """

        KPF1_Primitive.__init__(self, action, context)

        def get_args_value(key: str, args: Arguments, args_keys: list):
            v = None
            if key in args_keys:
                v = args[key]
            elif key in self.default_args_val.keys():
                v = self.default_args_val[key]
            return v

        # input arguments
        args_keys = [item for item in action.args.iter_kw() if item != "name"]

        self.l1_obj = self.action.args[0]
        self.quicklook = self.action.args[1]

        # look for and set optional keywords needed for LFC
        self.f0_key = get_args_value('f0', action.args, args_keys)
        self.frep_key = get_args_value('fr', action.args, args_keys)

        # look for and set other optional keywords
        self.master_wavelength = get_args_value('master_wavelength', action.args, args_keys)
        self.peak_wavelength_data = get_args_value('peak_wavelength_data', action.args, args_keys)
        self.data_type = get_args_value('data_type', action.args, args_keys)

        # possible keyword configurations:
        # LFC: self.master_wavelength always set (lamp solution)
        #  1) self.peak_wavelength_data set (expected mode nums & pixels)
        #  2) self.peak_wavelength_data NOT set (need to refind peaks)
        # ThAr/other lamp:
        #  1) self.master_wavelength NOT set,
        #     self.peak_wavelength_data set (expected line wavelengths & pixels)
        # Etalon: self.master_wavelength always set (lamp solution OR LFC solution)
        #  1) self.peak_wavelength_data set (expected peak numbers ("modes") & pixels)
        #  2) self.peak_wavelength_data NOT set (need to refind peaks)

        # input configuration
        self.config=configparser.ConfigParser()
        try:
            self.config_path=context.config_path['wavelength_cal']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        # start logger
        self.logger=None

        if not self.logger:
            self.logger=self.context.logger
        self.logger.info('Loading config from: {}'.format(self.config_path))

        # wavelength calibration algorithm setup
        self.alg = LFCWaveCalibration(self.config, self.logger)

        # preconditions
       
        # postconditions
        
    def _perform(self) -> None:
        """ Primitive action.

        Performs wavelength calibration by calling method 'run_wave_cal' 
        from alg.py, and saves result in .fits extensions.

        Returns:
            Level 1 data, containing wavelength-per-pixel result.
        """

        # extract master data
        if self.logger:
            self.logger.info("Wavelength Calibration: Extracting master data.")  

        master_data = self.alg.get_master_data(self.master_wavelength)

        # check that we have an image containing the matching calibration type
        if self.alg.config_type == 'LFC':
            if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('LFC'):
                raise ValueError('Not an LFC file!')
        elif self.alg.config_type == 'ThAr':
            if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('ThAr'):
                raise ValueError('Not a ThAr file!')
        elif self.alg.config_type == 'Etalon':
            if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('Etalon'):
                raise ValueError('Not an Etalon file!')
        else:
            raise ValueError(
                'config_type {} not recognized. Available options are LFC, ThAr, and Etalon.'.format(
                    self.alg.config_type
                )
            )

        # get comb frequency values if LFC
        if self.alg.config_type == 'LFC':
            
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

        if self.logger:
            self.logger.info("Wavelength Calibration: Starting wavelength calibration loop")
        
        for prefix in ['CALFLUX']: #change to recipe config: 'orderlette_names' 

            if self.l1_obj[prefix] is not None:
                self.logger.info("Wavelength Calibration: Running {prefix}")
                if self.logger:
                    self.logger.info("Wavelength Calibration: Extracting flux")
                
                flux = self.l1_obj[prefix]

                flux = np.nan_to_num(flux)
                if self.logger:
                    self.logger.info("Wavelength Calibration: Running algorithm")  

                wl_soln = self.alg.open_and_run(
                    flux, self.alg.config_type, master_data=master_data, f0=comb_f0, 
                    f_rep=comb_fr, quicklook=self.quicklook
                )

                if self.logger:
                    self.logger.info("Wavelength Calibration: Saving solution output")  
                self.l1_obj['CALWAVE'] = wl_soln

        if self.l1_obj is not None:
            self.l1_obj.receipt_add_entry(
                'Wavelength Calibration', self.__module__, 
                f'config_path={self.config_path}', 'PASS'
            )
        if self.logger:
            self.logger.info("Wavelength Calibration: Receipt written")

        if self.logger:
            self.logger.info("Wavelength Calibration: Done!")

        return Arguments(self.l1_obj)

