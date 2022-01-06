# standard dependencies
import configparser
import numpy as np

# pipeline dependencies
from kpfpipe.primitives.level1 import KPF1_Primitive

# external dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# local dependencies
from modules.wavelength_cal.src.alg import WaveCalibration

# global read-only variables
DEFAULT_CFG_PATH = 'modules/wavelength_cal/configs/LFC_NEID.cfg'

class WaveCalibrate(KPF1_Primitive):
    """
    This module defines class `WaveCalibrate`, which inherits from KPF1_Primitive 
    and provides methods to perform the event `WaveCalibration` in 
    the recipe.

    Args:
        KPF1_Primitive: Parent class
        action (keckdrpframework.models.action.Action): contains positional 
            arguments and keyword arguments passed by the `WaveCalibration` 
            event issued in recipe.
        context (keckdrpframework.models.processing_context.ProcessingContext): 
            contains path of config file defined for `wavelength_cal` module in 
            master config file associated with recipe.

    Attributes:
        l1_obj (kpfpipe.models.level1.KPF1): instance of `KPF1`, assigned by 
            `actions.args[0]`
        quicklook (bool): whether to run quicklook pipeline, assigned by 
            `actions.args[1]`
        f0_key (str or float): if float, the initial frequency of the LFC
            in Hz. If string, the fits header keyword used to look up this value.
            Default None.
        frep_key (str or float): if float, the repetition frequency of the LFC
            in Hz. If string, the fits header keyword used to look up this value.
            Default None.
        master_wavelength (np.array): (N_orders x N_pixels) wavelength solution
            that will be used to derive the wavelength solution of the 
            input frame. Ex: for an LFC frame, this might be a ThAr-derived
            solution.
        data_type (str): 'KPF', 'NEID', etc
        config_path (str): Path of config file for wavelength calibration.
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger
        alg (modules.wavelength_cal.src.alg.WaveCalibration): instance of 
            `WaveCalibration`, which has operation codes for wavelength 
            calibration.
    """

    # default_args_val = {
    #     'data_type': 'KPF'
    # }

    def __init__(self, action:Action, context:ProcessingContext) -> None:
        """
        WaveCalibrate constructor.

        Args:
            action (Action): Contains positional arguments and keyword arguments 
                passed by the `WaveCalibration` event issued in recipe:
              
                `action.args[0] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` 
                    containing level 1 file
                `action.args[1] (str)`: data type

            context (ProcessingContext): Contains path of config file defined for 
                `wavelength_cal` module in master config file associated with recipe.
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
        args_keys = [item for item in self.action.args.iter_kw() if item != "name"]

        self.l1_obj = self.action.args[0]
        self.quicklook = self.action.args[1]
        self.linelist_path = self.action.args[2]
        self.cal_orderlette_name = self.action.args[]
        self.data_type = self.action.args[]

        # look for and set optional keywords needed for LFC
        self.f0_key = get_args_value('f0', self.action.args, args_keys)
        self.frep_key = get_args_value('fr', self.action.args, args_keys)

        # look for and set other optional keywords
        self.master_wavelength = get_args_value('master_wavelength', self.action.args, args_keys)

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
        self.alg = WaveCalibration(self.config, self.logger)

        # preconditions
       
        # postconditions
        
    def _perform(self) -> None:
        """ Primitive action.

        Performs wavelength calibration by calling method `run_wavelength_cal()` 
        from alg.py, and saves result in .fits extensions.

        Returns:
            keckdrpframework.models.arguments.Arguments: a single
                kpfpipe.models.level1.KPF1 object containing 
                wavelength-per-pixel result in the CALWAVE fits header
        """

        # extract master data
        if self.logger:
            self.logger.info("Wavelength Calibration: Extracting master wavelength solution.")  

        rough_wls = self.alg.get_master_data(self.master_wavelength)

        # check that we have an image containing the matching calibration type
        if self.alg.cal_type == 'LFC':
            if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('LFC'):
                raise ValueError('Not an LFC file!')
        elif self.alg.cal_type == 'ThAr':
            if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('ThAr'):
                raise ValueError('Not a ThAr file!')
        elif self.alg.cal_type == 'Etalon':
            if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('Etalon'):
                raise ValueError('Not an Etalon file!')
        else:
            raise ValueError(
                'cal_type {} not recognized. Available options are LFC, ThAr, and Etalon.'.format(
                    self.alg.cal_type
                )
            )

        # get comb frequency values if LFC
        if self.alg.cal_type == 'LFC':
            
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
        else:
            lfc_allowed_wls = None

        if self.logger:
            self.logger.info("Wavelength Calibration: Starting wavelength calibration loop")

        for prefix in self.cal_orderlette_names: 

            if self.l1_obj[prefix] is not None:
                self.logger.info("Wavelength Calibration: Running {prefix}")
                if self.logger:
                    self.logger.info("Wavelength Calibration: Extracting flux")
                
                calflux = self.l1_obj[prefix]

                calflux = np.nan_to_num(calflux)
                if self.logger:
                    self.logger.info("Wavelength Calibration: Running algorithm")  
                

                if self.linelist_path is not None:
                    peak_wavelengths_ang = np.load(
                        self.linelist_path, allow_pickle=True
                    ).tolist()
                else:
                    peak_wavelengths_ang = None

                # perform wavelength calibration
                wl_soln, wls_and_pixels = self.alg.run_wavelength_cal(
                    calflux, self.alg.cal_type, rough_wls = rough_wls, 
                    peak_wavelengths_ang = peak_wavelengths_ang, 
                    lfc_allowed_wls = lfc_allowed_wls, 
                    quicklook = self.quicklook
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

        return Arguments(self.l1_obj, wls_and_pixels)

