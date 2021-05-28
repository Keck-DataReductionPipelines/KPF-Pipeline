#packages
import numpy as np

from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments
from modules.Utils.overscan_subtract import OverscanSubtraction as osub

class BiasSubtraction:
    """
    Bias subtraction calculation.

    This module defines 'BiasSubtraction' and methods to perform bias subtraction by subtracting a master bias frame from the raw data frame.  

    Args:
        rawimage (np.ndarray): The FITS raw data with image extensions
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger.
    
    Attributes:
        rawimage (np.ndarray): From parameter 'rawimage'.
    
    Raises:
        Exception: If raw image and bias frame don't have the same dimensions
    """


    def __init__(self,config=None, logger=None):
        """Inits BiasSubtraction class with raw data, config, logger.

        Args:
            config (configparser.ConfigParser, optional): Config context. Defaults to None.
            logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.
        """
        self.config=config
        self.logger=logger

        configpull = ConfigHandler(config,'PARAM')
        self.ffi_exts = configpull.get_config_value('ffi_exts', [6,12])
        # self.mode = configpull.get_config_value('overscan_mode', 1)
        # self.overscan_pixels = configpull.get_config_value('overscan_pixels', 160)
        # self.prescan_pixels = configpull.get_config_value('prescan_pixels', 0)
        # self.paralscan_pixels = configpull.get_config_value('parallelscan_pixels',0)
        
    def get_ffi_exts(self):
        return self.ffi_exts

    def bias_subtraction(self,frame,masterbias):
        """
            Subtracts bias data from raw data.
            In pipeline terms: inputs two L0 files, produces one L0 file. 

        Args:
            frame (np.ndarray): The raw, assembled FFI
            masterbias (np.ndarray): The master bias data.

        Raises:
            Exception: If raw image and bias frame don't have the same dimensions.
        """
        if frame.data.shape==masterbias.data.shape:
            frame.data = frame.data-masterbias.data
        else:
            raise Exception("Bias .fits Dimensions NOT Equal! Check Failed")
    
        raw_sub_bias = frame

        return raw_sub_bias

    # def get(self):
    #     """Returns bias-corrected raw image result.

    #     Returns:
    #         self.rawimage: The bias-corrected data
    #     """
    #     return self.rawimage
