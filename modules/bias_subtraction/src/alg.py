#packages
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
###

from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments

class BiasSubtractionAlg:
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


    def __init__(self,rawimage,ffi_exts,quicklook,config=None, logger=None):
        """Inits BiasSubtraction class with raw data, config, logger.

        Args:
            rawimage (np.ndarray): The FITS raw data.
            ffi_exts (np.ndarray): The extensions in L0 FITS files where FFIs (full frame images) are stored.
            config (configparser.ConfigParser, optional): Config context. Defaults to None.
            logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.
        """
        self.rawimage=rawimage
        self.ffi_exts=ffi_exts
        self.quicklook=quicklook
        self.config=config
        self.logger=logger
        #self.imagesize=
        
    def bias_subtraction(self,masterbias):
        """Subtracts bias data from raw data.
        In pipeline terms: inputs two L0 files, produces one L0 file. 

        Args:
            masterbias (np.ndarray): The master bias data.

        Raises:
            Exception: If raw image and bias frame don't have the same dimensions.
        """
        ###for testing purposes###
        #masterbias = fits.open(masterbias)
        #masterbias = masterbias[1].data
        # masterbias = np.zeros_like(frame)
        ###
        if self.quicklook == False: 
            if self.data_type == 'KPF':
                for ffi in self.ffi_exts:
                    print(self.rawimage.info)
                    print(masterbias.info())
                    assert self.rawimage[ffi].data.shape==masterbias[ffi].data.shape, "Bias .fits Dimensions NOT Equal! Check failed"
                    #self.rawimage[ffi].data=self.rawimage[ffi].data-masterbias[ffi].data
                    minus_bias = self.rawimage[ffi].data-masterbias[ffi].data
                    self.rawimage[ffi] = minus_bias
                
            if self.data_type == 'NEID':
                print(self.rawimage.info())
                print('shapes:',self.rawimage['DATA'].shape,masterbias['DATA'].shape)
                assert self.rawimage['DATA'].shape==masterbias['DATA'].shape, "Bias .fits Dimensions NOT Equal! Check failed"
                minus_bias=self.rawimage['DATA']-masterbias['DATA']
                self.rawimage['DATA'] = minus_bias
                 
    def get(self):
        """Returns bias-corrected raw image result.

        Returns:
            self.rawimage: The bias-corrected data.
        """
        return self.rawimage

        #raise flag when counts are significantly diff from master bias
        #identify bad pixels
