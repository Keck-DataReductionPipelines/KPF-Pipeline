#packages
import numpy as np
import matplotlib.pyplot as plt
from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments

class ImageProcessingAlg:
    """
    Bias subtraction calculation.

    This module defines 'BiasSubtraction' and methods to perform bias subtraction by subtracting a master bias frame from the raw data frame.  
    
    Attributes:
        rawimage (np.ndarray): From parameter 'rawimage'.
        ffi_exts (list): From parameter 'ffi_exts'.
        quicklook (bool): From parameter 'quicklook'.
        data_type (str): From parameter 'data_type'.
        config (configparser.ConfigParser, optional): From parameter 'config'.
        logger (logging.Logger, optional): From parameter 'logger'.
    
    Raises:
        Exception: If raw image and bias frame don't have the same dimensions.
    """

    def __init__(self,rawimage,ffi_exts,quicklook,data_type,config=None,logger=None):
        """Inits BiasSubtraction class with raw data, config, logger.

        Args:
            rawimage (np.ndarray): The FITS raw data.
            ffi_exts (list): The extensions in L0 FITS files where FFIs (full frame images) are stored.
            quicklook (bool): If true, quicklook pipeline version of bias subtraction is run, outputting information and plots.
            data_type (str): Instrument name, currently choice between KPF and NEID.
            config (configparser.ConfigParser, optional): Config context. Defaults to None.
            logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.
        """
        self.rawimage=rawimage
        self.ffi_exts=ffi_exts
        self.quicklook=quicklook
        self.data_type=data_type
        self.config=config
        self.logger=logger
        
    def bias_subtraction(self,masterbias):
        """Subtracts bias data from raw data.
        In pipeline terms: inputs two L0 files, produces one L0 file. 

        Args:
            masterbias (np.ndarray): The master bias data.

        Raises:
            Exception: If raw image and bias frame don't have the same dimensions.
        """
        if self.quicklook == False: 
            if self.data_type == 'KPF':
                for ffi in self.ffi_exts:
                    print(self.rawimage.info)
                    print(masterbias.info())
                    assert self.rawimage[ffi].data.shape==masterbias[ffi].data.shape, "Bias .fits Dimensions NOT Equal! Check failed"
                    #self.rawimage[ffi].data=self.rawimage[ffi].data-masterbias[ffi].data
                    minus_bias = self.rawimage[ffi].data-masterbias[ffi].data
                    self.rawimage[ffi] = minus_bias
                
            # if self.data_type == 'NEID':
            #     print(self.rawimage.info())
            #     print('shapes:',self.rawimage['DATA'].shape,masterbias['DATA'].shape)
            #     assert self.rawimage['DATA'].shape==masterbias['DATA'].shape, "Bias .fits Dimensions NOT Equal! Check failed"
            #     minus_bias=self.rawimage['DATA']-masterbias['DATA']
            #     self.rawimage['DATA'] = minus_bias
                 
    def get(self):
        """Returns bias-corrected raw image result.

        Returns:
            self.rawimage: The bias-corrected data.
        """
        return self.rawimage
    
    def dark_subtraction(self,dark_frame):
        for ffi in self.ffi_exts:
            assert self.rawimage[ffi].data.shape==dark_frame[ffi].data.shape, "Dark frame dimensions don't match raw image. Check failed."
            assert self.rawimage.header['PRIMARY']['EXPTIME'] == dark_frame.header['PRIMARY']['EXPTIME'], "Dark frame and raw image don't match in exposure time. Check failed."
            minus_dark = self.rawimage[ffi]-dark_frame[ffi]
            self.rawimage[ffi] = minus_dark
        
#quicklook TODO: raise flag when counts are significantly diff from master bias, identify bad pixels
        
