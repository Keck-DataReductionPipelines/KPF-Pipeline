#packages
import numpy as np
import matplotlib.pyplot as plt
from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments

class BiasSubtractionAlg:
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
                
            if self.data_type == 'NEID':
                print(self.rawimage.info())
                print('shapes:',self.rawimage['DATA'].shape,masterbias['DATA'].shape)
                assert self.rawimage['DATA'].shape==masterbias['DATA'].shape, "Bias .fits Dimensions NOT Equal! Check failed"
                minus_bias=self.rawimage['DATA']-masterbias['DATA']
                self.rawimage['DATA'] = minus_bias

        if self.quicklook == True:
            for no,ffi in enumerate(self.ffi_exts):
                print('shapes:',self.rawimage[ffi].data.shape,masterbias[ffi].data.shape)
                assert self.rawimage[ffi].data.shape==masterbias[ffi].data.shape, "Bias .fits Dimensions NOT Equal! Check failed"
                minus_bias = self.rawimage[ffi].data-masterbias[ffi].data
                self.rawimage[ffi] = minus_bias                 

                counts = masterbias[ffi].data 
                flatten_counts = np.ravel(counts)
                low, high = np.percentile(flatten_counts,[0.1,99.9])
                counts[(counts>high) | (counts<low)] = np.nan #bad pixels
                flatten_counts = np.ravel(counts)
                print(np.nanmedian(flatten_counts),np.nanmean(flatten_counts),np.nanmin(flatten_counts),np.nanmax(flatten_counts))

                plt.imshow(counts, cmap = 'cool')
                plt.colorbar()
                plt.savefig('2D_bias_frame.pdf')

                plt.close()
                plt.hist(flatten_counts, bins = 20)
                plt.savefig('Bias_histo.pdf')

    def get(self):
        """Returns bias-corrected raw image result.

        Returns:
            self.rawimage: The bias-corrected data.
        """
        return self.rawimage

        
#quicklook TODO: raise flag when counts are significantly diff from master bias, identify bad pixels
        
