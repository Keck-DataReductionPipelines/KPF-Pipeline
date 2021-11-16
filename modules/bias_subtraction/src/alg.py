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


    def __init__(self,rawimage,ffi_exts,quicklook,data_type,config=None,logger=None):
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
        self.data_type=data_type
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
                for no,ffi in enumerate(self.ffi_exts):
                    print('shapes:',self.rawimage[ffi].data.shape,masterbias[ffi].data.shape)
                    if self.rawimage[ffi].data.shape==masterbias[ffi].data.shape:
                        print ("Bias .fits Dimensions Equal, Check Passed")
                    else:
                        raise Exception ("Bias .fits Dimensions NOT Equal! Check failed")

                    self.rawimage[ffi].data=self.rawimage[ffi].data-masterbias[ffi].data
                #ext no+1 for mflat because there is a primary ext coded into the masterflat currently
                
            if self.data_type == 'NEID':
                print('shapes:',self.rawimage.DATA.shape,masterbias.DATA.shape)
                if self.rawimage.DATA.shape==masterbias.DATA.shape:
                    print ("Bias .fits Dimensions Equal, Check Passed")
                else:
                    raise Exception ("Bias .fits Dimensions NOT Equal! Check failed")
                self.rawimage.DATA=self.rawimage.DATA-masterbias.DATA


        if self.quicklook == True:
            for no,ffi in enumerate(self.ffi_exts):
                print('shapes:',self.rawimage[ffi].data.shape,masterbias[ffi].data.shape)
                #until data model for master files is added:
                if self.rawimage[ffi].shape==masterbias[ffi].data.shape:
                    print ("Bias .fits Dimensions Equal, Check Passed")
                else:
                    raise Exception ("Bias .fits Dimensions NOT Equal! Check failed")
                self.rawimage[ffi].data=self.rawimage[ffi].data-masterbias[ffi].data

                counts = masterbias[no+1].data #red and green potentially masters, no+1 means ignoring primary?
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

        #raise flag when counts are significantly diff from master bias
        #identify bad pixels
