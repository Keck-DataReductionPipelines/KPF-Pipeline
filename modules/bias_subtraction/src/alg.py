#packages
#Subtracting 2D array, function to subtract master bias frame from raw data image

from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments

class BiasSubtraction:
    """
    The BiasSubtraction class performs master bias frame subtraction from a raw observation frame. 

    Args:
        rawimage (np.ndarray): The FITS raw data
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger.
    
    Attributes:
        rawimage (np.ndarray): From parameter 'rawimage'.
    
    Raises:
        Exception: If raw image and bias frame don't have the same dimensions
    """


    def __init__(self,rawimage,config=None, logger=None):
        """Inits BiasSubtraction class with raw data, config, logger.

        Args:
            rawimage (np.ndarray): The FITS raw data
            config (configparser.ConfigParser, optional): Config context. Defaults to None.
            logger (logging.Lobber, optional): Instance of logging.Logger. Defaults to None.
        """
        self.rawimage=rawimage
        self.config=config
        self.logger=logger

#make bias subtraction just a function, takes 
    def bias_subtraction(self,masterbias):
        """
            Subtracts bias data from raw data.
            In pipeline terms: inputs two L0 files, produces one L0 file. 

        Args:
            masterbias (np.ndarray): The FITS master bias data.

        Raises:
            Exception: If raw image and bias frame don't have the same dimensions.
        """
        if self.rawimage.data.shape==masterbias.data.shape:
            print ("Bias .fits Dimensions Equal, Check Passed")
        else:
            raise Exception("Bias .fits Dimensions NOT Equal! Check Failed")
        self.rawimage.data=self.rawimage.data-masterbias.data
    
    def get(self):
        """Returns bias-corrected raw image result.

        Returns:
            self.rawimage: The now-bias-corrected data
        """
        return self.rawimage
