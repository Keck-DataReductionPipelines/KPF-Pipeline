#packages
#Subtracting 2D array, function to subtract master bias frame from raw data image

from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments

class BiasSubtractionAlg:
    """
    The BiasSubtraction class performs master bias frame subtraction from a raw science frame. 
    Steps include:
        - Reads in master bias frame
        - Reads in raw science
        - Checks whether both data arrays have same dimensions, prints "equal" or "not equal"
        - Subtracts master bias array values from raw array values
        - Returns array of bias-corrected science frame

    Args:
        rawimage (array): The FITS raw data
        masterbias (array): The FITS master bias data
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger.
    
    Attributes:
        logger (logging.Logger): Instance of logging.Logger.
    
    Raises:
        Exception: If raw image and bias frame don't have the same dimensions

    Returns:
        raw_bcorrect: The bias-corrected data
    """


    def __init__(self,rawimage,config=None, logger=None):
        
        self.rawimage=rawimage
        self.config=config
        self.logger=logger

#make bias subtraction just a function, takes 
    def bias_subtraction(self,masterbias):
        """
            Subtracts bias data from raw data.
            In pipeline terms: inputs two L0 files, outputs one L0 file. 
    
        Returns:
            raw_bcorrect (array): The bias corrected data

        Raises:
            Exception: If raw image and bias frame don't have the same dimensions
        """
        if self.rawimage.data.shape==masterbias.data.shape:
            print ("Bias .fits Dimensions Equal, Check Passed")
        else:
            raise Exception("Bias .fits Dimensions NOT Equal! Check Failed")
        self.rawimage.data=self.rawimage.data-masterbias.data
    
    def get(self):

        return self.rawimage
