#packages
#Subtracting 2D array, function to subtract master bias frame from raw data image

from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments

class BiasSubtraction:
    """
    The BiasSubtraction class performs master bias frame subtraction from a raw science frame. 
    Working on file input and export.
    
    Args:
        rawimage
        masterbias
        config
        logger
    
    Attributes:
        logger

    """

    def __init__(self,rawimage,masterbias,config=None, logger=None):
        
        self.rawimage=rawimage
        self.masterbias=masterbias
        self.logger=logger

    def bias_subtraction(self):
        """
        Steps:
            Reads in master bias frame
            Reads in raw science
            Checks whether both data arrays have same dimensions, prints "equal" or "not equal"
            Subtracts master bias array values from raw array values
            Returns array of bias-corrected science frame
        
            In pipeline terms: inputs two L0 files, outputs one L0 file
    
        Args:
            rawimage (str): The string to the raw science frame .fits file
            masterbias (str): The string to master bias .fits file - the result of combining/averaging several bias frames

        Returns:
            raw_bcorrect (array): bias corrected "raw" data (data no longer raw)
        """
    #check to see if both matrices have the same dimensions, Cindy's recommendation
        if self.rawimage.data.shape==self.masterbias.data.shape:
            print ("Bias .fits Dimensions Equal, Check Passed")
        if self.rawimage.data.shape!=self.masterbias.data.shape:
            raise Exception("Bias .fits Dimensions NOT Equal! Check Failed")
        #do I need to do KPF0 here?
        raw_bcorrect=KPF0()
        raw_bcorrect.data=self.rawimage.data-self.masterbias.data
        return Arguments(raw_bcorrect)

