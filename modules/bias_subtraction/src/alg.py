#packages
#Subtracting 2D array, function to subtract master bias frame from raw data image

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
            Reads in bias frame data
            Reads in raw science data
            Checks whether both data arrays have same dimensions, prints "equal" or "not equal"
            Stacks, creates average of bias frames element-wise - final average results is master bias
            Subtracts master bias array values from raw array values
            Returns array of bias-corrected science frame
        
            In pipeline terms: inputs two L0 files, outputs one L0 file
    
        Args:
            rawimage (str): The string to the raw science frame .fits file
            masterbias (str): The string to master bias .fits file - the result of combining/averaging several bias frames

        Returns:
            raw_bcorrect (array):
        """
        biasdata=(self.masterbias).data
        rawdata=(self.rawimage).data
    #check to see if both matrices have the same dimensions, Cindy's recommendation
        if biasdata.shape==rawdata.shape:
            print ("Bias .fits Dimensions Equal, Check Passed")
        if biasdata.shape!=rawdata.shape:
            print ("Bias .fits Dimensions NOT Equal! Check Failed")
        raw_bcorrect=rawdata-biasdata
        return raw_bcorrect
