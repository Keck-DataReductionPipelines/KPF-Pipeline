from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments

class FlatFielding:
    """
    The FlatFielding class performs master flat frame division on a raw science frame. 

    Args:
        rawimage
        masterflat
        config
        logger
    
    Attributes:
        logger

    """

    def __init__(self,rawimage,masterflat,config=None, logger=None):
        
        self.rawimage=rawimage
        self.masterflat=masterflat
        self.logger=logger

    def flat_fielding(self):
        """
        Steps:
            Reads in master flat
            Reads in raw image
            Checks whether both data arrays have the same dimensions, prints "equal" or raises error
            Divides raw image by master flat 
            Returns array of flat-corrected raw frame

            In pipeline terms: inputs two L0 files, outputs one L0 file
        Args:
            rawimage (str): The string to the raw science frame .fits file
            masterflat (str): The string to master bias .fits file - the result of combining/averaging several flat frames

        Raises:
            Exception: raw image and flat frame don't have same dimensions

        Returns:
            raw_flatcorrect: flat corrected "raw" data 
        """
        if self.rawimage.data.shape==self.masterflat.data.shape:
            print ("Flat .fits Dimensions Equal, Check Passed")
        if self.rawimage.data.shape!=self.masterflat.data.shape:
            raise Exception("Flat .fits Dimensions NOT Equal! Check Failed")
        #do I need to do KPF0 here?
        raw_flatcorrect=KPF0()
        raw_flatcorrect.data=self.rawimage.data/self.masterflat.data
        return Arguments(raw_flatcorrect)

    