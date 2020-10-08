from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments

class FlatFielding:
    """
    The FlatFielding class performs master flat frame division on a raw science frame. 
    Steps include:
        - Reads in master flat
        - Reads in raw image
        - Checks whether both data arrays have the same dimensions, prints "equal" or raises error
        - Divides raw image by master flat 
        - Returns array of flat-corrected raw data

    Args:
        rawimage (array): The FITS raw data
        masterflat (array): The FITS master flat data

    Attributes:
        logger (logging.Logger): Instance of logging.Logger.
        
    Raises:
        Exception: If raw image and flat frame don't have the same dimensions

    Returns:
        raw_flatcorrect: The flat-corrected data
    """

    def __init__(self,rawimage,masterflat,config=None, logger=None):
        
        self.rawimage=rawimage
        self.masterflat=masterflat
        self.logger=logger

    def flat_fielding(self):
        """
            Divides L0 data by master flat.
            In pipeline terms: inputs two L0 files, outputs one L0 file. 

        Raises:
            Exception: If raw image and flat frame don't have the same dimensions

        Returns:
            raw_flatcorrect: The flat-corrected data
        """

        if self.rawimage.data.shape==self.masterflat.data.shape:
            print ("Flat .fits Dimensions Equal, Check Passed")
        if self.rawimage.data.shape!=self.masterflat.data.shape:
            raise Exception("Flat .fits Dimensions NOT Equal! Check Failed")
        #do I need to do KPF0 here?
        raw_flatcorrect=KPF0()
        raw_flatcorrect.data=self.rawimage.data/self.masterflat.data
        return Arguments(raw_flatcorrect)

    