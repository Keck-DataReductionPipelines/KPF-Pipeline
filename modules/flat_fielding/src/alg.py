from astropy.io import fits

class FlatFielding:
    """
    The FlatFielding class performs master flat frame division on a raw science frame. 
    Working on file input and export.
    
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
        flatdata=(self.masterflat).data
        rawdata=(self.rawimage).data
    #check to see if both matrices have the same dimensions, Cindy's recommendation
        if flatdata.shape==rawdata.shape:
            print ("Flat .fits Dimensions Equal, Check Passed")
        if flatdata.shape!=rawdata.shape:
            print ("Flat .fits Dimensions NOT Equal! Check Failed")
        raw_flatcorrect=rawdata/flatdata
        return raw_flatcorrect


    