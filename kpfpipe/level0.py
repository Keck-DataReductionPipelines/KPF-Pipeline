"""
Define objects used in level zero data processing
"""

class KPF0(object):
    """
    Container object for level zero data
    
    To Do: Consider making an abstract base class...
    """
    def __init__(self):
        self.header # meta data from KPF, telesceope, and observatory (dict)
        self.red # Echellogram (image) from red CCD; 2D array (row, col)
        self.green # Echellogram (image) from green CCD; 2D array (row, col)
        self.hk # Echellogram (image) from HK spectrometer CCD; 2D array (row, col)
        self.expmeter # exposure meter sequence; 3D array = time series of 2D CCD images (time, row, col)
        self.guidecam # guidecam sequence; 3D array = time series of 2D CCD images (time, row, col) [consider whether guidecam should be included;  will it be analyzed?]

    def to_fits(self, fn):
        """
        Optional: collect all the level 0 data into a monolithic fits file
        """
        pass
    
class MasterFlat(KFP0):
    """
    Flat field derived from a stack of master flats
    """

class MasterBias(KPF0):
    """
    Bias frame derived from a stack of bias observations
    """
