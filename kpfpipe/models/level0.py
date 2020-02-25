"""
Data models for KPF data
"""


class KPF0(object):
    """Container object for level zero data"""
    def __init__(self):
        """

        """
        ## Internal members 
        ## all are private members (not accesible from the outside directly)
        ## to modify them, use the appropriate methods.

        # 1D spectrums
        # Contain 'object', 'sky', and 'calibration' fiber.
        # Each fiber is accessible through their key.
        self.__flux = {}
        self.__variance = {}

        # header keywords 
        self.__header = {}
        self.__source = {}      

    def from_fits(self, fn):
        """
        Contruct KPF0 object from a raw FITS file.

        Args:
            fn (string): Path to FITS file
        """
        pass


class MasterFlat(KPF0):
    """
    Flat field derived from a stack of master flats
    """
    def __init__(self):
        pass


class MasterBias(KPF0):
    """
    Bias frame derived from a stack of bias observations
    """
    def __init__(self):
        pass
