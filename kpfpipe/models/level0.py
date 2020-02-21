"""
Data models for KPF data
"""


class KPF0(object):
    """Container object for level zero data"""
    def __init__(self):
        """

        Attributes:
            data (dictionary): Keys correspond to chips (e.g. green, red). Values contain the 2D flux arrays
            flat (dictionary): Master flat files
            bias (dictionary): Master bias files
            header (dictionary): FITS header as dictionary
            hk (array): (optional) Echellogram (image) from HK spectrometer CCD; 2D array (row, col)
            expmeter (array): (optional) Exposure meter sequence; 3D array = time series of 2D CCD images (time, row, col)
            guidecam (array): (optional) Guidecam sequence; 3D array = time series of 2D CCD images (time, row, col)

        Todo:
            * implement `to_fits` and `from_fits` methods

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
