"""
Define objects used in level one data processing
"""

from astropy.io import fits


class KPF1(object):
    """
    Container object for level one data

    Attributes:
        Norderlets (dictionary): Number of orderlets per chip
        Norderlets_total (int): Total number of orderlets
        orderlets (dictionary): Collection of Spectrum objects for each chip, order, and orderlet
        hk (array): Extracted spectrum from HK spectrometer CCD; 2D array (order, col)
        expmeter (array): exposure meter sequence; 3D array (time, order, col)

    Todo:
        * implement `to_fits` and `from_fits` methods
    """

    def __init__(self):
        self.Norderlets = {'red': 10, 'green': 10}
        self.Norderlets_total = 0
        for key in self.Norderlets.keys():
             self.Norderlets_total += self.Norderlets[key] 
        self.orderlets = {}
        for key in self.Norderlets.keys():
            # This should not be a list
            self.orderlets[key] = [Spectrum() for i in range(self.Norderlets[key])] # collection of Orderlet1 objects
        self.hk = None # 1D CaII-HK spectrum 
        self.expmeter = None # time series of 1D exposure meter spectra

    def to_fits(self, fn):
        """
        Optional: collect all the level 1 data into a monolithic fits file
        """
        pass

    @classmethod
    def from_fits(cls, fn):
        """
        Construct a KPF1 object from a FITS file

        Args:
            fn (string): file name

        Returns:
            KPF1 object
        """
        hdu = fits.open(fn)
        data = hdu[0].data

        lvl0 = cls()

        return lvl0


class Spectrum(object):
    """
    Contanier for data that's associated with level one data products per orderlet

    Attributes:
        source (string): e.g. 'sky', 'sci', `cal`
        flux (array): flux values
        flux_err (array): flux uncertainties
        wav (array): wavelengths
        fiberid (int): fiber identifier
        ordernum (int): order identifier

    """
    def __init__(self):
        self.source = None
        self.flux = None
        self.flux_err = None
        self.wav = None
        self.fiberid = None
        self.ordernum = None  # [71-137]; 103-137 = green, 71-102 = red


class HK1(object):
    """
    Contanier for data associated with level one data products from the HK spectrometer

        Attributes:
        source (string): e.g. 'sky', 'sci', `cal`
        flux (array): flux values
        flux_err (array): flux uncertainties
        wav (array): wavelengths
    """
    def __init__(self):
        self.source # 'sky', 'sci', `cal`
        self.flux # flux from the spectrum
        self.flux_err # flux uncertainty
        self.wav # wavelenth solution
