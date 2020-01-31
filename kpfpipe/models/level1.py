"""
Define objects used in level one data processing
"""

from astropy.io import fits
from astropy.time import Time
import numpy as np

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

    def __init__(self) -> None:

        self.spectrums = {}
        self.hk = None       # 1D CaII-HK spectrum 
        self.expmeter = None # time series of 1D exposure meter spectra

    def from_fits(self, fn: str, 
                  overwrite: bool=True) -> None:
        '''
        Read data from .fits file
        Arg: 
            fn (str): .fits file name
            HDU (str): HDU (Deader Data Unit) to be read
            overwrite (bool): if this instance already contains data
                specifies whether newly read-in data should overwrite 
                existing data
        '''
        if fn.endswith('.fits') == False:
            # Can only read .fits files
            msg = 'input files must be .fits files'
            raise IOError(msg)
    
        if overwrite != True and not self.orderlets:
            # This instance already contains data
            msg = 'Cannot overwrite existing data'
            raise IOError(msg)
            
        self.filename = fn
        with fits.open(fn) as hdu_list:
            # First record relevant header information
            for hdu in hdu_list:
                
                # Some header keywords in primary hdu are global to 
                # the file. Record them here
                if isinstance(hdu, fits.PrimaryHDU):
                    self.julian = Time(hdu.header['bjd'], format='jd')
                    self.berv = hdu.header['beryVel']

                header = hdu.header
                spec = Spectrum(hdu.name, hdu.data, 
                                header['waveinterp deg'], 
                                header)
                self.spectrums[hdu.name] = spec
            
        
    def get_order(self, order: int, source: str) -> np.ndarray:
        '''
        Returns a wave-flux pair data
        Args:
            order (int):  order of data
            source (str): source of data
        '''

        return (self.spec[source].wave[order], 
                self.spec[source].flux[order])

    def to_fits(self, fn):
        """
        Optional: collect all the level 1 data into a monolithic fits file
        """
        pass

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
    def __init__(self, source: str, 
                 data: np.ndarray,
                 opwer: int,
                 header: fits.Header) -> None:
        
        self.source = source
        self.flux = data
        self.opwer = opwer

        NOrder, NPixel = data.shape

        # Generate 
        self.wave = np.zeros_like(self.flux)
        a = np.zeros(opwer+1)
        for order in range(0, NOrder):
            for i in range(0, self.opwer+1, 1):
                keyi = 'hierarch waveinterp ord ' + str(order) +\
                ' deg ' + str(i)
                a[i] = header[keyi]
            self.wave[order] = np.polyval(
                np.flip(a),
                np.arange(NPixel, dtype=np.float64)
            )

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
