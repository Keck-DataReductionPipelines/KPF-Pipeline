"""
Define objects used in level one data processing
"""

class KPF1(object):
    """
    Container object for level one data
    """
    

    def __init__(self):
        self.Norderlets = {'red': 10, 'green': 10} # These can also be passed as arguments to KPF1 instead of being statically defined
        self.Norderlets_total = 0
        for key in self.Norderlets.keys():
             self.Norderlets_total += self.Norderlets[key] 
        self.orderlets = {}
        for key in self.Norderlets.keys():
            self.orderlets[key] = [Orderlet1() for i in range(self.Norderlets[key])] # collection of Orderlet1 objects
        self.hk = None # 1D CaII-HK spectrum 
        self.expmeter = None # time series of 1D exposure meter spectra
        self.wavelength_solution = None #

    def to_fits(self, fn):
        """
        Optional: collect all the level 0 data into a monolithic fits file
        """
        pass

class Orderlet1(object):
    """
    Contanier for data that's associated with level one data products per orderlet
    """
    def __init__(self):
        self.source = None #'sky', 'sci', `cal`
        self.flux = None #flux from the spectrum
        self.flux_err = None #flux uncertainty
        self.wav = None #wavelenth solution
        self.fiberid = None #[1,2,3,4,5]
        self.ordernum = None #[71-137]; 103-137 = green, 71-102 = red

class HK1(object):
    """
    Contanier for data associated with level one data products from the HK spectrometer
    """
    def __init__(self):
        self.source # 'sky', 'sci', `cal`
        self.flux # flux from the spectrum
        self.flux_err # flux uncertainty
        self.wav # wavelenth solution
