
class KPF1(object):
    """
    Container object for level one data
    """
    def __init__(self):
        self.orderlets # collection of Orderlet1 objects
        self.hk # 1D CaII-HK spectrum 
        self.expmeter # time series of 1D exposure meter spectra

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
        self.source # 'sky', 'sci', `cal`
        self.flux = # flux from the spectrum
        self.flux_err # flux uncertainty
        self.wav # wavelenth solution
        self.fiberid # [1,2,3,4,5]
        self.ordernum # [71-137]; 103-137 = green, 71-102 = red

class HK1(object):
    """
    Contanier for data associated with level one data products from the HK spectrometer
    """
    def __init__(self):
        self.source # 'sky', 'sci', `cal`
        self.flux = # flux from the spectrum
        self.flux_err # flux uncertainty
        self.wav # wavelenth solution

