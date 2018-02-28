class KPF0(object):
    """
    Container object for level zero 
    """
    def __init__(self):
        self.header # meta data from the telesceope (dict)
        self.red # 2 spectrum from KPF red chip (row, col)
        self.green # 2 spectrum from KPF red chip (row, col)
        self.hk # 2 spectrum from HK spectrometer (wav, intensity)
        self.expmeter # (time, wavelength, intensity)
        self.guidecam # guidecam sequence (row, col, intensity)

    def to_fits(self, fn):
        """
        Optional: collect all the level 0 data into a monolithic fits file
        """
        pass

class KPF1(object):
    """
    Container object for level one data
    """

    def __init__(self):
        self.orderlets # collection orderlet objects
        self.hk # 1D CaII-HK spectrum 

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
        self.id # 

class KPF2(object):
    def __init__(self):
        self.orderlets # collection of level2 orderlet code
        self.rv

        # collection of different activity mectrics like svalue,
        # bisector span, halpha, etc

        self.activity 
    def to_fits(self):
        """
        Collect all the level 2 data into a monolithic fits file
        """
        pass
        
class Orderlet2(object):
    """
    Contanier for data that's associated with level two data products
    per orderlet
    """
    def __init__(self):
        self.ccf
        self.mask
        self.rv
        self.bc
        self.expmeter
        

    
        

        

