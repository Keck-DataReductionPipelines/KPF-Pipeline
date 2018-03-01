class KPF0(object):
    """
    Container object for level zero data
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

class KPF2(object):
    def __init__(self):
        self.orderlets # collection of Orderlet2
        self.rv # final radial-velocity (float)
        self.activity # collection of different activity mectrics like svalue, bspan, halpha (dict)

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
        self.ccf # cross-correlation function (1D array)
        self.dv # displacement in velocity space (1D array)
        self.mask # binary mask (1D array)
        self.rv # per orderlet rv (float)
        self.bc # per orderlet bary-centric correction (float)
        self.fiberid # [1,2,3,4,5]
        self.cameraid # ['green','red']
