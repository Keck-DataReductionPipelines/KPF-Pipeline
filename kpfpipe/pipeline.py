class KPF0(object):
    """
    Container object for level zero 
    """
    def __init__(self):
        self.header # meta data from the telesceope (dict)
        self.red # Echellogram red chip; 2D arrray (row, col)
        self.green # Echellogram from green chip; 2D array(row, col)
        self.hk # Echellogram from HK spectrometer (wav, intensity)
        self.expmeter # exposure meter sequence; 2D array (time, wavelength)
        self.guidecam # guidecame sequence;  3D array (time, row, col)

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
        self.orderlets # collection Orderlet1 objectz
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
        self.id # [1,2,3,4,5]

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
