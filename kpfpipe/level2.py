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
        self.ordernum # [71-137]; 103-137 = green, 71-102 = red
