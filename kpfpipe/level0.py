"""
Define objects used in data processing
"""

class KPF0(object):
    """
    Container object for level zero data
    
    To Do: Consider making an abstract base class...
    """
    def __init__(self):
        self.header = None# meta data from KPF, telesceope, and observatory (dict)
        self.red = None #Echellogram (image) from red CCD; 2D array (row, col)
        self.green = None #Echellogram (image) from green CCD; 2D array (row, col)
        self.hk = None #Echellogram (image) from HK spectrometer CCD; 2D array (row, col)
        self.expmeter = None #exposure meter sequence; 3D array = time series of 2D CCD images (time, row, col)
        self.guidecam = None #guidecam sequence; 3D array = time series of 2D CCD images (time, row, col) [consider whether guidecam should be included;  will it be analyzed?]
        self.bias = None #(2D array) master bias frame for that exposure 
        self.flat = None #(2D array) master flat frame
        
    def to_fits(self, fn):
        """
        Optional: collect all the level 0 data into a monolithic fits file
        """
        pass

class KPF1(object):
    """
    Container object for level zero data
    
    To Do: Consider making an abstract base class...
    """
    def __init__(self):
        self.header = None# meta data from KPF, telesceope, and observatory (dict)
        self.red = None #Echellogram (image) from red CCD; 2D array (row, col)
        self.green = None #Echellogram (image) from green CCD; 2D array (row, col)
        self.hk = None #Echellogram (image) from HK spectrometer CCD; 2D array (row, col)
        self.expmeter = None #exposure meter sequence; 3D array = time series of 2D CCD images (time, row, col)
        
    def to_fits(self, fn):
        """
        Optional: collect all the level 0 data into a monolithic fits file
        """
        pass

# Decided to just have HK be a part of KPF objects as above
#class HK0(object):
#    """
#    Container object for level zero data
#    
#    To Do: Consider making an abstract base class...
#    """
#    def __init__(self):
#        self.header # meta data from KPF, telesceope, and observatory (dict)
#        self.hk # Echellogram (image) from HK spectrometer CCD; 2D array (row, col)
#        
#    def to_fits(self, fn):
#        """
#        Optional: collect all the level 0 data into a monolithic fits file
#        """
#        pass
   

#class MasterFlat(KPF0):
#    """
#    Flat field derived from a stack of master flats
#    """
#
#class MasterBias(KPF0):
#    """
#    Bias frame derived from a stack of bias observations
#    """


# We won't do this since we will define Methods as functions in the Pipeline class
#
#class Level0Method(object):
#    """
#    Check if the correct level exists
#    """
#    def __init__(self):
#        self.
#    class __Level0Method__():
#        def __init__(self):
#            self.level0 = level0
#            if level0 is None:
#                raise     
#    class __Level1Method__():
#        def __init__(self):
#            self.level0 = level0
#            if level1 is None:
#                raise     
#    class __Level2Method__():
#        def __init__(self):
#            self.level0 = level0
#            if level2 is None:
#                raise     
#

