## Define pipeline class with methods for processing data

## Pursuing this architecture
#def Pipeline_Recipe(Pipeline_object):
#    p = Pipeline_object
#    p.level0_method()
#    p.level0_method()
#    # enforce in p that level 0 exists before running level1
#    p.level1_method()
#    p.level2_method()

#import functools
import numpy as np
from kpfpipe.level1 import KPF1

#Npx_x = 32
#Npx_y = 32
   
class Pipeline(object):
    """
    Pipeline must
    - record the receipt (multiple interfaces)
       - string-based receipt, 
          - everything needed to reproduce
       - database receipt
          - (not required for running/development)
          - file locations
          - levelX produced?

    - know how to talk to a database 
    - record git SHA
    - record running diagnostics (time, runtime)
    - vertical organization
       - pipe.level0.method()
       - pipe.level1.method()
       
    - each method must 
       - perform checks on input data (existence, quality, format of data)
       - quality control (assertions that would log errors)
       - input/output quality functions may re-use code.
       
    - logging
      - capture STDOUT/STDERR (prepend with function that produced the output)
      - logging.py
      

    """
    
    
    
    
    def __init__(self, level0=None, level1=None, level2=None):
        if type(level0) is str:
            level0 = KPF0(level0) # Construct KPFlevel0 object from file "level0" 
        if type(level1) is str:
            level1 = KPF1(level1) 
        if type(level2) is str: 
            level2 = KPF2(level2) 
        self.level0 = level0  
        self.level1 = level1
        self.level2 = level2
        self.method_list = []

    def __str__(self):
        s = ''
        if self.level0 is not None:
            s += str(self.level0.green)
            s += str(self.level0.red)
            s += str(self.method_list)
        if self.level1 is not None:
            s += str(self.level1.orderlets[0].flux) 
            s += str(self.level1.orderlets[0].flux_err) 
            s += str(self.level1.orderlets[0].wav) 
        return s


    def checklevel0(self):
        if self.level0 == None:
            raise
        if not valid_level0_data(self.level0):
            raise
    def checklevel0(self):
        if self.level0 == None:
            raise
    def checklevel0(self):
        if self.level0 == None:
            raise
    def checklevel(self, level):
        if level == 0:
            self.checklevel0()
        elif level == 1:
            self.checklevel1()
        elif level == 2:
            self.checklevel2()
        else:
            raise
    def checkhk(self, levelX):
        checklevel(levelX)
        if not hasattr(self, hklevelX):
            raise 

    def valid_level0_data(self):
        if not isinstance(self.green, np.ndarray):
            return False
        if not (self.green.shape == (Npx_x, Npx_y)):
            return False
        if not np.all(np.isfinite(self.green)):
            return False
        return True

########

    # These are all level 0 methods, which can be called in any order as long as there is a level0 object in this Pipeline instance 
    #    This is enforced with the checklevel(0) method
    #    We probably want to enforce this way instead of with classes to delay execution of the class's body
    #      without hiding the whole functionality in __init__ or something

    # Decorator to check level and data before and after each level0 method
    def level0_method(level0_method_function):
        def level0_method_wrapper(self, *args, **kwargs):
            self.method_list.append(str(level0_method_function.__name__))
            #checklevel(0)
            level0_method_function(self, *args, **kwargs)
            #checklevel(0)
        return level0_method_wrapper


    @level0_method
    def subtract_bias(self):
        # log basic facts, like RMS, median bias, no nans, etc. to logging.py for example 
        self.level0.green -= self.level0.bias_green
        self.level0.red -= self.level0.bias_red

    @level0_method
    def divide_flat(self):#, color=(green, red)):
        #for c in color:  # we may want some flexibility in picking what chips to operate on
        self.level0.green /= self.level0.flat_green
        self.level0.red /= self.level0.flat_red


    # This level0 method creates a level1 data object
    #   Needs to be the last level0 function you call
    #   And you must call this or have loaded a level1 data object to continue the pipeline
    @level0_method
    def extract_spectrum(self):
        #self.level1 = KPF1()    # A) enforce calling this function first
        if self.level1 is None:  # B) allow calibrate_wavelength to be called first (really, any order is OK)
            self.level1 = KPF1() #    - Which design would we prefer?
                                 # C) have separate create_level1() method which must be called before either
        for i in range(self.level1.Norderlets_green):
            self.level1.orderlets[i].flux = np.mean(self.level0.green, axis=1)
            self.level1.orderlets[i].flux_err = np.mean(self.level0.green, axis=1)
        for i in range(self.level1.Norderlets_green, self.level1.Norderlets_red):    
            self.level1.orderlets[i].flux = np.mean(self.level0.red, axis=1)
            self.level1.orderlets[i].flux_err = np.mean(self.level0.red, axis=1)

    @level0_method
    def calibrate_wavelengths(self):
        if self.level1 is None:
            self.level1 = KPF1()
        for i in range(self.level1.Norderlets_green):
            self.level1.orderlets[i].wav = np.mean(self.level0.green, axis=1)
        for i in range(self.level1.Norderlets_green, self.level1.Norderlets_red):    
            self.level1.orderlets[i].wav = np.mean(self.level0.red, axis=1)

#    def correct_brighter_fatter(self):
#        self.checklevel(0)
#        # maybe needed
#
#    def subtract_hk_bias(self):
#        self.checkhk(0)
#        hkbias = get_hk_master_bias()
#        bias_subtracted = self.hk - hkbias
#        self.levl0.hk_bias_subtracted = bias_subtracted
#        self.hk = bias_subtracted
#
#######

#    # Two ways to instantiate level1 object
#    #   1) From level0 object in this Pipeline object
#    def extract_spectra(self):
#        self.checklevel(0) 
#        level1 = KPF1()
#        # 2D images are converted into Orderlet1 objects 5 * norders
#        def extraction(self):
#            return extracted
#        oneDspectrum = extraction()
#        level1.green = oneDspectrum
#        self.level1 = level1
#        self.green = oneDspectrum   
#
#     #    2) or from saved level1 object
#    def create_level1(self, directory):
#        level1 = KPF1()
#        # read in raw fits files from instrument and populate attributes, knows about file structure
#        self.level1 = level1
#        self.green = self.level1.green.copy()
#        self.red = self.level1.red.copy()
#    
#    # Level 1 methods        
#        
#    def write_level1(self):
#        self.checklevel(1)
#        # save level 1 data to fits
#    def calibrate_wavelength_cal(self, *args):
#        self.checklevel(1)
#        # not dependent on other data from the night
#    def calibrate_wavelength_all(self):
#        self.checklevel(1)
#        # call function that is recipe for combining all of the etalon calibrations
#        # may also combine data from previous nights
#   
#    def extract_hk_spectrum(self):
#        self.checkhk(0)
#        self.checklevel(1) # This is necessary if we have hk as part of KPF[0-2] objects instead of its own object 
#        oneDspectrum = extraction()
#        self.level1.hk = oneDspectrum # Add the hk data to the level1 data product (for printing/saving purposes)
#        self.hk = oneDspectrum 
#        self.hklevel1 = True #??
#
########
#    
#    # Two ways to instantiate level2 object
#    #   2) From level1 object in this Pipeline object
#    def create_RV(self):
#        self.checklevel(1) 
#        def make_rv(self):
#            return rv 
#        rv = make_rv(self)
#        self.green = rv   
#
#     #    2) or from saved level2 object
#    def create_level2(self, directory):
#        level2 = KPF2()
#        # read in raw fits files from instrument and populate attributes, knows about file structure
#        self.green = self.level2.green.copy()
#
#    # Level 2 Methods 
#
#    def dosomething(self):
#        self.checklevel(2)
#        # do something
#
#    def RprimeHK(self):
#        self.checkhk(1)
#        self.checklevel(2)
#        self.rphk = computerphk()
#    def RemoveRprimeHK(self):
#        self.checklevel(2)
#        if not hasattr(self, rphk):
#            raise
#        self.green -= self.rphk
#
#
#######

    # Some methods may not require level0, 1, or 2 data products, and can be called at any time
    #    Even if in practice it is more useful to call them before certain methods  
    def create_master_flat(self):
        pass
        # create master flats file
    def to_fits(self):
        pass
        # write out current state of pipe object regardless of progress of data levels 


#######





