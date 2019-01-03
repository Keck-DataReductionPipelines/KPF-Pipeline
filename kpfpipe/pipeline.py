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
from kpfpipe.level0 import KPF0
from kpfpipe.level1 import KPF1
from kpfpipe.level2 import KPF2
import logging
   
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
        # Set up logging when a Pipeline object is instantiated
        #   - Do we actually want to do this elsewhere? This doesn't log the initialization of the levelX objects, but that's probably fine? 
        # Probably want to have Pipeline take a logging level as an optional argument, but for now we we'll just set it to debug
        # We also specify the format of all of the logging messages here 
        logging.basicConfig(filename='log.txt', 
              format='%(asctime)s || %(levelname)s || %(name)s : %(message)s',
              datefmt='%m/%d/%Y %I:%M:%S %p',
              level=logging.DEBUG)


    # Define how to dump a Pipeline object to a string with print
    def __str__(self):
        logging.debug('Entered Pipeline.__str__')
        s = 'Methods completed:\n'
        s += str(self.method_list)
        s += '\n'
        if self.level0 is not None:
            s += 'Level 0 Data:\n'
            for key in self.level0.data.keys():
                s += key + ': '
                s += str(self.level0.data[key])
                s += '\n'
        if self.level1 is not None:
            s += 'Level 1 Data:\n'
            for key in self.level1.Norderlets.keys():
                s += key + ': '
                s += str(self.level1.orderlets[key][0].flux) 
                #s += str(self.level1.orderlets[key][0].flux_err) 
                #s += str(self.level1.orderlets[key][0].wav) 
                s += '\n'
        if self.level2 is not None:
            s += 'Level 2 Data:\n'
            for key in self.level2.rv.keys():
                s += key + ': '
                s += str(self.level2.rv[key]) 
                s += '\n'
        return s

    # Functions for checking if the correct level object is in pipeline
    def checklevel0(self):
        if ((self.level0 == None) or 
              (not self.valid_level0_data())):
            raise
    def checklevel1(self):
        if ((self.level1 == None) or 
              (not self.valid_level1_data())):
            raise
    def checklevel2(self):
        if ((self.level2 == None) or 
              (not self.valid_level2_data())):
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
#    def checkhk(self, levelX):
#        checklevel(levelX)
#        if not hasattr(self, hklevelX):
#            raise 
   
    # Confirms that KPF0 object has necessary data/structure to operate on (after checking for its existence)
    def valid_level0_data(self):
        # The absolutely necessary data in a level0 array is in the self.data dictionary.
				# Check that it has not been corrupted 
        if type(self.level0.data) is not dict: 
            return False
        # Check that it contains some data
        if len(self.level0.data) <= 0:
            return False
        # And that the data is appropriate
        for key in self.level0.data:
            if ((not isinstance(self.level0.data[key], np.ndarray)) or
                    not np.all(np.isfinite(self.level0.data[key]))):
                return False
            # Could check for dimensionality, but we won't do this for flexibility
            #if self.data[key].shape == shape:
            #    return False
        # We will also want to check some other things eventually
        #if self.level0.header is None:
        #    return False
        return True
    
    # Confirms that KPF1 object has necessary data/structure to operate on (after checking for its existence)
    # The first checks are similar to valid_level0_data(), but we keep them separated because different checks may be necessary
    #   e.g., checking that we only have 1-d spectra for each chip now, the correct number of orderlets, etc. 
    def valid_level1_data(self):
        # The absolutely necessary data in a level1 array is in the self.data dictionary.
				# Check that it has not been corrupted 
        if type(self.level1.orderlets) is not dict: 
            return False
        # Check that it contains some data
        if len(self.level1.orderlets) <= 0:
            return False
        # And that the data is appropriate
        for key in self.level1.orderlets.keys():
            if ((not isinstance(self.level1.orderlets[key][0].flux, np.ndarray)) or
                    not np.all(np.isfinite(self.level1.orderlets[key][0].flux))):
                return False
            # Could check for dimensionality, but we won't do this for flexibility
            #if self.data[key].shape == shape:
            #    return False
        # We will also want to check some other things eventually
        #if self.level1.header is None:
        #    return False
        return True

########
    
    ## LEVEL 0 Section
    
    # Level 0 Decorator to 
    #   This decorator will be applied to every level 0 method
    #   It will:
    #     - Log the method's name (and arguments) 
    #     - Check that level0 object exists before method execution
    #     - Check that the minimum level0 data structure exists
    #     - Execute the actual method
    #     - Check that the level0 object + data still exists after method execution 
    def level0_method(level0_method_function):
        def level0_method_wrapper(self, *args, **kwargs):
            logging.info('Running level0 method %s' % str(level0_method_function))
            logging.debug('Appending method to method list')
            self.method_list.append(str(level0_method_function.__name__))
            logging.debug('Checking level0 before method')
            self.checklevel(0)
            logging.debug('Executing method')
            level0_method_function(self, *args, **kwargs)
            logging.debug('Checking level0 after method')
            self.checklevel(0)
        return level0_method_wrapper

    # These are all level 0 methods, which can be called in any order as long as there is a level0 object in this Pipeline instance 
    #   Each level0 method must be tagged with the @level0_method decorator 

    @level0_method
    def subtract_bias(self, chips=True):
        logging.debug('Entered subtract_bias')
        if chips is True:
            logging.warning('Chips have not been explicitly set in subtract_bias method')
            logging.warning('Setting chips to self.level0.data.keys()')
            chips = self.level0.data.keys()
        for chip in chips:
            logging.debug('Subtracting bias from chip: ' + chip)
            try:
                self.level0.data[chip] -= self.level0.bias[chip]
                rms = np.sqrt(np.mean(np.square(self.level0.data[chip])))
                med = np.median(self.level0.data[chip])
                logging.info('After bias subtraction on chip ' + chip + ', RMS = %f, median = %f' % (rms, med))
            except AttributeError:
                logging.error('There is no bias for chip ' + chip + ' so bias subtraction failed')
                pass # maybe we don't want this to crash the program, but we record an ERROR in the log 
        logging.debug('Finished subtract_bias method')

    
    @level0_method
    def divide_flat(self, chips=True):#, color=(green, red)):
        if chips is True:
            chips = self.level0.data.keys()
        for chip in chips:
            try:
                self.level0.data[chip] -= self.level0.flat[chip]
                rms = np.sqrt(np.mean(np.square(self.level0.data[chip])))
                med = np.median(self.level0.data[chip])
                logging.info('After dividing flat on chip ' + chip + ', RMS = %f, median = %f' % (rms, med))
            except AttributeError:
                logging.error('There is no bias for chip ' + chip + ' so dividing flat failed')
                pass # maybe we don't want this to crash the program, but we record an ERROR in the log 
        # Perhaps log some additional data if desired


    # This level0 method creates a level1 data object
    #   Needs to be the last level0 function you call
    #   And you must call this or have loaded a level1 data object to continue the pipeline
    @level0_method
    def extract_spectrum(self, chips=True):
        #self.level1 = KPF1()    # A) enforce calling this function first
        if self.level1 is None:  # B) allow calibrate_wavelength to be called first (really, any order is OK)
            self.level1 = KPF1() #    - Which design would we prefer?
                                 # C) have separate create_level1() method which must be called before either
        if chips is True:
            chips = self.level0.data.keys()
        for chip in chips:
            for i in range(self.level1.Norderlets[chip]):
                # This is where the extraction algorithm is called. For now we just use np.mean
                self.level1.orderlets[chip][i].flux = np.mean(self.level0.data[chip], axis=1)
                self.level1.orderlets[chip][i].flux_err = np.mean(self.level0.data[chip], axis=1)

    @level0_method
    def calibrate_wavelengths(self, chips=True):
        if self.level1 is None:
            self.level1 = KPF1()
        if chips is True:
            chips = self.level0.data.keys()
        for chip in chips: 
            for i in range(self.level1.Norderlets[chip]):
                self.level1.orderlets[chip][i].wav = np.mean(self.level0.data[chip], axis=1)

#    # Load data from saved level1 object
#    # Do we need this since we can call Pipeline(level1=level1)?
#    # Should we have a function that just creates a level1 object in Pipeline?
#    def create_level1(self, directory):
#        level1 = KPF1()
#        # load data here    

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
    
    ## LEVEL 1 Section
    
    # Level 1 Decorator to 
    #   This decorator will be applied to every level 1 method
    #   It will:
    #     - Log the method's name (and arguments) 
    #     - Check that level1 object exists before method execution
    #     - Check that the minimum level1 data structure exists
    #     - Execute the actual method
    #     - Check that the level1 object + data still exists after method execution 
    def level1_method(level1_method_function):
        def level1_method_wrapper(self, *args, **kwargs):
            logging.info('Running level1 method %s' % str(level1_method_function))
            logging.debug('Appending method to method list')
            self.method_list.append(str(level1_method_function.__name__))
            logging.debug('Checking level1 before method')
            self.checklevel(1)
            logging.debug('Executing method')
            level1_method_function(self, *args, **kwargs)
            logging.debug('Checking level1 after method')
            self.checklevel(1)
        return level1_method_wrapper

    # These are all level 1 methods, which can be called in any order as long as there is a level1 object in this Pipeline instance 
    #   Each level1 method must be tagged with the @level1_method decorator 

    @level1_method
    def remove_emission_line_regions(self, regions=True):
        if regions is True:
            logging.warning('Using default emission line region mask')
            regions = {'green': {0: [1,2,3], 3: [4,5]}, 'red': {1: [1,2,3], 2: [1,5]}} 
        logging.info('removing emission line regions: %s' % str(regions))
        # I don't know how this will work so I'm just zero masking some random points
        for key in regions.keys():
            for orderlet_index in regions[key].keys():
                for wavelength_index in regions[key][orderlet_index]:
                    try: 
                        self.level1.orderlets[key][orderlet_index].flux[wavelength_index] = 0.
                    except:
                        pass # log errors, etc. 

    @level1_method
    def remove_solar_regions(self, regions=True):
        if regions is True:
            logging.warning('Using default emission line region mask')
            regions = {'green': {0: [7], 2: [1,5]}, 'red': {3: [1,2,3], 4: [0,1,2,3,4]}} 
        logging.info('removing solar line regions: %s' % str(regions))
        # I don't know how this will work so I'm just zero masking some random points
        for key in regions.keys():
            for orderlet_index in regions[key].keys():
                for wavelength_index in regions[key][orderlet_index]:
                    try: 
                        self.level1.orderlets[key][orderlet_index].flux[wavelength_index] = 0.
                    except:
                        pass # log errors, etc. 

    
    @level1_method
    def correct_telluric_lines(self, correction_mask=True):
       if correction_mask is True:
           logging.warning('Using default emission line region mask')
           correction_mask = {'green': {0: [[0,0.1], [1,0.5], [7,0.1]], 2: [[1,0.5]]}} # no corrections in red 
       logging.info('correcting tellurics with correction_mask: %s' % str(correction_mask))
       # Here I'm just adding some random values to random positions in random orderlets
       for key in correction_mask.keys():
           for orderlet_index in correction_mask[key].keys():
               for index_correction in correction_mask[key][orderlet_index]:
                   try: 
                       self.level1.orderlets[key][orderlet_index].flux[index_correction[0]] += index_correction[1]
                   except:
                       pass # log errors, etc. 

    @level1_method
    def correct_wavelength_dependent_barycentric_velocity(self):
        for key in self.level1.orderlets.keys():
            for orderlet in self.level1.orderlets[key]:
                for i in range(len(orderlet.flux)):
                    orderlet.flux[i] += 0.02*i
     

    # Get the RV
    # Again, should we initialize the level2 object in this function, or have it separately
    @level1_method
    def calculate_RV_from_spectrum(self, chips=True):
        self.level2 = KPF2()
        if chips is True:
            chips = self.level1.orderlets.keys()
        for chip in chips:
            try:
                self.level2.rv[chip] = 1.0
            except AttributeError:
                pass #log Error


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





