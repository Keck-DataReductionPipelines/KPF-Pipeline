## Define pipeline class with methods for processing data

# Not choosing this architecture
#
## Architecture A
#def Pipeline_Master(directory):
#    level0_processed = level0_pipe(directory)
#    level1_processed = level1_pipe(level0_processed)
#    level2_processed = level2_pipe(level1_processed)
#    return level2_processed
#def level0_pipe(level0_object):
#    l = level0_object
#    l.level0_method()
#    l.level0_method()
#    return l

# Pursuing this architecture
#
# Architecture B
def Pipeline_Recipe(Pipeline_object):
    p = Pipeline_object
    p.level0_method()
    p.level0_method()
    # enforce in p that level 0 exists before running level1
    p.level1_method()
    p.level2_method()

   
class Pipeline(object):
    def __init__(self, level0=None, level1=None, level2=None):
				if level0 is not None:
						level0 = self.create_level0(level0)
				if level1 is not None:
						level1 = self.create_level1(level1)
				if level2 is not None:
						level2 = self.create_level2(level2)
				self.level0 = level0	
				self.level1 = level1
				self.level2 = level2

		def checklevel0(self):
				if self.level0 == None:
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

########

		# Must first create_level0 object with this method
    def create_level0(self, directory):
        level0 = KPF0()
        # read in fits files from instrument and populate attributes, knows about file structure
				# these need not be raw files, but could be files saved by to_fits() after some processing (e.g. flat fielding)
        self.green = self.level0.green.copy()
        self.red = self.level0.red.copy()
				if self.level0.hk is not none:
						self.hk = self.level0.hk.copy()
						self.hklevel0 = True #??
        
		# These are all level 0 methods, which can be called in any order as long as there is a level0 object in this Pipeline instance 
		#		This is enforced with the checklevel(0) method
		#		We probably want to enforce this way instead of with classes to delay execution of the class's body
		#			without hiding the whole functionality in __init__ or something

    def subtract_bias_2d(self):
        # check that self.green is not None
				self.checklevel(0) 
				# check if master bias exists
				try: 
						self.master_bias = get_master_bias() 
				except:
						make_master_bias() 
						self.master_bias = get_master_bias() 
				def subtract_bias(spectra2d, master_bias): # This can be written elsewhere so it can be called by different methods (e.g., subtract_hk_bias())			
						return spectra2d - bias
				bias_removed = subtract_bias(self.green, self.master_bias)
        self.bias_removed = bias_removed # (2D array)
        self.green = bias_removed 
        
    def divide_flat_2d(self):
				self.checklevel(0)
				# check if master flat exists (see master bias above)
        # this where brighter fatter calculations are done?
        self.flat_removed = divide_flat(self.green) # (2D array)
        self.green = divide_flat(self.green)
    
		def correct_brighter_fatter(self):
				self.checklevel(0)
        # maybe needed

		def subtract_hk_bias(self):
				self.checkhk(0)
				hkbias = get_hk_master_bias()
				bias_subtracted = self.hk - hkbias
				self.levl0.hk_bias_subtracted = bias_subtracted
				self.hk = bias_subtracted

#######

		# Two ways to instantiate level1 object
		#   1) From level0 object in this Pipeline object
    def extract_spectra(self):
				self.checklevel(0) 
				level1 = KPF1()
        # 2D images are converted into Orderlet1 objects 5 * norders
        def extraction(self):
						return extracted
				oneDspectrum = extraction()
				level1.green = oneDspectrum
				self.level1 = level1
				self.green = oneDspectrum   

 		#		2) or from saved level1 object
    def create_level1(self, directory):
        level1 = KPF1()
        # read in raw fits files from instrument and populate attributes, knows about file structure
				self.level1 = level1
        self.green = self.level1.green.copy()
        self.red = self.level1.red.copy()
    
		# Level 1 methods        
        
    def write_level1(self):
				self.checklevel(1)
        # save level 1 data to fits
    def calibrate_wavelength_cal(self, *args):
				self.checklevel(1)
        # not dependent on other data from the night
    def calibrate_wavelength_all(self):
				self.checklevel(1)
        # call function that is recipe for combining all of the etalon calibrations
        # may also combine data from previous nights
   
		def extract_hk_spectrum(self):
				self.checkhk(0)
				self.checklevel(1) # This is necessary if we have hk as part of KPF[0-2] objects instead of its own object 
				oneDspectrum = extraction()
				self.level1.hk = oneDspectrum # Add the hk data to the level1 data product (for printing/saving purposes)
				self.hk = oneDspectrum 
				self.hklevel1 = True #??

#######
		
		# Two ways to instantiate level2 object
		#   2) From level1 object in this Pipeline object
    def create_RV(self):
				self.checklevel(1) 
        def make_rv(self):
						return rv 
				rv = make_rv(self)
				self.green = rv   

 		#		2) or from saved level2 object
    def create_level2(self, directory):
        level2 = KPF2()
        # read in raw fits files from instrument and populate attributes, knows about file structure
        self.green = self.level2.green.copy()

		# Level 2 Methods 

		def dosomething(self):
				self.checklevel(2)
				# do something

		def RprimeHK(self):
				self.checkhk(1)
				self.checklevel(2)
				self.rphk = computerphk()
		def RemoveRprimeHK(self):
				self.checklevel(2)
				if not hasattr(self, rphk):
						raise
				self.green -= self.rphk


######

		# Some methods may not require level0, 1, or 2 data products, and can be called at any time
		#		Even if in practice it is more useful to call them before certain methods	
		def create_master_flat()
				# create master flats file
		def to_fits() 
				# write out current state of pipe object regardless of progress of data levels 


#######





