



# Example recipe
def Recipe(path):
    pipe = Pipeline()  # instantiate the pipeline object
    pipe.create_level0(path)  # Load the raw data
    pipe.subtract_bias_2d()  # perform some processing of level0 data
    pipe.extract_spectra()  # convert to level 1 data
    pipe.write_level1(path+'l1')  # save the data state for later
    pipe.create_RV()  # convert to level 2 data
    pipe.write_level2(path+'l2')  # save final answer 

# Example of picking up halfway through
def Recipe1(path):
    pipe = Pipeline()
    pipe.create_level1(path) # Load saved level1 data product
    pipe.calibrate_wavelength(method='method1') # perform different processing operations from before
    pipe.create_RV() # create level 2 data
    pipe.dosomething()  # more level 2 methods
    pipe.write_level2(path+'l2')  # save final product


# Example of picking up halfway through + R'HK processing
def Recipe2(path):
    pipe = Pipeline()
    pipe.create_level1(path) # Load saved level1 data product
    pipe.calibrate_wavelength(method='method1') # perform different processing operations from before
    pipe.create_RV() # create level 2 data
    pipe.dosomething()  # more level 2 methods
    #pipe.RemoveRprimeHK()  # This will fail if we don't process RHK first. So we process HK with, e.g.:
    pipe.subtract_hk_bias()
    pipe.extract_hk_spectrum() 
    pipe.RPrimeHK()
    pipe.RemoveRprimeHK() # Remove R'HK from RVs
    pipe.write_level2(path+'l2')  # save final product



# Example of Parallelization:
pathlist = [file1, file2, file3, file4]
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# In this example, for 2 cores, each would process two files. 
for f in pathlist[rank::size]:
    # Recipes remain unchanged in parallel version
    Recipe(path)

# But we must sometimes tell functions how to parallelize correctly
# Example
class Pipeline(object):
    '''
    An example of the slightly tricky subtract_bias_2d parallelization where all need to wait until 1 process
        completes a master task
    '''
    def subtract_bias_2d(self):
        # check that self.green is not None
        self.checklevel(0) 
        # check if master bias exists
        
        # If you are not the master node, wait until the bias is made or found
        if rank != 0: # this may be sub-optimal if rank 0's processing is slower than others before this.
            MPI.Bcast(None, root=0) 
        try: 
            self.master_bias = get_master_bias() 
        except:
            make_master_bias() 
            self.master_bias = get_master_bias()
        # If the first thread found or made the bias, tell the others to continue 
        if rank == 0:
            MPI.Bcast(None, root=0)

        def subtract_bias(spectra2d, master_bias):      
            return spectra2d - bias
        bias_removed = subtract_bias(self.green, self.master_bias)
        self.bias_removed = bias_removed # (2D array)
        self.green = bias_removed 

# More complicated schemes can be implemented for more complicated blocking tasks, rather than just having root do everything

