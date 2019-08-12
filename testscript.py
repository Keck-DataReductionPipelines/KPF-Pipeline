from kpfpipe.pipeline import Pipeline
from kpfpipe.level0 import KPF0
from kpfpipe.level1 import KPF1
from kpfpipe.level2 import KPF2
import numpy as np

# This dictionary-style storage of the data allows more flexibility for which data are present
# During testing, we may only have one chip online so we don't want to hard code 'red' and 'green' into everything
# This allows allows adding additional data groups in the future.  
# In this paradigm:
#  - Could either have green/red subtract_bias functions individually
#  - Check which data is available in this KPF0 object and subtract_bias only the available data
#

# We don't want to enforce dimensionality a priori, allow more flexible input

kpf0 = KPF0()
kpf0.data['green'] = np.ones((32,32), dtype=np.float) 
kpf0.data['red'] = np.ones((32,32), dtype=np.float) 
kpf0.bias['green'] = np.ones((32,32), dtype=np.float)*0.4
kpf0.bias['red'] = np.ones((32,32), dtype=np.float)*0.5 
kpf0.flat['green'] = np.ones((32,32), dtype=np.float)*0.2
kpf0.flat['red'] = np.ones((32,32), dtype=np.float)*0.1

# Recipe:
def MyRecipe(kpf0):

    with Pipeline(level0=kpf0, config='configs/default.cfg') as p:

        # Create configuration object, this is passed to methods as arguments
        # This will contain all arguments used by all methods
        # c = Configuration(params=params) # can be defined above
        #                                  # or be passed a file path where a config file is read in
        #                                  # should probably have some hierarchy: c.level0.variable_name
        print(p)
        p.subtract_bias()
        p.divide_flat()
        print(p)
        p.extract_spectrum()
        p.calibrate_wavelengths()
        print(p)
        p.correct_wavelength_dependent_barycentric_velocity()
        print(p)
        p.remove_emission_line_regions()
        p.remove_solar_regions()
        print(p)
        p.correct_telluric_lines()
        print(p)
        p.calculate_RV_from_spectrum()
        print(p)

MyRecipe(kpf0)

## To dos:
# x Implement logging.py to keep track of information/errors in each method
# - Discuss creating level1 object as independent mandatory pipeline method
# x Think about how to make the methods more flexible (e.g. bias subtract green only)
# x Create level 1 skeletons
# - Create level 2 skeletons
# - Think about what attributes each KPF Level objects need
#    - including vertical structure (e.g., orders, orderlets, etc.)
#    - pass around fits header (keeping some attributes/info from previous levels on creation of higher level object)
# 
#
# 1/11:
# - Arpita will make more detailed diagram
# - Erik will look at Airflow (might be better than Luigi)
# - Look at ESO pipeline
#
#









