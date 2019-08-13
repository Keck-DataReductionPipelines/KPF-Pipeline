from kpfpipe.pipeline import KPFPipe
from kpfpipe.level0 import KPF0
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


def MyRecipe(kpf0):
    """Sample custom recipe"""

    with KPFPipe(level0=kpf0, config='configs/default.cfg') as p:
        print(p)
        p.subtract_bias(chips=True)
        p.divide_flat()
        print(p)
        p.extract_spectrum(True, orders=[10, 11])
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


# run the recipe
MyRecipe(kpf0)
