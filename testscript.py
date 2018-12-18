from kpfpipe.pipeline import Pipeline
from kpfpipe.level0 import KPF0
from kpfpipe.level1 import KPF1
from kpfpipe.level2 import KPF2
import numpy as np

kpf0 = KPF0()
kpf0.green = np.ones((32,32), dtype=np.float) 
kpf0.red = np.ones((32,32), dtype=np.float) 
kpf0.bias_green = np.ones((32,32), dtype=np.float)*0.4
kpf0.bias_red = np.ones((32,32), dtype=np.float)*0.5 
kpf0.flat_green = np.ones((32,32), dtype=np.float)*0.2
kpf0.flat_red = np.ones((32,32), dtype=np.float)*0.1

p = Pipeline(level0=kpf0)
print(p)
p.subtract_bias()
p.divide_flat()
print(p)
p.extract_spectrum()
p.calibrate_wavelengths()
print(p)


