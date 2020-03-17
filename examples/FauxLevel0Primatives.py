# FauxLevel0Primatives.py

"""
Provides some faked primitives that do nothing but provide results
that allow testing of the pipeline parsing and state tree walking.
See example.recipe and simple_example.recipe for test usage.
"""
def read_data(input):
    return [0, 1, 2, 3, 4, 5, 6]

def Normalize(input):
    result = 0.5
    return (input, result)

def NoiseReduce(input, param=0.):
    result = (param != 0.)
    return (input, result)

def Spectrum1D(input, param=None):
    result = (param is not None)
    return (input, result)