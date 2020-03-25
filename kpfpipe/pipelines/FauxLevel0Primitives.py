# FauxLevel0Primitives.py

from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext
from keckdrpframework.primitives.base_primitive import BasePrimitive

"""
Provides some faked primitives that do nothing but provide results
that allow testing of the pipeline parsing and state tree walking.
See example.recipe and simple_example.recipe for test usage.
"""
class read_data(BasePrimitive):
    """
    read_data -- fake processing primitive
    """
    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)

    def _perform(self):
        """
        read_data
        args: data_set
        results: numpy.array
        """
        # unused example input (badly named as output)
        filename = self.output.arg0
        output = {'name': "read_data_output", 'arg0': [0, 1, 2, 3, 4, 5, 6]}
        return Arguments(**output)

class Normalize(BasePrimitive):
    """
    Normalize
    args:
        input: numpy.array
    results: (numpy.array, result: numeric)
    """
    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)

    def _perform(self):
        # expect argument "d"
        input = self.output.arg0 # badly named in base class
        output = {'name': "Normalize_output"}
        output['arg0'] = input
        output['arg1'] = 0.5
        return Arguments(**output)

class NoiseReduce(BasePrimitive):
    """
    NoiseReduce - fake primitive
    """
    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)

    def _perform(self):
        # (input, param=0.)
        input = self.output.arg0
        param = getattr(self.output, "arg1", 0.)
        result = (param != 0.)
        output = {'name': "NoiseReduce_output"}
        output['arg0'] = input
        output['arg1'] = result
        return Arguments(**output)

class Spectrum1D(BasePrimitive):
    """
    Spectrum1D - fake primitive
    """
    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)

    def _perform(self):
        # (input, param)
        input= self.output.arg0 # badly named in base class
        param = getattr(self.output, 'arg1', None)
        result = (param is not None)
        output = {'name': "Spectrum1D_output"}
        output['arg0'] = input
        output['arg1'] = result
        return Arguments(**output)
