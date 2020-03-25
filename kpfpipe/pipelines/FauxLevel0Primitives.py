# FauxLevel0Primitives.py

from keckdrpframework.models.action import Action
from keckdrpframework.models.processing_context import ProcessingContext
from keckdrpframework.primitives.base_primitive import BasePrimitive
from kpfpipe.models.kpf_arguments import KpfArguments

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
        filename = self.output[0]
        return KpfArguments([0, 1, 2, 3, 4, 5, 6], name='read_data_results')

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
        input = self.action.args[0]
        return KpfArguments(input, 0.5, name='Normalize_results')

class NoiseReduce(BasePrimitive):
    """
    NoiseReduce - fake primitive
    """
    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)

    def _perform(self):
        # (input, param=0.)
        param = 0.
        if len(self.action.args) == 0:
            raise Exception("NoiseReduce._perform: at least one argument is needed")
        input = self.action.args[0]
        if len(self.action.args) > 1:
            param = self.action.args[1]
        result = (param != 0.)
        return KpfArguments(input, result, name='NoiseReduce_results')

class Spectrum1D(BasePrimitive):
    """
    Spectrum1D - fake primitive
    """
    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)

    def _perform(self):
        # (input, param)
        param = None
        if len(self.action.args) == 0:
            raise Exception("Spectrum1D._perform: at least one argument is needed")
        input= self.action.args[0]
        if len(self.action.args) > 1:
            param = self.action.args[1]
        result = (param is not None)
        print(f"Spectrum1D: input is {input}, result is {result}")
        return KpfArguments(input, result, name='Spectrum1D_results')
