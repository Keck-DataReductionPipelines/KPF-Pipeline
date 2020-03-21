# FauxLevel0Primatives.py

from keckdrpframework.models.action import Action
from keckdrpframework.models.processing_context import ProcessingContext
from kpf_pipeline_args import KpfPipelineArgs

"""
Provides some faked primitives that do nothing but provide results
that allow testing of the pipeline parsing and state tree walking.
See example.recipe and simple_example.recipe for test usage.
"""
def read_data(action: Action, context: ProcessingContext):
    """
    read_data
    args: data_set
    results: numpy.array
    """

    output = ([0, 1, 2, 3, 4, 5, 6],)
    return KpfPipelineArgs(action.args.visitor, action.args.tree, (output,))

def Normalize(action: Action, context: ProcessingContext):
    """
    Normalize
    args:
        input: numpy.array
    results: (numpy.array, result: numeric)
    """
    result = 0.5
    input = action.args.args[0]
    return KpfPipelineArgs(action.args.visitor, action.args.tree, (input, result))

def NoiseReduce(action: Action, context: ProcessingContext):
    param = 0.
    input = action.args.args[0]
    if len(action.args.args) > 1:
        param = action.args.args[1]
    result = (param != 0.)
    return KpfPipelineArgs(action.args.visitor, action.args.tree, (input, result))

def Spectrum1D(action: Action, context: ProcessingContext):
    param = None
    input = action.args.args[0]
    if len(action.args.args) > 1:
        param = action.args.args[1]
    result = (param is not None)
    return KpfPipelineArgs(action.args.visitor, action.args.tree, (input, result))