# fits_primitives.py

from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext
from keckdrpframework.primitives.base_primitive import BasePrimitive
from kpfpipe.models.level0 import KPF0
from kpfpipe.models.level1 import KPF1
from kpfpipe.models.level2 import KPF2

#from kpfpipe.models.kpf_arguments import KpfArguments

"""
Provides pipeline primitive wrappers around data model to_fits and
from_fits methods.
"""
class to_fits(BasePrimitive):
    """
    to_fits: pipeline primitive to write data_model to FITS file
    """
    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)

    def _perform(self):
        """
        _perform

        inputs
            args[0]: data model (python subclassed from KPFDataModel)
            args[1]: FITS filename (path) as string.  Should be
                    extracted from config in recipe.
        outputs
            result:  Always True
        """
        data_model = self.action.args[0]
        file_name = self.action.args[1]
        data_model.to_fits(file_name)
        return Arguments(True, name='to_fits_result')

class FromFitsBasePrimitive(BasePrimitive):
    """
    FromFitsPrimitive: create a data model object and instiantiate
    its contents from a FITS file. 
    """
    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
    
    def _perform_common(self, data_model, name):
        """
        _perform_common
        
        arguments
            data_model: instance of a subclass of KPFDataModel
            name: name of data model class as str
        inputs
            args[0]: Name of FITS file (path). Should be extracted
                     from config in recipe.
            data_type: 'KPF' or 'NEID. Defaults to 'KPF'
        outputs
            python object of type data_model
        """
        filename = self.action.args[0]
        try:
            data_type = self.action.args['data_type']
        except KeyError:
            data_type = 'KPF'
        print(f"_perform_common: {filename} data_type is {data_type}")
        data_model = data_model.from_fits(filename, data_type)
        return Arguments(data_model, name=name+'_from_fits_result')


class kpf0_from_fits(FromFitsBasePrimitive):
    """
    kpf0_from_fits: create a KPF0 data model object and instantiate
                    its contents from the given FITS file
    """
    def __init__(self, action, context):
        FromFitsBasePrimitive.__init__(self, action, context)

    def _perform(self):
        return self._perform_common(KPF0, 'kpf0')


class kpf1_from_fits(FromFitsBasePrimitive):
    """
    kpf1_from_fits: create a KPF1 data model object and instantiate
                    its contents from the given FITS file
    """
    def __init__(self, action, context):
        FromFitsBasePrimitive.__init__(self, action, context)

    def _perform(self):
        return self._perform_common(KPF1, 'kpf1')


class kpf2_from_fits(FromFitsBasePrimitive):
    """
    kpf02from_fits: create a KPF2 data model object and instantiate
        its contents from the given FITS file
    """
    def __init__(self, action, context):
        FromFitsBasePrimitive.__init__(self, action, context)

    def _perform(self):
        return self._perform_common(KPF2, 'kpf2')

