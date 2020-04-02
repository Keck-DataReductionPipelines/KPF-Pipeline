# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level1 import KPF1_Primitive
from kpfpipe.models.level1 import KPF1

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext


class KPFModExample(KPF1_Primitive):

    def __init__(self, 
                 action: Action,
                 context: ProcessingContext) -> None:
        '''
        Example KPF module
        '''
        # Initialize parent class
        KPF1_Primitive.__init__(self, action, context)
        # start a logger
        self.logger = start_logger(self.__class__.__name__, None)

    def _perform(self) -> None:
        '''
        This primitive's action
        '''
        print('[{}] Performed!'.format(self.__class__.__name__))
