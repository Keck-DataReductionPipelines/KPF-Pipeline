import numpy as np

# Pipeline dependencies
from kpfpipe.logger import start_logger
from kpfpipe.primitives.level1 import KPF1_Primitive

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext


class save_wl_pixel_info(KPF1_Primitive):

    def __init__(self, 
                 action: Action,
                 context: ProcessingContext) -> None:
        """
        TODO: document
        """
        # Initialize parent class
        KPF1_Primitive.__init__(self, action, context)
        # start a logger
        self.logger = start_logger(self.__class__.__name__, None)

        self.filename = self.action.args[0]
        self.data = self.action.args[1]

    def _perform(self) -> None:
        """
        TODO: document
        """
        
        np.save(self.filename, self.data, allow_pickle=True)