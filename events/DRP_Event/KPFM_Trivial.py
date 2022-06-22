"""
    The file contains primitive KPFModeExample

    Attributes:
       KPFModeExample: primitive for simmple recipe test


"""

from kpfpipe.primitives.core import KPF_Primitive

# External dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext


class KPFModExample(KPF_Primitive):
    def __init__(self,
                 action: Action,
                 context: ProcessingContext) -> None:

        # Initialize parent class
        KPF_Primitive.__init__(self, action, context)

        self.logger = self.context.logger

    def _pre_condition(self) -> bool:
        """
        check if the extensions exist in the data model object
        """
        return True

    def _post_condition(self) -> bool:
        return True

    def _perform(self):
        result = 'KPFModeExample done'
        print('[{}] Performed!'.format(self.__class__.__name__))
        return Arguments(result)

