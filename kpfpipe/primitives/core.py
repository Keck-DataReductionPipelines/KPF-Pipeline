
import traceback

from keckdrpframework.primitives.base_primitive import BasePrimitive
from keckdrpframework.models.arguments import Arguments


class KPF_Primitive(BasePrimitive):
    """
    Base primitive for orther KPF primitives.
    All KPF primitive classes should ultimately inherit from this one.

    Args:
        action (keckdrpframework.models.action.Action): Keck DRPF Action object
        context (keckdrpframework.models.processing_context.ProcessingContext): Keck DRPF ProcessingContext object

    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)

        self.action = action
        self.context = context
        self.logger = self.context.logger

    def apply(self):
        try:
            if self._pre_condition():
                self.output = self._perform()
                if self._post_condition():
                    return self.output
        except Exception as e:
            self.logger.error(f"Failed executing primitive {self.__class__.__name__}: {e}\n{traceback.format_exc()}")
            raise(e)
        return None