
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
