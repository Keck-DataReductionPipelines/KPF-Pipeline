
from keckdrpframework.primitives.base_primitive import Base_primitive
from keckdrpframework.models.arguments import Arguments


class KPF_Primitive(Base_primitive):
    """
    Base primitive for orther KPF primitives.
    All KPF primitive classes should ultimately inherit from this one.

    Args:
        action (keckdrpframework.models.action.Action): Keck DRPF Action object
        context (keckdrpframework.models.processing_context.Processing_context): Keck DRPF Processing_context object

    """

    def __init__(self, action, context):
        Base_primitive.__init__(self, action, context)

        self.action = action
        self.context = context
