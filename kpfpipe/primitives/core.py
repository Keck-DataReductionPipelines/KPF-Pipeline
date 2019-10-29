
from keckdrpframework.primitives.base_primitive import Base_primitive
from keckdrpframework.models.arguments import Arguments


class KPF_Primitive(Base_primitive):
    """
    Base primitive for orther KPF primitives.
    All KPF primitive classes should ultimately inherit from this one.

    """

    def __init__(self, action, context):
        Base_primitive.__init__(self, action, context)

        self.action = action
        self.context = context
