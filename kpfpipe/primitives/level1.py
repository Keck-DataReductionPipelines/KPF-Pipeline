"""
Define primitives that operate on KPF data
"""

import numpy as np

from kpfpipe.primitives.core import KPF_Primitive
from kpfpipe.logger import start_logger

class KPF1_Primitive(KPF_Primitive):
    """
    Base primitive for other KPF1 primitives.
    All KPF1 primitives should inherit from this one.
    
    Args:
        action (keckdrpframework.models.action.Action): Keck DRPF Action object
        context (keckdrpframework.models.ProcessingContext.ProcessingContext): Keck DRPF ProcessingContext object

    """
    def __init__(self, action, context):
        KPF_Primitive.__init__(self, action, context)

<<<<<<< HEAD
        # Argument can be a KPF1 data type, or a
        # list of KPF1 data type
        self.arg = []
=======
>>>>>>> 348817e6f04f0058c0cdf81c7f78480c6773997e
