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
