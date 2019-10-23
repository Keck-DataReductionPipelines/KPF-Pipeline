
from keckdrpframework.pipelines.base_pipeline import Base_pipeline
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.core.framework import Framework
from keckdrpframework.config.framework_config import ConfigClass

from kpfpipe.primitives.level0 import *


class KPF_pipeline(Base_pipeline):
    """
    Pipeline to Process KPF data
    """

    event_table = {"process_level0": ("subtract_bias", "removing bias", 'divide_flat'),
                   "divide_flat": ("divide_flat", "dividing by flat", 'extract_spectrum'),
                   "extract_spectrum": ("extract_spectrum", "extracting spectrum", None)
                   }

    def __init__(self):
        """
        Constructor
        """
        Base_pipeline.__init__(self)

