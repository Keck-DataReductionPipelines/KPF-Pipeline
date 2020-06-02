"""
Define primitives that operate on KPF data
"""

import numpy as np

from kpfpipe.primitives.core import KPF_Primitive

class KPF0_Primitive(KPF_Primitive):
    """
    Base primitive for other KPF0 primitives.
    All KPF0 primitives should inherit from this one.
    
    Args:
        action (keckdrpframework.models.action.Action): Keck DRPF Action object
        context (keckdrpframework.models.ProcessingContext.ProcessingContext): Keck DRPF ProcessingContext object

    """
    def __init__(self, action, context):
        KPF_Primitive.__init__(self, action, context)

        self.level0 = action.args.level0

    def checklevel0(self):
        if ((self.level0 == None) or
                (not self.valid_level0_data())):
            raise (TypeError, "Invalid data")

    def valid_level0_data(self):
        """
        Confirms that KPF0 object has necessary data/structure to operate on (after checking for its existence)

        Returns:
            bool
        """

        # The absolutely necessary data in a level0 array is in the self.data dictionary.
        # Check that it has not been corrupted
        if type(self.level0.data) is not dict:
            return False
        # Check that it contains some data
        if len(self.level0.data) <= 0:
            return False
        # And that the data is appropriate
        for key in self.level0.data:
            if ((not isinstance(self.level0.data[key], np.ndarray)) or
                    not np.all(np.isfinite(self.level0.data[key]))):
                return False
            # Could check for dimensionality, but we won't do this for flexibility
            # if self.data[key].shape == shape:
            #    return False
        # We will also want to check some other things eventually
        # if self.level0.header is None:
        #    return False
        return True

    def _pre_condition(self):
        if (self.action.args.level0 is None) or (not self.valid_level0_data()):
            self.logger.error("Invalid level 0 data")
            return False
        else:
            return True