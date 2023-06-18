import numpy as np

class AnalyzeL0:

    """
    Description:
        This class [will] contain functions to analyze L0 images (storing 
        them as attributes) and functions to plot the results.  
        Currently, this class is a placeholder for future functions to be
        added

    Arguments:
        L0 - an L0 object

    Attributes:
        TBD
    """

    def __init__(self, L0, logger=None):
        self.L0 = L0
        if logger:
            self.logger = logger
            self.logger.debug('AnalyzeL0 class constructor')
        else:
            self.logger = None
            print('---->AnalyzeL0 class constructor')
