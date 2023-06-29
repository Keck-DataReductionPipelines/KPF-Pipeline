import numpy as np
import matplotlib.pyplot as plt

class Analyze2D:

    """
    Description:
        This class contains functions to analyze 2D images (storing them
        as attributes) and functions to plot the results.
        It is only a placeholder for now.

    Arguments:
        2D - a 2D object

    Attributes:
        TBD
    """

    def __init__(self, D2, logger=None):
        self.D2 = D2 # use D2 instead of 2D because variable names can't start with a number
        if logger:
            self.logger = logger
            self.logger.debug('Analyze2D class constructor')
        else:
            self.logger = None
            print('---->Analyze2D class constructor')

 
