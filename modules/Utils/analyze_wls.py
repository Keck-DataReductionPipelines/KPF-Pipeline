import numpy as np
import matplotlib.pyplot as plt
from modules.Utils.kpf_parse import HeaderParse
from datetime import datetime

class AnalyzeWLS:

    """
    Description:
        This class contains functions to analyze wavelength solutions 
        (storing them as attributes) and functions to plot the results.

    Arguments:
        L1 - an L1 object

    Attributes:
        None so far
    """

    def __init__(self, L1, logger=None):
        if self.logger:
            self.logger = logger
            self.logger.debug('Initializing AnalyzeWLS object.')
        else:
            self.logger = None
        self.L1 = L1
        primary_header = HeaderParse(L1, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        
        #self.ObsID = primary_header.get_obsid()
        # need a filename instead or in addition
