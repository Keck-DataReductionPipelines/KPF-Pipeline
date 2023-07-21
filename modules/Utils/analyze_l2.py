import numpy as np
import matplotlib.pyplot as plt
from modules.Utils.kpf_parse import HeaderParse

class AnalyzeL2:

    """
    Description:
        This class contains functions to analyze L1 spectra (storing them
        as attributes) and functions to plot the results.

    Arguments:
        L2 - an L2 object

    Attributes:
        TBD
    """

    def __init__(self, L2, logger=None):
        if logger:
            self.logger = logger
            self.logger.debug('Initializing AnalyzeL2 object')
        else:
            self.logger = None
        self.L1 = L1
        #self.header = L1['PRIMARY'].header
        primary_header = HeaderParse(L2, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        self.ObsID = primary_header.get_obsid()
