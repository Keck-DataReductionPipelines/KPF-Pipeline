# This file is currently a copy from the continuum normalization module.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib as mpl
mpl.use('Agg')
from astropy.io import fits
from matplotlib import gridspec
import os
import scipy.interpolate as inter
# import pyreduce
# import alphashape
# import shapely
# from math import ceil
# from scipy import linalg

from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0 
from keckdrpframework.models.arguments import Arguments


class LineActivityAlg:
    """
    Spectral Line Activity module algorithm. 
    Purpose is to measure mesaure the equivalent width of spectral line (or other metrics)
    Attributes:
        config_param(ConfigHandler): Instance representing pull from config file.
    """

    def __init__(self, config=None, logger=None):
        """Initializes line activity algorithm.
        Args:
#            mask_array_provided (boolean): Whether or not mask array is provided.
#           method (str): Method of continuum normalization within the following:
#                'Spline','AFS','Polynomial','Pyreduce'.
#            continuum_guess_provided (boolean): If initial guess of continuum normalization
#                is provided.
            plot_results (boolean): Whether to plot results
#            n_iter (int): Number of iterations
#            n_order (int): Number of order in polynomial fit
#            ffrac (float): Percentile above which is considered as continuum
#            a (int): Radius of AFS circle (1/a)
#            d (float): Window width in AFS
            config (configparser.ConfigParser, optional): Config context. Defaults to None.
            logger (logging.Lobber, optional): Instance of logging.Logger. Defaults to None.
        """
        configpull=ConfigHandler(config,'PARAM')
#        self.mask_array_provided=configpull.get_config_value('mask_array_provided',False)
#        self.method=configpull.get_config_value('method','Spline')
#        self.continuum_guess_provided=configpull.get_config_value('continuum_guess_provided',False)
        self.plot_results=configpull.get_config_value('plot_results',True)
#        self.n_iter=configpull.get_config_value('n_iter',5)
#        self.n_order=configpull.get_config_value('n_order',8)
#        self.ffrac=configpull.get_config_value('ffrac',0.98)
#        self.med_window=configpull.get_config_value('median_window',15)
#        self.std_window=configpull.get_config_value('std_window',15)
#        self.a=configpull.get_config_value('a',6)
#        self.d=configpull.get_config_value('d',.25)
#        self.edge_clip=configpull.get_config_value('edge_clip',1000)
#        self.output_dir=configpull.get_config_value('output_dir','/Users/paminabby/Desktop/cn_test')
        self.config=config
        self.logger=logger

    def CalcHalpha(self,x,y,window):
        """ Perform equivalent width calculation
        Args:
            x (np.array): X-data (wavelength).
            y (np.array): Y-data (flux).
            w0 (float)  : center of extraction region
            width(float): length of wavelength to be summed
        Returns:
            EW (float)  : Equivalent width
        """
        Halpha_EW = np.sum(x)
        return ss





