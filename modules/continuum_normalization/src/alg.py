#import pylab as pl
#import matplotlib.pyplot as plt
import numpy as np

from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0 
from keckdrpframework.models.arguments import Arguments

class ContinuumNorm:
    """
    Continuum normalization module algorithm. Purpose is to measure and remove variability in blaze
    of stellar spectrum.

    Args:

    Attributes:

    Raises:
    """

    def __init__(self, config=None, logger=None):
        """Initializes ContinuumNorm class with rawspec, config, and logger.

        Args:
            config (configparser.ConfigParser, optional): Config context. Defaults to None.
            logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.
        
        Attributes:

        """
        configpull=ConfigHandler(config,'PARAM')
        self.weight=configpull.get_config_value('weight',0.)
        self.flatspec_order=configpull.get_config_value('flatspec_order',4)
        self.config=config
        self.logger=logger

    def flatspec(self, rawspec):
        """
        Polynomial fit model of spectrum.

        Args:
            rawspec (np.ndarray): Raw spectrum data.
        """

        ffrac = .95 #get from config instead?
        x = np.arange(0,len(rawspec),1)

        #plt.plot(x,rawspec)
        #plt.show()

        pos = np.where((np.isnan(rawspec)==False)&(np.isnan(self.weight)==False))[0]
        coef = np.polyfit(x[pos],rawspec[pos],self.order)
        poly = np.poly1d(coef)
        yfit = poly(x)
        for i in range(8):
            normspec = rawspec / yfit
            pos = np.where((normspec >= ffrac))[0]#& (normspec <= 2.)
            coef = np.polyfit(x[pos],rawspec[pos],i+2)
            poly = np.poly1d(coef)
            yfit = poly(x)
        
        #pl.plot(rawspec,'k-')
        #pl.plot(x,yfit,'b-')
        #pl.show()

        normspec = rawspec / yfit

        return normspec, yfit

        