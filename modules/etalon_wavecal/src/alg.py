import numpy as np
from astropy.io import fits
import os.path
import matplotlib.pyplot as plt

from modules.wavelength_cal.src.alg import LFCWaveCalibration
from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments

class EtalonWaveCalAlg:

    def __init__(self, config=None, logger=None):
        configpull = ConfigHandler(config,'PARAM')
        self.min_order = configpull.get_config_value('min_order', 40)
        self.max_order = configpull.get_config_value('max_order', 45)
        self.height_sigma = configpull.get_config_value('height_sigma', .5)
        #self.saveplots = configpull.get_config_value('saveplots', 'Etalon_plots')
        self.run = LFCWaveCalibration()

    def run_on_all_orders(self,flux):
        """Run wavelength calibration on all orders.

        Args:
            flux (np.array): Flux data
        """
        for order in np.arange(self.min_order,self.max_order):
            new_peaks,peaks,peak_heights,gauss_coeffs = self.run.find_peaks_in_order(flux[order],self.height_sigma)

