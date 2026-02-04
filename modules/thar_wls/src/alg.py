import os
import sys
import glob
import warnings

from astropy.time import Time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter
from astropy.stats import mad_std
from numpy.polynomial import polynomial, legendre

from kpfpipe.models.level1 import KPF1
from modules.Utils.kpf_parse import get_datecode, HeaderParse


class WLSAlg:
    def __init__(self, obs_ids, rough_wls):
        self.obs_ids = obs_ids
        self.rough_wls = rough_wls
        self.l1_stack = self._load_stack()
        
        # config
        # logger

        self.polyorder = 4


    def _load_stack(self):
        self.nobs = len(self.obs_ids)
        self.l1_stack = [None]*self.nobs

        for i, obs_id in enumerate(self.obs_ids):
            datecode = get_datecode(self.obs_ids[i])
            filepath = f'/data/L1/{datecode}/{self.obs_ids[i]}_L1.fits'
            self.l1_stack[i] = KPF1.from_fits(filepath, data_type='KPF')

        return self.l1_stack


    @staticmethod
    def _get_orderlet_ext_from_fiber_name(chip, fiber):
        flux_dict = {'SKY': f'{chip}_SKY_FLUX',
                    'SCI1': f'{chip}_SCI_FLUX1',
                    'SCI2': f'{chip}_SCI_FLUX2',
                    'SCI3': f'{chip}_SCI_FLUX3',
                    'CAL': f'{chip}_CAL_FLUX'
                    }

        var_dict = {'SKY': f'{chip}_SKY_VAR',
                    'SCI1': f'{chip}_SCI_VAR1',
                    'SCI2': f'{chip}_SCI_VAR2',
                    'SCI3': f'{chip}_SCI_VAR3',
                    'CAL': f'{chip}_CAL_VAR'
                }

        wave_dict = {'SKY': f'{chip}_SKY_WAVE',
                    'SCI1': f'{chip}_SCI_WAVE1',
                    'SCI2': f'{chip}_SCI_WAVE2',
                    'SCI3': f'{chip}_SCI_WAVE3',
                    'CAL': f'{chip}_CAL_WAVE'
                }
        
        return flux_dict[fiber], var_dict[fiber], wave_dict[fiber]


    @staticmethod
    def _get_norder(chip):
        norder_dict = {'GREEN':35, 'RED':32}

        return norder_dict[chip]


    @staticmethod
    def clipped_mean(x, p):
        xmin, xmax = np.nanpercentile(x, [p/2,100-p/2])
        return np.nanmean(x[(x > xmin) & (x < xmax)])

    
    @staticmethod
    def gaussian(theta, x):
        mu, sigma, a, b = theta

        return b + a * np.exp(-(x-mu)**2/(2*sigma**2))

    
    @staticmethod
    def optimize_lsq(func, theta0, x, y):
        """
        optimize theta for a given function using non-linear least-squares
        """
        def _residuals(theta, x, y):
            return func(theta, x) - y
        
        result = least_squares(_residuals, theta0, args=(x,y))
        theta, rms = result.x, np.std(result.fun)
        
        return theta, rms
