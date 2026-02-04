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