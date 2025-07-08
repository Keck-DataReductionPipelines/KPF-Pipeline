import warnings
from datetime import datetime, timedelta

import astropy.constants as apc
from astropy.stats import mad_std
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as poly
import pandas as pd
from scipy.ndimage import median_filter, gaussian_filter
from scipy.interpolate import LSQUnivariateSpline, CubicSpline

from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger
from modules.Utils.config_parser import ConfigHandler

class StrayLightAlg:
    """
    This module defines 'StrayLight' and methods to perform stray (scattered) light estimation from inter-order pixels.

    Args:
        target_2D (KPF0): A KPF 2D science object
        master_order_mask (KPF0): a KPF master order mask
        config (configparser.ConfigParser): Config context
        logger (logging.Logger): Instance of logging.Logger
    
    Attributes:
        rawimage (np.ndarray): From parameter 'rawimage'.

    """
    def __init__(self, 
                 target_2D, 
                 master_order_mask,
                 default_config_path,
                 logger=None
                ):
        """
        Inits StrayLight class with raw data, order mask, config, logger.
        
        Args:
            target_2D (KPF0): A KPF 2D science object
            master_order_mask (KPF0): a KPF master order mask
            config (configparser.ConfigParser): Config context
            logger (logging.Logger): Instance of logging.Logger
        """

        # Input arguments
        self.config = ConfigClass(default_config_path)
        if logger == None:
            self.log = start_logger('StrayLight', default_config_path)
        else:
            self.log = logger
            
        cfg_params = ConfigHandler(self.config, 'PARAM')
        self.method = cfg_params.get_config_value('method')
        self.polyorder = int(cfg_params.get_config_value('polyorder'))
        self.edge_clip = int(cfg_params.get_config_value('edge_clip'))

        self.target_2D = target_2D        
        self.master_order_mask = master_order_mask
        self.drptag = self.target_2D.header['PRIMARY']['DRPTAG']

    
    def add_keywords(self, stray_light_image, inter_order_mask, chip):
        """
        Adds keywords to track basic stray light statistics (mean, rms, min, max)
        """
        header = self.target_2D.header['PRIMARY']

        if self.method == 'polynomial':
            method = f"{self.method}_{self.polyorder}"
        else:
            method = self.method

        if chip == 'GREEN':
            slg = stray_light_image['GREEN_CCD'][~inter_order_mask['GREEN_CCD']]

            header['SLGMETH'] = method            # COMMENT method used to estimate stray light for GREEN
            header['SLGMEAN'] = np.mean(slg)      # COMMENT mean of GREEN inter-order stray light
            header['SLGRMS']  = np.std(slg)       # COMMENT standard deviation of GREEN inter-order stray light
            header['SLGMIN']  = np.min(slg)       # COMMENT minimum of GREEN inter-order stray light
            header['SLGMAX']  = np.max(slg)       # COMMENT maximum of GREEN inter-order stray light

        elif chip == 'RED':
            slr = stray_light_image['RED_CCD'][~inter_order_mask['RED_CCD']]

            header['SLRMETH'] = method            # COMMENT method used to estimate stray light for RED
            header['SLRMEAN'] = np.mean(slr)      # COMMENT mean of RED inter-order stray light
            header['SLRRMS']  = np.std(slr)       # COMMENT standard deviation of RED inter-order stray light
            header['SLRMIN']  = np.min(slr)       # COMMENT minimum of RED inter-order stray light
            header['SLRMAX']  = np.max(slr)       # COMMENT maximum of RED inter-order stray light


    def estimate_stray_light(self, chip):
        """
        Main method used to estimate stray light
        Calls method defined in config file; allowed methods are 'zero', 'mean', and 'polynomial'

        Returns:
            stray_light_image (dict of ndarrays): 2D stray light images for GREEN and RED ccds
            inter_order_mask (dict of ndarrays): 2D boolean mask of inter-order pixels for GREEN and RED ccds
        """
        try:
            stray_light_method = self.__getattribute__(self.method)
        except AttributeError:
            #self.log.error(f'Stray light method {self.method} not implemented.')
            raise(AttributeError)

        stray_light_image, inter_order_mask = stray_light_method(chip)
        self.add_keywords(stray_light_image, inter_order_mask, chip)

        return stray_light_image, inter_order_mask

    
    def zero(self, chip):
        """
        Method to estimate stray light -- returns zero (i.e. no stray light)

        Args:
            chip (str) : 'GREEN' or 'RED' to select which CCD to fit

        Returns:
            stray_light (ndarray): 2D stray light image
            mask (ndarray): 2D boolean mask of inter-order pixels 
        """
        mask = self._inter_order_mask(chip).astype('bool')
        stray_light = np.zeros_like(self.target_2D[f'{chip}_CCD'])

        return stray_light, mask

    
    def mean(self, chip):
        """
        Method to estimate stray light -- returns mean of inter-order pixels

        Args:
            chip (str) : 'GREEN' or 'RED' to select which CCD to fit

        Returns:
            stray_light (ndarray): 2D stray light image
            mask (ndarray): 2D boolean mask of inter-order pixels 
        """
        data = np.array(self.target_2D[f'{chip}_CCD'].data)
        mask = self._inter_order_mask(chip).astype('bool')
        stray_light = np.mean(data[~mask])*np.ones_like(data)
    
        return stray_light, mask

    
    def polynomial(self, chip):
        """
        Method to estimate stray light -- fits a 2D polynomial to inter-order pixels
        
        Args:
            chip (str) : 'GREEN' or 'RED' to select which CCD to fit

        Returns:
            stray_light (ndarray): 2D stray light image
            mask (ndarray): 2D boolean mask of inter-order pixels 
        """
        data = np.array(self.target_2D[f'{chip}_CCD'].data)
        mask = self._inter_order_mask(chip).astype('bool')

        clip = self.edge_clip
        coeffs = self._polyfit2d(data[clip:-clip,clip:-clip],
                                    self.polyorder,
                                    mask[clip:-clip,clip:-clip]
                                )
    
        stray_light = self._polyval2d(coeffs, self.polyorder, data.shape)
        stray_light = np.maximum(stray_light, 0)

        return stray_light, mask
        

    def _polyfit2d(self, data_image, polyorder, mask=None):
        # coordinate grid
        nrow, ncol = data_image.shape
        y, x = np.mgrid[0:nrow, 0:ncol]  
        x = np.ravel(x)
        y = np.ravel(y)
        z = np.ravel(data_image)
    
        # mask exposed pixels
        if mask is not None:
            mask = np.array(mask, dtype='bool').ravel()
        else:
            mask = np.zeros((nrow,ncol), dtype='bool').ravel()

        x = x[~mask]
        y = y[~mask]
        z = z[~mask]
        
        # Build design matrix
        terms = []
        for i in range(polyorder + 1):
            for j in range(polyorder + 1 - i):
                terms.append((x**i) * (y**j))
        
        A = np.vstack(terms).T
        
        # Solve least squares
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    
        return coeffs
    
    
    def _polyval2d(self, coeffs, polyorder, shape):
        nrow, ncol = shape

        y,x = np.mgrid[0:nrow,0:ncol]
        x = x.ravel()
        y = y.ravel()
    
        terms = []
        for i in range(polyorder + 1):
            for j in range(polyorder + 1 - i):
                terms.append((x**i) * (y**j))
        A = np.vstack(terms)

        result = np.dot(coeffs, A).reshape((nrow,ncol))
    
        return result


    def _inter_order_mask(self, chip, dark_fibers=None):
        """
        dark_fibers is a list of integers 1-5 indicating any non-illuminated fibers
        """
        mask = self.master_order_mask[f'{chip}_CCD'] == 0

        if dark_fibers is not None:
            for fiber in dark_fibers:
                mask += self.master_order_mask[f'{chip}_CCD'] == fiber

        return ~mask