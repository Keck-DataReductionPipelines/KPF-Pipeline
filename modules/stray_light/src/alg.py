import warnings
from datetime import datetime, timedelta

import astropy.constants as apc
from astropy.stats import mad_std
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial.legendre import legval
import pandas as pd
from scipy.ndimage import median_filter, gaussian_filter, distance_transform_edt
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
        self.method = str(cfg_params.get_config_value('method'))
        self.polyorder = int(cfg_params.get_config_value('polyorder'))
        try:
            self.regularize = float(cfg_params.get_config_value('method'))
        except ValueError:
            self.regularize = str(cfg_params.get_config_value('regularize'))
        self.edge_clip = int(cfg_params.get_config_value('edge_clip'))
        self.mask_buffer = int(cfg_params.get_config_value('mask_buffer'))

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
            slg = stray_light_image[~inter_order_mask]

            header['SLGMETH'] = method            # COMMENT method used to estimate stray light for GREEN
            header['SLGMEAN'] = np.mean(slg)      # COMMENT mean of GREEN inter-order stray light
            header['SLGRMS']  = np.std(slg)       # COMMENT standard deviation of GREEN inter-order stray light
            header['SLGMIN']  = np.min(slg)       # COMMENT minimum of GREEN inter-order stray light
            header['SLGMAX']  = np.max(slg)       # COMMENT maximum of GREEN inter-order stray light

        elif chip == 'RED':
            slr = stray_light_image[~inter_order_mask]

            header['SLRMETH'] = method            # COMMENT method used to estimate stray light for RED
            header['SLRMEAN'] = np.mean(slr)      # COMMENT mean of RED inter-order stray light
            header['SLRRMS']  = np.std(slr)       # COMMENT standard deviation of RED inter-order stray light
            header['SLRMIN']  = np.min(slr)       # COMMENT minimum of RED inter-order stray light
            header['SLRMAX']  = np.max(slr)       # COMMENT maximum of RED inter-order stray light


    def remove_stray_light(self, 
                             chip, 
                             method=None, 
                             polyorder=None, 
                             regularize=None,
                             edge_clip=None, 
                             mask_buffer=None,
                             return_model=False
                             ):
        """
        Main method used to estimate and remove stray light
        Calls method defined in config file; allowed methods are 'zero', 'mean', and 'polynomial'

        Args:
            chip (str) : 'GREEN' or 'RED' to select which CCD to fit
            method (str) : fitting method, allowed methods are 'zero', 'mean' and 'polynomial'
            polyorder (int) : order of polynomial
            regularize : can be 'none', 'auto', or a positive float (lambda parameter for T2-ridge regression)
            edge_clip (int) : number of pixels to ignore on each edge (default=0)
            mask_buffer (int): illuminated mask region will be widened by N=mask_buffer pixels
            return_model (bool): True to return ndarrays for smooth stray light model and inter-order mask
            
        Returns:
            out_2D (KPF0): level 2D object with stray light removed from CCD image
            if return_model=True:
                stray_light_image (dict of ndarrays): stray light images for GREEN and RED ccds
                inter_order_mask (dict of ndarrays): boolean masks of inter-order pixels for GREEN and RED ccds
        """
        # set parameters
        if method is None:
            method = self.method
        if polyorder is None:
            polyorder = self.polyorder
        if regularize is None:
            regularize = self.regularize
        if edge_clip is None:
            edge_clip = self.edge_clip
        if mask_buffer is None:
            mask_buffer = self.mask_buffer

        # select method and estimate stray light
        try:
            stray_light_method = self.__getattribute__(method)
        except AttributeError:
            #self.log.error(f'Stray light method {self.method} not implemented.')
            raise(AttributeError)

        stray_light_image, inter_order_mask = stray_light_method(chip, 
                                                                 method=method, 
                                                                 polyorder=polyorder, 
                                                                 regularize=regularize,
                                                                 edge_clip=edge_clip, 
                                                                 mask_buffer=mask_buffer
                                                                )
        
        self.add_keywords(stray_light_image, inter_order_mask, chip)

        # remove stray light from science image
        out_2D = self.target_2D
        out_2D[f'{chip}_CCD'] -= stray_light_image

        if return_model:
            return out_2D, stray_light_image, inter_order_mask

        return out_2D

    
    def zero(self, chip, **kwargs):
        """
        Method to estimate stray light -- returns zero (i.e. no stray light)
        """
        print("zero")
        
        mask = self._inter_order_mask(chip).astype('bool')
        stray_light = np.zeros_like(self.target_2D[f'{chip}_CCD'])

        return stray_light, mask

    
    def mean(self, chip, edge_clip=0, mask_buffer=None, **kwargs):
        """
        Method to estimate stray light -- returns mean of inter-order pixels
        """
        data = np.array(self.target_2D[f'{chip}_CCD'].data)
        mask = self._inter_order_mask(chip, mask_buffer=mask_buffer).astype('bool')
        
        if edge_clip > 0:
            d = data[edge_clip:-edge_clip,edge_clip:-edge_clip]
            m = mask[edge_clip:-edge_clip,edge_clip:-edge_clip]
        
        stray_light = np.mean(d[~m])*np.ones_like(d)
    
        return stray_light, mask

    
    def polynomial(self, chip, polyorder, regularize='none', edge_clip=0, mask_buffer=None, **kwargs):
        """
        Method to estimate stray light -- fits a 2D polynomial to inter-order pixels
        """
        data = deepcopy(np.array(self.target_2D[f'{chip}_CCD'].data))
        mask = deepcopy(self._inter_order_mask(chip, mask_buffer=mask_buffer).astype(bool))

        if edge_clip > 0:
            edge_mask = np.ones_like(mask)
            edge_mask[edge_clip:-edge_clip,edge_clip:-edge_clip] = 0
            mask |= edge_mask

        coeffs = self._polyfit2d(data, polyorder, regularize=regularize, mask=mask)    
        stray_light = self._polyval2d(coeffs, polyorder, data.shape)

        stray_light = np.maximum(stray_light, 0)
        
        print(np.nanmedian((data - stray_light)[~mask]))

        stray_light += np.nanmedian((data - stray_light)[~mask])
        stray_light = np.maximum(stray_light, 0)

        print(np.nanmedian((data - stray_light)[~mask]))

        return stray_light, mask


    def columns(self, chip, polyorder, gaussian_sigma=128, edge_clip=0, mask_buffer=None, **kwargs):
        """
        Method to estimate stray light -- fits a 2D polynomial to inter-order pixels
        """
        data = np.array(self.target_2D[f'{chip}_CCD'].data)
        mask = self._inter_order_mask(chip, mask_buffer=mask_buffer).astype(bool)

        if edge_clip > 0:
            d = data[edge_clip:-edge_clip,edge_clip:-edge_clip]
            m = mask[edge_clip:-edge_clip,edge_clip:-edge_clip]

        else:
            d = deepcopy(data)
            m = deepcopy(mask)

        coeffs = self._polyfit_columns(d, polyorder, mask=m)
        stray_light = self._polyval_columns(coeffs, data.shape[0])
        stray_light = self._patch_nan_nearest(stray_light)
        stray_light = gaussian_filter(stray_light, gaussian_sigma, mode='reflect', truncate=4.0)

        return stray_light, mask
        

    def _polyfit2d(self, data_image, polyorder, regularize='none', mask=None):
        # coordinate grid
        nrow, ncol = data_image.shape
        y, x = np.mgrid[0:nrow, 0:ncol]

        # map to [-1,1] for Legendre polynomial basis
        x = 2*x/(ncol-1) - 1
        y = 2*y/(nrow-1) - 1

        # ravel arrays
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
        
        # build design matrix
        terms = []
        for i in range(polyorder + 1):
            for j in range(polyorder + 1 - i):
                cx = np.zeros(i+1)
                cy = np.zeros(j+1)
                cx[i] = 1
                cy[j] = 1
                Px = legval(x,cx)
                Py = legval(y,cy)
                terms.append(Px*Py)

        A = np.column_stack(terms)

        # solve least squares
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

        # apply regularization and recompute coefficients
        if isinstance(regularize, float):
            lam = regularize

        elif isinstance(regularize, str):
            if regularize == 'auto':
                lam = np.var(z)/np.var(coeffs)
            elif regularize == 'none':
                lam = 0
            else:
                raise ValueError("regularize must be 'auto', 'none', or a float value")

        if lam == 0:
            pass
        elif lam > 0:
            ATz = np.dot(A.T,z)
            ATA = np.dot(A.T,A) + lam*np.eye(A.shape[1])
            coeffs = np.linalg.solve(ATA,ATz)
        elif lam < 0:
            raise ValueError("regularization parameter lambda must not be negative")
            
        return coeffs
    
    
    def _polyval2d(self, coeffs, polyorder, shape):
        # coordinate grid
        nrow, ncol = shape
        y, x = np.mgrid[0:nrow, 0:ncol]

        # map to [-1,1] for Legendre polynomial basis
        x = 2*x/(ncol-1) - 1
        y = 2*y/(nrow-1) - 1

        # ravel arrays
        x = np.ravel(x)
        y = np.ravel(y)

        # build polynomial terms
        terms = []
        for i in range(polyorder + 1):
            for j in range(polyorder + 1 - i):
                cx = np.zeros(i+1)
                cy = np.zeros(j+1)
                cx[i] = 1
                cy[j] = 1
                Px = legval(x,cx)
                Py = legval(y,cy)
                terms.append(Px*Py)
        terms = np.array(terms)

        # calculate fitted polynomial
        result = np.dot(coeffs, terms).reshape((nrow,ncol))
    
        return result


    def _polyfit_columns(self, data_image, polyorder, mask=None):
        nrow, ncol = data_image.shape
        y = np.arange(nrow)
        
        # mask exposed pixels
        if mask is not None:
            mask = np.array(mask, dtype='bool')
        else:
            mask = np.zeros((nrow,ncol), dtype='bool')

        V = np.vander(y, polyorder + 1, increasing=True)
        coeffs = np.full((polyorder + 1, ncol), np.nan, dtype=float)

        for j in range(ncol):
            m = ~mask[:, j]
            
            if np.count_nonzero(m) > polyorder:
                Vj = V[m, :]
                yj = data_image[m, j]
                c, *_ = np.linalg.lstsq(Vj, yj, rcond=None)
                coeffs[:, j] = c
        
        return coeffs


    def _polyval_columns(self, coeffs, nrow):
        ncoeffs, ncol = coeffs.shape
        y = np.arange(nrow)
        V = np.vander(y, ncoeffs, increasing=True)
        
        return np.dot(V, coeffs)


    def _patch_nan_nearest(self, data_image):
        bad = np.isnan(data_image)
        
        if not np.any(bad):
            return data_image.copy()

        indices = distance_transform_edt(bad, return_distances=False, return_indices=True)

        return data_image[tuple(indices)]
    
    
    def _inter_order_mask(self, chip, dark_fibers=None, mask_buffer=None):
        """
        Args:
          dark_fibers (list) : list of integers 1-5 indicating any non-illuminated fibers
          mask_buffer (int) : number of pixels to expand intra-order region of mask
        """
        mask = self.master_order_mask[f'{chip}_CCD'] > 0

        if dark_fibers is not None:
            for fiber in dark_fibers:
                mask &= self.master_order_mask[f'{chip}_CCD'] != fiber

        if mask_buffer is not None:
            for i in range(mask_buffer):
                buffer = np.zeros_like(mask)
            
                row_diff = mask[:-1,:] != mask[1:,:]
                col_diff = mask[:,:-1] != mask[:,1:]
            
                buffer[:-1,:] |= row_diff
                buffer[1:,:] |= row_diff
                buffer[:,:-1] |= col_diff
                buffer[:,1:] |= col_diff
        
                mask |= buffer

        return mask