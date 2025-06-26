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
    Docstring
    """
    def __init__(self, 
                 target_2D, 
                 order_trace_green,
                 order_trace_red,
                 default_config_path,
                 logger=None
                ):
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
        self.order_trace = {}
        self.order_trace['GREEN_CCD'] = order_trace_green
        self.order_trace['RED_CCD'] = order_trace_red
        self.drptag = self.target_2D.header['PRIMARY']['DRPTAG']

    
    def add_keywords(self, stray_light_image, inter_order_mask):
        header = self.target_2D.header['PRIMARY']

        if self.method == 'polynomial':
            method = f"{self.method}_{self.polyorder}"
        else:
            method = self.method

        slg = stray_light_image['GREEN_CCD'][~inter_order_mask['GREEN_CCD']]
        slr = stray_light_image['RED_CCD'][~inter_order_mask['RED_CCD']]

        header['SL_METH'] = method             # COMMENT method used to estimate stray light
        header['SLG_MEAN'] = np.mean(slg)      # COMMENT mean of GREEN inter-order stray light
        header['SLG_RMS']  = np.std(slg)       # COMMENT root-mean-square of GREEN inter-order stray light
        header['SLG_MIN']  = np.min(slg)       # COMMENT minimum of GREEN inter-order stray light
        header['SLG_MAX']  = np.max(slg)       # COMMENT maximum of GREEN inter-order stray light
        header['SLR_MEAN'] = np.mean(slr)      # COMMENT mean of RED inter-order stray light
        header['SLR_RMS']  = np.std(slr)       # COMMENT root-mean-square of RED inter-order stray light
        header['SLR_MIN']  = np.min(slr)       # COMMENT minimum of RED inter-order stray light
        header['SLR_MAX']  = np.max(slr)       # COMMENT maximum of RED inter-order stray light


    def estimate_stray_light(self):
        try:
            stray_light_method = self.__getattribute__(self.method)
        except AttributeError:
            #self.log.error(f'Stray light method {self.method} not implemented.')
            raise(AttributeError)

        stray_light_image, inter_order_mask = stray_light_method()
        self.add_keywords(stray_light_image, inter_order_mask)

        return stray_light_image, inter_order_mask

    
    def zero(self):
        mask = {}
        stray_light = {}

        for chip in ['GREEN', 'RED']:
            mask[f'{chip}_CCD'] = self._inter_order_mask(chip).astype('bool')
            stray_light[f'{chip}_CCD'] = np.zeros_like(self.target_2D[f'{chip}_CCD'])

        return stray_light, mask

    
    def mean(self):
        mask = {}
        stray_light = {}

        for chip in ['GREEN', 'RED']:
            m = self._inter_order_mask(chip).astype('bool')
            d = np.array(self.target_2D[f'{chip}_CCD'].data)

            mask[f'{chip}_CCD'] = m.copy()
            stray_light[f'{chip}_CCD'] = np.mean(d[~m])*np.ones_like(d)
    
        return stray_light, mask

    
    def polynomial(self):
        mask = {}
        stray_light = {}

        for chip in ['GREEN', 'RED']:
            m = self._inter_order_mask(chip).astype('bool')
            d = np.array(self.target_2D[f'{chip}_CCD'].data)

            clip = self.edge_clip
            coeffs = self._polyfit2d(d[clip:-clip,clip:-clip],
                                     self.polyorder,
                                     m[clip:-clip,clip:-clip]
                                    )
    
            stray_light[f'{chip}_CCD'] = self._polyval2d(coeffs, self.polyorder, d.shape)
            stray_light[f'{chip}_CCD'] = np.maximum(stray_light[f'{chip}_CCD'], 0)

            mask[f'{chip}_CCD'] = m.copy()

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


    def _inter_order_mask(self, chip):
        data_image = self.target_2D[f'{chip}_CCD'].data
        order_trace = self.order_trace[f'{chip}_CCD']
        
        norder = len(order_trace)
        nrow, ncol = data_image.shape
    
        mask = np.zeros((nrow,ncol),dtype='int')
    
        # polynomial order trace
        for trace_index in range(norder):
            coeffs = np.array([float(order_trace[f'Coeff{i}'][trace_index]) for i in range(4)])
        
            # trace in pixel coorrdinates on detector
            trace_center = poly.polyval(np.arange(ncol), coeffs)
            trace_top    = trace_center + order_trace.TopEdge[trace_index]
            trace_bottom = trace_center - order_trace.BottomEdge[trace_index]
        
            # track edge pixel locations (+/- one pixel buffer)
            edge_pixel_top = np.array(np.floor(trace_top), dtype='int') + 1
            edge_pixel_bottom = np.array(np.floor(trace_bottom), dtype='int') - 1
        
            # broadcast vectors            
            _row = np.arange(nrow)[:,None]                  # shape (nrow, 1)
            _edge_pixel_top = edge_pixel_top[None,:]        # shape (1, ncol)
            _edge_pixel_bottom = edge_pixel_bottom[None,:]
            _trace_top = trace_top[None,:]
            _trace_bottom = trace_bottom[None,:]
            
            # set mask_ij for pixels inside trace
            mask[(_row > _edge_pixel_bottom) & (_row < _edge_pixel_top)] = 1
            
        return mask