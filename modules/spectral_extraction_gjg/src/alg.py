import glob
import os
import sys
import warnings

from astropy.io import fits
from astropy.stats import mad_std
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as poly
import pandas as pd
from scipy.ndimage import median_filter
from scipy.interpolate import LSQUnivariateSpline

from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger
from modules.Utils.config_parser import ConfigHandler


class SpectralExtractionAlg:
    """
    Args:
        target_2D (KPF0): A KPF 2D science object
        master_flat_2D (KPF0): A KPF 2D master flat
        stray_light_image (dict of ndarray): 2D stray light arrays for the GREEN and RED ccds
        order_trace_green (pd.DataFrame): a pre-loaded dataframe with order trace for GREEN ccd
        order_trace_red (pd.DataFrame): a pre-loaded dataframe with order trace for RED ccd
        config (configparser.ConfigParser): Config context
        logger (logging.Logger): Instance of logging.Logger
    """
    def __init__(self, 
                 target_2D, 
                 master_flat_2D, 
                 stray_light_image,
                 order_trace_green, 
                 order_trace_red,
                 default_config_path,
                 logger=None
                 ):
        # config inputs
        self.config = ConfigClass(default_config_path)
        if logger == None:
            self.log = start_logger('SpectralExtraction', default_config_path)
        else:
            self.log = logger
            
        cfg_params = ConfigHandler(self.config, 'PARAM')
        self.stray_light_method = str(cfg_params.get_config_value('stray_light_method'))
        self.stray_light_polyorder = int(cfg_params.get_config_value('stray_light_polyorder'))
        self.stray_light_edge_clip = int(cfg_params.get_config_value('stray_light_edge_clip'))

        self.extraction_method = cfg_params.get_config_value('extraction_method')
        self.extraction_sigma_clip = float(cfg_params.get_config_value('extraction_sigma_clip'))
        self.extraction_max_iter = int(cfg_params.get_config_value('extraction_max_iter'))

        self.profile_filter_size = int(cfg_params.get_config_value('profile_filter_size'))
        self.profile_sigma_clip = float(cfg_params.get_config_value('profile_sigma_clip'))
        self.profile_num_knots = int(cfg_params.get_config_value('profile_num_knots'))

        # data inputs
        self.target_2D = target_2D
        self.master_flat_2D = master_flat_2D
        self.stray_light_image = stray_light_image
        self.order_trace = {}
        self.order_trace['GREEN_CCD'] = order_trace_green
        self.order_trace['RED_CCD'] = order_trace_red


    def _orderlet_box(self, data_image, order_trace, trace_index, return_box_coords=False, verbose=False, do_plot=False):
        nrow, ncol = data_image.shape
    
        # polynomial order trace
        coeffs = np.array([float(order_trace[f'Coeff{i}'][trace_index]) for i in range(4)])
    
        # trace in pixel coorrdinates on detector
        trace_center = poly.polyval(np.arange(ncol), coeffs)
        trace_top    = trace_center + order_trace.TopEdge[trace_index]
        trace_bottom = trace_center - order_trace.BottomEdge[trace_index]

        # track where trace goes off detector
        off_detector = (trace_top > nrow-1) | (trace_bottom < 0)

        if np.any(off_detector):
            trace_top[off_detector] = np.minimum(trace_top, nrow-1)[off_detector]
            trace_center[off_detector] = np.minimum(trace_center, nrow-1)[off_detector]
            trace_bottom[off_detector] = np.minimum(trace_bottom, nrow-1)[off_detector]
    
            trace_top[off_detector] = np.maximum(trace_top, 0)[off_detector]
            trace_center[off_detector] = np.maximum(trace_center, 0)[off_detector]
            trace_bottom[off_detector] = np.maximum(trace_bottom, 0)[off_detector]
            
        # extract a box around the desired order
        box_zeropt = int(np.floor(trace_bottom.min()))
        box_height = int(np.ceil(trace_top.max())) - box_zeropt
    
        D = data_image[box_zeropt:box_zeropt + box_height]

        if do_plot:
            fig, ax = plt.subplots(8,1, figsize=(16,8))
            for i in range(8):
                ax[i].imshow(D, 
                             cmap='viridis', 
                             origin='lower',
                             vmin=np.percentile(D,0.1),
                             vmax=np.percentile(D,99)
                            )
                
                ax[i].plot(trace_center - box_zeropt, c='r', ls=':')
                ax[i].plot(trace_bottom - box_zeropt, c='r')
                ax[i].plot(trace_top - box_zeropt, c='r')
                
                ax[i].set_ylim(None, box_height)
                ax[i].set_xlim(512*i,512*(i+1))
            plt.tight_layout()
            plt.show()
    
        # track edge pixel locations within extraction box
        edge_pixel_top = np.array(np.floor(trace_top - box_zeropt), dtype='int')
        edge_pixel_bottom = np.array(np.floor(trace_bottom - box_zeropt), dtype='int')
    
        # broadcast vectors            
        _row = np.arange(box_height)[:,None]                  # shape (nrow, 1)
        _edge_pixel_top = edge_pixel_top[None,:]        # shape (1, ncol)
        _edge_pixel_bottom = edge_pixel_bottom[None,:]
        _trace_top = trace_top[None,:]
        _trace_bottom = trace_bottom[None,:]
        
        # set W_ij for pixels fully outsize (0) or inside (1) trace
        W = np.zeros_like(D)
        W[(_row > _edge_pixel_bottom) & (_row < _edge_pixel_top)] = 1
        
        # set W_ij for pixels at edges of trace using some complicated array broadcasting
        mask_top = _row == _edge_pixel_top
        frac_top = np.tile((_trace_top - box_zeropt - _edge_pixel_top), (box_height,1))
        W[mask_top] = frac_top[mask_top]
       
        mask_bot = _row == _edge_pixel_bottom
        frac_bot = np.tile((1 - (_trace_bottom - box_zeropt - _edge_pixel_bottom)), (box_height,1))
        W[mask_bot] = frac_bot[mask_bot]

        if return_box_coords:
            return D, W, (box_zeropt, box_zeropt+box_height)
        return D, W


    def spatial_profile(self, D, S, W, f, filter_size=101, sigma_clip_x=3.0, num_knots=32, do_plot=False):
        """
        Estimate the spatial profile of a 2D data array (typical use is for a single orderlet)
        Applies a spline along detector rows, interpolating over outlier pixels
    
        Args:
            D (np.ndarray): 2D data array, bias corrected and flat fielded
            S (np.ndarray): 2D stray light (sky/scattered/stray light background)
            W (np.ndarray): 2D weight array to handle order curvature/tilt
            f (np.ndarray): 1D spectrum
            filter_size (int): filter size for median filter, used to identify outliers
            sigma_clip_x (float): sigma clipping threshold, used to identify outliers
            num_knots (int): number of knots for spline

        """
        P = (D-S)/f

        nrow, ncol = np.shape(P)
        y = np.arange(nrow)
        x = np.arange(ncol)

        for i in range(nrow):
            med = median_filter(P[i], size=filter_size)
            out = np.abs(P[i]-med)/mad_std(P[i]-med) > sigma_clip_x

            try:
                knots = np.linspace(x[~out].min()+1, x[~out].max()-1, num_knots)[1:-1]
                spline = LSQUnivariateSpline(x[~out], P[i][~out], t=knots, ext='const')
            except ValueError:
                knots = np.linspace(x[~out].min()+1, x[~out].max()-1, num_knots//2)[1:-1]
                spline = LSQUnivariateSpline(x[~out], P[i][~out], t=knots, ext='const')
    
            if do_plot:
                plt.figure(figsize=(20,2))
                plt.plot(x, P[i], 'k')
                plt.plot(x[out], P[i][out], 'rx')
                plt.plot(x, spline(x), 'C1', lw=2)
                plt.ylim(P[i][~out].min(), P[i][~out].max())
                plt.show()
    
            P[i] = 1.0*spline(x)

        P = np.maximum(P,0)
        P /= np.sum(P*W,axis=0)
    
        return P


    def box_extraction(self, D, S, V0, Q=None, M=None, W=None, do_plot=False):
        """
        Box extraction on a data array D (typical use case is a single orderlet)
        Variable names follow Horne 1986
    
        Args
            D: data array
            S: sky/scattered/stray light array
            V0: variance array from detector (i.e. read noise)
            Q: quantum scaling (electrons/photons/ADU)
            M: mask (1 = good pixel, 0=bad)
            W: weights, typically to define order trace
        """
        # ensure mask and weight arrays exist
        if Q is None:
            Q = np.ones_like(D, dtype='float')
        if W is None:
            W = np.ones_like(D, dtype='float')/D.shape[0]
        if M is None:
            M = np.ones_like(D, dtype='int')

        # sanitize inputs
        D = np.asarray(D)
        S = np.asarray(S)
        Q = np.asarray(Q)
        M = np.asarray(M)
        W = np.asarray(W)
        
        # 2D variance array
        V = V0 + np.abs(D)/Q
    
        # 1D sum extraction of spectrum and variance
        f = np.sum((D-S)*W,axis=0)
        v = np.sum(V*W, axis=0)
    
        if do_plot:
            plt.figure(figsize=(20,4))
            plt.plot(f)
            plt.plot(v)
            plt.show()
            
        return f, v


    def optimal_extraction(self, 
                           D, 
                           S, 
                           V0, 
                           Q=None, 
                           M=None, 
                           W=None, 
                           P=None,
                           filter_size=101,
                           sigma_clip_x=3.0,
                           sigma_clip_y=5.0,
                           num_knots=32,
                           max_iter=20, 
                           verbose=False, 
                           do_plot=False
                           ):
        """
        Optimal extraction on a data array D (typical use case is a single orderlet)
        Variable names follow Horne 1986

        May optionally supply a pre-computed spatial profile P
        Pre-computing the spatial profile from a master flat is advised
    
        Args
            D: data array
            S: sky/scattered/stray light array
            V0: variance array from detector (i.e. read noise)
            Q: quantum scaling (electrons/photons/ADU)
            M: mask (1 = good pixel, 0=bad)
            W: weights, typically to define order trace
            P: pre-computed spatial profile
            filter_size (int): filter size for median filter, used to identify outliers in spatial profile
            sigma_clip_x (float): sigma clipping used to identify outliers during profile modeling 
            sigma_clip_y (float): sigma clipping used to identify cosmic rays and pixel defects
            num_knots (int): number of knots in smoothing spline for profile modeling
            max_iter (int): maximum number of iterations of algorithm
        """
        nrow, ncol = np.shape(D)
        
        # ensure all arrays exist
        if Q is None:
            Q = np.ones_like(D, dtype='float')
        if W is None:
            W = np.ones_like(D, dtype='float')/nrow
        if M is None:
            M = np.ones_like(D, dtype='int')

        # check for pre-computed spatial profile
        if P is not None:
            static_profile = True
        else:
            static_profile = False
    
        # sanitize inputs
        D = np.asarray(D)
        S = np.asarray(S)
        Q = np.asarray(Q)
        M = np.asarray(M)
        W = np.asarray(W)

        # mask inter-order pixels
        M[W == 0] = 0

        # initial estimates
        f, v = self.box_extraction(D, S, V0, Q=Q, M=M, W=W)
        
        if not static_profile:
            P = (D-S)/f
        
        V = V0 + np.abs(D)/Q
    
        # optimal extraction loop
        loop = 0
        while loop < max_iter:
            # spectrum
            f = np.sum(M*P*(D-S)*(W/V),axis=0)/np.sum(M*P**2*(W/V), axis=0)
            v = np.sum(M*P*W,axis=0)/np.sum(M*P**2*(W/V), axis=0)
        
            # profile
            if not static_profile:
                P = self.spatial_profile(D, S, W, f, filter_size, sigma_clip_x, num_knots)
            
            # variance
            V = V0 + np.abs(f*P + S)/Q
        
            # residuals
            R = (D - f*P - S)**2/V
            
            # mask cosmic rays
            bad_pixel_count = np.sum(M==0)
            worst_pixel_row = np.argmax(R*M, axis=0)
    
            if verbose:
                print(f"loop {loop} | {bad_pixel_count - np.sum(W==0)} pixels flagged")
        
            # plot spectrum
            if do_plot:
                plt.figure(figsize=(20,4))
                plt.plot(f)
                plt.plot(v)
                plt.show()
        
            for col in range(ncol):
                row = worst_pixel_row[col]
            
                if R[row,col] > sigma_clip_y**2:
                    M[row,col] = 0
    
                    if do_plot:
                        plt.figure(figsize=(4,3))
                        plt.step(np.arange(nrow), R[:,col], color='k', where='mid')
                        plt.plot(row, R[row,col], 'rx')
                        plt.title(f"Column {col}", fontsize=14)
                        plt.show()
        
            if np.sum(M==0) == bad_pixel_count:
                break
        
            loop += 1
    
        return f, v, P, M

    
    def extract_spectrum(self, method, chip, trace_index):
        """
        Extract 1D spectrum for a single orderlet

        Args:
            method (str): extraction method, can bo 'box' or 'optimal'
            chip (str): 'GREEN' or 'RED' ccd
            trace_index (int): integer identifying ordelet in order_trace

        *** This method should be updated to take, order number and orderlet name as inputs ***
        *** e.g. orderlet='SCI2', orderno=17 ***

        Returns:
            f (ndarray): extracted 1D spectrum
            v (ndarray): extracted 1D variance
            P (ndarray): 2D spatial profile (if 'optimal' extraction is used)
            M (ndarray): 2D boolean bad pixel mask (if 'optimal' extraction is used)
        """
        # target data
        D, W = self._orderlet_box(self.target_2D[f'{chip}_CCD'].data,
                                  self.order_trace[f'{chip}_CCD'],
                                  trace_index
                                 )

        # master flat data
        F, _ = self._orderlet_box(self.master_flat_2D[f'{chip}_CCD_STACK'].data,
                                  self.order_trace[f'{chip}_CCD'],
                                  trace_index
                                 )

        # stray light
        S, _ = self._orderlet_box(self.stray_light_image[f'{chip}_CCD'].data,
                                  self.order_trace[f'{chip}_CCD'],
                                  trace_index
                                 )

        V0 = float(np.mean([self.target_2D.header['PRIMARY'][f'RN{chip}1'],self.target_2D.header['PRIMARY'][f'RN{chip}2']]))
        M = np.ones_like(D, dtype='int')

        # box extraction
        f_box, v_box = self.box_extraction(D, S, V0, M=M, W=W)
        if method == 'box':
            return f_box, v_box

        f_flat, _ = self.box_extraction(F, np.zeros_like(F), V0, M=M, W=W)
        P = self.spatial_profile(F, 
                                 np.zeros_like(F), 
                                 W, 
                                 f_flat, 
                                 filter_size=self.profile_filter_size, 
                                 sigma_clip_x=self.profile_sigma_clip, 
                                 num_knots=self.profile_num_knots
                                 )

        f_opt, v_opt, P, M = self.optimal_extraction(D, 
                                                     S, 
                                                     V0, 
                                                     M=M, 
                                                     W=W, 
                                                     P=P,
                                                     sigma_clip_y=self.extraction_sigma_clip,
                                                     max_iter=self.extraction_max_iter, 
                                                     verbose=False, 
                                                     do_plot=False
                                                    )
        return f_opt, v_opt, P, M