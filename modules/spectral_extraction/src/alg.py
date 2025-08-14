import glob
import os
import sys
import warnings

from astropy.io import fits
from astropy.stats import mad_std
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as poly
import pandas as pd
from scipy.ndimage import median_filter
from scipy.interpolate import LSQUnivariateSpline

from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger
from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level1 import KPF1


class SpectralExtractionAlg:
    """
    Args:
        target_2D (KPF0): A KPF 2D science object
        master_flat_2D (KPF0): A KPF 2D master flat
        order_trace_green (str): path to csv with order trace for GREEN ccd
        order_trace_red (str): path to csv with order trace for RED ccd
        start_order (tuple): index to start order trace, see caldates/start_order.csv
        config (configparser.ConfigParser): Config context
        logger (logging.Logger): Instance of logging.Logger
    """
    def __init__(self, 
                 target_2D, 
                 master_flat_2D, 
                 order_trace_green, 
                 order_trace_red,
                 start_order_green,
                 start_order_red,
                 default_config_path,
                 bad_pixel_mask_green=None,
                 bad_pixel_mask_red=None,
                 background_image_green=None,
                 background_image_red=None,
                 logger=None,
                 ):
        # config inputs
        self.config = ConfigClass(default_config_path)
        if logger == None:
            self.log = start_logger('SpectralExtraction', default_config_path)
        else:
            self.log = logger
            
        cfg_params = ConfigHandler(self.config, 'PARAM')
        print(f"In spectral extraction")
        self.extraction_method = cfg_params.get_config_value('extraction_method')
        self.extraction_sigma_clip = float(cfg_params.get_config_value('extraction_sigma_clip'))
        self.extraction_max_iter = int(cfg_params.get_config_value('extraction_max_iter'))

        self.profile_filter_size = int(cfg_params.get_config_value('profile_filter_size'))
        self.profile_sigma_clip = float(cfg_params.get_config_value('profile_sigma_clip'))
        self.profile_num_knots = int(cfg_params.get_config_value('profile_num_knots'))

        # required data inputs
        self.target_2D = target_2D
        self.master_flat_2D = master_flat_2D
        self.order_trace = {}
        self.order_trace['GREEN_CCD'] = pd.read_csv(order_trace_green, index_col=0)
        self.order_trace['RED_CCD'] = pd.read_csv(order_trace_red, index_col=0)
        self.start_order = {}
        self.start_order['GREEN_CCD'] = start_order_green
        self.start_order['RED_CCD'] = start_order_red
        self.order_trace = self._fix_order_trace_indexing()

        # By default the KPF DRP subracts stray light from the data image
        # Only supply background_image if you suspect some additional contamination
        self.background_image = {}
        if background_image_green is not None:
            self.background_image['GREEN_CCD'] = background_image_green
        else:
            self.background_image['GREEN_CCD'] = np.zeros_like(self.target_2D['GREEN_CCD'])
        
        if background_image_red is not None:
            self.background_image['RED_CCD'] = background_image_red
        else:
            self.background_image['RED_CCD'] = np.zeros_like(self.target_2D['RED_CCD'])

        # variance: The variance is not currently populated for masters.
        for chip in ['GREEN', 'RED']:
            self._check_for_variance_frame(chip)
        
        # bad pixel mask
        self.bad_pixel_mask = {}
        if bad_pixel_mask_green is not None:
            self.bad_pixel_mask['GREEN_CCD'] = bad_pixel_mask_green
        else:
            self.bad_pixel_mask['GREEN_CCD'] = np.ones_like(self.target_2D['GREEN_CCD'], dtype='bool')

        if bad_pixel_mask_red is not None:
            self.bad_pixel_mask['RED_CCD'] = bad_pixel_mask_red
        else:
            self.bad_pixel_mask['RED_CCD'] = np.ones_like(self.target_2D['RED_CCD'], dtype='bool')

        #for chip in ['GREEN', 'RED']:
        #    self.bad_pixel_mask[f'{chip}_CCD'] &= self._make_bad_pixel_mask(chip)

        # initialize L1 object
        self.target_l1 = KPF1.from_l0(self.target_2D)

        
    def _check_for_variance_frame(self, chip):
        var_ext_name = f'{chip}_VAR'
        if var_ext_name not in self.target_2D.extensions:
            self.log.warning(f"Variance extension {var_ext_name} not found, setting variance equal to photon noise")
            self.target_2D[var_ext_name] = np.abs(self.target_2D[f'{chip}_CCD'])

        elif np.shape(self.target_2D[var_ext_name]) != np.shape(self.target_2D[f'{chip}_CCD']):
            self.log.warning(f"Variance extension {var_ext_name} has mismatched dimensions {np.shape(self.target_2D[var_ext_name])} vs {np.shape(self.target_2D[f'{chip}_CCD'])}, setting variance equal to photon noise")
            self.target_2D[var_ext_name] = np.abs(self.target_2D[f'{chip}_CCD'])

    
    def _make_bad_pixel_mask(self, chip, sigma_cut=5.0):
        # data, variance, mask
        D = self.target_2D[f'{chip}_CCD']
        V = self.target_2D[f'{chip}_VAR']
        M = np.ones(D.shape, dtype='bool')
        
        # check for NaN and inf
        M &= np.isfinite(D)
        M &= np.isfinite(V)
    
        # check for variance outliers
        V0 = np.abs(V-D)
        M &= np.abs(V0 - np.median(V0))/mad_std(V0, ignore_nan=True) < sigma_cut

        return M

        
    def _fix_order_trace_indexing(self):
        for chip in ['GREEN', 'RED']:
            if self.start_order[f'{chip}_CCD'] > 0:
                self.order_trace[f'{chip}_CCD'] = self.order_trace[f'{chip}_CCD'].drop(index=0).reset_index(drop=True)
            elif self.start_order[f'{chip}_CCD'] < 0:
                n = np.abs(self.start_order[f'{chip}_CCD'])
                df = self.order_trace[f'{chip}_CCD']
                nan_rows = pd.DataFrame(np.nan, index=range(n), columns=df.columns)
                self.order_trace[f'{chip}_CCD'] = pd.concat([nan_rows, df], ignore_index=True)
            elif self.start_order[f'{chip}_CCD'] == 0:
                pass

        return self.order_trace


    def _get_orderlet_ext_from_trace_index(self, chip, trace_index, start_order=None):
        if trace_index % 5 == 0:
            drp_f_ext = f'{chip}_SKY_FLUX'
            drp_v_ext = f'{chip}_SKY_VAR'
        elif trace_index % 5 == 1:
            drp_f_ext = f'{chip}_SCI_FLUX1'
            drp_v_ext = f'{chip}_SCI_VAR1'
        elif trace_index % 5 == 2:
            drp_f_ext = f'{chip}_SCI_FLUX2'
            drp_v_ext = f'{chip}_SCI_VAR2'
        elif trace_index % 5 == 3:
            drp_f_ext = f'{chip}_SCI_FLUX3'
            drp_v_ext = f'{chip}_SCI_VAR3'
        elif trace_index % 5 == 4:
            drp_f_ext = f'{chip}_CAL_FLUX'
            drp_v_ext = f'{chip}_CAL_VAR'

        return drp_f_ext, drp_v_ext


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
            return D, W, box_zeropt, box_zeropt+box_height
        return D, W


    def spatial_profile(self, 
                        D, 
                        S, 
                        W, 
                        f, 
                        filter_size=None, 
                        num_knots=None, 
                        sigma_clip=None, 
                        do_plot=False
                       ):
        """
        Estimate the spatial profile of a 2D data array (typical use is for a single orderlet)
        Applies a spline along detector rows, interpolating over outlier pixels
    
        Args:
            D (np.ndarray): 2D data array, bias corrected and flat fielded
            S (np.ndarray): 2D sky/scattered/stray light background
            W (np.ndarray): 2D weight array to handle order curvature/tilt
            f (np.ndarray): 1D spectrum
            filter_size (int): filter size for median filter, used to identify outliers
            num_knots (int): number of knots for spline
            sigma_clip (float): sigma clipping threshold, used to identify outliers

        """
        # populate kwargs
        if filter_size is None:
            filter_size = self.profile_filter_size
        if num_knots is None:
            num_knots = self.profile_num_knots
        if sigma_clip is None:
            sigma_clip = self.profile_sigma_clip

        P = (D-S)/f

        nrow, ncol = np.shape(P)
        y = np.arange(nrow)
        x = np.arange(ncol)

        for i in range(nrow):
            med = median_filter(P[i], size=filter_size)
            out = np.abs(P[i]-med)/mad_std(P[i]-med, ignore_nan=True) > sigma_clip

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


    def box_extraction(self, 
                       D, 
                       S, 
                       V, 
                       Q=1.0, 
                       M=None, 
                       W=None
                      ):
        """
        Box extraction on a data array D (typical use case is a single orderlet)
        Variable names follow Horne 1986
    
        Args
            D: data array
            S: sky/scattered/stray light background array
            V: variance array
            Q: quantum scaling (electrons/photons/ADU)
            M: mask (1 = good pixel, 0=bad)
            W: weights, typically to define order trace, assumed to be normalized
        """
        # ensure mask and weight arrays exist
        if M is None:
            M = np.ones_like(D, dtype=int)
        if W is None:
            W = np.ones_like(D, dtype=float)/D.shape[0]

        # sanitize inputs
        D = np.asarray(D)
        S = np.asarray(S)
        V = np.asarray(V)
        Q = np.asarray(Q)
        M = np.asarray(M, dtype=int)
        W = np.asarray(W, dtype=float)

        # 1D box extraction of spectrum and variance
        self.log.debug(f"Box extraction: D shape: "+str(D.shape)+ " S shape: "+str(S.shape)+ " V shape: "+str(V.shape)+ " M shape: "+str(M.shape)+ " W shape: "+str(W.shape))
        f = np.sum(M*(D-S)*W,axis=0)
        v = np.sum(M*V*W,axis=0)
                        
        # return four values to match optimal extraction
        return f, v, None, None


    def optimal_extraction(self, 
                           D, 
                           S, 
                           V, 
                           Q=1.0, 
                           M=None, 
                           W=None, 
                           P=None,
                           max_iter=None, 
                           profile_filter_size=None,
                           profile_num_knots=None,
                           profile_sigma_clip=None,
                           extraction_sigma_clip=None,
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
            S: sky/scattered/stray light background array
            V: variance array
            Q: quantum scaling (electrons/photons/ADU)
            M: mask (1 = good pixel, 0=bad)
            W: weights, typically to define order trace
            P: pre-computed spatial profile
            max_iter (int): maximum number of iterations of algorithm
            profile_filter_size (int): filter size for median filter, used to identify outliers in spatial profile
            profile_num_knots (int): number of knots in smoothing spline for profile modeling
            profile_sigma_clip (float): sigma clipping used to identify outliers during profile modeling 
            extraction_sigma_clip (float): sigma clipping used to identify cosmic rays and pixel defects
        """
        # check for pre-computed spatial profile
        if P is not None:
            static_profile = True
        else:
            static_profile = False
        
        # ensure mask and weight arrays exist
        if M is None:
            M = np.ones_like(D, dtype=int)
        if W is None:
            W = np.ones_like(D, dtype=float)/nrow

        # populate kwargs
        if max_iter is None:
            max_iter = self.extraction_max_iter
        if profile_filter_size is None:
            profile_filter_size = self.profile_filter_size
        if profile_num_knots is None:
            profile_num_knots = self.profile_num_knots
        if profile_sigma_clip is None:
            profile_sigma_clip = self.profile_sigma_clip
        if extraction_sigma_clip is None:
            extraction_sigma_clip = self.extraction_sigma_clip

        # get data image shape
        nrow, ncol = np.shape(D)
        
        # sanitize inputs
        D = np.asarray(D)
        S = np.asarray(S)
        V = np.asarray(V)
        Q = np.asarray(Q)
        M = np.asarray(M,dtype=int)
        W = np.asarray(W,dtype=float)
        
        # mask inter-order pixels
        M[W == 0] = 0

        # box extraction
        f, v, _, _ = self.box_extraction(D, S, V, Q=Q, M=M, W=W)

        # spatial profile
        if not static_profile:
            P = (D-S)/f
        
        # variance from non-photon sources
        V0 = V - np.abs(D)/Q

        # optimal extraction loop
        loop = 0
        while loop < max_iter:
            # spectrum
            f = np.sum(M*P*(D-S)*(W/V),axis=0)/np.sum(M*P**2*(W/V),axis=0)
            v = np.sum(M*P*W,axis=0)/np.sum(M*P**2*(W/V),axis=0)
        
            # profile
            if not static_profile:
                P = self.spatial_profile(D, 
                                         S, 
                                         W, 
                                         f,
                                         filter_size=profile_filter_size,
                                         num_knots=profile_num_knots,
                                         sigma_clip=profile_sigma_clip
                                        )
            
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
            
                if R[row,col] > extraction_sigma_clip**2:
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

    
    def extract_orderlet(self, 
                         chip, 
                         trace_index, 
                         method=None,
                         max_iter=None,
                         profile_filter_size=None,
                         profile_num_knots=None,
                         profile_sigma_clip=None,
                         extraction_sigma_clip=None,
                        ):
        """
        Extract 1D spectrum for a single orderlet

        Args:
            method (str): extraction method, can be 'box' or 'optimal'
            chip (str): 'GREEN' or 'RED' ccd
            trace_index (int): integer identifying ordelet in order_trace

        Returns:
            f (ndarray): extracted 1D spectrum
            v (ndarray): extracted 1D variance
            P (ndarray): 2D spatial profile (if 'optimal' extraction is used)
            M (ndarray): 2D boolean bad pixel mask (if 'optimal' extraction is used)
        """
        # populate kwargs
        if method is None:
            method = self.extraction_method
        if max_iter is None:
            max_iter = self.extraction_max_iter
        if profile_filter_size is None:
            profile_filter_size = self.profile_filter_size
        if profile_num_knots is None:
            profile_num_knots = self.profile_num_knots
        if profile_sigma_clip is None:
            profile_sigma_clip = self.profile_sigma_clip
        if extraction_sigma_clip is None:
            extraction_sigma_clip = self.extraction_sigma_clip

        # data image
        D, W, ymin, ymax = self._orderlet_box(self.target_2D[f'{chip}_CCD'].data,
                                              self.order_trace[f'{chip}_CCD'],
                                              trace_index,
                                              return_box_coords=True
                                             )

        # sky/scattered/stray light background
        V = self.target_2D[f'{chip}_VAR'][ymin:ymax]

        # sky/scattered/stray light background
        S = self.background_image[f'{chip}_CCD'][ymin:ymax]

        # mask
        #M = np.ones_like(D, dtype=int)
        M = self.bad_pixel_mask[f'{chip}_CCD'].astype(int)[ymin:ymax]

        # flat frame
        F = self.master_flat_2D[f'{chip}_CCD_STACK'][ymin:ymax]
        
        # zero frame
        Z = np.zeros_like(F)
        
        # box extraction
        f_box, v_box, _, _ = self.box_extraction(D, S, V, M=M, W=W)
        if method == 'box':
            return f_box, v_box, None, None

        f_flat, v_flat, _, _ = self.box_extraction(F, Z, V, M=M, W=W)
        
        P = self.spatial_profile(F, 
                                 Z, 
                                 W, 
                                 f_flat, 
                                 filter_size=profile_filter_size, 
                                 sigma_clip=profile_sigma_clip, 
                                 num_knots=profile_num_knots
                                )

        f_opt, v_opt, P, M = self.optimal_extraction(D, 
                                                     S, 
                                                     V, 
                                                     M=M, 
                                                     W=W, 
                                                     P=P,
                                                     max_iter=max_iter, 
                                                     extraction_sigma_clip=extraction_sigma_clip,
                                                     verbose=False, 
                                                     do_plot=False
                                                    )
        return f_opt, v_opt, P, M
    

    def extract_ccd(self, 
                    chip, 
                    method=None,
                    max_iter=None,
                    profile_filter_size=None,
                    profile_num_knots=None,
                    profile_sigma_clip=None,
                    extraction_sigma_clip=None
                   ):
        """
        Extract 1D spectrum and variance for all orders/orderlets on GREEN or RED ccd

        Args:
            chip (str): 'GREEN' or 'RED' ccd
            method (str): extraction method, can bo 'box' or 'optimal'

        Returns:
            l1_out: KPF L1 object populated with extracted 1D spectra and varaiance
        """
        # populate kwargs
        if method is None:
            method = self.extraction_method
        if max_iter is None:
            max_iter = self.extraction_max_iter
        if profile_filter_size is None:
            profile_filter_size = self.profile_filter_size
        if profile_num_knots is None:
            profile_num_knots = self.profile_num_knots
        if profile_sigma_clip is None:
            profile_sigma_clip = self.profile_sigma_clip
        if extraction_sigma_clip is None:
            extraction_sigma_clip = self.extraction_sigma_clip

        # set up container for arrays
        nrow, ncol = self.target_2D[f'{chip}_CCD'].shape
        ntrace = len(self.order_trace[f'{chip}_CCD'])
        norder = ntrace // 5

        l1_arrays = {}
        for trace_index in range(5):
            f_ext, v_ext = self._get_orderlet_ext_from_trace_index(chip, trace_index)

            l1_arrays[f_ext] = np.zeros((norder,ncol))
            l1_arrays[v_ext] = np.zeros((norder,ncol))

        # extract spectra
        for trace_index in range(ntrace):                      
            if any(np.isnan(self.order_trace[f'{chip}_CCD'].iloc[trace_index])):
                f = np.nan*np.ones(ncol)
                v = np.nan*np.ones(ncol)

            else:
                f, v, _, _ = self.extract_orderlet(chip, 
                                                   trace_index, 
                                                   method=method,
                                                   max_iter=max_iter,
                                                   profile_filter_size=profile_filter_size,
                                                   profile_num_knots=profile_num_knots,
                                                   profile_sigma_clip=profile_sigma_clip,
                                                   extraction_sigma_clip=extraction_sigma_clip
                                                  )

            f_ext, v_ext = self._get_orderlet_ext_from_trace_index(chip, trace_index)
            order_index = trace_index // 5

            l1_arrays[f_ext][order_index] = f.copy()
            l1_arrays[v_ext][order_index] = v.copy()

        # build KPF L1 object
        l1_out = self.target_l1
        for key in l1_arrays.keys():
            l1_out[key] = l1_arrays[key]

        return l1_out