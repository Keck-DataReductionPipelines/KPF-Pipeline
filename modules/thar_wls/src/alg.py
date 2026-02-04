import os
import sys
import glob
import warnings

from astropy.stats import mad_std
from astropy.time import Time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import polynomial, legendre
import pandas as pd
from scipy.ndimage import median_filter, gaussian_filter
from scipy.optimize import least_squares

from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger
from kpfpipe.models.level1 import KPF1
from modules.Utils.config_parser import ConfigHandler
from modules.Utils.kpf_parse import get_datecode, HeaderParse


class WLSAlg:
    def __init__(self, 
                obs_ids, 
                rough_wls, 
                linelist, 
                default_config_path, 
                logger=None
                ):
        """
        obs_ids : list of obs_ids
        rough_wls : KPF1 object with WAV extensions
        linelist : list of thorium wavelengths in Angstroms
        """
        # direct inputs
        self.obs_ids = obs_ids
        self.rough_wls = rough_wls
        self.linelist = linelist
        
        # config inputs
        self.config = ConfigClass(default_config_path)
        if logger == None:
            self.log = start_logger('SpectralExtraction', default_config_path)
        else:
            self.log = logger

        cfg_params = ConfigHandler(self.config, 'PARAM')
        self.polyorder_x = int(cfg_params.get_config_value('polyorder_x'))
        self.polyorder_m = int(cfg_params.get_config_value('polyorder_m'))
        self.polyorder_s = int(cfg_params.get_config_value('polyorder_s'))

        # init routines
        self._load_stack()
        self._set_linefunc(str(cfg_params.get_config_value('linefunc')))
        


    def _load_stack(self):
        self.nobs = len(self.obs_ids)
        self.l1_stack = [None]*self.nobs

        for i, obs_id in enumerate(self.obs_ids):
            datecode = get_datecode(self.obs_ids[i])
            filepath = f'/data/L1/{datecode}/{self.obs_ids[i]}_L1.fits'
            self.l1_stack[i] = KPF1.from_fits(filepath, data_type='KPF')


    @classmethod
    def _set_linefunc(cls, name):
        try:
            cls.linefunc = staticmethod(getattr(cls, name))
        except AttributeError:
            raise ValueError(f"No such line function: {name}")


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


    def fit_line_positions(self, flux1d, wave1d, linelist = None, linefunc = None, window=5, qc_sigma=2.5, do_plot=False):
        """
        Fit line postions (in pixel space) for all lines in a 1D flux array
            * line centers are determined by fitting a 1D function in pixel-vs-flux
            * quality control flags and rejects poorly conditioned fits
        """
        if linelist is None:
            linelist = self.linelist
        if linefunc is None:
            linefunc = self.linefunc

        assert len(flux1d) == len(wave1d), "length of flux and wave arrays are mismatched"
        ncol = len(flux1d)

        lines = {}
        lines['wav'] = np.sort(linelist[(linelist > wave1d.min()) & (linelist < wave1d.max())])
        nlines = len(lines['wav'])

        for key in ['pix', 'std', 'amp', 'rms']:
            lines[key] = np.zeros(nlines, dtype='float')

        for i, lw in enumerate(lines['wav']):
            loc = np.argmin(np.abs(wave1d-lw))
            cols = np.arange(loc-window,loc+window+1)
            cols = cols[(cols >= 0) & (cols < ncol)]

            x = cols
            y = flux1d[cols]
            
            theta0 = [loc, np.abs(np.mean(np.diff(x))), y.max(), 0]
            theta, rms = self.optimize_lsq(linefunc, theta0, x, y)
        
            lines['pix'][i] = theta[0]
            lines['std'][i] = theta[1]
            lines['amp'][i] = theta[2]
            lines['rms'][i] = rms / np.abs(theta[2] * np.sqrt(2*np.pi) * theta[1])
            
            #if do_plot:
            #    plt.figure()
            #    plt.plot(x, y, color='C0')
            #    plt.plot(x, linefunc(theta, x), color='C1', ls='--')
            #    plt.show()

        lines['bad'] = np.abs(lines['rms']-np.median(lines['rms']))/mad_std(lines['rms']) > qc_sigma
        lines['bad'] |= np.abs(lines['std']-np.median(lines['std']))/mad_std(lines['std']) > qc_sigma
        lines['bad'] |= lines['amp'] < 0    

        if do_plot:
            bad = lines['bad']
            plt.figure(figsize=(20,4))
            plt.plot(wave1d, flux1d)
            plt.plot(lines['wav'][~bad], 2500*np.ones(np.sum(~bad)), 'kd')
            plt.plot(lines['wav'][bad], 2500*np.ones(np.sum(bad)), 'rx')
            plt.ylim(0, 3000)
            plt.show()
                
        line_x = lines['pix'][~lines['bad']]
        line_w = lines['wav'][~lines['bad']]
        
        return line_x, line_w


    def fit_line_positions(self, 
                               obs_id, 
                               chip, 
                               fiber, 
                               linelist = None,
                               linefunc = None, 
                               window = 5, 
                               qc_sigma = 2.5,
                               verbose = True,
                               do_plot = False,
                               ):
        """
        Docstring 
        """
        if linelist is None:
            linelist = self.linelist
        if linefunc is None:
            linefunc = self.linefunc

        l1_obj = self.l1_stack[self.obs_ids.index(obs_id)]

        flux_ext, var_ext, wave_ext = self._get_orderlet_ext_from_fiber_name(chip, fiber)
        flux_arr = l1_obj[flux_ext]
        wave_arr = self.rough_wls[wave_ext]

        assert np.shape(flux_arr) == np.shape(wave_arr), "shape mismatch between flux array and rough WLS"

        norder, ncol = np.shape(flux_arr)

        line_x = [None]*norder
        line_w = [None]*norder
        line_m = [None]*norder
        
        for o in range(norder):
            if verbose:
                print(f"  order {o+1} of {norder}")
            
            line_x[o], line_w[o] = self.fit_line_positions(flux_arr[o], 
                                                           wave_arr[o],
                                                           linelist, 
                                                           linefunc, 
                                                           window = window, 
                                                           qc_sigma = qc_sigma, 
                                                           do_plot = do_plot,
                                                           )

            line_m[o] = (o+1)*np.ones_like(line_x[o], dtype=int)
             
        line_x = np.hstack(line_x)
        line_w = np.hstack(line_w)
        line_m = np.hstack(line_m)

        if do_plot:
            plt.figure(figsize=(5,6))
            plt.plot(line_x, line_w, 'k.')
            plt.xlabel("pixel column", fontsize=16)
            plt.ylabel("wavelength (A)", fontsize=16)
            plt.title(f"{obs_id} | {chip} {fiber}")
            plt.show()

        return line_x, line_w, line_m


    def calculate_wls_coeffs(self, 
                             obs_id, 
                             chip, 
                             method, 
                             linelist = None,
                             linefunc = None, 
                             window = 5, 
                             qc_sigma = 2.5,
                             polyorder_x = None,
                             polyorder_m = None,
                             polyorder_s = None,
                             verbose = True,
                             do_plot = False,
                             ):
        """
        Docstring 
        """
        # sanitize inputs
        if linelist is None:
            linelist = self.linelist
        if linefunc is None:
            linefunc = self.linefunc

        if polyorder_x is None:
            polyorder_x = self.polyorder_x
        if polyorder_m is None:
            polyorder_m = self.polyorder_m
        if polyorder_s is None:
            polyorder_s = self.polyorder_s

        flux_ext, _, _ = self._get_orderlet_ext_from_fiber_name(chip, 'CAL')
        norder, ncol = np.shape(self.rough_wls[flux_ext])

        if method == 'SCI':
            fibers = ['SCI1', 'SCI2', 'SCI3']
        elif method == 'CAL':
            fibers = ['CAL']
        else:
            raise ValueError("method must be 'SCI' or 'CAL'")
        
        # fit line positions order-by-order
        line = {}
        line['x'] = [None]*len(fibers)     # x = pixel
        line['w'] = [None]*len(fibers)     # w = wavelength
        line['m'] = [None]*len(fibers)     # m = order
        line['f'] = [None]*len(fibers)     # f = fiber

        for i, fiber in enumerate(fibers):
            if verbose:
                print(f"fitting line positions for {fiber} fiber")
            
            result = self.fit_line_positions_ffi(obs_id, 
                                                 chip, 
                                                 fiber, 
                                                 linelist = linelist,
                                                 linefunc = linefunc, 
                                                 window = 5, 
                                                 qc_sigma = 2.5,
                                                 verbose = True,
                                                 do_plot = False,
                                                 )

            line['x'][i], line['w'][i], line['m'][i] = result

            if method == 'SCI':
                line['f'][i] = int(fiber[-1])*np.ones_like(result[0])
            elif method == 'CAL':
                line['f'][i] = np.zeros_like(result[0])

        for k in line.keys():
            line[k] = np.hstack(line[k])

        # rescale position variables to [-1,1] for Legendre fitting
        _x = 2*line['x']/ncol - 1
        _m = 2*(line['m'] - line['m'].min())/(line['m'].max() - line['m'].min()) - 1

        if method == 'SCI':
            _f = line['f'] - 2

        # fit Legendre polynomials
        if method == 'SCI':
            V = legendre.legvander3d(_x, _m, _f, deg=[polyorder_x, polyorder_m, polyorder_s])

            coeffs, *_ = np.linalg.lstsq(V, line['w'], rcond=None)
            coeffs = coeffs.reshape(polyorder_x+1, polyorder_m+1, polyorder_s+1)

        elif method == 'CAL':
            V = legendre.legvander2d(_x, _m, deg=[polyorder_x, polyorder_m])

            coeffs, *_ = np.linalg.lstsq(V, line['w'], rcond=None)
            coeffs = coeffs.reshape(polyorder_x+1, polyorder_m+1)

        return coeffs


    @staticmethod
    def evaluate_wls_coeffs(coeffs, ncol, norder, nfiber):
        """
        Docstring
        """
        _x = np.linspace(-1, 1, ncol)
        _y = np.linspace(-1, 1, norder)
        _z = np.linspace(-1, 1, nfiber)

        if coeffs.ndim == 2:
            X, Y = np.meshgrid(_x, _y)
            W = legendgre.legval2d(X, Y, coeffs)
        
        elif coeffs.ndim == 3:
            X, Y, Z = np.meshgrid(_x, _y, _z)
            W = legendre.legval3d(X, Y, Z, coeffs)

        return W