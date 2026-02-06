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
        self.polyorder_f = int(cfg_params.get_config_value('polyorder_f'))

        # init routines
        self._load_stack()
        self._set_linefunc(str(cfg_params.get_config_value('linefunc')))
        self.ncol = 4080
        


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

        try:
            cls.linefunc_jac = staticmethod(getattr(cls, f'{name}_jac'))
        except AttributeError:
            raise ValueError(f"No such jacobian function: {f'{name}_jac'}")


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
    def gaussian_jac(theta, x):
        mu, sigma, a, b = theta
        dx = x - mu
        e = np.exp(-dx**2 / (2*sigma**2))

        J = np.empty((x.size, 4))
        J[:, 0] = a * e * dx / sigma**2
        J[:, 1] = a * e * dx**2 / sigma**3
        J[:, 2] = e
        J[:, 3] = 1.0
        
        return J


    @staticmethod
    def optimize_lsq(func, theta0, x, y, jac=None):
        """
        optimize theta for a given function using non-linear least-squares
        """
        def _residuals(theta, x, y):
            return func(theta, x) - y
        
        if jac is None:
            raise ValueError("why is jac None!?!?!?!")

        def _jac(theta, x, y):
                return jac(theta, x)
        
        result = least_squares(_residuals, theta0, jac=_jac, method='lm', args=(x,y))
        theta, rms = result.x, np.std(result.fun)
        
        return theta, rms


    def fit_line_positions_1D(self, 
                              flux1d, 
                              wave1d, 
                              linelist = None, 
                              linefunc = None, 
                              linefunc_jac = None, 
                              window=5, 
                              qc_sigma=2.5, 
                              do_plot=False
                              ):
        """
        Fit line postions (in pixel space) for all lines in a 1D flux array
            * line centers are determined by fitting a 1D function in pixel-vs-flux
            * quality control flags and rejects poorly conditioned fits
        """
        if linelist is None:
            linelist = self.linelist
        if linefunc is None:
            linefunc = self.linefunc
        if linefunc_jac is None:
            linefunc_jac = self.linefunc_jac

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
            theta, rms = self.optimize_lsq(linefunc, theta0, x, y, jac=linefunc_jac)
        
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
        
        return line_w, line_x


    def fit_line_positions_ffi(self, 
                               obs_id, 
                               chip, 
                               fibers, 
                               linelist = None,
                               linefunc = None, 
                               linefunc_jac = None,
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
        if linefunc_jac is None:
            linefunc_jac = self.linefunc_jac

        l1_obj = self.l1_stack[self.obs_ids.index(obs_id)]

        lines = {}
        for k in ['w', 'x', 'm', 'f']:
            lines[k] = [None]*len(fibers)

        for i, fiber in enumerate(fibers):
            if verbose:
                print(f"fitting {chip} {fiber} line positions")

            flux_ext, var_ext, wave_ext = self._get_orderlet_ext_from_fiber_name(chip, fiber)
            flux_arr = l1_obj[flux_ext]
            wave_arr = self.rough_wls[wave_ext]

            assert np.shape(flux_arr) == np.shape(wave_arr), "shape mismatch between flux array and rough WLS"

            norder, ncol = np.shape(flux_arr)

            for k in ['w', 'x', 'm', 'f']:
                lines[k][i] = [None]*norder
                    
            for o in range(norder):
                #print(f"  order {o+1} of {norder}")
                
                result = self.fit_line_positions_1D(flux_arr[o], 
                                                    wave_arr[o],
                                                    linelist, 
                                                    linefunc, 
                                                    linefunc_jac,
                                                    window = window, 
                                                    qc_sigma = qc_sigma, 
                                                    do_plot = do_plot,
                                                    )

                lines['w'][i][o] = result[0]
                lines['x'][i][o] = result[1]
                lines['m'][i][o] = (o+1)*np.ones_like(lines['x'][i][o], dtype=int)
                lines['f'][i][o] = np.array([fiber]*len(lines['x'][i][o]))

            if do_plot:
                plt.figure(figsize=(5,6))
                plt.plot(lines['x'][i][o], lines['w'][i][o], 'k.')
                plt.xlabel("pixel column", fontsize=16)
                plt.ylabel("wavelength (A)", fontsize=16)
                plt.title(f"{obs_id} | {chip} {fiber}")
                plt.show()

            for k in lines.keys():
                lines[k][i] = np.hstack(lines[k][i])        
        
        for k in lines.keys():
            lines[k] = np.hstack(lines[k])

        return lines


    def calculate_wls_coeffs(self, 
                             chip, 
                             lines,
                             polyorder_x = None,
                             polyorder_m = None,
                             polyorder_f = None,
                             verbose = True,
                             do_plot = False,
                             ):
        """
        Docstring 
        """
        # sanitize inputs
        if polyorder_x is None:
            polyorder_x = self.polyorder_x
        if polyorder_m is None:
            polyorder_m = self.polyorder_m
        if polyorder_f is None:
            polyorder_f = self.polyorder_f

        fibers = list(np.unique(lines['f']))

        if len(fibers) == 1:
            pass
        elif len(fibers) == 3:
            if not np.isin('SCI1', fibers) or not np.isin('SCI2', fibers) or not np.isin('SCI3', fibers):
                raise ValueError("expected SCI1 / SCI2 / SCI3")
        else:
            raise ValueError(f"expected 1 or 3 fibers, got {len(fibers)}")

        flux_ext, _, _ = self._get_orderlet_ext_from_fiber_name(chip, fibers[0])
        norder, ncol = np.shape(self.rough_wls[flux_ext])

        # rescale position variables to [-1,1] for Legendre fitting
        _x = 2*lines['x']/ncol - 1
        _m = 2*(lines['m'] - lines['m'].min())/(lines['m'].max() - lines['m'].min()) - 1

        if len(fibers) == 3:
            _f = np.array([fiber[-1] for fiber in lines['f']], dtype=int) - 2

        # fit Legendre polynomials
        if len(fibers) == 3:
            V = legendre.legvander3d(_x, _m, _f, deg=[polyorder_x, polyorder_m, polyorder_f])

            coeffs, *_ = np.linalg.lstsq(V, lines['w'], rcond=None)
            coeffs = coeffs.reshape(polyorder_x+1, polyorder_m+1, polyorder_f+1)

        elif len(fibers) == 1:
            V = legendre.legvander2d(_x, _m, deg=[polyorder_x, polyorder_m])

            coeffs, *_ = np.linalg.lstsq(V, lines['w'], rcond=None)
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
            W = legendre.legval2d(X, Y, coeffs)
        
        elif coeffs.ndim == 3:
            X, Y, Z = np.meshgrid(_x, _y, _z)
            W = legendre.legval3d(X, Y, Z, coeffs)

        return W


    def compute_wls_from_stack(self, 
                               chip, 
                               fibers, 
                               linelist = None,
                               linefunc = None, 
                               linefunc_jac = None,
                               window = 5, 
                               qc_sigma = 2.5,
                               polyorder_x = None,
                               polyorder_m = None,
                               polyorder_f = None,
                               verbose = True,
                               do_plot = False,
                               return_stacks = True,
                               ):
        """
        Docstring
        """
        if linelist is None:
            linelist = self.linelist
        if linefunc is None:
            linefunc = self.linefunc
        if polyorder_x is None:
            polyorder_x = self.polyorder_x
        if polyorder_m is None:
            polyorder_m = self.polyorder_m
        if polyorder_f is None:
            polyorder_f = self.polyorder_f

        lines_stack = [None]*self.nobs
        coeffs_stack = [None]*self.nobs

        for i, obs_id in enumerate(self.obs_ids):
            if verbose:
                print(f"\n{i+1} of {self.nobs} : {obs_id}")
            
            lines_stack[i] = self.fit_line_positions_ffi(obs_id, 
                                                         chip, 
                                                         fibers, 
                                                         linelist = linelist,
                                                         linefunc = linefunc, 
                                                         linefunc_jac = linefunc_jac,
                                                         window = window, 
                                                         qc_sigma = qc_sigma,
                                                         verbose = verbose,
                                                         do_plot = do_plot,
                                                         )



            coeffs_stack[i] = self.calculate_wls_coeffs(chip, 
                                                        lines_stack[i],
                                                        polyorder_x = polyorder_x,
                                                        polyorder_m = polyorder_m,
                                                        polyorder_f = polyorder_f,
                                                        verbose = True,
                                                        do_plot = False,
                                                        )

        coeffs_stack = np.array(coeffs_stack)
        bad = np.abs(coeffs_stack - np.median(coeffs_stack, axis=0)) / mad_std(coeffs_stack, axis=0) > qc_sigma
        coeffs_mean = np.sum(coeffs_stack * ~bad, axis=0)/np.sum(~bad, axis=0)

        W = self.evaluate_wls_coeffs(coeffs_mean, self.ncol, self._get_norder(chip), len(fibers))

        if return_stacks:
            return W, coeffs_mean, coeffs_stack, lines_stack

        return W, coeffs_mean