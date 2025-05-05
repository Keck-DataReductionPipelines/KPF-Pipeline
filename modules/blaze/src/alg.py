import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.ndimage import median_filter
from scipy.interpolate import LSQUnivariateSpline
from astropy.stats import mad_std
import astropy.constants as apc

import matplotlib.pyplot as plt


from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger

from modules.Utils.config_parser import ConfigHandler

class BlazeAlg:
    """Docstring
    
    """
    def __init__(self, target_l1, smooth_lamp_l1, default_config_path, logger=None):
        # Input arguments
        self.config = ConfigClass(default_config_path)
        if logger == None:
            self.log = start_logger('BlazeCorrection', default_config_path)
        else:
            self.log = logger
            
        cfg_params = ConfigHandler(self.config, 'PARAM')
        self.method = cfg_params.get_config_value('method')
        self.flattemp = float(cfg_params.get_config_value('flattemp'))
        
        self.target_l1 = target_l1
        self.smooth_lamp_l1 = smooth_lamp_l1
        self.date_mid = self.target_l1.header['PRIMARY']['DATE-MID']
        self.dt = datetime.strptime(self.date_mid, "%Y-%m-%dT%H:%M:%S.%f")
        self.drptag = self.target_l1.header['PRIMARY']['DRPTAG']
        
        try:
            self.readmode = self.target_l1.header['PRIMARY']['READSPED']
        except KeyError:
            if 'regular' in self.target_l1.header['PRIMARY']['GRACF']:
                self.readmode = 'regular'
            else:
                self.readmode = 'fast'
                
        self.add_extensions()
        
    def add_keywords(self):
        header = self.target_l1.header['PRIMARY']
        header['BLAZCORR'] = 1
        header['BLAZMETH'] = self.method    # COMMENT intra-order optical element modeling
        header['BLAZSED']  = 'none'         # COMMENT inter-order normalization
        
    def add_extensions(self):
        self.target_l1.create_extension('GREEN_SCI_BLAZE1', np.array)
        self.target_l1.create_extension('GREEN_SCI_BLAZE2', np.array)
        self.target_l1.create_extension('GREEN_SCI_BLAZE3', np.array)
        self.target_l1.create_extension('GREEN_SKY_BLAZE', np.array)
        self.target_l1.create_extension('GREEN_CAL_BLAZE', np.array)
        self.target_l1.create_extension('RED_SCI_BLAZE1', np.array)
        self.target_l1.create_extension('RED_SCI_BLAZE2', np.array)
        self.target_l1.create_extension('RED_SCI_BLAZE3', np.array)
        self.target_l1.create_extension('RED_SKY_BLAZE', np.array)
        self.target_l1.create_extension('RED_CAL_BLAZE', np.array)
        
    def apply_blaze_correction(self):
        try:
            blaze_method = self.__getattribute__(self.method)
        except AttributeError:
            self.log.error(f'Blaze correction method {self.method} not implemented.')
            raise(AttributeError)
            
        out_l1 = blaze_method()        
        self.add_keywords()
        
        return out_l1
    
    def uniform(self):        
        self.target_l1['GREEN_SCI_BLAZE1'] = np.ones_like(self.target_l1['GREEN_SCI_FLUX1'])
        self.target_l1['GREEN_SCI_BLAZE2'] = np.ones_like(self.target_l1['GREEN_SCI_FLUX2'])
        self.target_l1['GREEN_SCI_BLAZE3'] = np.ones_like(self.target_l1['GREEN_SCI_FLUX3'])
        self.target_l1['GREEN_SKY_BLAZE'] = np.ones_like(self.target_l1['GREEN_SKY_FLUX'])
        self.target_l1['GREEN_CAL_BLAZE'] = np.ones_like(self.target_l1['GREEN_CAL_FLUX'])
        self.target_l1['RED_SCI_BLAZE1'] = np.ones_like(self.target_l1['RED_SCI_FLUX1'])
        self.target_l1['RED_SCI_BLAZE2'] = np.ones_like(self.target_l1['RED_SCI_FLUX2'])
        self.target_l1['RED_SCI_BLAZE3'] = np.ones_like(self.target_l1['RED_SCI_FLUX3'])
        self.target_l1['RED_SKY_BLAZE'] = np.ones_like(self.target_l1['RED_SKY_FLUX'])
        self.target_l1['RED_CAL_BLAZE'] = np.ones_like(self.target_l1['RED_CAL_FLUX'])
        
        return self.target_l1
    
    def _blackbody_helper(self, ext):
        c = apc.c.value
        h = apc.h.value
        kB = apc.k_B.value
        
        w = self.target_l1[ext] * 1e-10   # Angstroms --> meters
        T = self.flattemp
        B = 2*h*c**2 / w**5 / (np.exp((h*c)/(w*kB*T))-1)
        
        return B
    
    def blackbody(self):
        c = apc.c.value
        h = apc.h.value
        kB = apc.k_B.value
        
        wave_ext = ['SCI_WAVE1', 'SCI_WAVE2', 'SCI_WAVE3', 'SKY_WAVE', 'CAL_WAVE']
        blaze_ext = ['SCI_BLAZE1', 'SCI_BLAZE2', 'SCI_BLAZE3', 'SKY_BLAZE', 'CAL_BLAZE']
        
        for ccd in ['GREEN', 'RED']:
            for i in range(5):
                blaze_array = self._blackbody_helper(f'{ccd}_{wave_ext[i]}')
                self.target_l1[f'{ccd}_{blaze_ext[i]}'] = blaze_array
                
        return self.target_l1
    
    def spline(self, filter_size=15, sigma_cut=20., num_knots=256, remove_lamp_blackbody=False):
        flux_ext = ['SCI_FLUX1', 'SCI_FLUX2', 'SCI_FLUX3', 'SKY_FLUX', 'CAL_FLUX']
        wave_ext = ['SCI_WAVE1', 'SCI_WAVE2', 'SCI_WAVE3', 'SKY_WAVE', 'CAL_WAVE']
        blaze_ext = ['SCI_BLAZE1', 'SCI_BLAZE2', 'SCI_BLAZE3', 'SKY_BLAZE', 'CAL_BLAZE']
        
        for ccd in ['GREEN', 'RED']:
            for i in range(5):
                flux = self.smooth_lamp_l1[f'{ccd}_{flux_ext[i]}']
                
                norder, npix = flux.shape
                blaze_array = np.zeros((norder,npix))
                lamp_bb_array = self._blackbody_helper(f'{ccd}_{wave_ext[i]}')

                for o in range(norder):
                    x = np.arange(npix)
                    f = np.array(flux[o], dtype='float')

                    mask = f == 0
                    med = median_filter(f[~mask], size=filter_size)             
                    mask[~mask] = np.abs(f[~mask]-med)/mad_std(f[~mask]-med) > sigma_cut
                                    
                    knots = np.linspace(x[~mask].min(), x[~mask].max(), num_knots)[1:-1]
                    spline = LSQUnivariateSpline(x[~mask], f[~mask], t=knots, ext='const')

                    blaze_array[o] = spline(x)
                    
                if remove_lamp_blackbody:
                    blaze_array /= lamp_bb_array
                
                self.target_l1[f'{ccd}_{blaze_ext[i]}'] = blaze_array
                        
        return self.target_l1