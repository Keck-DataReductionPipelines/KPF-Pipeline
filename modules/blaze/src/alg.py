import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.ndimage import median_filter
from scipy.interpolate import UnivariateSpline
from astropy.stats import mad_std
import astropy.constants as apc


from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger

from modules.Utils.config_parser import ConfigHandler

class BlazeAlg:
    """Docstring
    
    """
    def __init__(self, l1_obj, default_config_path, logger=None):
        # Input arguments
        self.config = ConfigClass(default_config_path)
        if logger == None:
            self.log = start_logger('BlazeCorrection', default_config_path)
        else:
            self.log = logger
            
        cfg_params = ConfigHandler(self.config, 'PARAM')
        
        self.l1_obj = l1_obj
        self.date_mid = self.l1_obj.header['PRIMARY']['DATE-MID']
        self.dt = datetime.strptime(self.date_mid, "%Y-%m-%dT%H:%M:%S.%f")
        self.drptag = self.l1_obj.header['PRIMARY']['DRPTAG']
        self.method = 'spline'    # TODO: FIX HARDCODED METHOD
        
        try:
            self.readmode = self.l1_obj.header['PRIMARY']['READSPED']
        except KeyError:
            if 'regular' in self.l1_obj.header['PRIMARY']['GRACF']:
                self.readmode = 'regular'
            else:
                self.readmode = 'fast'
                
        self.add_extensions()
                
                    
    def add_keywords(self):
        header = self.l1_obj.header['PRIMARY']
        header['BLAZCORR'] = 1
        header['BLAZMETH'] = self.method
        
    def add_extensions(self):
        self.l1_obj.create_extension('GREEN_SCI_BLAZE1', np.array)
        self.l1_obj.create_extension('GREEN_SCI_BLAZE2', np.array)
        self.l1_obj.create_extension('GREEN_SCI_BLAZE3', np.array)
        self.l1_obj.create_extension('RED_SCI_BLAZE1', np.array)
        self.l1_obj.create_extension('RED_SCI_BLAZE2', np.array)
        self.l1_obj.create_extension('RED_SCI_BLAZE3', np.array)
        
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
        self.l1_obj['GREEN_SCI_BLAZE1'] = np.ones_like(self.l1_obj['GREEN_SCI_FLUX1'])
        self.l1_obj['GREEN_SCI_BLAZE2'] = np.ones_like(self.l1_obj['GREEN_SCI_FLUX2'])
        self.l1_obj['GREEN_SCI_BLAZE3'] = np.ones_like(self.l1_obj['GREEN_SCI_FLUX3'])
        self.l1_obj['RED_SCI_BLAZE1'] = np.ones_like(self.l1_obj['RED_SCI_FLUX1'])
        self.l1_obj['RED_SCI_BLAZE2'] = np.ones_like(self.l1_obj['RED_SCI_FLUX2'])
        self.l1_obj['RED_SCI_BLAZE3'] = np.ones_like(self.l1_obj['RED_SCI_FLUX3'])
        
        return self.l1_obj
    
    def blackbody(self):
        c = apc.c.value
        h = apc.h.value
        kB = apc.k_B.value
        
        for ccd in ['GREEN', 'RED']:
            for i in range(3):
                ext = f'{ccd}_SCI_WAVE{i+1}'    
                
                wave = self.l1_obj[ext]    # Angstroms
                temp = 6000.               # TODO: FIX HARDCODED TEMPERATURE
                
                norder, npix = wave.shape
                blaze_array = np.zeros((norder,npix))
                
                for o in range(norder):
                    w = wave[o] * 1e-10   # Angstrom --> meter
                    T = temp * 1.0        # Kelvin
                    B = 2*h*c**2 / w**5 / (np.exp((h*c)/(w*kB*T))-1)
                    
                    blaze_array[o] = B/B.max()
                    
                self.l1_obj[f'{ccd}_SCI_BLAZE{i+1}'] = blaze_array
                
        return self.l1_obj
    
    def spline(self, filter_size=7, sigma_cut=20., smooth_factor=5, return_mask=False):
        for ccd in ['GREEN', 'RED']:
            for i in range(3):
                ext = f'{ccd}_SCI_FLUX{i+1}'                
                data = self.l1_obj[ext]
                
                norder, npix = data.shape
                blaze_array = np.zeros((norder,npix))

                for o in range(norder):
                    x = np.arange(npix)
                    f = data[o]
            
                    med = median_filter(f, size=filter_size)             
                    mask = np.abs(f-med)/mad_std(f-med) > sigma_cut
                    mask += f == 0
            
                    spline = UnivariateSpline(x[~mask], f[~mask], s=smooth_factor*npix)
                    blaze_array[o] = spline(x)
            
                self.l1_obj[f'{ccd}_SCI_BLAZE{i+1}'] = blaze_array
                
        if return_mask:
            return self.l1_obj, mask
        
        return self.l1_obj