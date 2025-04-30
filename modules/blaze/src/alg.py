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
        self.method = cfg_params.get_config_value('method')
        self.flattemp = float(cfg_params.get_config_value('flattemp'))
        
        self.l1_obj = l1_obj
        self.date_mid = self.l1_obj.header['PRIMARY']['DATE-MID']
        self.dt = datetime.strptime(self.date_mid, "%Y-%m-%dT%H:%M:%S.%f")
        self.drptag = self.l1_obj.header['PRIMARY']['DRPTAG']
        
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
        header['BLAZMETH'] = self.method    # COMMENT intra-order optical element modeling
        header['BLAZSED']  = 'none'         # COMMENT inter-order normalization
        
    def add_extensions(self):
        self.l1_obj.create_extension('GREEN_SCI_BLAZE1', np.array)
        self.l1_obj.create_extension('GREEN_SCI_BLAZE2', np.array)
        self.l1_obj.create_extension('GREEN_SCI_BLAZE3', np.array)
        self.l1_obj.create_extension('GREEN_SKY_BLAZE', np.array)
        self.l1_obj.create_extension('GREEN_CAL_BLAZE', np.array)
        self.l1_obj.create_extension('RED_SCI_BLAZE1', np.array)
        self.l1_obj.create_extension('RED_SCI_BLAZE2', np.array)
        self.l1_obj.create_extension('RED_SCI_BLAZE3', np.array)
        self.l1_obj.create_extension('RED_SKY_BLAZE', np.array)
        self.l1_obj.create_extension('RED_CAL_BLAZE', np.array)
        
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
        self.l1_obj['GREEN_SKY_BLAZE'] = np.ones_like(self.l1_obj['GREEN_SKY_FLUX'])
        self.l1_obj['GREEN_CAL_BLAZE'] = np.ones_like(self.l1_obj['GREEN_CAL_FLUX'])
        self.l1_obj['RED_SCI_BLAZE1'] = np.ones_like(self.l1_obj['RED_SCI_FLUX1'])
        self.l1_obj['RED_SCI_BLAZE2'] = np.ones_like(self.l1_obj['RED_SCI_FLUX2'])
        self.l1_obj['RED_SCI_BLAZE3'] = np.ones_like(self.l1_obj['RED_SCI_FLUX3'])
        self.l1_obj['RED_SKY_BLAZE'] = np.ones_like(self.l1_obj['RED_SKY_FLUX'])
        self.l1_obj['RED_CAL_BLAZE'] = np.ones_like(self.l1_obj['RED_CAL_FLUX'])
        
        return self.l1_obj
    
    def blackbody(self):
        c = apc.c.value
        h = apc.h.value
        kB = apc.k_B.value
        
        wave_ext = ['SCI_WAVE1', 'SCI_WAVE2', 'SCI_WAVE3', 'SKY_WAVE', 'CAL_WAVE']
        blaze_ext = ['SCI_BLAZE1', 'SCI_BLAZE2', 'SCI_BLAZE3', 'SKY_BLAZE', 'CAL_BLAZE']
        
        for ccd in ['GREEN', 'RED']:
            for i in range(5):
                wave = self.l1_obj[f'{ccd}_{wave_ext[i]}']    # Angstroms
                temp = self.flattemp                          # Kelvin
                
                norder, npix = wave.shape
                blaze_array = np.zeros((norder,npix))
                
                for o in range(norder):
                    w = wave[o] * 1e-10   # Angstroms --> meters
                    T = temp * 1.0
                    B = 2*h*c**2 / w**5 / (np.exp((h*c)/(w*kB*T))-1)
                    
                    blaze_array[o] = B
                    
                self.l1_obj[f'{ccd}_{blaze_ext[i]}'] = blaze_array
                
        return self.l1_obj
    
    def spline(self, filter_size=7, sigma_cut=20., smooth_factor=5, return_mask=False):
        flux_ext = ['SCI_FLUX1', 'SCI_FLUX2', 'SCI_FLUX3', 'SKY_FLUX', 'CAL_FLUX']
        blaze_ext = ['SCI_BLAZE1', 'SCI_BLAZE2', 'SCI_BLAZE3', 'SKY_BLAZE', 'CAL_BLAZE']
        
        for ccd in ['GREEN', 'RED']:
            for i in range(5):
                data = self.l1_obj[f'{ccd}_{flux_ext[i]}']
                
                norder, npix = data.shape
                blaze_array = np.zeros((norder,npix))

                for o in range(norder):
                    x = np.arange(npix)
                    f = data[o]
            
                    med = median_filter(f, size=filter_size)             
                    mask = np.abs(f-med)/mad_std(f-med) > sigma_cut
                    mask[f == 0] = 1
            
                    spline = UnivariateSpline(x[~mask], f[~mask], s=smooth_factor*npix)
                    blaze_array[o] = spline(x)
            
                self.l1_obj[f'{ccd}_{blaze_ext[i]}'] = blaze_array
                
        if return_mask:
            return self.l1_obj, mask
        
        return self.l1_obj