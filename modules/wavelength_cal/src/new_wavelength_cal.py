# standard dependencies
import configparser
import numpy as np

# pipeline dependencies
from kpfpipe.primitives.level1 import KPF1_Primitive

# external dependencies
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# local dependencies
from modules.wavelength_cal.src.alg import WaveCalibration

# global read-only variables

class WaveCalibrate(KPF1_Primitive):
    
    def __init__(self, action:Action, context:ProcessingContext) -> None:
        KPF1_Primitive.__init__(self, action, context)
        
        self.l1_obj = []
        self.cal_orderlette_names = self.action.args[]
        self.quicklook = self.action.args[]
        self.data_type =self.action.args[]

    def _perform(self) -> None: 
        
        for prefix in self.cal_orderlette_names:
            calflux = self.l1_obj[prefix]
        
        rough_wls = 
        
        peak_wavelengths_ang = 
        
        
        
        if self.alg.cal_type == 'LFC':
            if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('LFC'):
                raise ValueError('Not an LFC file!')
            
            lfc_allowed_wls = 
            
            wl_soln, wls_and_pixels = self.alg.run_wavelength_cal(
                calflux,rough_wls,peak_wavelengths_ang,lfc_allowed_wls)
            
        elif self.alg.cal_type == 'ThAr':
            if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('ThAr'):
                raise ValueError('Not a ThAr file!')
            
        elif self.alg.cal_type == 'Etalon':
            if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('Etalon'):
                raise ValueError('Not an Etalon file!')

            
        else:
            raise ValueError(
                'cal_type {} not recognized. Available options are LFC, ThAr, & Etalon'.format(
                    self.alg.cal_type))