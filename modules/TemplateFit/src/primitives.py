import copy 
import typing as tp
# Using Python3's PEP 484 typing

from astropy.io import fits
import numpy as np
import scipy.ndimage as img

import matplotlib.pyplot as plt

# Local dependencies
from modules.TemplateFit.src import macro as mc
from modules.TemplateFit.src import arg as sp
from modules.TemplateFit.src import rm_outlier as rmo

class ProcessSpec:
    '''
    Action Object class that process the input spec
    '''
    def __init__(self):
        '''

        '''
    
    def bary_correct(self, spec: sp.Spec) -> sp.Spec: 
        '''

        '''
        berv = spec.header['eso drs berv']
        dlamb = np.sqrt((1+berv/mc.C_SPEED)/(1-berv/mc.C_SPEED))

        for order in range(spec.NOrder): 
            spec.shift_wave(dlamb, order)

        return spec
    
    def run(self, spec: sp.Spec) -> sp.Spec:
        '''
        
        '''
        spec = self.bary_correct(spec)
        return spec

class PostProcess:
    '''
    Action Object class that process the input spec
    '''
    def __init__(self):
        '''

        '''
        self.correct = rmo.RemoveOutlier()
    def average(self, result: list) -> np.ndarray:
        '''
        input should be a list of tfa_res class
        sort
        '''
        earliest = min(result, key=lambda x: x.julian_day)
        jd0 = earliest.julian_day

        # [julian_day, rv, err]
        final_result = np.zeros([len(result), 3])

        for i, res in enumerate(result):
            rv = res.rv
            bad = self.correct.sigma_clip(res.rv, 2)
            rv[bad] = 0
            mu_e = np.mean(rv)
            err = np.mean(res.get_error())
            # sec acc correct
            year = np.divide(res.julian_day - jd0, 365.25)
            offset = np.multiply(year, mc.SEC_ACC)
            mu_e += offset

            final_result[i] = [res.julian_day, -mu_e, err]
        return final_result
    
    def weighted(self, result: list) -> np.ndarray:
        '''
        input should be a list of tfa_res class
        sort
        '''
        earliest = min(result, key=lambda x: x.julian_day)
        jd0 = earliest.julian_day

        # [julian_day, rv, err]
        final_result = np.zeros([len(result), 3])

        for i, res in enumerate(result):
            # eqn 10, 11
            rv = res.rv
            bad = self.correct.sigma_clip(res.rv, 2)
            
            weight = np.divide(1, np.square(res.get_error()))
            # weight is 1xn row array, while rv is nx1 column vector
            weight = np.reshape(weight, rv.shape)
            # remove outliers
            bad = self.correct.sigma_clip(res.rv, 2)
            weight[bad] = 0
            
            Z = np.sum(weight)
            mu_e = np.sum(np.multiply(rv, weight))/Z
            err = np.mean(res.get_error())

            # sec acc correct
            year = np.divide(res.julian_day - jd0, 365.25)
            offset = np.multiply(year, mc.SEC_ACC)
            mu_e += offset

            final_result[i] = [res.julian_day, -mu_e, err]
        return final_result

    def run(self, result: list) -> np.ndarray:
        '''
        
        '''
        final_result = self.sec_acc(result)
        return final_result

class RemoveOutlier:

    def __init__(self):
        '''

        '''
    def sigma_clip(self, x, factor):
        '''
        perform a sigma clipping on the given data
        outlier is set to val, default 0
        '''
        x_mu = np.mean(x)
        x_sig = np.std(x)
        # boundaries
        up = x_mu + np.multiply(factor, x_sig)
        down = x_mu - np.multiply(factor, x_sig)

        too_high = (x - x_mu) > up
        too_low = (x - x_mu) < down
        bad = np.where(np.logical_or(too_high, too_low))
        return bad

if __name__ == '__main__':
    pass
