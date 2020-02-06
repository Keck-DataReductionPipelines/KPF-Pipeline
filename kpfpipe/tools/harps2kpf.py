
import numpy as np
from astropy.io import fits
from astropy.time import Time

import sys
import os

in_fpath = 'modules/TemplateFit/data/HARPS_Barnards_Star_benchmark'
out_fpath = 'examples/data/KPF1_Barnards_Star_benchmark'

in_ftype = 'HARPS'
out_ftype = 'KPF1'

def findfiles(fpath, extension):
    '''
    find all the files in the sub directories with relevant extension
    '''
    lst_fname = []
    for dirpath,_, filenames in os.walk(fpath):
        for filename in [f for f in filenames if f.endswith(extension)]:
            lst_fname.append(os.path.join(dirpath, filename))
    return lst_fname

class Converter: 

    def __init__(self):

        self.flux = None
        self.wave = None
        self.coef = []
        self.header = {}

    def read(self, fn: str, dtype: str,
             overwrite: bool=True) -> None:

        if fn.endswith('.fits') == False:
            # Can only read from .fits files
            msg = 'input files must be .fits files'
            raise IOError(msg)
    
        if overwrite != True and self.flux != None:
            # This instance already contains data
            msg = 'Cannot overwrite existing data'
            raise IOError(msg)
        
        self.filename = fn
        if dtype == 'HARPS':
            self.from_harps(fn, source='PRIMARY')
        
        if dtype == 'KPF1':
            self.from_kpf1(fn, source='PRIMARY')
 
    def write(self, fn:str, dtype: str) -> None:

        if fn.endswith('.fits') == False:
            # Can only read from .fits files
            msg = 'can only write to .fits files'
            raise IOError(msg)
    
        if dtype == 'KPF1':
            self.to_kpf1(fn, source=dtype)
        
        if dtype == 'HARPS':
            self.to_harps(fn, source=dtype)

    def from_kpf1(self, fn:str, source:str='primary') -> None:
        '''
        Read KPF Level 1 data
        Arg:
            fn (str): file name
            source (str): equivalent to .fits HDU
        '''
        with fits.open(fn) as hdu_list:
            
            # First record relevant header information
            header = hdu_list[source].header
            self.opower = header['waveinterp deg']
            self.berv = header['beryVel']
            NOrder = header['naxis2']
            NPixel = header['naxis1']

            self.flux = hdu_list[source].data
            NOrder, NPixel = self.flux.shape
            if (NOrder, NPixel) != self.flux.shape:
                msg = 'data array size does not agree with header'
                raise ValueError(msg)

            self.wave = np.zeros_like(self.flux)
            for order in range(0, NOrder):
                a = np.zeros(self.opower+1)
                for i in range(0, self.opower+1, 1):
                    keyi = 'hierarch waveinterp ord ' + str(order) +\
                    ' deg ' + str(i)
                    a[i] = header[keyi]
                self.wave[order] = np.polyval(
                    np.flip(a), np.arange(NPixel, dtype=np.float64)
                )
                self.coef.append(a)

            self.julian = Time(header['bjd'], format='jd')
            self.header = header

    def from_harps(self, fn:str, source:str='primary') -> None:
        '''
        Read HARPS data
        Arg:
            fn (str): file name
            source (str): equivalent to .fits HDU
        '''
        with fits.open(fn) as hdu_list:
            # First record relevant header information
            header = hdu_list[source].header
            # print(header)
            self.opower = header['eso drs cal th deg ll']
            self.berv = header['eso drs berv']
            NOrder = header['naxis2']
            NPixel = header['naxis1']

            self.flux = hdu_list[source].data
            NOrder, NPixel = self.flux.shape
            if (NOrder, NPixel) != self.flux.shape:
                msg = 'data array size does not agree with header'
                raise ValueError(msg)

            self.wave = np.zeros_like(self.flux)
            for order in range(0, NOrder):
                a = np.zeros(self.opower+1)
                for i in range(0, self.opower+1, 1):
                    keyi = 'eso drs cal th coeff ll' + str((self.opower+1)*order+i)
                    a[i] = header[keyi]
                self.wave[order] = np.polyval(
                    np.flip(a), np.arange(NPixel, dtype=np.float64)
                )
                self.coef.append(a)
            self.julian = Time(header['eso drs bjd'], format='jd')
            self.header['HARPS'] = header
    
    def to_harps(self, fn:str, source:str) -> None:

        # Initialize a new instance of HDU and save data to primary
        hdu = fits.PrimaryHDU(self.flux)
        hdu_header = hdu.header

        # Record
        NOrder, NPixel = self.flux.shape
        hdu_header.set('naxis2', NOrder)
        hdu_header.set('naxis1', NPixel)

        # degree of interpolation for wavelength
        deg_key = 'hierarch eso drs cal th deg ll'
        hdu_header.set(deg_key, self.opower)

        hdu_header.set('hierarch eso drs berv', self.berv)
        hdu_header.set('hierarch eso drs bjd', self.julian.jd)

        # Record polynomial interpolation results to headers
        if self.coef == []:
            print('wht?')
            for order in range(NOrder):
                c = np.polyfit(np.arange(NPixel), self.wave[order], self.opower)
                c = np.flip(c)
                for i, ci in enumerate(c):
                    key = 'hierarch eso drs cal th coeff ll' + str((self.opower+1)*order+i)
                    hdu_header.set(key, ci)
        else:
            for order, c in enumerate(self.coef):
                for i, ci in enumerate(c):
                    key = 'hierarch eso drs cal th coeff ll' + str((self.opower+1)*order+i)
                    hdu_header.set(key, ci)
        hdul = fits.HDUList([hdu])
        hdul.writeto(fn, overwrite=True)
        
    def to_kpf1(self, fn:str, source:str) -> None:

        hdu = fits.PrimaryHDU(self.flux)
        hdu_header = hdu.header

        # Record
        NOrder, NPixel = self.flux.shape
        hdu_header.set('naxis2', NOrder)
        hdu_header.set('naxis1', NPixel)
        hdu_header.set('beryVel', self.berv)
        hdu_header.set('bjd', self.julian.jd)
        hdu_header.set('hierarch waveinterp deg', self.opower)

        # Interpolation information for wavelength
        if self.coef == []:
            print('wht?')
            for order in range(0, NOrder):
                c = np.polyfit(np.arange(NPixel), self.wave[order], self.opower)
                c = np.flip(c)
                for i, ci in enumerate(c):
                    key = 'hierarch waveinterp ord ' + \
                        str(order) + ' deg ' + str(i)
                    hdu_header.set(key, ci)
        else:
            for order, c in enumerate(self.coef):
                for i, ci in enumerate(c):
                    key = 'hierarch waveinterp ord ' + \
                        str(order) + ' deg ' + str(i)
                    hdu_header.set(key, ci)
        hdul = fits.HDUList([hdu])
        hdul.writeto(fn, overwrite=True)

if __name__ == '__main__':
    for fn in findfiles(in_fpath, '.fits'):
        # change the names to KPF
        name = os.path.basename(fn)
        parts = name.split('.')[1:]
        parts.insert(0, 'KPF')
        out_name = '.'.join(parts)
        C = Converter()
        C.read(fn, 'HARPS')
        print(out_name)
        C.write(out_fpath +'/'+ out_name, 'KPF1')



