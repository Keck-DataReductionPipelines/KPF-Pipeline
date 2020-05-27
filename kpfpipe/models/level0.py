"""
Level 0 Data Model
"""
# Standard dependencies
import os
import copy

# External dependencies
import astropy
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
import numpy as np
import pandas as pd

from kpfpipe.models.data_model import KPFDataModel



class KPF0(KPFDataModel):
    """
    Container for level 0 data
    """

    def __init__(self):
        """
        Constructor
        """
        KPFDataModel.__init__(self)
        # level 0 contain only 1 array
        self.data: np.ndarray = None
        self.variance: np.ndarray = None

        self.read_methods: dict = {
            'KPF':  self._read_from_KPF,
            'NEID': self._read_from_NEID
        }
    
    def _read_from_NEID(self, hdul: fits.HDUList,
                        force: bool=True) -> None:
        '''
        Parse the HDUL based on NEID standards
        '''
        for hdu in hdul:
            this_header = hdu.header

            # depending on the name of the HDU, store them with corresponding keys
            if hdu.name == 'DATA':
                self.header['DATA'] = this_header
                self.data = np.asarray(hdu.data, dtype=np.float64)
            elif hdu.name == 'VARIANCE':
                self.header['VARIANCE'] = this_header
                self.variance = np.asarray(hdu.data, dtype=np.float64)
            else: 
                raise KeyError('Unrecognized')
    
    def _read_from_KPF(self, hdul: fits.HDUList,
                        force: bool=True) -> None:
        '''
        Parse the HDUL based on KPF standards
        '''
        for hdu in hdul:
            if isinstance(hdu, fits.PrimaryHDU):
                self.header['PRIMARY'] = hdu.header
                # Primary HDU should not contain any data
                if hdu.data is not None:
                    raise ValueError('Detected data in Primary HDU')
            
            elif hdu.name == 'DATA':
                # This HDU contains the 2D image array 
                self.data = hdu.data
                self.header['DATA'] = hdu.header
            
            elif hdu.name == 'VARIANCE':
                # This HDU contains the 2D variance array
                self.variance = hdu.data
                self.header['VARIANCE'] = hdu.header
    
    def info(self):
        '''
        Pretty print information about this data to stdout 
        '''
        if self.filename is not None:
            print('File name: {}'.format(self.filename))
        else: 
            print('Empty KPF0 Data product')
        # a typical command window is 80 in length
        head_key = '|{:20s} |{:20s} \n{:40}'.format(
            'Header Name', '# Cards',
            '='*80 + '\n'
        )
        for key, value in self.header.items():
            row = '|{:20s} |{:20} \n'.format(key, len(value))
            head_key += row
        print(head_key)
        head = '|{:20s} |{:20s} |{:20s} \n{:40}'.format(
            'Data Name', 'Data Type', 'Data Dimension',
            '='*80 + '\n'
        )
        if self.data is not None and self.variance is not None:
            row = '|{:20s} |{:20s} |{:20s}\n'.format('Data', 'array', str(self.data.shape))
            head += row
            row = '|{:20s} |{:20s} |{:20s}\n'.format('Variance', 'array', str(self.variance.shape))
            head += row
            row = '|{:20s} |{:20s} |{:20s}\n'.format('Receipt', 'table', str(self.receipt.shape))
            head += row
        
        for name, aux in self.extension.items():
            row = '|{:20s} |{:20s} |{:20s}\n'.format(name, 'table', str(aux.shape))
            head += row
        print(head)

    def create_hdul(self):
        '''
        create an hdul in FITS format
        '''
        hdu_list: list = []
        for name, header_keys in self.header.items():
            if name == 'PRIMARY':
                hdu = fits.PrimaryHDU()
            elif name == 'DATA': 
                hdu = fits.ImageHDU(data=self.data)
            elif name == 'VARIANCE':
                hdu = fits.ImageHDU(data=self.variance)
            else: 
                continue

            for key, value in header_keys.items():
                hdu.header.set(key, value)
            hdu.name = name

            hdu_list.append(hdu)
        return hdu_list

        
if __name__ == "__main__":
    pass