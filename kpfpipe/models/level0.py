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

from kpfpipe.models.base_model import KPFDataModel
from kpfpipe.models.metadata import KPF_definitions

class KPF0(KPFDataModel):
    """
    The level 0 KPF data. Initialized with empty fields

    Attributes:
        data (numpy.ndarray): 2D numpy array storing raw image
        variance (numpy.ndarray): 2D numpy array storing pixel variance
        read_methods (dict): Dictionaries of supported parsers. 
        
            These parsers are used by the base model to read in .fits files from other
            instruments

            Supported parsers: ``KPF``, ``NEID``, ``PARAS``

    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.level = 0
        self.extensions = KPF_definitions.LEVEL0_EXTENSIONS.items()
        python_types = KPF_definitions.FITS_TYPE_MAP

        # add level0 header keywords
        self.header_definitions = KPF_definitions.LEVEL0_HEADER_KEYWORDS.items()
        for key, value in self.header_definitions:
            if key == 'NAXIS':
                self.header[key] = 2
            else:
                self.header[key] = value()

        # add empty level0 extensions
        for key, value in self.extensions:
            if key == 'PRIMARY':
                atr = self.header
            else:
                atr = python_types[value]([])
            setattr(self, key, atr)

        self.read_methods: dict = {
            'KPF':   self._read_from_KPF,
            'NEID':  self._read_from_NEID,
            'PARAS': self._read_from_PARAS
        }

    def _read_from_KPF(self, hdul: fits.HDUList) -> None:
        '''
        Parse the HDUL based on NEID standards

        Args:
            hdul (fits.HDUList): List of HDUs parsed with astropy.

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


    def _read_from_NEID(self, hdul: fits.HDUList) -> None:
        '''
        Parse the HDUL based on NEID standards

        Args:
            hdul (fits.HDUList): List of HDUs parsed with astropy.

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
            

    def _read_from_PARAS(self, hdul: fits.HDUList,
                        force: bool=True) -> None:
        '''
        Parse the HDUL based on PARAS standards

        Args:
            hdul (fits.HDUList): List of HDUs parsed with astropy.
            
        '''
        for hdu in hdul: 
            if isinstance(hdu, fits.PrimaryHDU):
                # PARAS data is stored in primary only
                self.header['DATA'] = hdu.header
                self.data = hdu.data
                self.variance = np.zeros_like(self.data)
            
            else: 
                raise NameError('cannot recognize HDU {}'.format(hdu.name))

    
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

    def _create_hdul(self):
        '''
        Create an hdul in FITS format. 
        This is used by the base model for writing data context to file
        '''
        hdu_list = []
        hdu_definitions = self.extensions
        for key, value in hdu_definitions:
            if key == 'PRIMARY':
                head = fits.Header(cards=self.header)
                hdu = fits.PrimaryHDU(header=head)
            elif value == fits.ImageHDU:
                data = getattr(self, key)
                hdu = value(data=data)
            elif value == fits.TableHDU:
                table = Table.from_pandas(getattr(self, key))
                hdu = fits.TableHDU.from_columns(table)
            else:
                print("Can't translate {} into a valid FITS format.".fotmat(type(getattr(self, key))))
                continue

            hdu_list.append(hdu)

        return hdu_list
    
if __name__ == "__main__":
    pass