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
from kpfpipe.models.metadata.receipt_columns import RECEIPT_COL

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
            # assume 2D image
            if key == 'NAXIS':
                self.header[key] = 2
            else:
                self.header[key] = value()

        # add empty level0 extensions
        for key, value in self.extensions:
            if key not in ['PRIMARY', 'RECEIPT']:
                atr = python_types[value]([])
            else:
                continue
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
            if isinstance(hdu, fits.ImageHDU):
                setattr(self, hdu.name, hdu.data)
            elif isinstance(hdu, fits.BinTableHDU):
                table = Table(hdu.data).to_pandas()
                setattr(self, hdu.name, table)
            elif hdu.name != 'PRIMARY' and hdu.name != 'RECEIPT':
                print("Unrecognized extension {}".format(hdu.name))
                continue

    def _read_from_NEID(self, hdul: fits.HDUList) -> None:
        '''
        Parse the HDUL based on NEID standards

        Args:
            hdul (fits.HDUList): List of HDUs parsed with astropy.

        '''
        for hdu in hdul:
            this_header = hdu.header

            # depending on the name of the HDU, store them with corresponding keys
            if hdu.name == 'PRIMARY':
                self.header = this_header
            # since KPF L0 data does not have equivilant 'data' and 'variance' 
            # extensions we will put them into the GREEN_CCD and RED_CCD 
            # extensions respectively. We will also populate the data and
            # variance attributes here.
            if hdu.name == 'DATA':
                self.GREEN_CCD = np.asarray(hdu.data, dtype=np.float64)
                self.data = self.GREEN_CCD
            elif hdu.name == 'VARIANCE':
                self.RED_CCD = np.asarray(hdu.data, dtype=np.float64)
                self.variance = self.RED_CCD
            else: 
                raise KeyError('Unrecognized extension {}'.format(hdu.name))
            

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
        
        for name, aux in self.extra_extensions.items():
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
            if value == fits.PrimaryHDU:
                head = fits.Header(cards=self.header)
                hdu = fits.PrimaryHDU(header=head)
            elif value == fits.ImageHDU:
                data = getattr(self, key)
                hdu = value(data=data)
            elif value == fits.BinTableHDU:
                table = Table.from_pandas(getattr(self, key))
                hdu = fits.table_to_hdu(table)
            else:
                print("Can't translate {} into a valid FITS format.".format(type(getattr(self, key))))
                continue
            hdu.name = key
            hdu_list.append(hdu)

        return hdu_list
    
if __name__ == "__main__":
    pass