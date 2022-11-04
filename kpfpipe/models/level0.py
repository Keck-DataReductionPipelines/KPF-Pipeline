"""
Level 0 Data Model
"""
# Standard dependencies
from collections import OrderedDict
import os
import copy
import warnings

# External dependencies
import astropy
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
import numpy as np
from numpy.lib.shape_base import get_array_prepare
import pandas as pd

from kpfpipe.models.base_model import KPFDataModel
from kpfpipe.models.metadata import KPF_definitions
from kpfpipe.models.metadata.receipt_columns import RECEIPT_COL


class KPF0(KPFDataModel):
    """
    The level 0 KPF data. Initialized with empty fields.
    Attributes inherited from KPFDataModel, additional attributes below.

    Attributes:
        read_methods (dict): Dictionaries of supported parsers. 
        
            These parsers are used by the base model to read in .fits files from
            other instruments.

            Supported parsers: ``KPF``, ``NEID``, ``PARAS``

    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.level = 0
        extensions = copy.copy(KPF_definitions.LEVEL0_EXTENSIONS)
        python_types = copy.copy(KPF_definitions.FITS_TYPE_MAP)
        # add empty level0 extensions and empty headers for each extension
        for key, value in extensions.items():
            if key not in ['PRIMARY', 'RECEIPT', 'CONFIG', 'TELEMETRY']:
                if python_types[value] == np.ndarray:
                    atr = np.array([])
                else:    
                    atr = python_types[value]([])
                self.header[key] = fits.Header()
            else:
                continue
            self.create_extension(key, python_types[value])
            setattr(self, key, atr)

        # add level0 header keywords for PRIMARY header
        self.header_definitions = pd.read_csv(KPF_definitions.LEVEL0_HEADER_FILE)
        for i, row in self.header_definitions.iterrows():
            ext_name = row['Ext']
            key = row['Keyword']
            val = row['Value']
            desc = row['Description']
            if val is np.nan:
                val = None
            if desc is np.nan:
                desc = None
            self.header[ext_name][key] = (val, desc)

        self.read_methods: dict = {
            'KPF':   self._read_from_KPF,
            'NEID':  self._read_from_NEID,
            'PARAS': self._read_from_PARAS
        }        

    def _read_from_KPF(self, hdul: fits.HDUList) -> None:
        '''
        Parse the HDUL based on KPF standards

        Args:
            hdul (fits.HDUList): List of HDUs parsed with astropy.

        '''
        for hdu in hdul:
            if isinstance(hdu, fits.ImageHDU):
                if hdu.name not in self.extensions:
                    self.create_extension(hdu.name, np.ndarray)
                setattr(self, hdu.name, hdu.data)
            elif isinstance(hdu, fits.BinTableHDU):
                if hdu.name not in self.extensions:
                    self.create_extension(hdu.name, pd.DataFrame)
                table = Table(hdu.data).to_pandas()
                setattr(self, hdu.name, table)
            elif hdu.name != 'PRIMARY' and hdu.name != 'RECEIPT':
                warnings.warn("Unrecognized extension {} of type {}".format(hdu.name, type(hdu)))
                continue
            
            self.header[hdu.name] = hdu.header
        if self.level == 0:
            self.l0filename = self.filename
            self.l1filename = self.filename.replace('.fits', '_L1.fits')
            self.l2filename = self.filename.replace('.fits', '_L2.fits')
        if self.level == 1:
            self.l0filename = self.filename.replace('_L1.fits', '.fits')
            self.l1filename = self.filename
            self.l2filename = self.filename.replace('_L1.fits', '_L2.fits')
        if self.level == 2:
            self.l0filename = self.filename.replace('_L2.fits', '.fits')
            self.l1filename = self.filename.replace('_L2.fits', '_L1.fits')
            self.l2filename = self.filename

    def _read_from_NEID(self, hdul):
        '''
        Parse the HDUL based on NEID standards

        Args:
            hdul (fits.HDUList): List of HDUs parsed with astropy.

        '''

        for hdu in hdul:
            this_header = hdu.header
            # depending on the name of the HDU, store them with corresponding 
            # keys primary HDU is named 'DATA'
            if hdu.name == 'PRIMARY':
                self.header['PRIMARY'] = this_header
            elif hdu.name == 'DATA':
                self.create_extension('DATA', np.array)
                self.data = hdu.data
                self.DATA = self.data
            elif hdu.name == 'VARIANCE':
                self.create_extension('VARIANCE', np.array)
                self.variance = hdu.data
                self.VARIANCE = self.variance
            elif isinstance(hdu, fits.ImageHDU):
                if hdu.name not in self.extensions.keys():
                    self.create_extension(hdu.name, np.array)
                setattr(self, hdu.name, hdu.data)
            elif isinstance(hdu, fits.BinTableHDU):
                if hdu.name not in self.extensions.keys():
                    self.create_extension(hdu.name, pd.DataFrame)
                table = Table(hdu.data).to_pandas()
                setattr(self, hdu.name, table)
            else:
                warnings.warn('Unrecognized NEID extension {} of type {}'.format(hdu.name, type(hdu)))
                continue
                # raise KeyError('Unrecognized NEID extension {} of type {}'.format(hdu.name, type(hdu)))

            self.header[hdu.name] = this_header

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
            print('Empty {:s} Data product'.format(self.__class__.__name__))
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
            'Extension Name', 'Data Type', 'Data Dimension',
            '='*80 + '\n'
        )

        for name in self.extensions.keys():
            if name == 'PRIMARY':
                continue
            
            ext = getattr(self, name)
            if isinstance(ext, (np.ndarray, np.generic)):
                row = '|{:20s} |{:20s} |{:20s}\n'.format(name, 'image',
                                                        str(ext.shape))
                head += row
            elif isinstance(ext, pd.DataFrame):
                row = '|{:20s} |{:20s} |{:20s}\n'.format(name, 'table',
                                                        str(len(ext)))
                head += row
        print(head)
        
    def _create_hdul(self):
        '''
        Create an hdul in FITS format. 
        This is used by the base model for writing data context to file
        '''
        hdu_list = []
        hdu_definitions = self.extensions.items()
        for key, value in hdu_definitions:
            if value == fits.PrimaryHDU:
                head = self.header[key]
                hdu = fits.PrimaryHDU(header=head)
            elif value == fits.ImageHDU:
                data = getattr(self, key)
                if data is None:
                    ndim = 0
                    # data = np.array([])
                else:
                    ndim = len(data.shape)
                self.header[key]['NAXIS'] = ndim
                if ndim == 0:
                    self.header[key]['NAXIS1'] = 0
                else:
                    for d in range(ndim):
                        self.header[key]['NAXIS{}'.format(d+1)] = data.shape[d]
                head = self.header[key]
                hdu = fits.ImageHDU(data=data, header=head)
            elif value == fits.BinTableHDU:
                table = Table.from_pandas(getattr(self, key))
                self.header[key]['NAXIS1'] = len(table)
                head = self.header[key]
                hdu = fits.BinTableHDU(data=table, header=head)
            else:
                print("Can't translate {} into a valid FITS format."\
                      .format(type(getattr(self, key))))
                continue
            hdu.name = key
            if hdu.name == 'PRIMARY':
                hdu_list.insert(0, hdu)
            else:
                hdu_list.append(hdu)

        return hdu_list
    
if __name__ == "__main__":
    pass