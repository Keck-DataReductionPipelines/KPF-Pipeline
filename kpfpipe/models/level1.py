'''
KPF Level 1 Data Model
'''
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
from kpfpipe.models.level0 import KPF0

MAPPING = {
    # Order:  (header-key, dimension, data-key)
    'PRIMARY' : ('PRIMARY', None, None),
    'SCIFLUX' : ('SCI1_FLUX', 0, 'SCI1'),
    'SKYFLUX' : ('SKY_FLUX', 0, 'SKY'),
    'CALFLUX' : ('CAL_FLUX', 0, 'CAL'),
    'SCIVAR'  : ('SCI1_VARIANCE', 2, 'SCI1'),
    'SKYVAR'  : ('SKY_VARIANCE', 2, 'SKY'),
    'CALVAR'  : ('CAL_VARIANCE', 2, 'CAL'),
    'SCIWAVE' : ('SCI1_WAVE', 1, 'SCI1'),
    'SKYWAVE' : ('SKY_WAVE', 1, 'SKY'),
    'CALWAVE' : ('CAL_WAVE', 1, 'CAL'),
}

class KPF1(KPF0):
    '''
    The level 1 KPF data. Initialize with empty fields

    Attributes:
        data (dict): A dictionary of 5 orderlettes' 1D extracted spectrum.

            This is the attribute of the instance that contains all image data.
            The keys are the name of each orderlette, and the values are image data
            asscoaited with that orderlette. 
            
            Each image data is a stack by row by column 3D numpy array. The first dimension
            (stack) is fixed at 3. The first stack is the 1D extracted spectrum (2D ndarray),
            the second stack is the wavelength calibration, and the 3rd stack is the pixel variance.
            The second dimension (row) specifies a 1D extracted spectrum, and each row is an order.

            There are five orderlettes (valid keys to the dict) in total:
                - ``CAL``: Calibration fiber
                - ``SKY``: Sky fiber
                - ``SCI1``: Science fiber 1
                - ``SCI2``: Science fiber 2
                - ``SCI3``: Science fiber 3


        read_methods (dict): Dictionaries of supported parsers. 
        
            These parsers are used by the base model to read in .fits files from other
            instruments

            Supported parsers: ``KPF``, ``NEID``
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super().__init__()

        self.level = 1

        extensions = copy.copy(KPF_definitions.LEVEL1_EXTENSIONS)
        self.header_definitions = KPF_definitions.LEVEL1_HEADER_KEYWORDS.items()
        python_types = copy.copy(KPF_definitions.FITS_TYPE_MAP)

        for key, value in extensions.items():
            if key not in ['PRIMARY', 'RECEIPT', 'CONFIG']:
                atr = python_types[value]([])
                self.header[key] = fits.Header()
            else:
                continue
            self.create_extension(key, python_types[value])
            setattr(self, key, atr)

        for key, value in self.header_definitions:
            # assume 2D image
            if key == 'NAXIS':
                self.header['PRIMARY'][key] = 2
            else:
                self.header['PRIMARY'][key] = value()

        self.read_methods: dict = {
            'KPF':  self._read_from_KPF,
            'NEID': self._read_from_NEID
        }


