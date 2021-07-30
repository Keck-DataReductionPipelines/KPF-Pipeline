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

class KPF1(KPF0):
    '''
    The level 1 KPF data. Initialize with empty fields.
    Attributes inherited from KPF0

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


