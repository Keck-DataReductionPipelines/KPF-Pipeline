'''
KPF Level 1 Data Model
'''
# Standard dependencies
from collections import OrderedDict
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

    def add_default_headers(self):
        """Adds the default header keywords as defined in KPF_headers_L1.csv"""

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

    @classmethod
    def from_l0(self, l0):
        """Create a level1 object from a level0 object in order to inherit headers."""
        l1 = KPF1()
        l1.header['PRIMARY'] = l0.header['PRIMARY']
        l1.add_default_headers()

        return l1

    def __init__(self):
        '''
        Constructor
        '''
        super().__init__()

        self.level = 1

        self.extensions = copy.copy(KPF_definitions.LEVEL1_EXTENSIONS)
        self.header_definitions = pd.read_csv(KPF_definitions.LEVEL1_HEADER_FILE)
        python_types = copy.copy(KPF_definitions.FITS_TYPE_MAP)

        for key, value in self.extensions.items():
            if key not in ['PRIMARY', 'RECEIPT', 'CONFIG']:
                atr = python_types[value]([])
                self.header[key] = fits.Header()
            else:
                continue
            self.create_extension(key, python_types[value])
            setattr(self, key, atr)

        header_keys = self.header.keys()
        del_keys = []
        for key in header_keys:
            if key not in self.extensions.keys():
                del_keys.append(key)
        for key in del_keys:
            del self.header[key]

        self.add_default_headers()

        self.read_methods: dict = {
            'KPF':  self._read_from_KPF,
            'NEID': self._read_from_NEID
        }

