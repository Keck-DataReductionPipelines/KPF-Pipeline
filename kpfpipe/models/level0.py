"""
Data models for KPF data
"""
# Standard dependencies
import os
import copy 
import warnings
import time
# External dependencies
import astropy
from astropy.io import fits
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
# Pipeline dependencies
from kpfpipe.models.metadata.KPF_headers import HEADER_KEY, LVL1_KEY
from kpfpipe.models.metadata.HARPS_headers import HARPS_HEADER_E2DS, HARPS_HEADER_RAW
from kpfpipe.models.metadata.NEID_headers import NEID0, NEID1

class KPF0(object):
    """Container object for level zero data"""
    def __init__(self):
        """

        """
        ## Internal members 
        ## all are private members (not accesible from the outside directly)
        ## to modify them, use the appropriate methods.

        # 1D spectrums
        # Contain 'object', 'sky', and 'calibration' fiber.
        # Each fiber is accessible through their key.
        self.data = None
        self.variance = None

        # header keywords 
        self.header = {}

        # supported data types
        self.read_methods = {
            'KPF1': self._read_from_KPF0,
            'NEID': self._read_from_NEID
        }

    @classmethod
    def from_fits(cls, fn: str,
                  data_type: str) -> None:
        '''
        Create a KPF1 data from a .fits file
        '''
        # Initialize an instance of KPF1
        this_data = cls()
        # populate it with self.read()
        this_data.read(fn, data_type=data_type)
        # Return this instance
        return this_data

    def read(self, fn: str,
             data_type: str,
             overwrite: bool=True) -> None:
        '''
        Read the content of a .fits file and populate this 
        data structure. Note that this is not a @classmethod 
        so initialization is required before calling this function
        '''

        if not fn.endswith('.fits'):
            # Can only read .fits files
            raise IOError('input files must be FITS files')

        if not overwrite and not self.flux:
            # This instance already contains data, and
            # we don't want to overwrite 
            raise IOError('Cannot overwrite existing data')
        
        self.filename = os.path.basename(fn)
        with fits.open(fn) as hdu_list:
            # Use the reading method for the provided data type
            try:
                self.read_methods[data_type](hdu_list)
            except KeyError:
                # the provided data_type is not recognized, ie.
                # not in the self.read_methods list
                raise IOError('cannot recognize data type {}'.format(data_type))
    def _read_from_KPF0(self, hdul: fits.HDUList,
                        force: bool=True) -> None:
        # For now the KPF0 is the same as NEID 0 
        self._read_from_NEID(hdul)

    def _read_from_NEID(self, hdul: fits.HDUList,
                        force: bool=True) -> None:
        all_keys = {**NEID0}
        for hdu in hdul:
            this_header = {}
            for key, value in hdu.header.items():
                # convert key to KPF keys
                try: 
                    expected_type, kpf_key = all_keys[key]
                    if not kpf_key: # kpf_key != None
                        this_header[kpf_key] = expected_type(value)
                    else: 
                        # dont save the key for now
                        # --TODO-- this might change soon
                        pass
                except KeyError: 
                    # Key is in FITS file header, but not in metadata files
                    if force:
                        this_header[key] = value
                    else:
                        # dont save the key for now
                        pass
                except ValueError: 
                    # Expecte value in metadata file does not match value in FITS file
                    if force:
                        this_header[key] = value
                    else:
                        # dont save the key for now
                        pass
            # depending on the name of the HDU, store them with corresponding keys
            if hdu.name == 'DATA':
                self.header['DATA'] = this_header
                self.data = np.asarray(hdu.data, dtype=np.float64)
            elif hdu.name == 'VARIANCE':
                self.header['VARIANCE'] = this_header
                self.variance = np.asarray(hdu.data, dtype=np.float64)
        
    def to_fits(self):
        """
        Collect all the level 1 data into a monolithic FITS file
        Can only write to KPF1 formatted FITS 
        """
        if not fn.endswith('.fits'):
            # we only want to write to a '.fits file
            raise NameError('filename must ends with .fits')
        hdu_list = []

        for name, header_keys in self.header.items():
            if name == 'DATA':
                hdu = fits.PrimaryHDU(data=self.data)
            else: 
                hdu = fits.ImageHDU(data=self.variance)

            for key, value in header_keys.items():
                hdu.header.set(key, value)
            hdu.name = name
            hdu_list.append(hdu)

        # finish up writing 
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(fn, overwrite=True)