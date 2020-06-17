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
from astropy.table import Table
import numpy as np
import pandas as pd
import git
import datetime

# Pipeline dependencies
from kpfpipe.models.metadata.receipt_columns import *

class KPFDataModel:
    '''
    The base class for all KPF data models.
    Attributes:
        header (dict): a dictionary of headers of each extension (HDU)
        receipt (pandas.DataFrame): a table that records the history of this data
        extension (dict): a dictionary of auxiliary extensions.

    Warning: 
        This class (KPFDataModel) should not be used directly.
        Based on the data level of your .fits file, used the appropriate data model.
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.filename: str = ''

        self.header: dict = {'PRIMARY': {}, 
                             'RECEIPT': {}}

        # Construct the receipt table
        self.receipt: pd.DataFrame = pd.DataFrame(columns=RECEIPT_COL)

        # list of auxiliary extensions 
        self.extension: dict = {}

# =============================================================================
# I/O related methods
    @classmethod
    def from_fits(cls, fn: str,
                  data_type: str):
        """
        Create a data instance from a file
        Args: 
            fn (str): file path (relative to the repository)
            data_type (str): instrument type of the file
        Returns: 
            cls (data model class): the data instance containing the file content

        """
        this_data = cls()
        # populate it with self.read()
        this_data.read(fn, data_type=data_type)
        # Return this instance
        return this_data

    def read(self, fn: str,
             data_type: str,
             overwrite: bool=False) -> None:
        """
        Read the content of a .fits file and populate this 
        data structure. 
        Args: 
            fn (str): file path (relative to the repository)
            data_type (str): instrument type of the file
            overwrite (bool): if this instance is not empty, specifies
                 whether to overwrite
        Note:
            This is not a @classmethod so initialization is 
            required before calling this function
        """

        if not fn.endswith('.fits'):
            # Can only read .fits files
            raise IOError('input files must be FITS files')
        
        if not overwrite and self.filename != '':
            # This instance already contains data, and
            # we don't want to overwrite 
            raise IOError('Cannot overwrite existing data')
        
        self.filename = os.path.basename(fn)
        with fits.open(fn) as hdu_list:
            # Handles the Receipt and the auxilary HDUs 
            for hdu in hdu_list:
                if isinstance(hdu, fits.BinTableHDU):
                    t = Table.read(hdu)
                    if hdu.name == 'RECEIPT':
                        # Table contains the RECEIPT
                        self.header['RECEIPT'] = hdu.header
                        self.receipt = t.to_pandas()
                
                    else:
                        if 'AUX' in hdu.header.keys():
                            if hdu.header['AUX'] == True:
                                # This is an auxiliary extension
                                self.header[hdu.name] = hdu.header
                                self.extension[hdu.name] = t.to_pandas()
            # Leave the rest of HDUs to level specific readers
            try:
                self.read_methods[data_type](hdu_list)
            except KeyError:
                # the provided data_type is not recognized, ie.
                # not in the self.read_methods list
                raise IOError('cannot recognize data type {}'.format(data_type))
    
    def to_fits(self, fn:str) -> None:
        """
        Collect the content of this instance into a monolithic FITS file
        Args: 
            fn (str): file path

        Note:
            Can only write to KPF formatted FITS 
        """
        if not fn.endswith('.fits'):
            # we only want to write to a '.fits file
            raise NameError('filename must ends with .fits')    

        gen_hdul = getattr(self, 'create_hdul', None)
        if gen_hdul is None:
            raise TypeError('Write method not found. Is this the base class?')
        else: 
            hdu_list = gen_hdul()
        
        # handles receipt
        t = Table.from_pandas(self.receipt)
        hdu = fits.table_to_hdu(t)
        for key, value in self.header['RECEIPT'].items():
            hdu.header.set(key, value)
        hdu.name = 'RECEIPT'
        hdu_list.append(hdu)

        # handles any auxiliary extensions
        for name, table in self.extension.items():
            t = Table.from_pandas(table)
            hdu = fits.table_to_hdu(t)
            for key, value in self.header[name].items():
                hdu.header.set(key, value)
            hdu.name = name
            hdu_list.append(hdu)

        # finish up writing
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(fn, overwrite=True)

# =============================================================================
# Receipt related members
    def receipt_add_entry(self, Mod: str, param: str, status: str) -> None:
        '''
        Add an entry to the receipt
        Args:
            Mod (str): Name of the module making this entry
            param (str): param to be recorded
            status (str): status to be recorded
        '''
        
        # time of execution in ISO format
        time = datetime.datetime.now().isoformat()

        # get version control info (git)
        repo = git.Repo(search_parent_directories=True)
        git_commit_hash = repo.head.object.hexsha
        git_branch = repo.active_branch.name

        # add the row to the bottom of the table
        row = [time, '---', git_branch, git_commit_hash, \
               Mod, '---', '---', param, status]
        self.receipt.loc[len(self.receipt)] = row

    def receipt_info(self):
        '''
        Print the short version of the receipt
        '''
        print(self.receipt[['Time', 'Module_Name', 'Status']])

# =============================================================================
# Auxiliary related extension
    def create_extension(self, ext_name: str):
        '''
        Create an Auxiliary extension to be saved to FITS 
        Args:
            ext_name (str): extension name
        '''
        # check whether the extension already exist
        if ext_name in self.header.keys():
            raise NameError('name {} already exist as extension'.format(ext_name))
        
        self.extension[ext_name] = pd.DataFrame()
        self.header[ext_name] = {'AUX': True}
    
    def del_extension(self, ext_name: str):
        '''
        Delete an existing auxiliary extension
        Args:
            ext_name (str): extension name
        '''
        if ext_name not in self.extension.keys():
            raise KeyError('extension {} could not be found'.format(ext_name))
        
        del self.extension[ext_name]
        del self.header[ext_name]

if __name__ == '__main__':
    pass
