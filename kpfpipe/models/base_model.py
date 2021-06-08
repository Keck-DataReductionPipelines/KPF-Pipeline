"""
Data models for KPF data
"""
# Standard dependencies
import os
import sys
import copy 
import warnings
import time
from collections import OrderedDict

# External dependencies
import astropy
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
import numpy as np
import pandas as pd
import git
import datetime
import hashlib

# Pipeline dependencies
from kpfpipe.models.metadata.receipt_columns import *

class KPFDataModel(object):
    '''The base class for all KPF data models.

    Warning: 
        This class (KPFDataModel) should not be used directly.
        Based on the data level of your .fits file, used the appropriate
        level specific data model.

    This is the base model for all KPF data models. Level specific data inherit from this 
    class, so any attribute and method listed here applies to all data models.

    Attributes:
        header (dict): a dictionary of headers of each extension (HDU)

            Header stores all header information from the FIT file. Since Each file is
            organized into extensions (HDUs), and Astropy parses each extension's 
            header cards into a dictionary, this attribute is structured as a dictionary
            of dictionaries. The first layer is the name of the header, and the second layer 
            is the name of the key.
            
            Note: 
                For KPF, FITS extensions are identified by their name

            Examples
                >>> from kpfpipe.models.level0 import KPF0
                # Assume we have an NEID level 0 file called "level0.fits"
                >>> level0 = KPF0.from_fits('level0.fits', 'NEID')
                # Accessing key 'OBS_TIME' from the 'PRIMARY' HDU
                >>> obs_time = level0.header['PRIMARY']['OBS_TIME'] 

        receipt (pandas.DataFrame): a table that records the history of this data

            The receipt keeps track of the data process history, so that the information
            stored by this instance can be reproduced from the original data. It is 
            structured as a pandas.DataFrame table, with each row as an entry

            Primitives that modifies the content of a data product are expected to also 
            write to the receipt. Three string inputs from the primitive are required: name, 
            any relevant parameters, and a status. The receipt will also automatically fill 
            in additional information, such as the time of execution, code release version, 
            current branch, ect. 

            Note: 
                It is not recommended to modify the Dataframe directly. Use the provided methods
                to make any adjustments.

            Examples:
                >>> from kpfpipe.models.level0 import KPF0
                >>> data = KPF0()
                # Add an entry into the receipt
                # Three args are required: name_of_primitive, param, status
                >>> data.receipt_add_entry('primitive1', 'param1', 'PASS')
                >>> data.receipt
                                        Time     ...  Module_Param Status
                0  2020-06-22T15:42:18.360409     ...        input1   PASS

        extension (dict): a dictionary of auxiliary extensions.

            This attribute stores any additional information that any primitive may wish to 
            record to FITS. Creating an auxiliary creates an empty pandas.DataFrame table, and
            one may modify it directly. Creating an auxiliary will also create a new key-value 
            in header, so that one can write header keywords to the extension. When writing to
            FITS, Auxiliary extensions are stored as binary tables.

            Examples:
                >>> from kpfpipe.models.level0 import KPF0
                >>> data = KPF0()
                # Add an auxiliary extension
                # A unique name is required
                >>> data.create_extension('extension 1')
                # Access the extension by using its name as the dict key
                # Add a column called 'col1' to the Dataframe
                >>> data.extension['extension 1']['col1'] = [1, 2, 3]
                >>> data.extension['extension 1']
                col1
                0     1
                1     2
                2     3
                # add a key-value pair to the header
                >>> data.header['extension 1']['key'] = 'value'
                # delete the extension we just made
                >>> data.del_extension['extension 1']

    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.filename: str = None

        self.header: OrderedDict = {}

        # list of auxiliary extensions 
        self.extension: dict = {}

        # level of data model
        self.level = None # set in each derived class

# =============================================================================
# I/O related methods
    @classmethod
    def from_fits(cls, fn: str,
                  data_type: str):
        """Create a data instance from a file

        This method emplys the ``read`` method for reading the file. Refer to 
        it for more detail.

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
        """Read the content of a .fits file and populate this 
        data structure. 

        Args: 
            fn (str): file path (relative to the repository)
            data_type (str): instrument type of the file
            overwrite (bool): if this instance is not empty, specifies whether to overwrite
        
        Raises:
            IOError: when a invalid file is presented

        Note:
            This is not a @classmethod so initialization is 
            required before calling this function
        
        """

        if not fn.endswith('.fits'):
            # Can only read .fits files
            raise IOError('input files must be FITS files')
        
        if not overwrite and self.filename is not None:
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
            if data_type in self.read_methods.keys():
                self.read_methods[data_type](hdu_list)
            else:
                # the provided data_type is not recognized, ie.
                # not in the self.read_methods list
                raise IOError('cannot recognize data type {}'.format(data_type))

        # compute MD5 sum of source file and write it into a receipt entry for tracking.
        # Note that MD5 sum has known security vulnerabilities, but we are only using
        # this to ensure data integrity, and there is no known reason for someone to try
        # to hack astronomical data files.  If something more secure is is needed,
        # substitute hashlib.sha256 for hashlib.md5
        md5 = hashlib.md5()
        with open(fn, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        self.receipt_add_entry('from_fits', self.__module__, f'md5_sum={md5.hexdigest()}', 'PASS')

    
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

        gen_hdul = getattr(self, '_create_hdul', None)
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
                hdu.header[key] = value           # value could be a single value or a 2-D tuple with (value, comment)
                # hdu.header.set(key, value)
            hdu.name = name
            hdu_list.append(hdu)

        # check that no card in any HDU is greater than 80
        # this is a hard limit by FITS 
        for hdu in hdu_list:
            if 'OBS FILE' in hdu.header.keys():
                del hdu.header['OBS FILE']

        # finish up writing
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(fn, overwrite=True)

# =============================================================================
# Receipt related members
    def receipt_add_entry(self, Mod: str, mod_path: str, param: str, status: str) -> None:
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
        try:
            git_commit_hash = repo.head.object.hexsha
            git_branch = repo.active_branch.name
            git_tag = str(repo.tags[-1])
        except TypeError:  # expected if running in testing env
            git_commit_hash = ''
            git_branch = ''
            git_tag = ''
        # add the row to the bottom of the table
        row = [time, git_tag, git_branch, git_commit_hash, \
               Mod, str(self.level), mod_path, param, status]
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
