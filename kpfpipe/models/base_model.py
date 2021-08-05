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
from astropy.io.fits import verify
from astropy.io.fits.hdu.image import PrimaryHDU
from astropy.time import Time
from astropy.table import Table
import numpy as np
import pandas as pd
import git
import datetime
import hashlib

# Pipeline dependencies
from kpfpipe.models.metadata.receipt_columns import *
from kpfpipe.models.metadata.config_columns import *
from kpfpipe.models.metadata.KPF_definitions import FITS_TYPE_MAP

class KPFDataModel(object):
    '''The base class for all KPF data models.

    Warning: 
        This class (KPFDataModel) should not be used directly.
        Based on the data level of your .fits file, used the appropriate
        level specific data model.

    This is the base model for all KPF data models. Level specific data inherit
    from this class, so any attribute and method listed here applies to all data
    models.

    Attributes:
        header (dict): a dictionary of headers of each extension (HDU)

            Header stores all header information from the FITS file. Since Each file is
            organized into extensions (HDUs), and Astropy parses each extension's 
            header cards into a dictionary, this attribute is structured as a dictionary
            of Astropy header objects. The first layer is the name of the header, and the second layer 
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

        extensions (dict): a dictionary of extensions.

            This attribute stores any additional information that any primitive may wish to 
            record to FITS. Creating an extension creates an empty extension of the given type and
            one may modify it directly. Creating an extension will also create a new key-value 
            pair in header, so that one can write header keywords to the extension. When writing to
            FITS extensions are stored in the FITS data type as specified in 
            kpfpipe.models.metadata.KPF_definitions.FITS_TYPE_MAP (image or binary table). Whitespace or 
            any symbols that may be interpreted by Python as an operator (e.g. -) are not
            allowed in extension names.

            Examples:
                >>> from kpfpipe.models.level0 import KPF0
                >>> data = KPF0()
                # Add an extension
                # A unique name is required
                >>> data.create_extension('extension1', pd.DataFrame)
                # Access the extension by using its name as an attribute
                # Add a column called 'col1' to the Dataframe
                >>> data.extension1['col1'] = [1, 2, 3]
                >>> data.extension1['extension1']
                col1
                0     1
                1     2
                2     3
                # add a key-value pair to the header
                >>> data.header['extension1']['key'] = 'value'
                # delete the extension we just made
                >>> data.del_extension['extension1']
        config (DataFrame): two-column dataframe that stores each line of the input configuration file
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.filename: str = None

        self.header = OrderedDict()
        self.header['PRIMARY'] = fits.Header()
        self.header['RECEIPT'] = fits.Header()
        self.header['CONFIG'] = fits.Header()

        self.receipt = pd.DataFrame([], columns=RECEIPT_COL)
        self.RECEIPT = self.receipt

        self.config = pd.DataFrame([], columns=CONFIG_COL)
        self.CONFIG = self.config

        self.primary = OrderedDict()
        self.PRIMARY = self.primary

        self.extensions = OrderedDict(PRIMARY=fits.PrimaryHDU,
                                      RECEIPT=fits.BinTableHDU,
                                      CONFIG=fits.BinTableHDU)


        # level of data model
        self.level = None # set in each derived class
        self.read_methods = dict()

    def __getitem__(self, key):
        return getattr(self, key.upper())

    def __setitem__(self, key, value):
        if key.upper() in self.extensions:
            setattr(self, key.upper(), value)
        else:
            data_type = type(value)
            self.create_extension(key.upper(), data_type)
            setattr(self, key.upper(), value)

    def __delitem__(self, key):
        self.del_extension(key.upper())

# =============================================================================
# I/O related methods
    @classmethod
    def from_fits(cls, fn, data_type='KPF'):
        """Create a data instance from a file

        This method emplys the ``read`` method for reading the file. Refer to 
        it for more detail.

        Args: 
            fn (str): file path (relative to the repository)
            data_type (str): (optional) instrument type of the file [default='KPF']
            
        Returns: 
            cls (data model class): the data instance containing the file content

        """
        this_data = cls()
        # populate it with self.read()
        this_data.read(fn, data_type=data_type)
        # Return this instance
        return this_data

    def read(self, fn, data_type, overwrite=False):
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
                if isinstance(hdu, fits.PrimaryHDU):
                    self.header[hdu.name] = hdu.header
                elif isinstance(hdu, fits.BinTableHDU):
                    t = Table.read(hdu)
                    if 'RECEIPT' in hdu.name:
                        # Table contains the RECEIPT
                        df = t.to_pandas()
                        df = df.reindex(df.columns.union(RECEIPT_COL,
                                                         sort=False),
                                                         axis=1, fill_value='')
                        setattr(self, hdu.name, df)
                        setattr(self, hdu.name.lower(), getattr(self, hdu.name))
                    self.header[hdu.name] = hdu.header
                    setattr(self, hdu.name, t.to_pandas())
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
        self.receipt_add_entry('from_fits', self.__module__,
                               f'md5_sum={md5.hexdigest()}', 'PASS')

    
    def to_fits(self, fn):
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
        
        # check that no card in any HDU is greater than 80
        # this is a hard limit by FITS 
        for hdu in hdu_list:
            if 'OBS FILE' in hdu.header.keys():
                del hdu.header['OBS FILE']
            elif 'PRIMARY' in hdu.header.keys():
                del hdu.header['PRIMARY']
            
        # finish up writing
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(fn, overwrite=True, output_verify='silentfix')

# =============================================================================
# Receipt related members
    def receipt_add_entry(self, module, mod_path, param, status, chip='all'):
        '''
        Add an entry to the receipt

        Args:
            module (str): Name of the module making this entry
            param (str): param to be recorded
            status (str): status to be recorded
            chip (str): (optional) which ccd [default='all']
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
        row = {'Time': time,
               'Code_Release': git_tag,
               'Commit_Hash': git_commit_hash,
               'Branch_Name': git_branch,
               'Chip': chip,
               'Module_Name': module,
               'Module_Level': str(self.level),
               'Module_Path': mod_path,
               'Module_Param': param,
               'Status': status}
        self.receipt = self.receipt.append(row, ignore_index=True)
        self.RECEIPT = self.receipt

    def receipt_info(self, receipt_name):
        '''
        Print the short version of the receipt

        Args:
            receipt_name (string): name of the receipt
        '''
        rec = getattr(self, receipt_name)
        msg = rec['Time', 'Module_Name', 'Status']
        print(msg)

# =============================================================================
# Auxiliary related extension
    def create_extension(self, ext_name, ext_type=pd.DataFrame):
        '''
        Create a new empty extension to be saved to FITS.
        Will not overwrite an existing extensions

        Args:
            ext_name (str): extension name
            ext_type (object): Python object type for this extension. 
                Must be present in kpfpipe.models.metadata.FITS_TYPE_MAP.keys().

        '''
        if ext_type not in FITS_TYPE_MAP.values():
            if ext_type == np.ndarray:
                ext_type = np.array
            else:
                raise TypeError("Unknown extension type {}. Available extension types: {}".format(ext_type, 
                                                                                                  FITS_TYPE_MAP.values()))
        reverse_map = OrderedDict(zip(FITS_TYPE_MAP.values(), FITS_TYPE_MAP.keys()))

        # check whether the extension already exist
        if ext_name in self.extensions.keys() and ext_name in self.__dir__():
            raise NameError('Name {} already exists as extension'.format(ext_name))

        setattr(self, ext_name, None)
        self.header[ext_name] = fits.Header()
        self.extensions[ext_name] = reverse_map[ext_type]
    
    def del_extension(self, ext_name):
        '''
        Delete an existing auxiliary extension

        Args:
            ext_name (str): extension name
            
        '''
        base = KPFDataModel()
        core_extensions = base.header.keys()
        if ext_name in core_extensions:
            raise KeyError('Can not remove any of the core extensions: {}'.format(core_extensions))
        elif ext_name not in self.extensions.keys():
            raise KeyError('Extension {} could not be found'.format(ext_name))
        
        delattr(self, ext_name)
        del self.header[ext_name]
        del self.extensions[ext_name]

if __name__ == '__main__':
    pass
