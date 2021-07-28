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


        segments (dict): A dictionary of 5 tables of spectrum segments

            A segment is a meaningful part of a 1D spectrum, identified by a 
            ``begin_index`` and ``end_index``. Both are 2-element tuples that specifies
            the row-column coordinate on the data array. Additionally, each segment has 
            a corresponding string label that uniquely defines it. Each segment can also 
            be attached with a string comment.

            Each orderlette can have its own list of segments, so segments are sorted in
            a dictionary, with the keys being names of the orderlettes. The value to each
            key is a pandas.DataFrame table. Each row of the table represent a unique 
            segment.

            Examples:
                >>> from kpfpipe.models.level1 import KPF1
                # Assume we have an NEID level 1 file called "level1.fits"
                >>> level1 = KPF1.from_fits('level1.fits', 'NEID')
                # By default each order comes as its own segment. 
                # Access the default segments for the 'SKY' orderlette
                >>> data.segments['SKY']
                    Label   Order   Begin_idx        End_idx   Length   Comment 
                1     '1'       0      (0, 0)      (0, 9216)     9217   1st order 
                2     '2'       1      (1, 0)      (1, 9216)     9217   2nd order 
                3     '3'       2      (2, 0)      (2, 9216)     9217   3rd order 
                ...
                118 '118'     117    (117, 0)    (117, 9216)     9217   127th order 
                # Creating a segments for 'SCI1' that begins on 10th pixel of 0th order
                # and ends on 300th pixel of 0th order 
                >>> level1.add_segment('SKY', (0, 10), (0, 300), 'example', 'an example order')
                # Access the segment we just added (the very last entry)
                # Any pandas.dataframe method will work here
                >>> example_segment= level1.segments['SKY'].loc[119]
                >>> example_segment
                Label                              example
                Order                                    0
                Begin_idx                           (0, 0)
                End_idx                           (0, 300)
                Length                                 301
                Comment                   an example order
                Name: 119, dtype: object

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

        # create blank segments extension
        self.clear_segment()

        self.read_methods: dict = {
            'KPF':  self._read_from_KPF,
            'NEID': self._read_from_NEID
        }

    def _read_from_NEID(self, hdul: fits.HDUList) -> None:
        '''
        Parse the HDUL based on NEID standards
        Args:
            hdul (fits.HDUList): List of HDUs parsed with astropy.
        '''
        for hdu in hdul:
            t = MAPPING.get(hdu.name)
            if t is None:
                raise ValueError(f'Unrecognized header "{hdu.name}"')

            (header_key, dimension, data_key) = t
            self.header[header_key] = hdu.header

            if data_key is not None and dimension is not None: 

                if self.data[data_key] is None:
                    nx, ny = hdu.data.shape
                    self.data[data_key] = np.zeros((3, nx, ny))

                if dimension is not None:
                    self.data[data_key][dimension, :, :]= np.asarray(hdu.data, dtype=np.float64)

        # populate wave for SKY and CAL
        self.data['SKY'][1, :, :] = self.data['SCI1'][1, :, :]
        self.data['CAL'][1, :, :] = self.data['SCI1'][1, :, :]

        # Generate default segments
        for fiber in self.data.keys():
            if self.data[fiber] is not None:
                for order in range(self.data[fiber].shape[1]):
                    label = 'Order {}'.format(order)
                    begin_idx = (order, 0)
                    end_idx = (order, self.data[fiber].shape[2])
                    length = self.data[fiber].shape[2]
                    comment = 'default segemnt for order {}'.format(order)

                    row = [label, order, begin_idx, end_idx, length, comment]
                    self.segments[fiber].loc[len(self.segments[fiber])] = row
    
# ============================================================================
# Segment related operations

    def add_segment(self, fiber: str, begin_idx: tuple, end_idx: tuple,
                    label=None, comment=None) -> None:
        '''
        Add an entry in the segments

        Args:
            fiber (str): name of the orderlette 
            begin_index (tuple): xy coordinate of the start of segments
            end_idx (tuple): xy coordinate of the end of segments
            label (str): segment label. Auto generated if None
            comment (str): comment attached to the segment

        '''

        if self.segments[fiber] is None:
            raise ValueError('Cannot add segment for empty data')

        if label is None:
            label = 'Custom segment {}'.format(self.n_custom_seg)
        
        if label in list(self.segments[fiber]['Label']):
            raise NameError('provided label already exist for another segment')

        # make sure that the index provided is valid
        if begin_idx[0] != end_idx[0]:
            # segments must be on same order
            raise ValueError('Segment begin on order {}, end on order {}'.format(
                                begin_idx[0], end_idx[0]))
        if begin_idx[1] > end_idx[1]:
            raise ValueError('Segment begin location must be before end location')
    
        # make sure index are not out of bound
        if begin_idx[0] < 0 or begin_idx[0] > self.data[fiber].shape[0]:
            raise ValueError('Invalid beginning row index')
        if begin_idx[1] < 0 or begin_idx[1] > self.data[fiber].shape[1]:
            raise ValueError('Invalid beginning column index')
    
        if end_idx[0] < 0 or end_idx[0] > self.data[fiber].shape[0]:
            raise ValueError('Invalid end row index')
        if end_idx[1] < 0 or end_idx[1] > self.data[fiber].shape[1]:
            raise ValueError('Invalid end column index')

        length = end_idx[1] - begin_idx[1] + 1
        order = begin_idx[0]
        row = [label, order, begin_idx, end_idx, length, comment]
        self.segments[fiber].loc[len(self.segments[fiber])] = row
        self.n_custom_seg += 1
    
    def clear_segment(self) -> None:
        '''
        Reset the table containing segments
        '''
        seg_header = ['Label', 'Order', 'Begin_idx',\
                      'End_idx', 'Length', 'Comment']
        self.segments = {}
        for fiber, value in self.data.items():
            self.segments[fiber] = pd.DataFrame(columns=seg_header)
        self.n_custom_seg = 0
    
    def remove_segment(self, fiber: str, label: str) -> None:
        '''
        Remove a segment based on label

        Args: 
            label (str): label of the segment to be removed
        '''
        if label not in list(self.segments[fiber]['Label']):
            raise ValueError('{} not found'.format(label))
        idx = self.segments[fiber].index[self.segments[fiber]['Label'] == label].tolist()
        self.segments[fiber] = self.segments[fiber].drop(idx, axis=0)


