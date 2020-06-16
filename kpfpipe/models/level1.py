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

from kpfpipe.models.data_model import KPFDataModel

MAPPING = {
    # Order:  (header-key, data-target, data-key)
    'PRIMARY' : ('PRIMARY', None, None),
    'SCIFLUX' : ('SCI1_FLUX', 'flux', 'SCI1'),
    'SKYFLUX' : ('SKY_FLUX', 'flux', 'SKY'),
    'CALFLUX' : ('CAL_FLUX', 'flux', 'CAL'),
    'SCIVAR'  : ('SCI1_VARIANCE', 'variance', 'SCI1'),
    'SKYVAR'  : ('SKY_VARIANCE', 'variance', 'SKY'),
    'CALVAR'  : ('CAL_VARIANCE', 'variance', 'CAL'),
    'SCIWAVE' : ('SCI1_WAVE', 'wave', 'SCI1'),
    'SKYWAVE' : ('SKY_WAVE', 'wave', 'SKY'),
    'CALWAVE' : ('CAL_WAVE', 'wave', 'CAL'),
}

class KPF1(KPFDataModel):

    def __init__(self):
        '''
        Constructor
        '''
        KPFDataModel.__init__(self)
        
        self.flux: dict = {'CAL':  None,
                           'SCI1': None,
                           'SCI2': None,
                           'SCI3': None,
                           'SKY':  None}
        self.wave: dict = copy.deepcopy(self.flux)
        self.variance: dict = copy.deepcopy(self.flux)

        # start an empty segment table
        self.clear_segment()

        self.read_methods: dict = {
            'KPF':  self._read_from_KPF,
            'NEID': self._read_from_NEID
        }

    def _read_from_NEID(self, hdul: fits.HDUList,
                        force: bool=True) -> None:
        '''
        Parse the HDUL based on NEID standards
        '''
        for hdu in hdul:
            t = MAPPING.get(hdu.name)
            if t is None:
                raise ValueError(f'Unrecognized header "{hdu.name}"')

            (header_key, data_target, data_key) = t
            self.header[header_key] = hdu.header

            if data_target is not None:
                getattr(self, data_target)[data_key] = np.asarray(hdu.data, dtype=np.float32)

        self.wave['CAL'] = self.wave['SCI1']
        self.wave['SKY'] = self.wave['SCI1']

        # NEID files do not contain these keys
        for k in ['SCI2_FLUX', 'SCI2_WAVE', 'SCI2_VARIANCE', 
                'SCI3_FLUX', 'SCI3_WAVE', 'SCI3_VARIANCE']:
            self.header[k] = {}

        # Generate default segments
        for order in range(self.flux['SCI1'].shape[0]):
            label = 'Order {}'.format(order)
            begin_idx = (order, 0)
            end_idx = (order, self.flux['SCI1'].shape[1])
            length = self.flux['SCI1'].shape[1]
            comment = 'default segemnt for order {}'.format(order)

            row = [label, order, begin_idx, end_idx, length, comment]
            self.segments.loc[len(self.segments)] = row
    
    def _read_from_KPF(self, hdul: fits.HDUList,
                        force: bool=True) -> None:
        '''
        Parse the HDUL based on KPF standards
        '''
        for hdu in hdul:
            this_header = hdu.header
            if hdu.name == 'PRIMARY':
                # PRIMARY extension does not contain data
                self.header['PRIMARY'] = this_header
            elif hdu.name == 'SEGMENTS':
                self.header['SEGMENTS'] = hdu.header
                t = Table.read(hdu)
                self.segments = t.to_pandas()
            elif hdu.name == 'RECEIPT':
                # this is handled by the base class
                pass
            else: 
                try: 
                    fiber, datatype = hdu.name.split('_')
                except ValueError:
                    raise NameError('Extension {} not recognized'.format(hdu.name))

                if datatype == 'FLUX':
                    self.flux[fiber] = hdu.data
                elif datatype == 'VARIANCE':
                    self.variance[fiber] = hdu.data
                elif datatype == 'WAVE':
                    self.wave[fiber] = hdu.data
                else: 
                    raise ValueError('HDU name {} not recognized'.format(hdu.name))
                self.header[hdu.name] = hdu.header
    
    def create_hdul(self) -> list:
        '''
        create an hdul in FITS format
        '''
        hdu_list: dict = []
        # Add primary HDU 
        hdu = fits.PrimaryHDU()
        for key, value in self.header['PRIMARY'].items():
            hdu.header.set(key, value)
        hdu_list.append(hdu)

        # add fiber data
        for fiber, data in self.flux.items():
            flux_hdu = fits.ImageHDU(data=data)
            header_key = fiber + '_FLUX'
            for key, val in self.header[header_key].items():
                flux_hdu.header.set(key, val)
            flux_hdu.name = header_key

            variance_hdu = fits.ImageHDU(data=self.variance[fiber])
            header_key = fiber + '_VARIANCE'
            for key, val in self.header[header_key].items():
                variance_hdu.header.set(key, val)
            variance_hdu.name = header_key

            wave_hdu = fits.ImageHDU(data=self.wave[fiber])
            header_key = fiber + '_WAVE'
            for key, val in self.header[header_key].items():
                wave_hdu.header.set(key, val)
            wave_hdu.name = header_key  

            hdu_list.append(flux_hdu)
            hdu_list.append(variance_hdu)
            hdu_list.append(wave_hdu)

        # Add segments
        t = Table.from_pandas(self.receipt)
        hdu = fits.table_to_hdu(t)
        hdu.name = 'SEGMENTS'
        for key, value in self.header['SEGMENTS'].items():
            hdu.header.set(key, value)
        hdu_list.append(hdu)
        
        return hdu_list

# ============================================================================
# Segment related operations

    def add_segment(self, begin_idx: tuple, end_idx: tuple,
                    label=None, comment=None) -> None:
        '''
        Add an entry in the segments
        '''
        # data.segment

        if label is None:
            label = 'Custom segment {}'.format(self.n_custom_seg)
        
        if label in list(self.segments['Label']):
            raise NameError('provided label already exist for another segment')

        # make sure that the index provided is valid
        if begin_idx[0] != end_idx[0]:
            # segments must be on same order
            raise ValueError('Segment begin on order {}, end on order {}'.format(
                                begin_idx[0], end_idx[0]))
        if begin_idx[1] > end_idx[1]:
            raise ValueError('Segment begin location must be before end location')

        length = end_idx[1] - begin_idx[1] + 1
        order = begin_idx[0]
        row = [label, order, begin_idx, end_idx, length, comment]
        self.segments.loc[len(self.segments)] = row
        self.n_custom_seg += 1
    
    def clear_segment(self) -> None:
        '''
        Reset the table containing segments
        '''
        seg_header = ['Label', 'Order', 'Begin_idx',\
                      'End_idx', 'Length', 'Comment']
        self.segments = pd.DataFrame(columns=seg_header)
        self.header['SEGMENTS'] = {}
        self.n_custom_seg = 0
    
    def remove_segment(self, label: str) -> None:
        '''
        Remove a segment based on label
        '''
        if label not in list(self.segments['Label']):
            raise ValueError('{} not found'.format(label))
        idx = self.segments.index[self.segments['Label'] == label].tolist()
        self.segments = self.segments.drop(idx, axis=0)

    def info(self) -> None:
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
            'Fiber Name', 'Data Type', 'Data Dimension',
            '='*80 + '\n'
        )
        for fiber, data in self.flux.items():
            if data is not None:
                row = '|{:20s} |{:20s} |{:20s}\n'.format(
                    fiber+'_FLUX', 'array', str(data.shape))
                head += row
            if self.variance[fiber] is not None:
                row = '|{:20s} |{:20s} |{:20s}\n'.format(
                    fiber+'_VARIANCE', 'array', str(self.variance[fiber].shape))
                head += row
            if self.wave[fiber] is not None:
                row = '|{:20s} |{:20s} |{:20s}\n'.format(
                    fiber+'_WAVE', 'array', str(self.wave[fiber].shape))
                head += row
        row = '|{:20s} |{:20s} |{:20s}\n'.format(
            'Receipt', 'table', str(self.receipt.shape))
        head += row
        row = '|{:20s} |{:20s} |{:20s}\n'.format(
            'Segment', 'table', str(self.segments.shape))
        head += row
        
        for name, aux in self.extension.items():
            row = '|{:20s} |{:20s} |{:20s}\n'.format(name, 'table', str(aux.shape))
            head += row

        print(head)


