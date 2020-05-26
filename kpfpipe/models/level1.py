'''
KPF Level 1 Data Model
'''

from kpfpipe.models.data_model import *
import copy

class KPF1(KPFDataModel):

    def __init__(self):
        '''
        Constructor
        '''
        KPFDataModel.__init__(self)
        
        self.flux = {'CAL':  None,
                     'SCI1': None,
                     'SCI2': None,
                     'SCI3': None,
                     'SCI4': None,
                     'SCI5': None,
                     'SKY':  None}
        self.wave = copy.deepcopy(self.flux)
        self.variance = copy.deepcopy(self.flux)

        # start an empty segment table
        self.clear_segment()

        self.read_methods = {
            'KPF':  self._read_from_KPF,
            'NEID': self._read_from_NEID
        }

    def _read_from_NEID(self, hdul: fits.HDUList,
                        force: bool=True) -> None:
        
        for hdu in hdul:
            this_header = hdu.header
            if hdu.name == 'PRIMARY':
                # no data is actually stored in primary HDU, as it contains only eader keys
                self.header['PRIMARY'] = this_header
            elif hdu.name == 'SCIFLUX':
                self.flux['SCI1'] = np.asarray(hdu.data, dtype=np.float32)
                self.header['SCI1_FLUX'] = this_header
            elif hdu.name == 'SKYFLUX':
                self.flux['SKY'] = np.asarray(hdu.data, dtype=np.float32)
                self.header['SKY_FLUX'] = this_header
            elif hdu.name == 'CALFLUX':
                self.flux['CAL'] = np.asarray(hdu.data, dtype=np.float32)
                self.header['CAL_FLUX'] = this_header
            elif hdu.name == 'SCIVAR':
                self.variance['SCI1'] = np.asarray(hdu.data, dtype=np.float32)
                self.header['SCI1_VARIANCE'] = this_header
            elif hdu.name == 'SKYVAR':
                self.variance['SKY'] = np.asarray(hdu.data, dtype=np.float32)
                self.header['SKY_VARIANCE'] = this_header
            elif hdu.name == 'CALVAR':
                self.variance['CAL'] = np.asarray(hdu.data, dtype=np.float32)
                self.header['CAL_VARIANCE'] = this_header
            elif hdu.name == 'SCIWAVE':
                self.wave['SCI1'] = np.asarray(hdu.data, dtype=np.float32)
                self.header['SCI1_WAVE'] = this_header
            elif hdu.name == 'SKYWAVE':
                self.wave['SKY'] = np.asarray(hdu.data, dtype=np.float32)
                self.header['SKY_WAVE'] = this_header
            elif hdu.name == 'CALWAVE':
                self.wave['CAL'] = np.asarray(hdu.data, dtype=np.float32)
                self.header['CAL_WAVE'] = this_header
        self.wave['CAL'] = self.wave['SCI1']
        self.wave['SKY'] = self.wave['SCI1']

        # Generate default segments
        for order in range(self.flux['SCI1'].shape[0]):
            label = 'Order {}'.format(order)
            begin_idx = (order, 0)
            end_idx = (order, self.flux['SCI1'].shape[1])
            length = self.flux['SCI1'].shape[1]
            comment = 'default segemnt for order {}'.format(order)

            row = [label, order, begin_idx, end_idx, length, comment]
            self.segments.loc[len(self.segments)] = row
    
    def _read_from_KPF(self):
        pass
    
    def create_hdul(self):

        hdu_list = []
        # Add primary HDU 
        hdu = fits.PrimaryHDU()
        for key, value in self.header['PRIMARY']:
            hdu.header.set(key, value)
        # add fiber data
        for fiber, data in self.flux:
            flux_hdu = fits.ImageHDU(data=data)
            header_key = fiber + '_FLUX'
            hdu.name = header_key
            for key, val in self.header[header_key]:
                flux_hdu.header.set(key, val)

            variance_hdu = fits.ImageHDU(data=self.variance[fiber])
            header_key = fiber + '_VARIANCE'
            hdu.name = header_key
            for key, val in self.header[header_key]:
                variance_hdu.header.set(key, val)
            
            wave_hdu = fits.ImageHDU(data=self.wave[fiber])
            header_key = fiber + '_WAVE'
            hdu.name = header_key
            for key, val in self.header[header_key]:
                wave_hdu.header.set(key, val)
            
            hdu_list.append(flux_hdu)
            hdu_list.append(variance_hdu)
            hdu_list.append(wave_hdu)

        # Segments
        t = Table.from_pandas(self.receipt)
        hdu = fits.table_to_hdu(t)
        hdu.name = 'SEGMENTS'
        for key, value in self.header['SEGMENTS'].items():
            hdu.header.set(key, value)
        hdu_list.append(hdu)
# =============================================================================
# Segment related operations

    def add_segment(self, begin_idx: tuple, end_idx: tuple,
                    label=None, comment=None) -> None:
        '''

        '''
        if not label:
            label = 'Custom segment {}'.format(self.n_custom_seg)
        
        if label in self.segments['Label']:
            raise ValueError('provided lable already exist for another segment')

        # make sure that the index provided is valid
        if begin_idx[0] != end_idx[0]:
            # segments must be on same order
            raise ValueError('Segment begin on order {}, end on order {}'.format(
                                begin_idx[0], end_idx[0]))
        if begin_idx[1] <= end_idx[0]:
            raise ValueError('Segment begin location must be before end location')

        length = end_idx[1] - begin_idx[1]
        order = begin_idx[0]
        row = [label, order, begin_idx, end_idx, length, comment]
        self.segments.loc[len(self.segments)] = row
        self.n_custom_seg += 1
    
    def clear_segment(self):
        '''

        '''
        seg_header = ['Label', 'Order', 'Begin_idx',\
                      'End_idx', 'Length', 'Comment']
        self.segments = pd.DataFrame(columns=seg_header)
        self.header['SEGMENTS'] = {}
        self.n_custom_seg = 0
    
    def remove_segment(self, label: str) -> None:
        index = self.segments.index[
            self.segments['Label'] == label 
        ].tolist()
        print(index)
        self.segments.drop(index, axis=0)

    def info(self):
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
                row = '|{:20s} |{:20s} |{:20s}\n'.format(
                    fiber+'_VARIANCE', 'array', str(self.variance.shape))
                head += row
                row = '|{:20s} |{:20s} |{:20s}\n'.format(
                    fiber+'_WAVE', 'array', str(self.wave.shape))
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


