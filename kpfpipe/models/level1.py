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

class SpecDict(dict):
    '''
    This is a dictionary with modified __getitem__ and __setitem__
    for monitored access to its members
    '''
    def __init__(self, dictionary: dict, type_of_dict: str) -> None:
        '''
        Set the type of dict this is (array or header_key)
        '''
        dict.__init__(dictionary)
        if type_of_dict != 'header' and type_of_dict != 'array':
            # This should never happen since this classed is not 
            # intended for users 
            raise ValueError('invalid type')
        self.__type = type_of_dict 
    
    def __getitem__(self, key: str) -> np.ndarray:
        '''
        returns a copy of dict[key] instead, so that the original 
        value is not affected
        '''
        value_copy = copy.deepcopy(dict.__getitem__(self, key))
        return value_copy
    
    def __setitem__(self, key: str, value: type) -> None:
        '''
        Setting dict[key] with special constraint
        '''
        # depending on whether this dict stores header keywords
        # or np.ndarrays, apply different checks
        if self.__type == 'header':
            self.__set_header(key, value)
        elif self.__type == 'array':                
            self.__set_array(key, value)
        else: 
            # this should never happen
            pass
    
    def __set_array(self, key: str, value: np.ndarray) -> None:
        '''
        Setting a array 
        '''
        if not np.all(np.isnan(value)):
            try:
                assert(isinstance(value, np.ndarray))
                assert(len(value.shape) == 2)
            except AssertionError:
                # np.nan represent empty array
                raise TypeError('Value can only be 2D numpy arrays')
            
        # passed all tests, setting value
        dict.__setitem__(self, key, value)
    
    def __set_header(self, key: str, value: type):
        # if key is defined in KPF headers, make sure that
        # the values are the proper types
        
        # combine the two header key dictionaries 
        # all_keys = {**HEADER_KEY, **LVL1_KEY}
        # if key in all_keys.keys():
        #     try:
        #         assert(isinstance(value, all_keys[key]))
        #     except AssertionError:
        #         warnings.warn('expected value as {}, but got {}'.format(
        #                         all_keys[key], type(value)))
        # else: 
        #     # print(key, type(value), value)
        #     warnings.warn('{} not found in KPF_header')
        
        # this point is reached if no exception is raised 
        dict.__setitem__(self, key, value)

def find_nearest_idx(array: np.ndarray, value: np.float64) -> tuple:
    '''
    A helper function that finds the index of the nearest value in the array
    '''
    x = np.abs(array - value)
    idx = np.where(x == x.min())
    return idx

class KPF1(object):
    """
    Container object for level one data

    Attributes:

    """

    def __init__(self) -> None:
        '''
        Constructor
        Initializes an empty KPF1 data class
        '''
        # 1D spectrums
        # Each fiber is accessible through their key.
        # Contains 'data', 'wavelength', 'variance'
        self.flux = SpecDict({}, 'array')
        # header keywords
        # dict key contains 'data', 'wavelength', 'variance'
        self.header = SpecDict({}, 'header')
        self.wave = SpecDict({}, 'array')
        self.variance = SpecDict({}, 'array')

        self.segments = {}

        # supported data types
        self.read_methods = {
            'KPF1': self._read_from_KPF1,
            'HARPS': self._read_from_HARPS,
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

    def _read_from_KPF1(self, hdul: astropy.io.fits.HDUList,
                        force: bool=False) -> None:
        '''
        Populate the current data object with data from a KPF1 FITS file
        '''
        # check keys in both HEADER_KEY and LVL1_KEY
        all_keys = {**HEADER_KEY, **LVL1_KEY}
        # we assume that all keywords are stored in PrimaryHDU
        # loop through the 
        for hdu in hdul:
            # --TODO--
            # verify that all required keywords are present in header
            # and provided values are expected types

            if hdu.name == 'PRIMARY':
                self.header['PRIMARY'] = hdu.header
                continue

            # For each fiber, there are two HDU: flux and wave
            # We assume that the names of HDU follow the convention 
            # 'fiberName_flux' / 'fiberName_wave'
            try:
                fiber, array_type = hdu.name.split('_')
            except: 
                raise NameError('invalid HUD name: {}'.format(hdu.name))

            if array_type == 'WAVE':
                self.wave[fiber] = hdu.data
            elif array_type == 'FLUX':
                self.flux[fiber] = hdu.data
            elif array_type == 'VARIANCE':
                self.variance[fiber] = hdu.data
            else: 
                raise NameError('Array type must be "wave" or "flux", got {} instead'.format(array_type))  
            
            self.header[array_type] = hdu.header

    def _read_from_HARPS(self, hdul: astropy.io.fits.HDUList,
                        force: bool=True) -> None:
        '''
        Populate the current data object with data from a HARPS FITS file
        '''
        t0 = time.time()
        all_keys = {**HARPS_HEADER_E2DS, **HARPS_HEADER_RAW}
        this_header = {}

        # HARPS E2DS contains only 1 HDU, which stores the flux data
        # wave data need to be interpolated from the header keywords
        hdu = hdul[0]
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
            # Interpolate wave from headers
            # 
        header = hdu.header
        opower = header['ESO DRS CAL TH DEG LL']
        a = np.zeros(opower+1)
        wave = np.zeros_like(hdu.data, dtype=np.float64)
        for order in range(0, header['NAXIS2']):
            for i in range(0, opower+1, 1): 
                keyi = 'ESO DRS CAL TH COEFF LL' + str((opower+1)*order+i)
                a[i] = header[keyi]
            wave[order, :] = np.polyval(
                np.flip(a),
                np.arange(header['NAXIS1'], dtype=np.float64)
            )
        self.wave['SCI1'] = wave

        self.flux['SCI1'] = np.asarray(hdu.data, dtype=np.float64)
        # no variance is given in E2DS file, so set all to zero
        self.variance['SCI1'] = np.zeros_like(hdu.data, dtype=np.float64)
        # Save the header key to 'data'
        self.header['SCI1'] = this_header

    def _read_from_NEID(self, hdul: astropy.io.fits.HDUList,
                        force: bool=True) -> None:
        all_keys = {**NEID0, **NEID1}
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
            if hdu.name == 'PRIMARY':
                # no data is actually stored in primary HDU, as it contains only eader keys
                self.header['PRIMARY'] = this_header
            elif hdu.name == 'SCIFLUX':
                self.flux['SCI1'] = np.asarray(hdu.data, dtype=np.float64)
                self.header['SCI1_FLUX'] = this_header
            elif hdu.name == 'SKYFLUX':
                self.flux['SKY'] = np.asarray(hdu.data, dtype=np.float64)
                self.header['SKY_FLUX'] = this_header
            elif hdu.name == 'CALFLUX':
                self.flux['CAL'] = np.asarray(hdu.data, dtype=np.float64)
                self.header['CAL_FLUX'] = this_header
            elif hdu.name == 'SCIVAR':
                self.variance['SCI1'] = np.asarray(hdu.data, dtype=np.float64)
                self.header['SCI1_VARIANCE'] = this_header
            elif hdu.name == 'SKYVAR':
                self.variance['SKY'] = np.asarray(hdu.data, dtype=np.float64)
                self.header['SKY_VARIANCE'] = this_header
            elif hdu.name == 'CALVAR':
                self.variance['CAL'] = np.asarray(hdu.data, dtype=np.float64)
                self.header['CAL_VARIANCE'] = this_header
            elif hdu.name == 'SCIWAVE':
                self.wave['SCI1'] = np.asarray(hdu.data, dtype=np.float64)
                self.header['SCI1_WAVE'] = this_header
            elif hdu.name == 'SKYWAVE':
                self.wave['SKY'] = np.asarray(hdu.data, dtype=np.float64)
                self.header['SKY_WAVE'] = this_header
            elif hdu.name == 'CALWAVE':
                self.wave['CAL'] = np.asarray(hdu.data, dtype=np.float64)
                self.header['CAL_WAVE'] = this_header
        self.wave['CAL'] = self.wave['SCI1']
        self.wave['SKY'] = self.wave['SCI1']

    def segment_data(self, seg: np.ndarray=[], fiber: str=None) -> None:
        '''
        Segment the data based on the given array. 
        If an empty list is given, reset segment to default (1 segment/order)
        '''
        if len(seg) == 0:
            # empty segment array. reset to default 
            self.segments.clear() # first clear any value in 

        else: 
            # seg is a list of 2-element tuples. Each tuple contains the boundires defined
            # by wavelength 
            for wave_range in seg:
                try: 
                    # range must be valid
                    assert(wave_range[0] < wave_range[1])
                    start = find_nearest_idx(self.wave[fiber], wave_range[0])
                    stop = find_nearest_idx(self.wave[fiber], wave_range[1])
                    # must be in same order
                    assert(np.all(start[0] == stop[0]))

                    length = (stop[1] - start[1])[0]
                    # add this segment 
                    self.segments.add_segment(start, length, fiber)

                except AssertionError:
                    warnings.warn('invalid wavelength range: {}'.format(wave_range))
    
    def get_segmentation(self) -> list:
        '''
        returns the current segmenting of data as a list 
        '''
        # --TODO-- implement this
        return

    def get_segment(self, fiber: str, i: int) -> tuple:
        '''
        Get a copy of ith segemnt from a specific fiber.
        Whatever happens to the returned copy will not affect 
        the original piece of data

        '''
        
        # --TODO-- add to changes

    def to_fits(self, fn:str) -> None:
        """
        Collect all the level 1 data into a monolithic FITS file
        Can only write to KPF1 formatted FITS 
        """
        if not fn.endswith('.fits'):
            # we only want to write to a '.fits file
            raise NameError('filename must ends with .fits')
        hdu_list = []

        for name, header_keys in self.header.items():
            if name == 'PRIMARY':
                hdu = fits.PrimaryHDU()
            else: 
                fiber, array_type = name.split('_')

                if array_type == 'WAVE':
                    hdu = fits.ImageHDU(data=self.wave[fiber])
                elif array_type == 'VARIANCE':
                    hdu = fits.ImageHDU(data=self.variance[fiber])
                elif array_type == 'FLUX':
                    hdu = fits.ImageHDU(data=self.flux[fiber])

            for key, value in header_keys.items():
                hdu.header.set(key, value)

            hdu.name = name
            hdu_list.append(hdu)

        # finish up writing 
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(fn, overwrite=True)

    def verify(self) -> bool:
        '''
        Verify that the data stored in the current instance is valid
        '''
        return True

    def info(self) -> None: 
        '''
        Pretty print information about this data to stdout 
        '''
        # a typical command window is 80 in length
        head_key = '|{:20s} |{:20s} \n{:40}'.format(
            'Header_Name', '# Cards',
            '='*40 + '\n'
        )
        for key, value in self.header.items():
            row = '|{:20s} |{:20} \n'.format(key, len(value))
            head_key += row
        print(head_key)
        head = '|{:20s} |{:20s} |{:20s} \n{:40}'.format(
            'Data_Name', 'Data_Dimension', 'N_Segment',
            '='*60 + '\n'
        )
        for key, value in {**self.flux, **self.wave, **self.variance}.items():
            row = '|{:20s} |{:20s} |{:20} \n'.format(key, str(value.shape), len(self.segments))
            head += row
        print(head)

class Segement:
    '''
    Data wrapper that contains a segment of wave flux pair in the 
    KPF1 class. 
    '''
    def __init__(self):
        '''
        constructor
        '''
        # dictionary of segments in the data
        # This is a dictionary of arrays requiring two argument
        # locate a specific segement: fiber name and index
        self.seg_list = {}


    def add_segment(self, start_coordinate: tuple,
                       length: int,
                       fiber: str, 
                       order: int):
        ''' 
        Add a new segment to the current connection
        '''

        # check tat input is valid
        try: 
            assert(isinstance(start_coordinate, tuple))
            assert(len(start_coordinate) == 2)
        except AssertionError:
            raise ValueError('start_coordinate must be a tuple of 2')

        if fiber not in self.seg_list:
            # no segment has been created for this fiber yet
            self.seg_list[fiber] = [(start_coordinate, length, order)]
        else: 
            self.seg_list[fiber].append((start_coordinate, length, order))
    
    def clear(self):
        '''
        Clear the entire segment collection
        '''
        self.seg_list.clear()
    
    def delete(self, fiber: str, index: int):
        '''
        delete a single segment, given the fiber and index
        '''
        self.seg_list[fiber].pop(index)

class HK1(object):
    """
    Contanier for data associated with level one data products from the HK spectrometer

        Attributes:
        source (string): e.g. 'sky', 'sci', `cal`
        flux (array): flux values
        flux_err (array): flux uncertainties
        wav (array): wavelengths
    """
    def __init__(self):
        self.source # 'sky', 'sci', `cal`
        self.flux # flux from the spectrum
        self.flux_err # flux uncertainty
        self.wav # wavelenth solution

if __name__ == '__main__':
    K = KPF1()
    print(K.__flux)