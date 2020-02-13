# Standard dependencies
import copy 

# External dependencies
import astropy
from astropy.io import fits
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt

# Pipeline dependencies
from kpfpipe.models.required_header import HEADER_KEY, LVL1_KEY

class KPF1(object):
    """
    Container object for level one data

    Attributes:
        Norderlets (dictionary): Number of orderlets per chip
        Norderlets_total (int): Total number of orderlets
        orderlets (dictionary): Collection of Spectrum objects for each chip, order, and orderlet
        hk (array): Extracted spectrum from HK spectrometer CCD; 2D array (order, col)
        expmeter (array): exposure meter sequence; 3D array (time, order, col)

    """

    def __init__(self) -> None:
        '''
        Constructor
        Initializes an empty KPF1 data class
        '''
        ## Internal members 
        ## all are private members (not accesible from the outside directly)
        ## to modify them, use the appropriate methods.

        # 1D spectrums
        # Contain 'object', 'sky', and 'calibration' fiber.
        # Each fiber is accessible through their key.
        self.__flux = {}
        # Contain error for each of the fiber
        self.__variance= {}
        self.__wave = np.nan
        # Ca H & K spectrum
        self.__hk = np.nan
        # header keywords
        self.__headers = None
        # dictionary of segments in the data
        # This is a 2D dictionary requiring 2 input to 
        # locate a specific segement: fiber name and index
        self.__segments = {}

        # supported data types
        self.read_methods = {
            'KPF1': self._read_from_KPF1,
            'HARPS': self._read_from_HARPS
        }

    @classmethod
    def from_fits(cls, fn: str,
                  data_type: str=None) -> None:
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
             overwrite: bool=True,
             data_type: str=None) -> None:
        '''
        Read the content of a .fits file and populate this 
        data structure. Note that this is not a @classmethod 
        so initialization is required before calling this function
        '''
        if not fn.endswith('.fits'):
            # Can only read .fits files
            raise IOError('input files must be FITS files')

        if not overwrite and not self.__flux:
            # This instance already contains data, and
            # we don't want to overwrite 
            raise IOError('Cannot overwrite existing data')
        
        with fits.open(fn) as hdu_list:
            # a data type is not provided, so try all the available
            # read methods. 
            if data_type == None:
                for dtype, read_method in self.read_methods.items():
                    try:
                        read_method(hdu_list, force=False)
                        # if this point is reached, then the file is successfully
                        # read. Break out of the for loop.
                        break
                    except: 
                        # an error is raised inside the read_method
                        # so try the next one instead
                        continue
                # if this point is reached, then none of the read_method work.
                raise IOError('Failed to read {} implicitly. Try again with an \
                                explicit data_type'.format(fn))
            else:
                # a data_type is actually provided. Use the 
                # corresponding reading method
                try:
                    self.read_methods[data_type](hdu_list)
                except KeyError:
                    # the provided data_type is not recognized, ie.
                    # not in the self.read_methods list
                    raise IOError('cannot recognize data type {}'.format(data_type))

    def _read_from_KPF1(self, hdul: astropy.io.fits.HDUList,
                        force: bool=True) -> None:
        '''
        Populate the current data object with data from a KPF1 FITS file
        '''
        # first parse header keywords
        self.header = {}
        # check keys in both HEADER_KEY and LVL1_KEY
        HEADER_KEY.update(LVL1_KEY)
        # we assume that all keywords are stored in PrimaryHDU
        # loop through the 
        for hdu in hdu_list:
            # we assume that all keywords are stored in PrimaryHDU
            if isinstance(hdu, astropy.io.fits.PrimaryHDU):
                # verify that all required keywords are present in header
                # and provided values are expected types
                for key, value_type in HEADER_KEY.items():
                    try: 
                        value = hdu.header[key]
                        if isinstance(value_type, Time):
                            # astropy time object requires time format 
                            # as additional initailization parameter
                            # setup format in Julian date
                            self.header[key] = value_type(value, format='jd')
                        
                        # add additional handling here, if required
                        else:
                            self.header[key] = value_type(value)
                    
                    except KeyError: 
                        # require key is not present in FITS header
                        msg =  'cannot read {} as KPF1 data: \
                                cannot find keyword {} in FITS header'.format(
                                self.filename, key)
                        if force:
                            self.header[key] = None
                        else:
                            raise IOError(msg)

                    except ValueError: 
                        # value in FITS header is not the anticipated type
                        msg = 'cannot read {} as KPF1 data: \
                            expected type {} from value of keyword {}, got {}'.format(
                            self.filename, value_type.__name__, type(value).__name__)
                        if force:
                            self.header[key] = None
                        else:
                            raise IOError() 

            # populate the _spectrum with data contain in current hdu
            # assuming that the hdu name is the fiber name 
            # ('SCI', 'CALIBRATION', 'SKY')
            self.__flux[hdu.name] = hdu.data  

            ## --TODO--
            # 1. calculate wavelength 
            #    How is wavelength information stored in KPF1 FITS file?
            
            ## set default segments (1 segment for each order)
            self.segment_data([])

    def _read_from_HARPS(self, hdul: astropy.io.fits.HDUList,
                        force: bool=True) -> None:
        '''
        Populate the current data object with data from a HARPS FITS file
        '''
        # --TODO--: implement this
        return

    def set_flux(self, fiber: str, value: np.ndarray) -> None:
        '''
        overwrite self.__flux[fiber] with the new value
        '''
        # --TODO-- implement some more checks
        try: 
            assert(isinstance(value, np.ndarray))
            self.__flux[fiber] = value
            self.segment_data() # resegment data to default (1 segement/order)
        except KeyError:
            # This happens when fiber is not a key in self.__flux 
            raise ValueError('{} fiber not recognized'.format(fiber))
        except AssertionError:
            # Value is not a np.ndarray
            raise ValueError('expected {} for value, got {}'.format(
                type(np.ndarray), type(value)))

    def set_variance(self, fiber: str, value: np.ndarray) -> None:
        '''
        overwrite self.__flux[fiber] with the new value
        '''
        # --TODO-- implement some more checks
        try: 
            assert(isinstance(value, np.ndarray))
            self.__variance[fiber] = value
            self.segment_data() # resegment data to default (1 segement/order)
        except KeyError:
            # This happens when fiber is not a key in self.__flux 
            raise ValueError('{} fiber not recognized'.format(fiber))
        except AssertionError:
            # Value is not a np.ndarray
            raise ValueError('expected {} for value, got {}'.format(
                type(np.ndarray), type(value)))
    
    def set_wave(self, value: np.ndarray) -> None:
        '''
        overwrite self.__wave with new value
        '''
        # --TODO-- implement some more checks
        try: 
            assert(isinstance(value, np.ndarray))
            self.__wave = value
            self.segment_data() # resegment data to default (1 segement/order)
        except AssertionError:
            # Value is not a np.ndarray
            raise ValueError('expected {} for value, got {}'.format(
                type(np.ndarray), type(value)))
        
    def segment_data(self, seg: np.ndarray=[]) -> None:
        '''
        Segment the data based on the given array. 
        If an empty list is given, reset segment to default (1 segment/order)
        '''
        if len(seg) == 0:
            # empty segment array. reset to default 
            self.__segments.clear() # first clear any value in 
            for fiber, data in self.__flux.items():
                self.__segments[fiber] = {}
                for i, order in enumerate(data):
                    # Segment(start_coordinate, length_of_segment, fiber_name)
                    self.__segments[fiber][i] = Segement((i, 0), len(order), fiber)
        else: 
            # --TODO-- implement non-trivial segment
            pass

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
        
        seg_info = self.__segments[fiber][i]
        # The coordinate information of the piece of spectrum 
        order = seg_info.start_coordinate[0]
        start = order
        finish = start + seg_info.length
        # the actual data
        flux_data = copy.deepcopy(self.__flux[fiber][order, start:finish])
        wave_data = copy.deepcopy(self.__wave[fiber][order, start:finish])
        # --TODO--: implement a wrapper class that contains this data
        return (wave_data, flux_data)

    def to_fits(self, fn:str) -> None:
        """
        Collect all the level 1 data into a monolithic FITS file
        Can only write to KPF1 formatted FITS 
        """
        if not fn.endswith('.fits'):
            # we only want to write to a '.fits file
            raise NameError('filename must ends with .fits')

        hdu_list = []
        for i, fiber in enumerate(self.__flux.items()):
            if i == 0: 
                # use the first fiber as the primary HDU
                data = self.__flux[fiber]
                hdu = fits.PrimaryHDU(data)
                # set all keywords
                for keyword, value in self.header:
                    if len(keyword) >= 8:
                        # for keywords longer than 8, a "HIERARCH" prefix
                        # must be added to the keyword.
                        keyword = '{} {}'.format('HIERARCH', keyword)
                    hdu.header.set(keyword, value)
            else: 
                # store the data only. Header keywords are only 
                # stored to PrimaryHDU
                hdu = fits.ImageHDU(self.__flux[key])
            hdu.name = fiber # set the name of hdu to the name of the fiber
            hdu_list.append(hdu)

        # finish up writing 
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(fn)

class Segement:
    '''
    Data wrapper that contains a segment of wave flux pair in the 
    KPF1 class. 
    '''
    def __init__(self, start_coordinate: tuple
                       length: int
                       fiber: str):
        '''
        constructor
        '''
        # check tat input is valid
        try: 
            assert(len(start_coordinate) == 2)
            assert(isinstance(start_coordinate[1], int))
            assert(isinstance(start_coordinate[0], int))
        except AssertionError
            raise ValueError('start_coordinate must be a tuple of 2 integers')

        self.start_coordinate = start_coordinate # coordinate of segment's beginning
        self.length = length                     # how long the segment is
        self.fiber = fiber                       # name of the fiber

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