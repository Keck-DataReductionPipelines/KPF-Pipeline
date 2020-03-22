import pytest
import warnings

from kpfpipe.models.level1 import *

## All unit test should follow the AAA (arrange, act, assert) style

@pytest.fixture
def kpf1():
    '''
    Initializes an KPF1 data product from 
    a testing HARPS file.
    This fixture assumes that basic init/IO of KPF1 
    is functioning
    '''
    # --TODO--
    pass

## custom structs and helper functions
def test_spec_dict():
    '''
    Check that the get/set item methods are working 
    '''
    ## Arrange
    # Init empty dicts of available type
    array_dict = SpecDict({}, 'header')
    header_dict = SpecDict({}, 'array')

    ## Act
    # we are testing on empty arrays, so nothing to be done here
    
    ## Assert
    # make sure that they are still dictionaries
    # this make sure that the newly defined dicts still behave as default dicts
    assert(isinstance(array_dict, dict))
    assert(isinstance(header_dict, dict))

def test_array_dict_set_attr():
    '''
    Test that array dicts only 2D accept numpy arrays
    '''
    ## Arrange
    array_dict = SpecDict({}, 'array')

    ## act/assert
    with pytest.raises(TypeError):
        # Key is not a string
        array_dict[10] = np.ndarray([[1, 2], [3, 4]], dtype=np.float64)

    with pytest.raises(TypeError):
        # not a np.ndarray
        array_dict['test'] = [[1, 2], [3, 4]]
    
    with pytest.raises(TypeError):
        # ndarray is not 2D
        array_dict['test'] = np.asarray([1, 2, 3, 4], dtype=np.float64) #1D
    
    with pytest.raises(ValueError):
        # ndarray is not np.float64
        array_dict['test'] = np.asarray([[1, 2], [3, 4]], dtype=int)
    


def test_array_dict_get_attr():
    '''
    Test that getting an attribute only gets a copy
    '''
    ## Arrange
    array_dict = SpecDict({}, 'array')
    data = np.asarray([[1, 2], [3, 4]], dtype=np.float64)
    array_dict['test'] = data
    
    ## Act
    result = array_dict['test']

    ## Assert
    assert(np.any(result == data)) # result values should be equal
    assert(id(result) != id(array_dict['test'])) # instances should be differnet


## Constructor and Deconstructor
def test_init():
    '''
    Check that level1 data initializes correctly
    '''
    pass

## Initializer through I/O
def test_from_harps():
    '''
    Check that the _read_from_HARP() works
    '''
    ## Arrange
    warnings.filterwarnings("ignore")
    fpath = 'resource/HARPS_E2DS/HARPS.2007-04-04T09_17_51.376_e2ds_A.fits'

    ## Act
    data = KPF1.from_fits(fpath, 'HARPS')
    data.info()
    
def test_form_neid():
    '''
    Check that the _read_from_HARP() works
    '''
    ## Arrange
    warnings.filterwarnings("ignore")
    fpath = 'resource/NEID/TAUCETI_20191217/L1/neidL1_20191217T023129.fits'

    ## Act
    data = KPF1.from_fits(fpath, 'NEID')
    data.info()
def test_from_kpf1(): 
    '''
    Check that the _read_from_KPF() works as intended
    '''
    # --TODO--
    pass


def test_from_fits():
    '''
    check that the class method is functioning 
    '''
    # --TODO--
    pass

def test_to_fits():
    '''
    Check that data can be written to FITS files
    '''
    # --TODO--
    pass


## Interface
def test_get_attr():
    '''
    Check that access of information functions as intended
    '''
    # --TODO--
    pass

def test_set_attr():
    '''
    Check that data can only be modified in expected ways
    '''
    # --TODO--
    pass

if __name__ == '__main__':
    test_from_harps()
    test_form_neid()