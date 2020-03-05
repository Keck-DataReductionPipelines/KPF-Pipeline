import pytest

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
    
    with pytest.raises(ValueError):
        # ndarray contains invalid data
        array_dict['test'] = np.asarray([[-1, -2], [3, 4]], dtype=np.float64)


def test_array_dict_get_attr():
    '''
    Test that getting an attribute only gets a copy
    '''
    ## Arrange
    array_dict = SpecDict({}, 'array')
    array_dict['test'] = np.asarray([[1, 2], [3, 4]], dtype=np.float64)
    
    ## Act
    result = array_dict['test']
    result = np.asarray([[2, 3], [4, 5]], dtype=np.float64)

    ## Assert
    assert(np.any(result != array_dict['test'])) # Original should not be modified


## Constructor and Deconstructor
def test_init():
    '''
    Check that level1 data initializes correctly
    '''
    # --TODO--
    pass

## Initializer through I/O
def test_from_harps():
    '''
    Check that the _read_from_HARP() works
    '''
    # --TODO--
    pass

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
    test_spec_dict()