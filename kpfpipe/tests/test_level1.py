import pytest

import kpfpipe.models.level1 as lvl1

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
    # --TODO-- 
    pass

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


