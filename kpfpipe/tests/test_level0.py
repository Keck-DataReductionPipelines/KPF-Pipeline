import pytest
import warnings
import os
import shutil
from dotenv import load_dotenv

from kpfpipe.models.level0 import *

# Load .env file for test path 
load_dotenv()

# RECEIPT
# =============================================================================
def test_receipt():
    '''
    Add an entry in the receipt
    '''
    data = KPF0()
    data.receipt_add_entry('test', 'test', 'PASS')
    assert(len(data.receipt) == 1)

# =============================================================================
# AUXILIARY

def test_aux():
    '''
    Create and then delete an auxiliary extension 
    '''
    data = KPF0()
    data.create_extension('hello')
    # At this point only one extenion should exist
    assert(len(data.extension) == 1)
    assert('hello' in data.extension.keys())
    assert('hello' in data.header.keys())

    data.del_extension('hello')
    # No extension should exist
    assert(len(data.extension) == 0)
    assert('hello' not in data.header.keys())

def test_aux_exceptions():
    '''
    Test that proper exceptions are raised when invalid 
    input are given
    '''
    data = KPF0()
    data.create_extension('test1')

    with pytest.raises(NameError):
        # creating extension with duplicate name
        data.create_extension('test1')
    
    with pytest.raises(KeyError):
        # deleting a non-existent extension
        data.del_extension('test2')
    
    with pytest.raises(KeyError):
        # deleting a core HDU
        data.del_extension('PRIMARY')

# =============================================================================
# IO
# Level 0 path: 
fpath = os.environ['KPFPIPE_TEST_DATA'] + '/NEIDdata/TAUCETI_20191217/L0'
flist = [f for f in os.listdir(fpath)]
    
def test_from_NEID():
    '''
    Read all available level 0 data and check for data
    '''
    for f in flist:
        data = KPF0.from_fits(os.path.join(fpath, f), 'NEID')

        assert(isinstance(data.data, np.ndarray))
        assert(data.variance.shape == data.data.shape)
    
def test_NEID2KPF():
    '''
    Check that data 
    '''
    # Make a temporary folder
    try:
        os.mkdir('temp')
    except FileExistsError:
        pass

    for f in flist:
        # read NEID data and convert to KPF data
        data = KPF0.from_fits(os.path.join(fpath, f), 'NEID')
        to_path = 'temp/' + f
        data.to_fits(to_path)
        # read the converted data
        data2 = KPF0.from_fits(to_path, 'KPF')
        # compare the data value of the two
        assert(np.all(data2.data == data.data))
        assert(np.all(data2.variance == data.variance))
    # Clean up 
    shutil.rmtree('temp')
        


if __name__ == '__main__':
    pass
    
