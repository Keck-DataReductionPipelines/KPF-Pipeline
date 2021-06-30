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
    data.receipt_add_entry('test', 'test_path', 'test', 'PASS')
    assert(len(data.receipt) == 1)

# =============================================================================
# AUXILIARY

def test_add():
    '''
    Create and then delete a new extension 
    '''
    data = KPF0()
    data.create_extension('hello')
    assert('hello' in data.extensions.keys())
    assert('hello' in data.header.keys())
    assert('hello' in data.__dir__())

    data.del_extension('hello')
    assert('hello' not in data.header.keys())
    assert('hello' not in data.__dir__())

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
flist = [f for f in os.listdir(fpath)][0:1]
    
def test_NEID():
    '''
    Check that we can read and write NEID data using the KPF data model 
    '''
    # Make a temporary folder
    try:
        os.mkdir('temp_level0')
    except FileExistsError:
        pass

    for f in flist:
        # read NEID data and convert to KPF data
        data = KPF0.from_fits(os.path.join(fpath, f), 'NEID')
        to_path = 'temp_level0/' + f
        data.to_fits(to_path)
        # read the converted data
        data2 = KPF0.from_fits(to_path, 'NEID')
        # compare the data value of the two
        assert(np.all(data2.data == data.data))
        assert(np.all(data2.variance == data.variance))
    # Clean up 
    shutil.rmtree('temp_level0')

def test_exceptions():
    
    f = flist[0]
    f_naught = f.split('.')[0] # same file without .fits file extension
    data = KPF0.from_fits(os.path.join(fpath, f), 'NEID')
    
    with pytest.raises(IOError):
        data2 = KPF0()
        # Invalid file extension
        data2.from_fits(os.path.join(fpath, f_naught), 'NEID')

    with pytest.raises(IOError):
        # overwriting without setting overwrite to True
        data.read(os.path.join(fpath, f), 'NEID')


if __name__ == '__main__':
    pass
    
