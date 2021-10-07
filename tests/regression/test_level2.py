import pytest
import warnings
import os
import sys
import shutil
from dotenv import load_dotenv

from kpfpipe.models.level2 import *

# Load .env file for test path 
load_dotenv()

# =============================================================================
# IO
fpath = os.environ['KPFPIPE_TEST_DATA'] + '/NEIDdata/TAUCETI_20191217/L1'
flist = [f for f in os.listdir(fpath)]

def test_from_NEID():
    '''
    Read all available level 0 data and check for data
    '''
    for f in flist:
        data = KPF1.from_fits(os.path.join(fpath, f), 'NEID')

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
        data = KPF2.from_fits(os.path.join(fpath, f), 'NEID')
        to_path = 'temp/' + f
        data.to_fits(to_path)
        # read the converted data
        data2 = KPF2.from_fits(to_path, 'KPF')
        # compare the data value of the two
        for key in data2.extensions.keys():
            if key not in data.extensions.keys() or key == 'RECEIPT':
                continue
            value = data2[key]
            if value is None:
                assert(getattr(data, key) is None)
            elif isinstance(value, pd.DataFrame):
                assert(np.all(value.values == getattr(data, key).values))
            else:
                assert(np.all(value == getattr(data, key)))

    # Clean up 
    shutil.rmtree('temp')

def test_io_exception():

    data = KPF2()
    with pytest.raises(FileNotFoundError):
        # file does not exist 
        data.read('not_exist.fits', 'NEID')

    with pytest.raises(IOError):
        # valid path, but no .fits extension
        path = os.path.join(fpath, flist[0])
        path = path.split('.')[0] # remove the '.fit' from extension
        data.read(path, 'NEID')
    
    data = KPF2()
    path = os.path.join(fpath, flist[0])
    data.read(path, 'NEID')

    with pytest.raises(IOError):
        #trying to overwrite existing data
        data.read(path, 'NEID')

if __name__ == "__main__":
    test_NEID2KPF()