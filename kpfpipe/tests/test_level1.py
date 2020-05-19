import pytest
import warnings
import os
import shutil
from dotenv import load_dotenv

from kpfpipe.models.level1 import *

# Load .env file for test path 
load_dotenv()

# =============================================================================
# Segments

def test():
    data = KPF1()
    data.add_segment((0, 1), (0, 5), label='test')
    data.add_segment((0, 1), (0 ,5))
    print(list(data.segments['Label']))

def test_add_segments():
    '''
    Check that segments are added properly
    '''
    data = KPF1()
    data.add_segment((0, 1), (0, 5), label='test')
    data.add_segment((0, 1), (0 ,5))
    assert('Custom segment 1' in list(data.segments['Label']))
    assert('test' in list(data.segments['Label']))

def test_remove_segment():
    '''
    Check that segments are removed properly
    '''
    data = KPF1()
    data.add_segment((0, 1), (0, 5), label='test')
    data.add_segment((0, 1), (0 ,5))
    data.remove_segment('test')
    data.remove_segment('Custom segment 1')
    assert(len(data.segments) == 0)

def test_segments_exceptions():
    '''
    Check that proper exceptions are raised with an invalid input
    '''
    data = KPF1()
    data.add_segment((0, 1), (0, 2), label='test')

    with pytest.raises(ValueError):
        # end index is less than beginning index
        data.add_segment((0, 3), (0, 2))
    
    with pytest.raises(ValueError):
        # segment not on same order
        data.add_segment((0, 1), (1, 2))
    
    with pytest.raises(NameError):
        # duplicate label 
        data.add_segment((0, 1), (0, 2), label='test')
    
    with pytest.raises(ValueError):
        # Non-existent label
        data.remove_segment('what')


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
        data = KPF1.from_fits(os.path.join(fpath, f), 'NEID')
        to_path = 'temp/' + f
        data.to_fits(to_path)
        # read the converted data
        data2 = KPF1.from_fits(to_path, 'KPF')
        # compare the data value of the two
        for key, value in data2.flux.items():
            assert(np.all(value == data.flux[key]))
        for key, value in data2.wave.items():
            assert(np.all(value == data.wave[key]))
        for key, value in data2.variance.items():
            assert(np.all(value == data.variance[key]))
    # Clean up 
    shutil.rmtree('temp')
if __name__ == "__main__":
    test()