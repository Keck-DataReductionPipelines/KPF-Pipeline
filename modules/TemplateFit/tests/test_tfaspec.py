import pytest

# Test file for arg.py
from modules.TemplateFit.src import arg

# This serves as the file to test on
testfile = 'modules/TemplateFit/data/HARPS_Barnards_Star_benchmark/HARPS.2007-04-06T09_02_30.189_e2ds_A.fits'

@pytest.fixture(scope='module')
def spectrum():
    spec = arg.TFASpec(filename=testfile)
    return spec

# def test_op(spectrum):
#     spectrum2 = spectrum.copy()
#     assert(spectrum == spectrum2)
#     assert(id(spectrum) != id(spectrum2))