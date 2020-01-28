import pytest

from kpfpipe.models.level1 import KPF1

def test_from_fits():
    # setup 
    in_file = 'kpfpipe/tests/data/KPF1.fits'
    K = KPF1()
    K.from_fits(in_file)