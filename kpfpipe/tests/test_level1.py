import pytest

from kpfpipe.models.level1 import KPF1

def test_from_fits():
    # setup
    in_file = 'kpfpipe/tests/data/KPF.2007-04-04T09_17_51.376_e2ds_A.fits'
    K = KPF1()
    K.from_fits(in_file)