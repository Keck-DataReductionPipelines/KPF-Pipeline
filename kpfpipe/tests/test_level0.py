import pytest
import warnings

from kpfpipe.models.level0 import *

def test_from_NEID():
    fn = 'resource/NEID/FLAT/neidTemp_2D20191214T001924.fits'
    data = KPF0.from_fits(fn, 'NEID')