import pytest
import numpy as np
from kpfpipe.tools.harps2kpf import Converter

def test_harps2kpf():
    # Setup
    original = 'kpfpipe/tests/data/HARPS.2007-04-04T09_17_51.376_e2ds_A.fits'
    out1_file = 'kpfpipe/tests/data/KPF1.fits'
    out2_file = 'kpfpipe/tests/data/HARPS_out.fits'
    C1 = Converter()
    C2 = Converter()
    C3 = Converter()

    C1.read(original, 'HARPS')
    C1.write(out1_file, 'KPF1')
    C1.write(out2_file, 'HARPS')

    # compare the data from both types of files by C1
    C2.read(out1_file, 'KPF1')
    C3.read(out2_file, 'HARPS')
    assert(np.all(C2.flux == C3.flux))
    # due to interpolation, wave values are not exactly identical, 
    # but are equal within a certain tolerance. Will this be 
    # an issue?
    assert(np.all(C2.wave - C3.wave < 1e-3))