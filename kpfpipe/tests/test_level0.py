import pytest
import warnings

from kpfpipe.models.level0 import *

def test_read_NEID():
    fpath = '/home/qwang3/Desktop/Work/KPF-Pipeline-TestData\
/NEIDdata/TAUCETI_20191217/L0/neidTemp_2D20191217T023129.fits'

    to_path = 'test.fits'
    
    data = KPF0.from_fits(fpath, 'NEID')
    data.info()
    data.receipt_info()
    data.receipt_add_entry(int, 'test', 'yay!')
    data.receipt_add_entry(int, 'test2', 'double yay!')
    data.create_extension('hello')

    data.to_fits(to_path)
    data2 = KPF0.from_fits(to_path, 'KPF')

    # data2.info()
    print('===================')
    data2.info()
    data2.receipt_info()


if __name__ == '__main__':
    test_read_NEID()
