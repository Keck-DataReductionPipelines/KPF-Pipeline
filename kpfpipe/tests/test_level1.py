import pytest

from kpfpipe.models.level1 import KPF1

def test_from_fits():
    # setup
    in_file = 'kpfpipe/tests/data/KPF.2007-04-04T09_17_51.376_e2ds_A.fits'

    # Constructor
    K = KPF1()
    
    # read in data from a .fits file
    # Each order is treated as a segment
    K.from_fits(in_file)
    
    # segment each orders into different segments
    # by default each order is its own segment
    # segment cannot be larger than an order
    K.segment_data(seg_list)

    # print out information in each segment of the data
    K.info()

    # get a copy of specific segment by its index in the list 
    # of segments
    K.get_segment_index(ind)

    # 