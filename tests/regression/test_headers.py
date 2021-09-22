# Test file for FITS headers in ../models/metadata

import pytest
# By importing all the headers, we are making sure that 
# the script runs without any syntax error. 
import kpfpipe.models.metadata.KPF_definitions as KPF_definitions
import kpfpipe.models.metadata.HARPS_headers as HARPS_headers

# KPF Data product related tests
class test_kpf_structure():
    '''
    Making sure that all KPF headers are dictionaries with 
    strings as keys and "types" (such as str, int, etc.) as values 
    '''
    # Check that all header keys are stored in dictionaries
    assert(isinstance(KPF_definitions.LEVEL0_HEADER_KEYWORDS, dict))
    assert(isinstance(KPF_definitions.LEVEL1_HEADER_KEYWORDS, dict))
    assert(isinstance(KPF_definitions.LEVEL2_HEADER_KEYWORDS, dict))

    # check that each key-value pair are expected types
    for key, value in KPF_definitions.LEVEL0_HEADER_KEYWORDS.items():
        # keys must be strings (name of header keywords)
        assert(isinstance(key, str))
        # value must be the expected types of data
        assert(isinstance(value, type))
    
    # do the same for lvl1 and lvl2 data:
    for key, value in KPF_definitions.LEVEL1_HEADER_KEYWORDS.items():
        assert(isinstance(key, str))
        assert(isinstance(value, type))

    for key, value in KPF_definitions.LEVEL2_HEADER_KEYWORDS.items():
        assert(isinstance(key, str))
        assert(isinstance(value, type))

def test_harps_structure():
    '''
    Making sure that all HARPS headers are dictionaries containing
    key as strings, and value as 2-element tuples that containe 
    expected value types and equivalent KPF values
    '''
    # Check that HARPS headers are dictionary
    assert(HARPS_headers.HARPS_HEADER_RAW)
    assert(HARPS_headers.HARPS_HEADER_E2DS)

    # Check that each key-value pair are in expected structures
    for key, value in HARPS_headers.HARPS_HEADER_RAW.items():
        # Check that key is a string
        assert(isinstance(key, str))
        # Check that value is a two element tuple
        assert(isinstance(value, tuple))
        assert(len(value) == 2)
        # Check that each element is their proper type
        keyword_type, kpf_key = value
        assert(isinstance(keyword_type, type))
        assert(isinstance(kpf_key, str) or kpf_key is None)
    
    # do the same for other dict
    for key, value in HARPS_headers.HARPS_HEADER_E2DS.items():
        assert(isinstance(key, str))
        assert(isinstance(value, tuple))
        assert(len(value) == 2)
        keyword_type, kpf_key = value
        assert(isinstance(keyword_type, type))
        assert(isinstance(kpf_key, str) or kpf_key is None)

if __name__ == '__main__':
    pass

