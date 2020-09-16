import pytest
import warnings
import os
import shutil
import time
from dotenv import load_dotenv

from kpfpipe.models.level0 import *
from tests.regression.test_level0 import test_from_NEID

# Load .env file for test path 
load_dotenv()

# Decorator function to assert that the execution time of a wrapped function 
# takes less than time_limit in seconds
def execution_time_limit(time_limit=300):
    def timer(func):
        def check(*args, **kwargs):
            t1 = time.time()
            func(*args, **kwargs)
            t2 = time.time()
            tt = t2 - t1

            print("Function {:s} completed in {:.0f} seconds.".format(func.__name__, tt))
            assert tt < time_limit, "Execution time for function {:s} took longer than the allocated limit ({:.0f} s > {:.0f} s)" \
                                    .format(func.__name__, tt, time_limit)

        return check
    return timer

# Test time to create level 0 files from NEID data. Should be pretty quick, 5 s limit
# =============================================================================
@execution_time_limit(time_limit=5)
def test_level0_creation():
    test_from_NEID()

if __name__ == '__main__':
    test_level0_creation()
    
