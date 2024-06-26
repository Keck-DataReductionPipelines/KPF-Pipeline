import pytest
import warnings
import os
import socket
import shutil
import time
import pandas as pd
from dotenv import load_dotenv

from kpfpipe.tools.recipe_test_unit import run_recipe
from kpfpipe.pipelines.kpf_parse_ast import RecipeError

neid_config = "examples/default_neid.cfg"

neid_recipe = """test_data_dir = KPFPIPE_TEST_DATA + '/NEIDdata' 
data_type = config.ARGUMENT.data_type
output_dir = config.ARGUMENT.output_dir

input_flat_pattern = config.ARGUMENT.input_flat_file_pattern
input_lev0_prefix = config.ARGUMENT.input_lev0_file_prefix
input_lev1_prefix = config.ARGUMENT.input_lev1_file_prefix

obs_list = config.ARGUMENT.obs_list
flat_stem_suffix = config.ARGUMENT.output_flat_suffix
lev1_stem_suffix = config.ARGUMENT.output_lev1_suffix
lev2_stem_suffix = config.ARGUMENT.output_lev2_suffix

max_result_order = config.ARGUMENT.max_result_order
start_result_order = config.ARGUMENT.start_result_order
rect_method = config.ARGUMENT.rectification_method

invoke_subrecipe("./examples/neid_recipe/test_order_trace.recipe")
invoke_subrecipe("./examples/neid_recipe/test_optimal_extraction.recipe")
invoke_subrecipe("./examples/neid_recipe/test_neid_radial_velocity.recipe")
"""

# Load .env file for test path 
load_dotenv()
try:
    hostname = os.environ['HOST_HOSTNAME']
except KeyError:
    hostname = socket.gethostname()

# load or set up file to track execution times
execution_limits_file = os.path.join(os.environ['KPFPIPE_TEST_DATA'], 'execution_times.csv')
if os.path.isfile(execution_limits_file):
    exe_limits = pd.read_csv(execution_limits_file)
else:
    exe_limits = pd.DataFrame([], columns=['hostname', 'function_name', 'min_time',
                                           'max_time', 'meas_time'])
    exe_limits.to_csv(execution_limits_file, header=True, index=False)

def get_execution_limits(func):
    """Get execution time limits for a given function.

    Args:
        func (function): function to look for in the execution limits file
    
    Returns:
        tuple or None: (min_execution_time, max_execution_time)
    """
    row = exe_limits.query('hostname == "{}" & function_name == "{}"' \
                           .format(hostname, func.__name__))
    if row['hostname'].count() > 0:
        limits = (row['min_time'].values[0], row['max_time'].values[0])
    else:
        limits = (None, None)
    
    return limits


def set_execution_limits(func, execution_time, low_frac=0.8, high_frac=1.2):
    """Set execution time limits for a given function and update the execution limits file.

    Args:
        func (function): function to set liits for
        execution time (float): measured execution time in seconds
        low_frac (float): execution time should not be shorter than low_frac*execution_time
        high_frac (float): execution time should not be longer than high_frac*execution_time
    
    """
    short_limit = low_frac * execution_time
    long_limit = high_frac * execution_time

    with open(execution_limits_file, 'a') as f:
        print("{:s},{:s},{:.2f},{:.2f},{:.2f}".format(hostname, func.__name__, short_limit,
                                                      long_limit, execution_time), file=f)

def execution_time_limit(time_limit=None):
    """
       Decorator function to assert that the execution time of a wrapped function 
       takes less than the time limits specified in the execution_times_file.

       Args:
           time_limit (int): manual override for max execution time
    """
    def timer(func):
        def check(*args, **kwargs):
            t1 = time.time()
            func(*args, **kwargs)
            t2 = time.time()
            tt = t2 - t1

            short_limit, long_limit = get_execution_limits(func)
            if time_limit is not None:
                long_limit = time_limit

            print("Function {:s} completed in {:.2f} seconds.".format(func.__name__, tt))

            if long_limit is not None:
                assert tt <= long_limit, "Execution time for function {:s} took longer than the allocated limit ({:.2f} s > {:.2f} s)" \
                                        .format(func.__name__, tt, short_limit)
            
            if short_limit is not None:
                assert tt >= short_limit, "Execution time for function {:s} took much less time than expected ({:.2f} s < {:.2f} s)" \
                                        .format(func.__name__, tt, short_limit)
            
            if short_limit is None and long_limit is None:
                set_execution_limits(func, tt)

        return check
    return timer

# Test time to create level 0 files from NEID data. Should be pretty quick.
# =============================================================================
@execution_time_limit(time_limit=None)
def test_neid_recipe():
    run_recipe(neid_recipe, neid_config)


if __name__ == '__main__':
    test_neid_recipe()
    
