# test_recipe.py
"""
Tests of the recipe mechanisms
"""

from kpfpipe.tools.recipe_test_unit import run_recipe
from kpfpipe.pipelines.kpf_parse_ast import RecipeError
import tempfile
import pytest

basics_recipe = """# test recipe basics
snr_thresh = config.ARGUMENT.snr_threshold
snr_thresh_subscript = config.ARGUMENT['snr_threshold']
input_filename = config.ARGUMENT.input_filename

# test within loop to cover reset_visited_state logic in lots of node types
for a in [1, 2, 3]:
    sum = 1 + 4
    dif = sum - 2.
    prod = 2 * 3
    div = prod / 3.

    test_primitive_validate_args(sum, 5, dif, 3, prod, 6, div, 2., snr_thresh, 3.5, snr_thresh_subscript, 3.5)
    test_primitive_validate_args(sum > dif, dif < sum, sum >= dif, dif <= sum, sum != dif, not sum == dif)

    uadd = +3
    usub = -3
    test_primitive_validate_args(uadd, 3, uadd + usub, 0)

    if sum > snr_thresh:
        bool1 = True
    else:
        bool1 = False

    if sum < snr_thresh:
        bool2 = True
    else:
        bool2 = False
    
    bool3 = (div == 42)

    s = 'panama'
    bool4 = 'nam' in s
    bool5 = 'man' in s
    test_primitive_validate_args(bool1, True, bool2, False, bool3, False, bool4, True, bool5, False)
"""

builtins_recipe = """# test recipe built-ins
# test within loop to cover reset_visited_state logic in lots of node types
for i in [1, 2]:
    a = int(1.1)
    l = find_files('kpfpipe/pipelines/*.py')
    n = len(l)
    for file in l:
        t, e = splitext(file)

    bool1 = not l
    test_primitive_validate_args(a, 1, n, 4, e, '.py', bool1, False)
"""

environment_recipe = """# test recipe environment
test_data = KPFPIPE_TEST_DATA
test_outputs = KPFPIPE_TEST_OUTPUTS
test_data_bool = test_data != None
test_outputs_bool = test_outputs != None
test_primitive_validate_args(test_data_bool, True, test_outputs_bool, True)
"""

undefined_variable_recipe = """# test recipe with undefined variable
b = a + 1
"""

bad_assignment_recipe = """# test recipe with bad assignment statement
a, b = 19
"""

level0_from_to_recipe = """# test level0 fits reader recipe
fname = "../ownCloud/KPF-Pipeline-TestData/NEIDdata/TAUCETI_20191217/L0/neidTemp_2D20191217T023129.fits"
kpf0 = kpf0_from_fits(fname, data_type="NEID")
result = to_fits(kpf0, "temp_level0.fits")
"""

level1_from_to_recipe = """# test level1 fits reader recipe
fname = "../ownCloud/KPF-Pipeline-TestData/NEIDdata/TAUCETI_20191217/L1/neidL1_20191217T023129.fits"
kpf1 = kpf1_from_fits(fname, data_type="NEID")
result = to_fits(kpf1, "temp_level1.fits")
"""

subrecipe_sub_recipe = """# subrecipe to invoke from main
b = 42
test_primitive_validate_args(a, b)
"""

subrecipe_main_recipe = """# test subrecipe handling
a = 42
invoke_subrecipe("{}")
"""

# experimental_recipe = """s = 'panama'
# t = 'nam' in s
# f = 'man' in s
# test_primitive_validate_args(t, True, f, False)
# """

def test_recipe_basics():
    try:
        run_recipe(basics_recipe)
    except Exception as e:
        assert False, f"test_recipe_basics: unexpected exception {e}"

def test_recipe_builtins():
    try:
        run_recipe(builtins_recipe)
    except Exception as e:
        assert False, f"test_recipe_builtins: unexpected exception {e}"

def test_recipe_environment():
    try:
        run_recipe(environment_recipe)
    except Exception as e:
        assert False, f"test_recipe_environment: unexpected exception {e}"

def test_recipe_undefined_variable():
    with pytest.raises(RecipeError) as excinfo:
        run_recipe(undefined_variable_recipe)
    assert "Name a on line 2 of recipe not defined" in str(excinfo.value)

def test_recipe_bad_assignment():
    with pytest.raises(RecipeError) as excinfo:
        run_recipe(bad_assignment_recipe)
    assert "Error during assignment" in str(excinfo.value)

def test_recipe_subrecipe():
    with tempfile.NamedTemporaryFile(mode='w+') as f:
        f.write(subrecipe_sub_recipe)
        f.seek(0)
        run_recipe(subrecipe_main_recipe.format(f.name))


# def test_recipe_experimental():
#     try:
#         run_recipe(experimental_recipe)
#     except Exception as e:
#         assert False, f"test_recipe_experimental: unexpected exception {e}"

def main():
    test_recipe_basics()
    test_recipe_builtins()
    test_recipe_environment()
    test_recipe_undefined_variable()
    test_recipe_bad_assignment()
    test_recipe_subrecipe()
