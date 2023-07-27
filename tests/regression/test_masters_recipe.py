# test_recipe.py
"""
Tests of the recipe mechanisms
"""

from kpfpipe.tools.recipe_test_unit import recipe_test
from kpfpipe.pipelines.kpf_parse_ast import RecipeError
import tempfile
import pytest

master_stacks_recipe = open('recipes/kpf_masters_drp.recipe', 'r').readlines()
master_stacks_config = 'configs/kpf_masters_drp.cfg'

def test_master_stacks():
    recipe_test(master_stacks_recipe, master_stacks_config)

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
