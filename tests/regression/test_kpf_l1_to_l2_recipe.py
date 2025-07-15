# test_kpf_l1_to_l2_recipe.py
"""
Regression test for the kpf_l1_to_l2.recipe on a limited data set.

This test validates the L1 to L2 processing pipeline, which includes:
- Radial velocity calculation
- Barycentric correction (if enabled)
- Quality control and diagnostics

The test runs in watch mode, monitoring L1 files and processing them to L2.
"""

from kpfpipe.tools.recipe_test_unit import recipe_test
from kpfpipe.pipelines.kpf_parse_ast import RecipeError
import os

# Test configuration
test_date = '20250519'
l1_test_file = f'/data/L1/{test_date}/KP.{test_date}.55029.51_L1.fits'

# Read the actual recipe file
l1_to_l2_recipe = open('recipes/kpf_l1_to_l2.recipe', 'r').read()
l1_to_l2_config = 'configs/kpf_single_recipes.cfg'


def test_kpf_l1_to_l2_recipe_watch_mode():
    """
    Test the L1 to L2 processing recipe in watch mode.
    This simulates monitoring an L1 file and processing it to L2.
    """
    recipe_test(l1_to_l2_recipe, l1_to_l2_config, 
                date_dir=test_date, 
                file_path=l1_test_file,
                watch=True)


def test_kpf_l1_to_l2_recipe_non_watch_mode():
    """
    Test the L1 to L2 processing recipe in non-watch mode.
    This processes L1 files from a specified date directory.
    """
    recipe_test(l1_to_l2_recipe, l1_to_l2_config, 
                date_dir=test_date, 
                file_path=f'/data/L1/{test_date}/',
                watch=False)


if __name__ == '__main__':
    print("Testing L1 to L2 recipe in watch mode...")
    test_kpf_l1_to_l2_recipe_watch_mode()
    
    print("Testing L1 to L2 recipe in non-watch mode...")
    test_kpf_l1_to_l2_recipe_non_watch_mode()
    
    print("All L1 to L2 recipe tests completed successfully!")
