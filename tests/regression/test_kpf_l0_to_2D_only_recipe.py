# test_kpf_l0_to_2D_only_recipe.py
"""
Regression test for the kpf_l0_to_2D_only.recipe on a limited data set.

This test validates the L0 to 2D processing pipeline, which includes:
- L0 to 2D conversion (overscan subtraction, bias correction, etc.)
- CA-HK extraction (if enabled)
- Quality control and diagnostics

The test runs in watch mode, monitoring L0 files and processing them to 2D format only.
This recipe is designed to stop at the 2D level without proceeding to L1 or L2.
"""

from kpfpipe.tools.recipe_test_unit import recipe_test
from kpfpipe.pipelines.kpf_parse_ast import RecipeError
import os

# Test configuration
test_date = '20250519'
l0_test_file = f'/data/L0/{test_date}/KP.{test_date}.55029.51.fits'

# Read the actual recipe file
l0_to_2d_recipe = open('recipes/kpf_l0_to_2D_only.recipe', 'r').read()
l0_to_2d_config = 'configs/kpf_single_recipes.cfg'


def test_kpf_l0_to_2D_only_recipe_watch_mode():
    """
    Test the L0 to 2D only processing recipe in watch mode.
    This simulates monitoring an L0 file and processing it to 2D format only.
    """
    recipe_test(l0_to_2d_recipe, l0_to_2d_config, 
                date_dir=test_date, 
                file_path=l0_test_file,
                watch=True)


def test_kpf_l0_to_2D_only_recipe_non_watch_mode():
    """
    Test the L0 to 2D only processing recipe in non-watch mode.
    This processes L0 files from a specified date directory to 2D format only.
    """
    recipe_test(l0_to_2d_recipe, l0_to_2d_config, 
                date_dir=test_date, 
                file_path=f'/data/L0/{test_date}/',
                watch=False)


if __name__ == '__main__':
    print("Testing L0 to 2D only recipe in watch mode...")
    test_kpf_l0_to_2D_only_recipe_watch_mode()
    
    print("Testing L0 to 2D only recipe in non-watch mode...")
    test_kpf_l0_to_2D_only_recipe_non_watch_mode()
    
    print("All L0 to 2D only recipe tests completed successfully!")
