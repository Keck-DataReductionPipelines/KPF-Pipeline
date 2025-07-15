# test_kpf_2d_to_l1_recipe.py
"""
Regression test for the kpf_2d_to_l1.recipe on a limited data set.

This test validates the 2D to L1 processing pipeline, which includes:
- Spectral extraction from 2D images
- Wavelength calibration and interpolation
- Barycentric correction (for historical reasons, done at L1 level)
- CA-HK extraction (if enabled)
- Blaze function application
- Drift correction
- Quality control and diagnostics

The test runs in watch mode, monitoring 2D files and processing them to L1 format.
This recipe stops at the L1 level and includes drift correction as the final step.
"""

from kpfpipe.tools.recipe_test_unit import recipe_test
from kpfpipe.pipelines.kpf_parse_ast import RecipeError
import os

# Test configuration
test_date = '20250519'
twod_test_file = f'/data/2D/{test_date}/KP.{test_date}.55029.51_2D.fits'

# Read the actual recipe file
twod_to_l1_recipe = open('recipes/kpf_2d_to_l1.recipe', 'r').read()
twod_to_l1_config = 'configs/kpf_single_recipes.cfg'


def test_kpf_2d_to_l1_recipe_watch_mode():
    """
    Test the 2D to L1 processing recipe in watch mode.
    This simulates monitoring a 2D file and processing it to L1 format.
    """
    recipe_test(twod_to_l1_recipe, twod_to_l1_config, 
                date_dir=test_date, 
                file_path=twod_test_file,
                watch=True)


def test_kpf_2d_to_l1_recipe_non_watch_mode():
    """
    Test the 2D to L1 processing recipe in non-watch mode.
    This processes 2D files from a specified date directory to L1 format.
    """
    recipe_test(twod_to_l1_recipe, twod_to_l1_config, 
                date_dir=test_date, 
                file_path=f'/data/2D/{test_date}/',
                watch=False)


def test_kpf_2d_to_l1_recipe_masters_mode():
    """
    Test the 2D to L1 processing recipe with masters files.
    This simulates processing master calibration files through the pipeline.
    """
    masters_test_file = f'/data/masters/{test_date}/kpf_{test_date}_master_flat.fits'
    recipe_test(twod_to_l1_recipe, twod_to_l1_config, 
                date_dir=test_date, 
                file_path=masters_test_file,
                watch=True)


if __name__ == '__main__':
    print("Testing 2D to L1 recipe in watch mode...")
    test_kpf_2d_to_l1_recipe_watch_mode()
    
    print("Testing 2D to L1 recipe in non-watch mode...")
    test_kpf_2d_to_l1_recipe_non_watch_mode()
    
    print("Testing 2D to L1 recipe with masters files...")
    test_kpf_2d_to_l1_recipe_masters_mode()
    
    print("All 2D to L1 recipe tests completed successfully!")
