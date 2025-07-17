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

The test runs using the same date as the masters recipe test to ensure
master calibration files are available for testing.
"""
import tempfile

from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.tools.recipe_test_unit import recipe_test
from kpfpipe.pipelines.kpf_parse_ast import RecipeError

# Use the same test date as masters recipe for consistency
masters_test_date = '20230730'

# Read the actual recipe file
twod_to_l1_recipe = open('recipes/kpf_2d_to_l1.recipe', 'r').read()
twod_to_l1_config = ConfigClass('configs/kpf_single_recipes.cfg')

# Configure the recipe to use the test date's master files
twod_to_l1_config.set('WATCHFOR_L0', 'masterbias_path', f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_bias_autocal-bias.fits')
twod_to_l1_config.set('WATCHFOR_L0', 'masterdark_path', f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_dark_autocal-dark.fits')
twod_to_l1_config.set('WATCHFOR_L0', 'masterflat_path', f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_flat.fits')
twod_to_l1_config.set('ARGUMENT', 'wls_fits', str([f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_WLS_autocal-lfc-all-eve_L1.fits',
                                                   f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_WLS_autocal-lfc-all-eve_L1.fits']))
twod_to_l1_config.set('ARGUMENT', 'do_db_query_for_one_nearest_wls_master_file', 'False')
twod_to_l1_config.set('WATCHFOR_L0', 'do_db_query_for_master_files', 'False')

# Create temporary config file
f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
twod_to_l1_config.write(f)
twod_to_l1_config_path = f.name
f.close()


def test_kpf_2d_to_l1_recipe():
    """
    Test the 2D to L1 processing recipe.
    This processes 2D files from the test date directory to L1 format.
    """
    recipe_test(twod_to_l1_recipe, twod_to_l1_config_path, date_dir=masters_test_date)


def main():
    test_kpf_2d_to_l1_recipe()


if __name__ == '__main__':
    main()
