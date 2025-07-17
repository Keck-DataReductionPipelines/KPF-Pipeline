# test_kpf_l1_to_l2_recipe.py
"""
Regression test for the kpf_l1_to_l2.recipe on a limited data set.

This test validates the L1 to L2 processing pipeline, which includes:
- Radial velocity calculation
- Barycentric correction (if enabled)
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
l1_to_l2_recipe = open('recipes/kpf_l1_to_l2.recipe', 'r').read()
l1_to_l2_config = ConfigClass('configs/kpf_single_recipes.cfg')

# Configure the recipe to use the test date's master files
l1_to_l2_config.set('WATCHFOR_L0', 'masterbias_path', f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_bias_autocal-bias.fits')
l1_to_l2_config.set('WATCHFOR_L0', 'masterdark_path', f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_dark_autocal-dark.fits')
l1_to_l2_config.set('WATCHFOR_L0', 'masterflat_path', f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_flat.fits')
l1_to_l2_config.set('ARGUMENT', 'wls_fits', str([f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_WLS_autocal-lfc-all-eve_L1.fits',
                                                 f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_WLS_autocal-lfc-all-eve_L1.fits']))
l1_to_l2_config.set('ARGUMENT', 'do_db_query_for_one_nearest_wls_master_file', 'False')
l1_to_l2_config.set('WATCHFOR_L0', 'do_db_query_for_master_files', 'False')

# Create temporary config file
f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
l1_to_l2_config.write(f)
l1_to_l2_config_path = f.name
f.close()


def test_kpf_l1_to_l2_recipe():
    """
    Test the L1 to L2 processing recipe.
    This processes L1 files from the test date directory to L2 format.
    """
    recipe_test(l1_to_l2_recipe, l1_to_l2_config_path, date_dir=masters_test_date)


def main():
    test_kpf_l1_to_l2_recipe()


if __name__ == '__main__':
    main()
