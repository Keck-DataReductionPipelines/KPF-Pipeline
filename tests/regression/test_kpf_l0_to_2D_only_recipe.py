# test_kpf_l0_to_2D_only_recipe.py
"""
Regression test for the kpf_l0_to_2D_only.recipe on a limited data set.

This test validates the L0 to 2D processing pipeline, which includes:
- L0 to 2D conversion (overscan subtraction, bias correction, etc.)
- CA-HK extraction (if enabled)
- Quality control and diagnostics

The test runs using the same date as the masters recipe test to ensure
master calibration files are available for testing.
This recipe is designed to stop at the 2D level without proceeding to L1 or L2.
"""
import tempfile

from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.tools.recipe_test_unit import recipe_test
from kpfpipe.pipelines.kpf_parse_ast import RecipeError

# Use the same test date as masters recipe for consistency
masters_test_date = '20230730'

# Read the actual recipe file
l0_to_2d_recipe = open('recipes/kpf_l0_to_2D_only.recipe', 'r').read()
l0_to_2d_config = ConfigClass('configs/kpf.cfg')

# Configure the recipe to use the test date's master files
l0_to_2d_config.set('WATCHFOR_L0', 'masterbias_path', f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_bias_autocal-bias.fits')
l0_to_2d_config.set('WATCHFOR_L0', 'masterdark_path', f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_dark_autocal-dark.fits')
l0_to_2d_config.set('WATCHFOR_L0', 'masterflat_path', f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_flat.fits')
l0_to_2d_config.set('ARGUMENT', 'wls_fits', str([f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_WLS_autocal-lfc-all-eve_L1.fits',
                                                 f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_WLS_autocal-lfc-all-eve_L1.fits']))
l0_to_2d_config.set('ARGUMENT', 'do_db_query_for_one_nearest_wls_master_file', 'False')
l0_to_2d_config.set('WATCHFOR_L0', 'do_db_query_for_master_files', 'False')

# Create temporary config file
f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
l0_to_2d_config.write(f)
l0_to_2d_config_path = f.name
f.close()


def test_kpf_l0_to_2D_only_recipe():
    """
    Test the L0 to 2D only processing recipe.
    This processes L0 files from the test date directory to 2D format only.
    """
    recipe_test(l0_to_2d_recipe, l0_to_2d_config_path, date_dir=masters_test_date)


def main():
    test_kpf_l0_to_2D_only_recipe()


if __name__ == '__main__':
    main()
