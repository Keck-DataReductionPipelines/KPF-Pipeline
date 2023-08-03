"""
Tests a subset of a single night's data with a couple SoCal frames also.
This should be the same night as the masters recipe test so that changes to masters
are tested.
"""
import tempfile
import pytest

from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.tools.recipe_test_unit import recipe_test
from kpfpipe.pipelines.kpf_parse_ast import RecipeError
from test_masters_recipe import masters_test_date


drp_recipe = open('recipes/kpf_drp.recipe', 'r').read()
drp_config = ConfigClass('configs/kpf_drp.cfg')

drp_config.set('WATCHFOR_L0', 'masterbias_path', f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_bias_autocal-bias.fits')
drp_config.set('WATCHFOR_L0', 'masterdark_path', f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_dark_autocal-dark.fits')
drp_config.set('WATCHFOR_L0', 'masterflat_path', f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_flat.fits')
drp_config.set('ARGUMENT', 'wls_fits', str([f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_WLS_autocal-lfc-all-eve_L1.fits',
                                        f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_WLS_autocal-lfc-all-eve_L1.fits']))
drp_config.set('ARGUMENT', 'do_db_query_for_one_nearest_wls_master_file', 'False')
drp_config.set('WATCHFOR_L0', 'do_db_query_for_master_files', 'False')

f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
drp_config.write(f)
drp_config_path = f.name
f.close()

@pytest.mark.run(after='test_masters_recipe')
def test_kpf_night():
    recipe_test(drp_recipe, drp_config_path, date_dir=masters_test_date)

def main():
    test_kpf_night()

if __name__ == '__main__':
    main()