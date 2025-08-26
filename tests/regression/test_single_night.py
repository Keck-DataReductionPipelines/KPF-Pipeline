"""
Tests a subset of a single night's data with a couple SoCal frames also.
This should be the same night as the masters recipe test so that changes to masters
are tested.
"""
import tempfile
import os
from glob import glob
import concurrent.futures

from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.tools.recipe_test_unit import recipe_test
from kpfpipe.pipelines.kpf_parse_ast import RecipeError
# from .test_masters_recipe import masters_test_date

masters_test_date = '20240228'

drp_recipe = open('recipes/kpf_drp.recipe', 'r').read()
drp_config = ConfigClass('configs/kpf_drp.cfg')

drp_config.set('WATCHFOR_L0', 'masterbias_path', f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_bias_autocal-bias.fits')
drp_config.set('WATCHFOR_L0', 'masterdark_path', f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_dark_autocal-dark.fits')
drp_config.set('WATCHFOR_L0', 'masterflat_path', f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_flat.fits')
drp_config.set('ARGUMENT', 'wls_fits', str([f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_WLS_autocal-lfc-all-eve_L1.fits',
                                        f'/masters/{masters_test_date}/kpf_{masters_test_date}_master_WLS_autocal-lfc-all-eve_L1.fits']))
drp_config.set('ARGUMENT', 'do_db_query_for_one_nearest_wls_master_file', 'True')
drp_config.set('WATCHFOR_L0', 'do_db_query_for_master_files', 'True')

f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
drp_config.write(f)
drp_config_path = f.name
f.close()


# Emulate --reprocess <nightdir>
nightdir = f"/data/L0/{masters_test_date}/"  # Adjust path as needed
fits_files = sorted(glob(os.path.join(nightdir, "*.fits")), reverse=True)


# Must be at module level for multiprocessing

def run_one(args):
    file_path, drp_recipe, drp_config_path, masters_test_date = args
    recipe_test(drp_recipe, drp_config_path, date_dir=masters_test_date, file_path=file_path, watch=True)

def test_kpf_night(ncpus=10):
    # Prepare args for map
    args = [(file_path, drp_recipe, drp_config_path, masters_test_date) for file_path in fits_files]
    with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
        list(executor.map(run_one, args))


def main():
    test_kpf_night(ncpus=1)

if __name__ == '__main__':
    main()