# test_radial_velocity_recipe.py
from kpfpipe.tools.recipe_test_unit import recipe_test

radial_velocity_neid_reweighting_recipe = """# test recipe for readial velocity ccf orders reweighting on NEID data
from modules.radial_velocity.src.radial_velocity_init import RadialVelocityInit
from modules.radial_velocity.src.radial_velocity import RadialVelocity
from modules.radial_velocity.src.radial_velocity_reweighting_ref import RadialVelocityReweightingRef
from modules.radial_velocity.src.radial_velocity_reweighting import RadialVelocityReweighting

test_data_dir = KPFPIPE_TEST_DATA + '/NEIDdata' 
reweighting_method = config.ARGUMENT.reweighting_method
total_order = config.ARGUMENT.total_order
input_lev2_prefix = config.ARGUMENT.input_lev2_file_prefix

rv_init = RadialVelocityInit()
lev2_files_pattern = test_data_dir + input_lev2_prefix + '*.fits'
all_lev2_files = find_files(lev2_files_pattern)
lev2_list = []
for f in all_lev2_files:
    lev2_obj = kpf2_from_fits(f, data_type='KPF')
    lev2_list = lev2_list + [lev2_obj]
ratio_ref = RadialVelocityReweightingRef(lev2_list, reweighting_method, total_order, ccf_hdu_name='CCFS')

for f in all_lev2_files:
    lev2_obj = kpf2_from_fits(f, data_type='KPF')
    reweighted_lev2 = RadialVelocityReweighting(lev2_obj, reweighting_method, ratio_ref, total_order, rv_init,
                        ccf_ext='CCFS')
"""

radial_velocity_neid_reweighting_config = "examples/default_recipe_neid_hd127334.cfg"

def test_recipe_radial_velocity_neid():
    recipe_test(radial_velocity_neid_reweighting_recipe, radial_velocity_neid_reweighting_config)

