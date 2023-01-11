# test_radial_velocity_recipe.py
from kpfpipe.tools.recipe_test_unit import recipe_test

radial_velocity_neid_reweighting_recipe = """# test recipe for readial velocity ccf orders reweighting on NEID data
from modules.radial_velocity.src.radial_velocity_init import RadialVelocityInit
from modules.radial_velocity.src.radial_velocity_reweighting_ref import RadialVelocityReweightingRef
from modules.radial_velocity.src.radial_velocity_reweighting import RadialVelocityReweighting

test_data_dir = KPFPIPE_TEST_DATA + '/radial_velocity_test/for_pytest/' 
reweighting_method = config.ARGUMENT.reweighting_method

rv_init = RadialVelocityInit()
lev2_files_pattern = test_data_dir + '*_L1_L2.fits'
all_lev2_files = find_files(lev2_files_pattern)
lev2_list = []
start_seg=10
end_seg=89
total_segment=end_seg-start_seg+1
for f in all_lev2_files:
    lev2_obj = kpf2_from_fits(f, data_type='KPF')
    lev2_list = lev2_list + [lev2_obj]
ratio_ref = RadialVelocityReweightingRef(lev2_list, reweighting_method, total_segment, ccf_hdu_name='CCF', 
                                             ccf_start_index=start_seg)

for f in all_lev2_files:
    lev2_obj = kpf2_from_fits(f, data_type='KPF')
    reweighted_lev2 = RadialVelocityReweighting(lev2_obj, reweighting_method, ratio_ref, total_segment, rv_init,
                        ccf_ext='CCF', rv_ext='RV', rv_ext_idx=0, ccf_start_index=start_seg)
"""

radial_velocity_neid_reweighting_config = "examples/default_recipe_test_neid.cfg"

def test_recipe_radial_velocity_neid():
    recipe_test(radial_velocity_neid_reweighting_recipe, radial_velocity_neid_reweighting_config)

