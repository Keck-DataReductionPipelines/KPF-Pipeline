# test_radial_velocity_recipe.py
from kpfpipe.tools.recipe_test_unit import recipe_test

radial_velocity_neid_recipe = """# test recipe for readial velocity on NEID data
from modules.radial_velocity.src.radial_velocity_init import RadialVelocityInit
from modules.radial_velocity.src.radial_velocity import RadialVelocity

lev1_data_dir = KPFPIPE_TEST_DATA + '/radial_velocity_test/for_pytest/'
output_dir = config.ARGUMENT.output_dir
lev1_stem_suffix = config.ARGUMENT.output_lev1_suffix
lev2_stem_suffix = config.ARGUMENT.output_lev2_suffix
rect_method = config.ARGUMENT.rectification_method
order_name = config.ARGUMENT.orderlet_names

rv_init = RadialVelocityInit()
input_lev1_pattern = lev1_data_dir + '*' + '_' + str(rect_method) + lev1_stem_suffix + '.fits'
for input_L1_file in find_files(input_lev1_pattern):
    _, short_lev1 = split(input_L1_file)
    lev1_stem, lev1_ext = splitext(short_lev1)
    output_lev2_file = output_dir + lev1_stem + '_recipe' + lev2_stem_suffix + lev1_ext
    lev1_data = kpf1_from_fits(input_L1_file, data_type='KPF')
    rv_data = RadialVelocity(lev1_data, rv_init, None, 'SCIFLUX', ccf_ext='CCF', rv_ext='RV',
                            start_seg=20, end_seg=24, ins='neid')
"""

radial_velocity_neid_config = "examples/default_neid.cfg"


def test_recipe_radial_velocity_neid():
    recipe_test(radial_velocity_neid_recipe, radial_velocity_neid_config)

