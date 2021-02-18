# test_radial_velocity_recipe.py
from kpfpipe.tools.recipe_test_unit import recipe_test

barycentric_correction_neid_recipe = """# test recipe for barycentric correction for Tau Ceti from NEID
from modules.barycentric_correction.src.barycentric_correction import BarycentricCorrection
period = 20
input_path = config.ARGUMENT.input_dir
output_path = config.ARGUMENT.output_dir + 'bc_test_20.csv'

start_time = config.ARGUMENT.start_time
obs_list = config.ARGUMENT.obs_list
rectification = config.ARGUMENT.rectification_method
L1_prefix = config.ARGUMENT.lev1_prefix

L1_file_prefix = input_path + L1_prefix
L1 = config.ARGUMENT.lev1_suffix

L1_dataset = []
L1_files = []

for code in obs_list:
    L1_file = L1_file_prefix + code + '_' + rectification + L1 + '.fits'
    if find_files(L1_file):
        lev1_data = kpf1_from_fits(L1_file)
        L1_dataset = L1_dataset + [lev1_data]
        L1_files = L1_files + [L1_file]

result = BarycentricCorrection(start_time=start_time, period = period, bc_corr_path = input_path, bc_corr_output=output_path, dataset=L1_dataset)

i = 0
for L1_data in L1_dataset:
    L1_file = L1_files[i]
    if find_files(L1_file):
        result = to_fits(L1_data, L1_file)
        i = i+1
"""

barycentric_correction_neid_config = "examples/default_bc.cfg"


def test_barycentric_correction_neid():
    recipe_test(barycentric_correction_neid_recipe, barycentric_correction_neid_config)

