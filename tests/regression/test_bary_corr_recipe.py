# test_bary_corr_recipe.py
from kpfpipe.tools.recipe_test_unit import recipe_test

bary_corr_recipe = """# test recipe for creating BARY_CORR table 
from modules.Utils.string_proc import str_replace
from modules.spectral_extraction.src.bary_corr import BaryCorrTable

data_type = config.ARGUMENT.data_type
orders_per_ccd = config.ARGUMENT.orders_per_ccd
s_bary_idx = [0, orders_per_ccd[0]]
ccd_idx = config.ARGUMENT.ccd_idx
output_dir = config.ARGUMENT.output_dir
lev0_stem_suffix = config.ARGUMENT.output_lev0_suffix
lev1_stem_suffix = config.ARGUMENT.output_lev1_suffix
wave_to_ext = config.ARGUMENT.wave_to_ext
ccd_list = config.ARGUMENT.ccd_list
date_dir='20230114/'
input_lev0_file = config.ARGUMENT.input_dir_root + date_dir + 'KP.20230114.03263.62_2D.fits'
lev0_data = kpf0_from_fits(input_lev0_file, data_type=data_type)
_, short_lev0_file = split(input_lev0_file)
lev1_stem, lev1_ext = splitext(short_lev0_file)
if lev0_stem_suffix != None and lev0_stem_suffix in lev1_stem:
    lev1_stem = str_replace(lev1_stem, lev0_stem_suffix, '')
output_extraction = output_dir + config.ARGUMENT.output_extraction + date_dir
#output_lev1_file = output_extraction + lev1_stem + lev1_stem_suffix + '.fits'
output_lev1_file = output_extraction + 'KP.20230114.03263.62_L1.fits'

if exists(output_lev1_file):
    output_data = kpf1_from_fits(output_lev1_file, data_type=data_type)
    t_order = orders_per_ccd[0] + orders_per_ccd[1]
    for idx in ccd_idx:
        result = BaryCorrTable(lev0_data, output_data, t_order, orders_per_ccd[idx], start_bary_index=s_bary_idx[idx], wls_ext=wave_to_ext[idx][0])

"""

bary_corr_config = "configs/kpf_bc_test.cfg"

def test_bary_corr_kpf():
    recipe_test(bary_corr_recipe, bary_corr_config)

if __name__ == '__main__':
    test_bary_corr_kpf()