# test_order_trace_recipe.py
from kpfpipe.tools.recipe_test_unit import recipe_test


order_trace_paras_recipe = """# test recipe for order trace on paras data
test_data_dir = KPFPIPE_TEST_DATA
data_type = config.ARGUMENT.data_type
output_dir = config.ARGUMENT.output_dir
paras_data_dir = config.ARGUMENT.paras_data_dir
input_flat_file = test_data_dir + paras_data_dir + 'paras.flatA.fits'

flat_stem_suffix = config.ARGUMENT.output_flat_suffix
row_range = config.ARGUMENT.row_range

from modules.order_trace.src.order_trace import OrderTrace

_, short_flat_file = split(input_flat_file)
flat_stem, flat_ext = splitext(short_flat_file)
output_lev0_file = output_dir + flat_stem + '_recipetest'+flat_stem_suffix + flat_ext
flat_data = kpf0_from_fits(input_flat_file, data_type=data_type)
ot_data = OrderTrace(flat_data, data_row_range=row_range)
"""

paras_config = "examples/default_recipe_test_paras.cfg"


order_trace_neid_recipe = """# test recipe for order trace on neid data
test_data_dir = KPFPIPE_TEST_DATA
data_type = config.ARGUMENT.data_type
output_dir = config.ARGUMENT.output_dir
neid_flat_file = config.ARGUMENT.neid_flat_data
input_flat_file = test_data_dir + neid_flat_file

flat_stem_suffix = config.ARGUMENT.output_flat_suffix
row_range = config.ARGUMENT.row_range

from modules.order_trace.src.order_trace import OrderTrace

_, short_flat_file = split(input_flat_file)
flat_stem, flat_ext = splitext(short_flat_file)
output_lev0_file = output_dir + flat_stem + '_recipetest'+flat_stem_suffix + flat_ext

flat_data = kpf0_from_fits(input_flat_file, data_type=data_type)
ot_data = OrderTrace(flat_data, data_row_range=row_range)
"""

neid_config = "examples/default_recipe_test_neid.cfg"

def test_recipe_order_trace_paras():
    recipe_test(order_trace_paras_recipe, paras_config)


def test_recipe_order_trace_neid():
    recipe_test(order_trace_neid_recipe, neid_config)