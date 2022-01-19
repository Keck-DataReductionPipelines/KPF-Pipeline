# test_optimal_extraction_recipe.py
from kpfpipe.tools.recipe_test_unit import recipe_test

optimal_extraction_neid_recipe = """# test recipe for optimal extraction on NEID data
from modules.spectral_extraction.src.spectral_extraction import SpectralExtraction

test_data_dir = KPFPIPE_TEST_DATA
flat_data_dir = test_data_dir + config.ARGUMENT.flat_data_dir
data_type = config.ARGUMENT.data_type
output_dir = config.ARGUMENT.output_dir
input_lev0_prefix = config.ARGUMENT.input_lev0_file_prefix
input_lev1_prefix = config.ARGUMENT.input_lev1_file_prefix
obs_list = config.ARGUMENT.obs_list
lev1_stem_suffix = config.ARGUMENT.output_lev1_suffix
max_result_order = config.ARGUMENT.max_result_order
start_result_order = config.ARGUMENT.start_result_order
rect_method = config.ARGUMENT.rectification_method
order_name = config.ARGUMENT.orderlet_names

lev0_flat_file = flat_data_dir + 'stacked_2fiber_flat_L0.fits'

lev0_flat_data = kpf0_from_fits(lev0_flat_file, data_type=data_type)
_, short_lev0_flat = split(lev0_flat_file)
lev0_flat_stem, lev0_flat_ext = splitext(short_lev0_flat)

for code in obs_list:
    input_lev0_file = test_data_dir + input_lev0_prefix + code + '.fits'
    if input_lev1_prefix != '':
        input_lev1_file = test_data_dir + input_lev1_prefix + code + '.fits'
    else:
        input_lev1_file = None

    _, short_lev0_file = split(input_lev0_file)
    lev0_stem, lev0_ext = splitext(short_lev0_file)
    lev0_data = kpf0_from_fits(input_lev0_file, data_type=data_type)
    output_lev1_file = output_dir + lev0_flat_stem + '_' + lev0_stem + '_' + str(rect_method) + '_recipe' + lev1_stem_suffix + '.fits'
    op_data = SpectralExtraction(lev0_data, lev0_flat_data, None, orderlet_names=order_name, 
                                                rectification_method=rect_method, max_result_order=max_result_order,
                                                start_order=start_result_order, wavecal_fits=input_lev1_file,
                                                data_extension='DATA', trace_extension='ORDER_TRACE_RESULT')
"""

rectification_neid_recipe = """# test recipe for rectification on NEID data
from modules.spectral_extraction.src.order_rectification import OrderRectification

test_data_dir = KPFPIPE_TEST_DATA
flat_data_dir = test_data_dir + config.ARGUMENT.flat_data_dir
data_type = config.ARGUMENT.data_type
output_dir = config.ARGUMENT.output_dir
input_lev0_prefix = config.ARGUMENT.input_lev0_file_prefix
obs_list = config.ARGUMENT.obs_list
max_result_order = 2
start_result_order = 20
rect_method = config.ARGUMENT.rectification_method
order_name = config.ARGUMENT.orderlet_names

lev0_flat_file = flat_data_dir + 'stacked_2fiber_flat_L0.fits'

lev0_flat_data = kpf0_from_fits(lev0_flat_file, data_type='NEID')
_, short_lev0_flat = split(lev0_flat_file)
lev0_flat_stem, lev0_flat_ext = splitext(short_lev0_flat)

lev0_flat_rect = OrderRectification(None, lev0_flat_data, orderlet_names=order_name,
                                    rectification_method=rect_method, max_result_order=max_result_order,
                                    data_extension='DATA',
                                    start_order=start_result_order, trace_extension='ORDER_TRACE_RESULT')
for code in obs_list:
    input_lev0_file = test_data_dir + input_lev0_prefix + code + '.fits'
    _, short_lev0_file = split(input_lev0_file)
    lev0_stem, lev0_ext = splitext(short_lev0_file)
    lev0_data = kpf0_from_fits(input_lev0_file, data_type=data_type)
    output_lev0_file = output_dir + 'spectrum_rect.fits'
    lev0_data_rect = OrderRectification(lev0_data, lev0_flat_rect, None, orderlet_names=order_name,
                            rectification_method=rect_method, max_result_order=max_result_order,
                            start_order=start_result_order, 
                            data_extension='DATA', trace_extension='ORDER_TRACE_RESULT')
"""
optimal_extraction_neid_config = "examples/default_recipe_test_neid.cfg"


def test_recipe_optimal_extraction_neid():
    recipe_test(optimal_extraction_neid_recipe, optimal_extraction_neid_config)


def test_recipe_rectification_neid():
    recipe_test(rectification_neid_recipe, optimal_extraction_neid_config)
