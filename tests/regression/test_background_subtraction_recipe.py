# test_background_subtraction_recipe.py
from kpfpipe.tools.recipe_test_unit import recipe_test

background_sub_kpf_recipe = """# test recipe for background subtraction on L0 data
data_type = config.ARGUMENT.data_type
flat_file_pattern = config.ARGUMENT.flat_file

from modules.order_trace.src.order_mask import OrderMask
from modules.image_processing.src.image_process import ImageProcessing
from modules.Utils.string_proc import date_from_kpffile

date_dir_flat = date_from_kpffile(flat_file_pattern)
date_dir_flat = date_dir_flat + '/'
orderlet_names = config.ARGUMENT.orderlet_names
start_order= config.ARGUMENT.start_order
ccd_list = config.ARGUMENT.ccd_list
input_2d_dir = config.ARGUMENT.input_dir_root + '20221217/'
output_order_trace = config.ARGUMENT.output_dir + config.ARGUMENT.output_trace + date_dir_flat
lev0_science_pattern = input_2d_dir + 'KP.20221217.04646.08_2D.fits'
ccd_idx = 0
csv_ext = '.csv'
trace_list = [output_order_trace + flat_file_pattern + '_' + ccd_list[ccd_idx] + csv_ext]

for input_lev0_file in find_files(lev0_science_pattern):
    order_mask = None
    lev0_data = kpf0_from_fits(input_lev0_file, data_type=data_type)
    order_mask = OrderMask(lev0_data, order_mask, orderlet_names=orderlet_names[ccd_idx],
                               start_order=start_order[ccd_idx], trace_file=trace_list[ccd_idx],
                               data_extension=ccd_list[ccd_idx])
    lev0_data = ImageProcessing(lev0_data, order_mask, [ccd_list[ccd_idx]], data_type, False)
"""

background_sub_config = "configs/kpf_bs_test.cfg"


def test_recipe_background_sub_kpf():
    recipe_test(background_sub_kpf_recipe, background_sub_config)

