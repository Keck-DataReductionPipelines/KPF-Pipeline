"""
Test of interpolation between two wavelength solutions.
"""
from kpfpipe.tools.recipe_test_unit import recipe_test

this_recipe = """from modules.Utils.string_proc import level_from_kpffile
from modules.Utils.string_proc import date_from_path
from modules.quality_control.src.quality_control_framework import QualityControlFramework
from modules.quicklook.src.diagnostics_framework import DiagnosticsFramework

data_type = 'KPF'
input_dir = '/testdata/L0/'
input_fits_filenames = [
'KP.20240424.32556.27', # star (add solar)
'KP.20240424.64534.64', # autocal-dark
'KP.20240424.61670.98', # autocal-bias
'KP.20240424.85791.84', # autocal-flat-all
'KP.20240424.80626.13', # autocal-etalon-morning-all
'KP.20240424.67285.36', # autocal-lfc-morning-all
'KP.20240424.66460.68', # autocal-une-morning-all
'KP.20240424.64028.50', # autocal-thar-morning-all
]

for input_file_file in input_fits_filenames:
    datecode = date_from_path(input_file_file)
    input_fits_filename = input_dir + datecode + '/' + input_file_file
    data_level_str = level_from_kpffile(input_fits_filename)
    if 'L0' in data_level_str:
        kpf_object = kpf0_from_fits(input_dir + input_fits_filename, data_type = data_type)
    elif '2D' in data_level_str:
        kpf_object = kpf0_from_fits(input_dir + input_fits_filename, data_type = data_type)
    elif 'L1' in data_level_str:
        kpf_object = kpf1_from_fits(input_dir + input_fits_filename, data_type = data_type)
    elif 'L2' in data_level_str:
        kpf_object = kpf2_from_fits(input_dir + input_fits_filename, data_type = data_type)

    qc_return_list = QualityControlFramework(data_type, data_level_str, kpf_object, 0)

"""

this_config = "examples/default_neid.cfg" #dummy cfg

def test_wls_interpolation():
    recipe_test(this_recipe, this_config)

if __name__ == '__main__':
    test_wls_interpolation()
