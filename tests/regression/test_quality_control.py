"""
Test of interpolation between two wavelength solutions.
"""
from kpfpipe.tools.recipe_test_unit import recipe_test

this_recipe = """from modules.quality_control.src.quality_control_framework import QualityControlFramework
from modules.quicklook.src.diagnostics_framework import DiagnosticsFramework

#wls1_file = '/data/reference_fits/kpf_20240101_master_WLS_autocal-lfc-all-eve_L1.fits'
#wls2_file = '/data/reference_fits/kpf_20240101_master_WLS_autocal-lfc-all-morn_L1.fits'
#wls1_l1 = kpf1_from_fits(wls1_file, data_type=data_type)
#wls2_l1 = kpf1_from_fits(wls2_file, data_type=data_type)
#obsid_l1 = 'KP.20240101.24368.88'
#l1_file = '/data/reference_fits/' + obsid_l1 + '_L1.fits'
#l1 = kpf1_from_fits(l1_file, data_type=data_type)
#l1_out = WaveInterpolate(wls1_l1, wls2_l1, l1)

data_type = 'KPF'
input_dir = '/data/reference_fits/'

#determine data_level_str
data_level_str = config.ARGUMENT.data_level_str
input_fits_filename = config.ARGUMENT.input_fits_filename
output_fits_filename = config.ARGUMENT.output_fits_filename


fits_arr = []
for fits_arr in fits_arr:
	#wave_fits = wave_fits + [output_dir + wls]



    if exists(input_fits_filename):
        if 'L0' in data_level_str:
            kpf_object = kpf0_from_fits(input_fits_filename, data_type = data_type)
        elif '2D' in data_level_str:
            kpf_object = kpf0_from_fits(input_fits_filename, data_type = data_type)
        elif 'L1' in data_level_str:
            kpf_object = kpf1_from_fits(input_fits_filename, data_type = data_type)
        elif 'L2' in data_level_str:
            kpf_object = kpf2_from_fits(input_fits_filename, data_type = data_type)
    
        qc_return_list = QualityControlFramework(data_type, data_level_str, kpf_object, 0)
        exit_code = qc_return_list[0]
        if exit_code == 1:
            kpf_object = qc_return_list[1]



"""

this_config = "examples/default_neid.cfg" #dummy cfg

def test_wls_interpolation():
    recipe_test(this_recipe, this_config)

if __name__ == '__main__':
    test_wls_interpolation()
