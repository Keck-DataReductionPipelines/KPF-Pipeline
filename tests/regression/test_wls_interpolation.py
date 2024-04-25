"""
Test of interpolation between two wavelength solutions.
"""
from kpfpipe.tools.recipe_test_unit import recipe_test

this_recipe = """from modules.wavelength_cal.src.wavelength_cal import WaveInterpolate

data_type = 'KPF'
wls1_file = '/data/reference_fits/kpf_20240101_master_WLS_autocal-lfc-all-eve_L1.fits'
wls2_file = '/data/reference_fits/kpf_20240101_master_WLS_autocal-lfc-all-morn_L1.fits'
wls1_l1 = kpf1_from_fits(wls1_file, data_type=data_type)
wls2_l1 = kpf1_from_fits(wls2_file, data_type=data_type)
obsid_l1 = 'KP.20240101.24368.88'
l1_file = '/data/reference_fits/' + obsid_l1 + '_L1.fits'
l1 = kpf1_from_fits(l1_file, data_type=data_type)
l1_out = WaveInterpolate(wls1_l1, wls2_l1, l1)
"""

this_config = "examples/default_neid.cfg" #dummy cfg

def test_wls_interpolation():
    recipe_test(this_recipe, this_config)

if __name__ == '__main__':
    test_wls_interpolation()
