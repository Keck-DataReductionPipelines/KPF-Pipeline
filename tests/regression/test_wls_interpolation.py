"""
Test of interpolation between two wavelength solutions.
"""
import os
from kpfpipe.tools.recipe_test_unit import recipe_test

recipe = """from modules.Utils.string_proc import date_from_kpffile
from modules.wavelength_cal.src.wavelength_cal import WaveInterpolate

data_type = 'KPF'
wls1_file = os.getenv('KPFPIPE_DATA') + '/masters/20240101/kpf_20240101_master_WLS_autocal-lfc-all-eve_L1.fits'
wls2_file = os.getenv('KPFPIPE_DATA') + '/masters/20240101/kpf_20240101_master_WLS_autocal-lfc-all-morn_L1.fits'
wls1_l1 = kpf1_from_fits(wls1_file, data_type=data_type)
wls2_l1 = kpf1_from_fits(wls2_file, data_type=data_type)
obsid_l1 = 'KP.20240101.24368.88'
datecode_l1 = date_from_kpffile(obsid_l1)
l1_file = os.getenv('KPFPIPE_DATA') + '/L1/' + datecode_l1 + '/' + obsid_l1 + '_L1.fits'
l1 = kpf1_from_fits(l1_file, data_type=data_type)
l1_out = WaveInterpolate(wls1_l1, wls2_l1, l1)
"""

cfg = this_config = "examples/default_recipe_test_neid.cfg" #dummy cfg

def test_wls_interpolation():
    recipe_test(recipe)

if __name__ == '__main__':
    test_wls_interpolation()
