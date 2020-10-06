bias_recipe = """# recipe experiment for bias subtraction
from modules.utils.frame_combine import FrameCombinePrimitive
from modules.bias_subtraction.src.bias_subtraction import BiasSubtraction 

bias_files=find_files(KPFPIPE_TEST_DATA+'/NEIDdata/BIAS/neidTemp*.fits')
master_bias_data=FrameCombinePrimitive(bias_files, 'NEID')
master_result = to_fits(master_bias_data, KPFPIPE_TEST_OUTPUTS + '/TestMasterBias.fits')

raw_files=find_files('KPFPIPE_TEST_DATA+'/NEIDdata/TAUCETI_20191217/L0/neidTemp_2D20191217T023129.fits')
master_bias=find_files(KPFPIPE_TEST_OUTPUTS + '/TestMasterBias.fits')
raw_min_bias=BiasSubtraction(raw_files, master_bias, 'NEID')
bias_result=to_fits(raw_min_bias, KPFPIPE_TEST_OUTPUTS + '/TestRawMinusBias.fits')
"""
#can insert different files in when necessary
def test_bias_recipe():
    try:
        run_recipe(bias_recipe)
    except Exception as e:
        assert False, f"test_bias_recipe: unexpected exception {e}"

