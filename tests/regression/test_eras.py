from kpfpipe.tools.recipe_test_unit import recipe_test

recipe = """from modules.Utils.era_specific_parameters import EraSpecific

test_param = 'order_trace_files'
fname = "/testdata/kpf/L0/20230730/KP.20230730.29130.27.fits"

l0 = kpf0_from_fits(fname)

value = EraSpecific(l0, test_param)

"""

cfg = master_stacks_config = "examples/default_recipe_test_neid.cfg"

def test_kpf_eras():
    recipe_test(recipe)

if __name__ == '__main__':
    test_kpf_eras()