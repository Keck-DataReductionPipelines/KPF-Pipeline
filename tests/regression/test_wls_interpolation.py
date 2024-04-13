"""
Test of interpolation between two wavelength solutions.
"""
from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.tools.recipe_test_unit import recipe_test

this_recipe = open('recipes/test_wls_interpolation.recipe', 'r').read()
this_config = ConfigClass('configs/test_wls_interpolation.cfg')

def test_wls_interp():
    recipe_test(this_recipe, this_config)

def main():
    test_kpf_night()

if __name__ == '__main__':
    main()
