"""
Test of interpolation between two wavelength solutions.
"""
from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.tools.recipe_test_unit import recipe_test

recipe = open('recipes/test_wls_interpolation.recipe', 'r').read()
config = 'configs/test_wls_interpolation.cfg'

def test_wls_interpolation_recipe():
    recipe_test(recipe, config, date_dir='20240101')

if __name__ == '__main__':
    test_wls_interpolation_recipe()
