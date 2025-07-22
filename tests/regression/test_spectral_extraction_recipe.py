from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.tools.recipe_test_unit import recipe_test

recipe = open('tests/recipes/test_spectral_extraction.recipe', 'r').read()
config = 'tests/configs/test_spectral_extraction_light.cfg'
date_dir = '20241022'

def test_spectral_extraction_recipe():
    recipe_test(recipe, config, date_dir=date_dir)

if __name__ == '__main__':
    test_spectral_extraction_recipe()