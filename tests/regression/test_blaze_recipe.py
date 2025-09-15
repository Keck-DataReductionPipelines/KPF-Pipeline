from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.tools.recipe_test_unit import recipe_test

recipe = open('tests/recipes/test_blaze.recipe', 'r').read()
config = 'tests/configs/test_blaze.cfg'
date_dir = '20241022'

def test_blaze_recipe():
    recipe_test(recipe, config, date_dir=date_dir)

if __name__ == '__main__':
    test_blaze_recipe()