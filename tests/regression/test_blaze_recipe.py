from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.tools.recipe_test_unit import recipe_test

recipe = open('tests/recipes/test_blaze.recipe', 'r').read()
config = 'modules/blaze/configs/default.cfg'

def test_blaze_recipe():
    recipe_test(recipe, config)

if __name__ == '__main__':
    test_blaze_recipe()