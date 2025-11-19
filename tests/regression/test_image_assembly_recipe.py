from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.tools.recipe_test_unit import recipe_test

recipe = open('tests/recipes/test_image_assembly.recipe', 'r').read()
config = 'tests/configs/test_image_assembly.cfg'
date_dir = '20240405'

def test_image_assembly_recipe():
    recipe_test(recipe, config, date_dir=date_dir)

if __name__ == '__main__':
    test_image_assembly_recipe()