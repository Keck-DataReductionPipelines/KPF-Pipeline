# test_recipe.py
"""
Tests of the recipe mechanisms
"""

from kpfpipe.tools.recipe_test_unit import recipe_test
from kpfpipe.pipelines.kpf_parse_ast import RecipeError
import os

masters_test_date = '20230730'
master_stacks_recipe = open('recipes/kpf_masters_drp.recipe', 'r').read()
master_stacks_config = 'configs/kpf_masters_drp.cfg'

master_l1l2_recipe = open('recipes/kpf_drp.recipe', 'r').read()
master_l1l2_config = 'configs/kpf_masters_l1.cfg'

master_wls_recipe = open('recipes/wls_auto.recipe', 'r').read()
master_wls_config = 'configs/wls_auto.cfg'


def test_master_stacks():
    recipe_test(master_stacks_recipe, master_stacks_config, date_dir=masters_test_date)
    os.system(f'mkdir /data/masters/{masters_test_date}; mv -v /testdata/kpf_{masters_test_date}_master_*.fits /data/masters/{masters_test_date}/')
    os.system(f'rm -vf /data/2D/{masters_test_date}/*.fits')

def test_master_l1l2():
    recipe_test(master_l1l2_recipe, master_l1l2_config,
                date_dir=masters_test_date, watch=False)

def test_master_wls():
    recipe_test(master_wls_recipe, master_wls_config,
            date_dir=masters_test_date, watch=False)

def main():
    # test_master_stacks()
    # test_master_l1l2()
    test_master_wls()

if __name__ == '__main__':
    main()