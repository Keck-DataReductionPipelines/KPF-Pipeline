# test_recipe.py

import sys, traceback
import ast

from keckdrpframework.core.framework import Framework
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.action import Action
from keckdrpframework.models.processing_context import ProcessingContext
from kpfpipe.pipelines.kpf_parse_ast import KpfPipelineNodeVisitor, RecipeError
from kpfpipe.pipelines.kpfpipeline import KPFPipeline
from kpfpipe.logger import start_logger

basics_recipe = """# test recipe basics
from test_primitives import test_primitive_validate_args

sum = 1 + 4
dif = sum - 2.
prod = 2 * 3
div = prod / 3.

# snr_thresh = config.argument.snr_threshold
snr_thresh = config.ARGUMENT.snr_threshold
snr_thresh = float(config['ARGUMENT']['snr_threshold'])
snr_thresh = config.ARGUMENT['snr_threshold']
test_primitive_validate_args(sum, 5, dif, 3, prod, 6, div, 2., snr_thresh, 3.5)

input_filename = config.ARGUMENT['input_filename']

if sum > snr_thresh:
    bool1 = True
else:
    bool1 = False
test_primitive_validate_args(bool1, True)
"""

# undefined_variable_recipe = """# test recipe with undefined variable
# b = a + 1
# """

# bad_assignment_recipe = """# test recipe with bad assignment statement
# a, b = 19
# """

level0_from_to_recipe = """# test level0 fits reader recipe
fname = "../ownCloud/KPF-Pipeline-TestData/NEIDdata/TAUCETI_20191217/L0/neidTemp_2D20191217T023129.fits"
kpf0 = kpf0_from_fits(fname, data_type="NEID")
result = to_fits(kpf0, "temp_level0.fits")
"""

level1_from_to_recipe = """# test level1 fits reader recipe
fname = "../ownCloud/KPF-Pipeline-TestData/NEIDdata/TAUCETI_20191217/L1/neidL1_20191217T023129.fits"
kpf1 = kpf1_from_fits(fname, data_type="NEID")
result = to_fits(kpf1, "temp_level1.fits")
"""

level2_from_to_recipe = """# test level2 fits reader recipe
fname = "../ownCloud/KPF-Pipeline-TestData/NEIDdata/TAUCETI_20191217/L2/neidL2_20191217T023129.fits"
kpf2 = kpf2_from_fits(fname, data_type="NEID")
result = to_fits(kpf2, "temp_level2.fits")
"""

class TestKpfPipeline(KPFPipeline):
    """
    Test pipeline class extending KpfPipeline
    """

    def __init__(self, context: ProcessingContext):
        """ constructor """
        KPFPipeline.__init__(self, context)
        self.event_table['test_start_recipe'] = ("test_start_recipe", "starting recipe", None)

    def test_start_recipe(self, action, context):
        """
        Starts evaluating the recipe file (Python syntax) specified in context.config.run.recipe.
        All actions are executed consecutively in the high priority queue

        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.ProcessingContext.ProcessingContext): Keck DRPF ProcessingContext object
        """
        try: 
            recipe_str = action.args.recipe
            self._recipe_ast = ast.parse(recipe_str)
            self._recipe_visitor = KpfPipelineNodeVisitor(pipeline=self, context=context)
            self._recipe_visitor.visit(self._recipe_ast)
        except:
            print(sys.exc_info())

    def test_primitive_validate_args(self, action: Action, context: ProcessingContext):
        """
        for each pair of arguments, validate that they are equal
        """
        args = action.args
        if len(args) % 2 != 0:
            assert False, f"test_primitive_validate_args called with an odd number of arguments, {len(args)}"
        arg_iter = iter(args)
        while True:
            try:
                arg1 = next(arg_iter)
                arg2 = next(arg_iter)
            except StopIteration:
                break
            except Exception as e:
                assert False, f"Unexpected exception in test_primitive_validate_args: {e}"
            assert arg1 == arg2, f"values didn't match as expected, {arg1} vs {arg2}"

# This is the default framework configuration file path
framework_config = 'configs/framework.cfg'
framework_logcfg= 'configs/framework_logger.cfg'

pipe_config = "examples/default_simple.cfg"

def run_recipe(recipe: str, pipe_config: str=pipe_config):
    """
    This is the code that runs the recipe given as the only argument.
    It mimics the kpf framework/pipeline startup code in cli.py, but
    uses strings instead of file names for the recipes.
    """
    pipe = TestKpfPipeline

    # Setup a pipeline logger
    # This is to differentiate between the loggers of framework and pipeline
    # and individual modules.
    # The configs related to the logger is under the section [LOGGER]

    # Try to initialize the framework 
    try:
        framework = Framework(pipe, framework_config)
        # Overwrite the framework logger with this instance of logger
        # using framework default logger creates some obscure problem
        """
        framework.logger = start_logger('DRPFrame', framework_logcfg)
        """
        framework.pipeline.start(pipe_config)
    except Exception as e:
        print("Failed to initialize framework, exiting ...", e)
        # traceback.print_exc()
        # sys.exit(1)

    # python code
    arg = Arguments(name="test_start_recipe_args", recipe=recipe)
    framework.append_event('test_start_recipe', arg)
    framework.append_event('exit', Arguments())
    framework.start()

def test_recipe_basics():
    try:
        run_recipe(basics_recipe)
    except Exception as e:
        assert False, f"test_recipe_basics: unexpected exception {e}"

# The framework doesn't return control after an exception, so we can't run
# the following two tests at the moment.
#
# def test_recipe_undefined_variable():
#     try:
#         run_recipe(undefined_variable_recipe)
#     except RecipeError:
#         pass
#     except Exception as e:
#         assert False, f"Unexpected error: {e}"
#     else:
#         assert False, "test_recipe_undefined_variable should have raised an exception, but didn't"

# def test_recipe_bad_assignment():
#     try:
#         run_recipe(bad_assignment_recipe)
#     except RecipeError:
#         pass
#     except Exception as e:
#         assert False, f"Unexpected error: {e}"
#     else:
#         assert False, "test_recipe_bad_assignment should have raised an exception, but didn't"

def test_recipe_level0_from_to():
    try:
        run_recipe(level0_from_to_recipe)
    except Exception as e:
        assert False, f"test_recipe_level0_from_to: unexpected exception {e}"

def test_recipe_level1_from_to():
    try:
        run_recipe(level1_from_to_recipe)
    except Exception as e:
        assert False, f"test_recipe_level1_from_to: unexpected exception {e}"

def test_recipe_level2_from_to():
    try:
        run_recipe(level2_from_to_recipe)
    except Exception as e:
        assert False, f"test_recipe_level2_from_to: unexpected exception {e}"

def main():
    test_recipe_basics()