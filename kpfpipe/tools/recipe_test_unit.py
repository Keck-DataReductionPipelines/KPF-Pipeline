# test_recipe_unit.py
import sys, os, traceback
import tempfile

sys.path.insert(0, os.path.abspath('../KeckDRPFramework'))

from keckdrpframework.core.framework import Framework
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.action import Action
from keckdrpframework.models.processing_context import ProcessingContext
from kpfpipe.pipelines.kpfpipeline import KPFPipeline
from kpfpipe.logger import start_logger


class KpfPipelineForTesting(KPFPipeline):
    """
    Test pipeline class extending KpfPipeline
    """

    def __init__(self, context: ProcessingContext):
        """ constructor """
        KPFPipeline.__init__(self, context)
        self.event_table['test_primitive_validate_args'] = ("test_primitive_validate_args", "processing", "resume_recipe")

    @staticmethod
    def test_primitive_validate_args(action: Action, context: ProcessingContext):
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
    This is the code that runs the given recipe.
    It mimics the kpf framework/pipeline startup code in cli.py, but writes
    the recipe string into a temporary file before invoking the framework
    with start_recipe as the initial event.
    The framework is put in testing mode so that it passes exceptions
    on to this testing code.  That we can test the proper handling of
    recipe errors, e.g. undefined variables.
    """
    pipe = KpfPipelineForTesting

    # Setup a pipeline logger
    # This is to differentiate between the loggers of framework and pipeline
    # and individual modules.
    # The configs related to the logger is under the section [LOGGER]

    # Try to initialize the framework 
    try:
        framework = Framework(pipe, framework_config, testing=True)
        # Overwrite the framework logger with this instance of logger
        # using framework default logger creates some obscure problem
        """
        framework.logger = start_logger('DRPFrame', framework_logcfg)
        """
        framework.pipeline.start(pipe_config)
        framework.start_action_loop()

    except Exception as e:
        print("Failed to initialize framework, exiting ...", e)
        traceback.print_exc()
        # sys.exit(1)

    # python code
    with tempfile.NamedTemporaryFile(mode='w+') as f:
        f.write(recipe)
        f.seek(0)
        arg = Arguments(name="start_recipe_args", recipe=f.name)
        framework.append_event('start_recipe', arg)
        framework.start_action_loop()


def recipe_test(recipe: str, pipe_config: str=pipe_config):
    try:
        run_recipe(recipe, pipe_config)
    except Exception as e:
        assert False, f"test_recipe_basics: unexpected exception {e}"

