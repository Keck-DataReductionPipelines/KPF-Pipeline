# An example pipeline that is used to test the template fitting 
# algorithm module. 
import os

from keckdrpframework.pipelines.base_pipeline import Base_pipeline

class TestPipeline(Base_pipeline):
    """
    Pipeline to Process KPF data using the KeckDRPFramework

    Attributes:
        event_table (dictionary): table of actions known to framework. All primitives must be registered here.
    """
    name = 'KPF-Pipe'
    event_table = {
                "read_recipe":
                    ("read_recipe", "executing recipe", None),
                "evaluate_recipe":
                    ("evaluate_recipe", "executing recipe", None),
                "TFAMain":
                    ("TFAMain", "Calculating RV", None),

                "exit":
                    ("exit_loop", "killing framework", None)
                }

    def __init__(self):
        Base_pipeline.__init__(self)

    def read_recipe(self, action, context):
        """
        Read the recipe file (list of actions) specified in context.config.run.recipe.
        All actions are executed consecutively in the high priority queue

        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.processing_context.Processing_context): Keck DRPF Processing_context object
        """

        recipe_file = action.args.recipe
        context.logger.info('Executing recipe file {}'.format(recipe_file))
        f = open(recipe_file)
        for event in f.readlines():
            event = event.strip()
            context.logger.info(event)
            context.push_event(event, action.args)
        f.close()

    def evaluate_recipe(self, action, context):
        """
        Evaluates the recipe file (Python file) specified in context.config.run.recipe.
        All actions are executed consecutively in the high priority queue

        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.processing_context.Processing_context): Keck DRPF Processing_context object
        """

        recipe_file = action.args.recipe
        context.logger.info('Executing recipe file {}'.format(recipe_file))
        f = open(recipe_file)
        fstr = f.read()
        exec(fstr)
        f.close()

    def exit_loop(self, action, context):
        """
        Force the Keck DRP Framework to exit the infinite loop

        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.processing_context.Processing_context): Keck DRPF Processing_context object
        """
        context.logger.info("Goodbye")
        os._exit(0)
