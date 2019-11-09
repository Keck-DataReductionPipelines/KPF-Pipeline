"""
Pipeline to run under the Keck DRP Framework
"""


import os

from keckdrpframework.pipelines.base_pipeline import Base_pipeline

from kpfpipe.primitives.level0 import *
from kpfpipe.primitives.level1 import *


class KPF_pipeline(Base_pipeline):
    """
    Pipeline to Process KPF data using the KeckDRPFramework

    Attributes:
        event_table (dictionary): table of actions known to framework. All primitives must be registered here.
    """

    event_table = {"reduce_level0":
                       ("subtract_bias", "removing bias", 'divide_flat'),
                   "subtract_bias":
                       ("subtract_bias", "removing bias", None),
                   "divide_flat":
                       ("divide_flat", "dividing by flat", 'extract_spectrum'),
                   "extract_spectrum":
                       ("extract_spectrum", "extracting spectrum", None),

                   "reduce_level1":
                       ("level1to2", "reducing level 1->2", None),
                   "calibrate_wavelengths":
                       ("calibrate_wavelengths", "calibrating wavelengths", None),
                   "remove_emission_line_regions":
                       ("remove_emission_line_regions", "removing emission lines", None),
                   "remove_solar_regions":
                       ("remove_solar_regions", "removing solar regions", None),
                   "correct_telluric_lines":
                       ("correct_telluric_lines", "removing telluric lines", None),
                   "correct_wavelength_dependent_barycentric_velocity":
                       ("correct_wavelength_dependent_barycentric_velocity", "correcting to barycenter", None),
                   "calculate_RV_from_spectrum":
                       ("calculate_RV_from_spectrum", "calculating RV", None),

                   "execute_recipe":
                       ("execute_recipe", "executing recipe", None),
                   "evaluate_recipe":
                       ("evaluate_recipe", "executing recipe", None),

                   "exit":
                       ("exit_loop", "killing framework", None)
                   }

    def __init__(self):
        Base_pipeline.__init__(self)

    def execute_recipe(self, action, context):
        """
        Executes the recipe file (list of actions) specified in context.config.run.recipe.
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
