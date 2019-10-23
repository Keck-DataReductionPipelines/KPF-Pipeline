
import os

from keckdrpframework.pipelines.base_pipeline import Base_pipeline
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.core.framework import Framework
from keckdrpframework.config.framework_config import ConfigClass

from kpfpipe.primitives.level0 import *
from kpfpipe.primitives.level1 import *


class KPF_pipeline(Base_pipeline):
    """
    Pipeline to Process KPF data
    """

    event_table = {"reduce_level0":
                       ("subtract_bias", "removing bias", 'divide_flat'),
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

                   "exit":
                       ("exit_loop", "killing framework", None)
                   }


    def __init__(self):
        """
        Constructor
        """
        Base_pipeline.__init__(self)

    def level1to2(self, action, context):
        context.push_event('calibrate_wavelengths', action.args)
        context.push_event('remove_emission_line_regions', action.args)
        context.push_event('remove_solar_regions', action.args)
        context.push_event('correct_telluric_lines', action.args)
        context.push_event('correct_wavelength_dependent_barycentric_velocity', action.args)
        context.push_event('calculate_RV_from_spectrum', action.args)

    def exit_loop(self, action, context):
        context.logger.info("Goodbye")
        os._exit(0)
