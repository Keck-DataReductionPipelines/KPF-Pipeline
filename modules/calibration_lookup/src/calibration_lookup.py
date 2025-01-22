
import configparser

from kpfpipe.primitives.level0 import KPF0_Primitive
from modules.calibration_lookup.src.alg import GetCalibrations
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/calibration_lookup/configs/default.cfg'

class CalibrationLookup(KPF0_Primitive):
    """This utility looks up the associated calibrations for a given datetime and
       returns a dictionary with all calibration types.

    Description:
        * Method `__init__`:

            CalibrationLookup constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `BarycentricCorrection` event issued in the recipe:

                    - `action.args[0] (dict)`: Datetime string in ISO format
                
                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the default config file defined for the CalibrationLookup module.


        * Method `__perform`:

            CalibrationLookup returns the result in `Arguments` object which contains a dictionary of calibration file paths
            for the input datetime

    Usage:
        For the recipe, the CalibrationLookup primitive is called like::

            :
            dt_string = GetHeaderValue(l1_obj, 'DATE-MID')
            cals = CalibrationLookup(dt_string)
            :


    """
    def __init__(self, action, context):

        #Initialize parent class
        KPF0_Primitive.__init__(self, action, context)

        #Input arguments
        self.datetime = self.action.args[0]   # ISO datetime string
        try:
            self.subset = self.action.args["subset"]
        except KeyError:
            self.subset = None

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['calibration_lookup']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        self.caldate_files = self.config['PARAM']['date_files']
        self.caltypes = self.config['PARAM']['lookup_map']

    def _perform(self):

        cal_look = GetCalibrations(self.datetime, self.config_path)
        output_cals = cal_look.lookup(subset=self.subset)

        return Arguments(output_cals)

