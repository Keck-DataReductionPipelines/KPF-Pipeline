
import configparser

from kpfpipe.primitives.level1 import KPF1_Primitive
from modules.drift_correction.src.alg import ModifyWLS
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/drift_correction/configs/default.cfg'

class DriftCorrection(KPF1_Primitive):
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
        KPF1_Primitive.__init__(self, action, context)

        #Input arguments
        self.l1_obj = self.action.args[0]   # KPF L1 object
        self.method = self.action.args[1]   # string of method to use

        # input configuration
        self.config = configparser.ConfigParser()
        try:
            self.config_path = context.config_path['drift_correction']
        except:
            self.config_path = DEFAULT_CFG_PATH
        self.config.read(self.config_path)

        self.ts_db = self.config['PARAM']['ts_db_path']

    def _perform(self):

        dc = ModifyWLS(self.l1_obj, self.config_path)
        out_l1 = dc.apply_drift(method=self.method)

        return Arguments(out_l1)

