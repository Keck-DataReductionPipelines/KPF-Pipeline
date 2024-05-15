
import configparser

from kpfpipe.primitives.level0 import KPF0_Primitive
from modules.calibration_lookup.src.alg import GetCalibrations
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/calibration_lookup/configs/default.cfg'

class CalibrationLookup(KPF0_Primitive):
    """This utility looks up the associated calibrations for a given datetime and
       returns a dictionary with all calibration types.

    """
    def __init__(self, action, context):

        #Initialize parent class
        KPF0_Primitive.__init__(self, action, context)

        #Input arguments
        self.datetime = self.action.args[0]   # ISO datetime string

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
        output_cals = cal_look.lookup()

        return Arguments(output_cals)

