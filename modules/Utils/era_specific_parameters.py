
from datetime import datetime

from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments
from kpfpipe.config.pipeline_config import ConfigClass

import pandas as pd

class EraSpecific(KPF0_Primitive):
    """This utility looks up the KPFERA for a file and then returns the
       appropriate era-specific configuration parameters.

    """
    def __init__(self, action, context):
        "FrameSubtract constructor."
        
        #Initialize parent class
        KPF0_Primitive.__init__(self, action, context)

        #Input arguments
        self.input_file = self.action.args[0]   # L0 object
        self.parameter_name = self.action.args[1]

        era_file = 'static/kpfera_definitions.csv'
        config_file = 'configs/era_specific.cfg'
        self.config = ConfigClass(config_file)

        self.eras = pd.read_csv(era_file, dtype='str',
                                sep='\s*,\s*')

    def _perform(self):
        
        dt = datetime.strptime(self.input_file.header['PRIMARY']['DATE-OBS'], "%Y-%m-%d")
        for i,row in self.eras.iterrows():
            start = datetime.strptime(row['UT_start_date'], "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(row['UT_end_date'], "%Y-%m-%d %H:%M:%S")
            if dt > start and dt <= end:
                break

        era = row['KPFERA']
        options = eval(self.config.ARGUMENTS[self.parameter_name])
        value = options[era]

        return Arguments(value)
