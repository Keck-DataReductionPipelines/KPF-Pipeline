
from datetime import datetime
import pandas as pd

from keckdrpframework.models.arguments import Arguments
from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger
from astropy.io.fits import getheader

class ModifyWLS:
    """This utility determines the drift correction derived from etalon frames and
    modifies the WLS then adds the appropriate keywords.

    """
    def __init__(self, l1_obj, default_config_path, logger=None):

        # Connect to TS DB
        # self.db_lookup = AnalyzeTSDatabase(self.action, self.context)

        #Input arguments
        self.l1_obj = l1_obj   # KPF L1 object
        self.config = ConfigClass(default_config_path)
        if logger == None:
            self.log = start_logger('DriftCorrection', default_config_path)
        else:
            self.log = logger


    def apply_drift(self, method):
        date_mid = self.l1_obj.header['PRIMARY']['DATE-MID']
        wls_file1 = self.l1_obj.header['PRIMARY']['WLS_FILE']
        drptag = self.l1_obj.header['PRIMARY']['DRPTAG']

        dt = datetime.strptime(date_mid, "%Y-%m-%dT%H:%M:%S.%f")
        date_str = datetime.strftime(dt, "%Y%m%d")

        return out_l1


    def nearest_neighbor(self):
        return self.l1_obj


    def nearest_interpolation(self):
        return self.l1_obj