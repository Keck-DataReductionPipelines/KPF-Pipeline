
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
    def __init__(self, l1_obj, default_config_path, logger=None, etalon_table=None):

        # Connect to TS DB
        if etalon_table is None:
            pass
            # self.db_lookup = AnalyzeTSDatabase(self.action, self.context)
        else:
            self.df = pd.read_csv(etalon_table)

        #Input arguments
        self.l1_obj = l1_obj   # KPF L1 object
        self.config = ConfigClass(default_config_path)
        if logger == None:
            self.log = start_logger('DriftCorrection', default_config_path)
        else:
            self.log = logger

        self.date_mid = self.l1_obj.header['PRIMARY']['DATE-MID']
        self.wls_file1 = self.l1_obj.header['PRIMARY']['WLSFILE']
        self.drptag = self.l1_obj.header['PRIMARY']['DRPTAG']


    def apply_drift(self, method):

        dt = datetime.strptime(self.date_mid, "%Y-%m-%dT%H:%M:%S.%f")
        date_str = datetime.strftime(dt, "%Y%m%d")

        try:
            clsmethod = self.__getattribute__(method)
        except AttributeError:
            self.log.error(f'Drift correction method {method} not implemented.')

            raise(AttributeError)

        out_l1 = clsmethod()

        return out_l1


    def nearest_neighbor(self):
        return self.l1_obj


    def nearest_interpolation(self):
        return self.l1_obj