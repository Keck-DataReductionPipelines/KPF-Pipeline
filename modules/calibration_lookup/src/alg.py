
import sys
if sys.version_info >= (3, 9):
    from importlib import resources
else:
    import importlib_resources as resources

import pandas as pd
from datetime import datetime

from database.modules.utils.kpf_db import KPFDB
from keckdrpframework.models.arguments import Arguments
from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger
from astropy.io.fits import getheader

class GetCalibrations:
    """This utility looks up the associated calibrations for a given datetime and
       returns a dictionary with all calibration types.

    """
    def __init__(self, datetime, default_config_path, use_db=True, logger=None):

        """
        use_db (boolean) - to disable db access, set to False (e.g., when looking up file-based keywords only)
        """
        
        # Initialize DB class
        # self.db_lookup = QueryDBNearestMasterFilesFramework(self.action, self.context)

        #Input arguments
        self.datetime = datetime   # ISO datetime string
        self.config = ConfigClass(default_config_path)
        if logger == None:
            self.log = start_logger('GetCalibrations', default_config_path)
        else:
            self.log = logger

        self.caldate_files = eval(self.config['PARAM']['date_files'])
        self.lookup_map = eval(self.config['PARAM']['lookup_map'])
        self.db_cal_types = eval(self.config['PARAM']['db_cal_types'])
        self.db_cal_file_levels = eval(self.config['PARAM']['db_cal_file_levels'])
        self.wls_cal_types = eval(self.config['PARAM']['wls_cal_types'])
        self.max_age = eval(self.config['PARAM']['max_cal_age'])
        self.defaults = eval(self.config['PARAM']['defaults'])
        if use_db:
            self.db = KPFDB(logger=self.log)

    def lookup(self, subset=None):
        dt = datetime.strptime(self.datetime, "%Y-%m-%dT%H:%M:%S.%f")
        date_str = datetime.strftime(dt, "%Y%m%d")

        output_cals = {}
        db_results = None
        if subset == None:
            subset = self.lookup_map.keys()
        for cal,lookup in self.lookup_map.items():
            if cal not in subset:
                continue
            if lookup == 'file':
                filename = self.caldate_files[cal]
                fndir, fn = filename.split("/", 1)
                # Use resources.open_text() to read the .csv because it has a relative path within repo
                with resources.open_text(fndir, fn) as f:
                    df = pd.read_csv(f, header=0, skipinitialspace=True)
                for i, row in df.iterrows():
                    start = datetime.strptime(row['UT_start_date'], "%Y-%m-%d %H:%M:%S")
                    end = datetime.strptime(row['UT_end_date'], "%Y-%m-%d %H:%M:%S")
                    if start <= dt < end:
                        try:
                            output_cals[cal] = eval(row['CALPATH'])
                        except SyntaxError:
                            output_cals[cal] = row['CALPATH']
            elif lookup == 'database':
                for lvl, cal_type in zip(self.db_cal_file_levels, self.db_cal_types):
                    if cal_type[0] in output_cals.keys() or cal_type[0].lower() not in subset:
                        continue
                    db_results = self.db.get_nearest_master(self.datetime, lvl, cal_type)
                    if db_results[0] == 0:
                        output_cals[cal_type[0].lower()] = db_results[1]
                    else:
                        output_cals[cal_type[0].lower()] = self.defaults[cal_type[0].lower()]
            elif lookup == 'wls' or lookup == 'etalon':
                for cal_type in self.wls_cal_types:
                    wls_results = self.db.get_bracketing_wls(self.datetime, cal_type[1], max_cal_delta_time=self.max_age)
                    if len(wls_results) > 1 and (wls_results[0] == 0 or wls_results[2] == 0):
                        wls_files = [wls_results[1], wls_results[3]]
                        if wls_files[0] == None:
                            wls_files[0] = wls_files[1]
                        if wls_files[1] == None:
                            wls_files[1] = wls_files[0]
                        output_cals[cal] = wls_files
                        break
                    else:
                        output_cals[cal] = self.defaults[cal]
                        wls_files = output_cals[cal]

                if lookup == 'etalon':
                    try:
                        new_dt = getheader(wls_files[0])['DATE-BEG']
                        self.datetime = new_dt
                    except:  # no DB available
                        pass
                    self.lookup_map['etalonmask'] = 'database'
                    output_cals[cal] = self.lookup(subset=['etalonmask'])['etalonmask']

        return output_cals

