import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.ogger import start_logger

from modules.Utils.config_parser import ConfigHandler
from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

db_path = '/data/ime_series/kpf_ts_dec5b.db'

class BlazeAlg:
    """Docstring
    
    """
    def __init__(self, l1_obj, default_config_path, logger=None):
        # Input arguments
        self.config = ConfigXClass(default_config_path)
        if logger == None:
            self.log = start_logger('BlazeCorrection', default_config_path)
        else:
            self.log = logger
            
        cfg_params = ConfigHandler(self.config, 'PARAM')
        self.db_path = cfg_params.get_config_value('ts_db_path')
        print(self.db_path)
        
        self.l1_obj = l1_obj
        self.date_mid = self.l1_obj.header['PRIMARY']['DATE-MID']
        self.dt = datetime.strptime(self.date_mid, "%Y-%m-%dT%H:%M:%S.%f")
        self.drptag = self.l1_obj.header['PRIMARY']['DRPTAG']
        
        try:
            self.readmode = self.l1_obj.header['PRIMARY']['READSPED']
        except KeyError:
            if 'regular' in self.l1_obj.header['PRIMARY']['GRACF']:
                self.readmode = 'regular'
            else:
                self.readmode = 'fast'
                
                
        # Connect to TS DB
        myTS = AnalyzeTimeSeries(db_path=self.db_path)
        
        date = self.dt.strftime(format='%Y%m%d')
        start_date = datetime(int(date[:4]), int(date[4:6]), int(date[6:8])) - timedelta(days=1)
        end_date   = datetime(int(date[:4]), int(date[4:6]), int(date[6:8])) + timedelta(days=1)

        cols = ['ObsID', 'OBJECT', 'DATE-MID', 'DRPTAG', 'NOTJUNK', 'READSPED']
        self.df = myTS.dataframe_from_db(start_date=start_date, end_date=end_date, columns=cols)
        self.df = self.prepare_table()
        
    def prepare_table(self):
        df = self.df
        
        df = df[(df['NOTJUNK'] == True)]
        df = df[(df['DRPTAG'] == self.drptag)]
        df = df[(df['READSPED'] == self.readmode)]
        
        # Extract 8-digit date code from utctime
        df['date_code_utctime'] = df['DATE-MID'].str[:10].str.replace('-', '')
        df['datetime'] = pd.to_datetime(df['DATE-MID'])
        
        return df
    
    def add_keywords(self):
        header = self.l1_ojb.header['PRIMARY']
        header['BLAZECORR'] = 1
        header['BLAZEMETH'] = self.method
        
    def add_extensions(self):
        self.l1_obj.create_extension('GREEN_SCI_BLAZE1', np.array)
        self.l1_obj.create_extension('GREEN_SCI_BLAZE2', np.array)
        self.l1_obj.create_extension('GREEN_SCI_BLAZE3', np.array)
        self.l1_obj.create_extension('RED_SCI_BLAZE1', np.array)
        self.l1_obj.create_extension('RED_SCI_BLAZE2', np.array)
        self.l1_obj.create_extension('RED_SCI_BLAZE3', np.array)
        
    def apply_blaze_correction(self, method):
        self.method = method
        
        try:
            blaze_method = self.__getattribute(method)
        except AttributeError:
            self.log.error(f'Blaze correction method {method} not implemented.')
            raise(AttributeError)
            
        out_l1 = blaze_method()
        
        return out_l1
    
    def _uniform(self):
        self.l1_obj['GREEN_SCI_BLAZE1'] = np.ones_like(self.l1_obj['GREEN_SCI_FLUX1'])
        self.l1_obj['GREEN_SCI_BLAZE2'] = np.ones_like(self.l1_obj['GREEN_SCI_FLUX2'])
        self.l1_obj['GREEN_SCI_BLAZE3'] = np.ones_like(self.l1_obj['GREEN_SCI_FLUX3'])
        self.l1_obj['RED_SCI_BLAZE1'] = np.ones_like(self.l1_obj['RED_SCI_FLUX1'])
        self.l1_obj['RED_SCI_BLAZE2'] = np.ones_like(self.l1_obj['RED_SCI_FLUX2'])
        self.l1_obj['RED_SCI_BLAZE3'] = np.ones_like(self.l1_obj['RED_SCI_FLUX3'])
        
        return self.l1_obj