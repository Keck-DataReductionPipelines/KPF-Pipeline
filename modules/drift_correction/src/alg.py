
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from astropy.constants import c

from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger

from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

# db_path = '/data/time_series/kpf_ts.db' # this is the standard database used for plotting, etc.
db_path = '/data/time_series/kpf_ts_dec5b.db'

class ModifyWLS:
    """This utility determines the drift correction derived from etalon frames and
    modifies the WLS then adds the appropriate keywords.

    """
    def __init__(self, l1_obj, default_config_path, logger=None):
        #Input arguments
        self.config = ConfigClass(default_config_path)
        if logger == None:
            self.log = start_logger('DriftCorrection', default_config_path)
        else:
            self.log = logger

        self.l1_obj = l1_obj   # KPF L1 object
        self.date_mid = self.l1_obj.header['PRIMARY']['DATE-MID']
        self.dt = datetime.strptime(self.date_mid, "%Y-%m-%dT%H:%M:%S.%f")
        self.wls_file1 = self.l1_obj.header['PRIMARY']['WLSFILE']
        self.drptag = self.l1_obj.header['PRIMARY']['DRPTAG']
        self.readmode = self.l1_obj.header['PRIMARY']['READSPED']
        for session in ['eve', 'morn', 'midnight']:
            if session in self.l1_obj.header['PRIMARY']['WLSFILE']:
                self.wls_session = session
        print(self.wls_session)

        # Connect to TS DB
        myTS = AnalyzeTimeSeries(db_path=db_path)

        date = self.dt.strftime(format='%Y%m%d')
        start_date = datetime(int(date[:4]), int(date[4:6]), int(date[6:8])) - timedelta(days=1)
        end_date   = datetime(int(date[:4]), int(date[4:6]), int(date[6:8])) + timedelta(days=1)

        cols=['ObsID', 'OBJECT', 'DATE-MID', 'DRPTAG','WLSFILE','WLSFILE2', 'CCFRV', 'CCD1RV', 'CCD2RV', 'NOTJUNK','READSPED','SNRSC548']
        self.df = myTS.dataframe_from_db(start_date=start_date, end_date=end_date,columns=cols)
        self.df = self.prepare_table()

    def apply_drift(self, method):
        self.method = method

        is_solar = self.l1_obj.header['PRIMARY']['SCI-OBJ'].startswith('SoCal')
        if is_solar:
            self.log.warning(f'Drift correction not implemented for SoCal data')
            return self.l1_obj

        try:
            drift_ext = self.l1_obj['DRIFT']
            drift_done = self.l1_obj.header['DRIFT']['DRFTCOR']
            if drift_done:
                self.log.warning(f'Drift correction already performed on file {self.l1_obj.filename}')
                method_performed = self.l1_obj.header['DRIFT']['DRFTMETH']
                if self.method != method_performed:
                    self.log.warning(f'Drift correction method {self.method} requested but method {method_performed} already performed')
                return self.l1_obj

        except (KeyError, AttributeError):
            pass

        try:
            clsmethod = self.__getattribute__(method)
        except AttributeError:
            self.log.error(f'Drift correction method {method} not implemented.')

            raise(AttributeError)

        out_l1 = clsmethod()

        return out_l1


    def prepare_table(self):
        df = self.df

        # df = df[(df['WLSFILE'] != df['WLSFILE2'])]
        df = df[(df['NOTJUNK'] == True)]

        # All fibers illuminated:
        df = df[df['OBJECT'].str.contains("etalon-all", na=False)]

        current_drp_tag = self.drptag
        df = df[(df['DRPTAG'] == current_drp_tag)]

        df = df[(df['READSPED'] == self.readmode)]

        snr_low_lim548  = 500 # needs double checked
        snr_high_lim548 = 2500 # needs double checked
        df = df[(df['SNRSC548'] > snr_low_lim548)]
        df = df[(df['SNRSC548'] <= snr_high_lim548)]

        # Extract 8-digit date code from utctime
        df['date_code_utctime'] = df['DATE-MID'].str[:10].str.replace('-', '')

        # Extract 8-digit date code from wls_file and wls_file2
        df['date_code_wls_file'] = df['WLSFILE'].str.extract(r'(\d{8})')
        df['date_code_wls_file2'] = df['WLSFILE2'].str.extract(r'(\d{8})')

        # df['etalon_mask_date'] = df['SCIMPATH'].str.split('/')[-1][0:8]
        # df = df[df['date_code_wls_file'] == df['etalon_mask_date']]

        # TODO check that etalon mask came from same calibration session as WLSFILE
        # df[df['SCIMPATH'].contains(self.wls_session)]


        df['datetime'] = pd.to_datetime(df['DATE-MID'])

        # don't allow it to choose itself as the drift correction
        df = df[(df['ObsID'] != self.l1_obj.filename.split('_')[0])]

        return df


    def adjust_wls(self, drift_rv):
        for ext in self.l1_obj.extensions:
            if 'WAVE' in ext.upper() and 'CA_HK' not in ext.upper():
                self.l1_obj[ext] = self.l1_obj[ext] * (1 - drift_rv/c.to('km/s').value)



    def add_keywords(self, drift_rv):
        header = self.l1_obj.header['DRIFT']
        header['DRFTCOR'] = 1
        header['DRFTRV'] = np.median(drift_rv)
        header['DRFTMETH'] = self.method


    def nearest_neighbor(self):
        # Ensure self.dt is a pandas Timestamp
        self.dt = pd.Timestamp(self.dt)
        
        # Calculate time delta in seconds
        self.df['time_delta'] = (self.df['datetime'] - self.dt).abs().dt.total_seconds()
        
        # Find the index of the closest etalon RV
        best_match = np.argmin(self.df['time_delta'])
        best_row = self.df.iloc[best_match]
        drift_rv = best_row['CCFRV']
        
        # Add DRIFT extension if it doesn't already exist
        if 'DRIFT' not in self.l1_obj.extensions:
            self.l1_obj.create_extension('DRIFT', np.array)
        
        # Update DRIFT extension
        self.l1_obj['DRIFT'] = np.array([drift_rv])  # Ensure correct data structure
        
        # Update headers with observational details
        self.l1_obj.header['DRIFT']['DRFTOBS'] = best_row['ObsID']
        time_delta_hours = best_row['time_delta'] / 3600  # Convert seconds to hours
        self.l1_obj.header['DRIFT']['DRFTDEL'] = time_delta_hours

        # Apply drift correction in RV space
        self.adjust_wls(drift_rv)
        self.add_keywords(drift_rv)

        return self.l1_obj


    def nearest_interpolation(self):
        self.df['time_delta'] = (self.df['datetime'] - self.dt).dt.total_seconds()
        after = self.df.query('time_delta >= 0')
        before = self.df.query('time_delta < 0')

        best_before = np.argmax(before['time_delta'])
        best_after = np.argmin(after['time_delta'])

        before_rv = before.iloc[best_before]['CCFRV']
        after_rv = after.iloc[best_after]['CCFRV']

        before_dt = before.iloc[best_before]['time_delta']
        after_dt = after.iloc[best_after]['time_delta']

        # linear interpolation
        drift_rv = before_rv + (before_dt / (before_dt - after_dt)) * (after_rv - before_rv)

        # Add DRIFT extension if it doesn't already exist
        if 'DRIFT' not in self.l1_obj.extensions:
            self.l1_obj.create_extension('DRIFT', np.array)
        
        # Update DRIFT extension
        self.l1_obj['DRIFT'] = np.array([drift_rv])  # Ensure correct data structure
        
        # Update headers with observational details
        self.l1_obj.header['DRIFT']['DRFTOBS'] = before.iloc[best_before]['ObsID']
        self.l1_obj.header['DRIFT']['DRFTOBS2'] = after.iloc[best_after]['ObsID']

        before_time_delta_hours = before_dt / 3600  # Convert seconds to hours
        after_time_delta_hours = after_dt / 3600

        self.l1_obj.header['DRIFT']['DRFTDEL'] = before_time_delta_hours
        self.l1_obj.header['DRIFT']['DRFTDEL2'] = after_time_delta_hours

        self.adjust_wls(drift_rv)
        self.add_keywords(drift_rv)

        print(before_rv*1000, after_rv*1000, drift_rv*1000)
        print(before_dt, after_dt)


        return self.l1_obj