
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from astropy.constants import c

from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger

from modules.Utils.config_parser import ConfigHandler
from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries


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

        cfg_params = ConfigHandler(self.config, 'PARAM')
        self.db_path = cfg_params.get_config_value('ts_db_path')
        self.backend = cfg_params.get_config_value('ts_backend')

        self.l1_obj = l1_obj   # KPF L1 object
        self.date_mid = self.l1_obj.header['PRIMARY']['DATE-MID']
        self.dt = datetime.strptime(self.date_mid, "%Y-%m-%dT%H:%M:%S.%f")
        self.wls_file1 = self.l1_obj.header['PRIMARY']['WLSFILE']
        self.drptag = self.l1_obj.header['PRIMARY']['DRPTAG']

        try:
            self.readmode = self.l1_obj.header['PRIMARY']['READSPED']
        except KeyError:
            if 'regular' in self.l1_obj.header['PRIMARY']['GRACF']:
                self.readmode = 'regular'
            else:
                self.readmode = 'fast'
        
        for session in ['eve', 'morn', 'midnight']:
            if session in self.l1_obj.header['PRIMARY']['WLSFILE']:
                self.wls_session = session

        # Connect to TS DB
        myTS = AnalyzeTimeSeries(db_path=self.db_path, backend=self.backend)

        date = self.dt.strftime(format='%Y%m%d')
        start_date = datetime(int(date[:4]), int(date[4:6]), int(date[6:8])) - timedelta(days=60) # this should be as long as our longest time witout LFC.
        end_date   = datetime(int(date[:4]), int(date[4:6]), int(date[6:8])) + timedelta(days=60)
        cols=['ObsID', 'OBJECT', 'DATE-MID', 'DRPTAGL1','WLSFILE','WLSFILE2', 'CCFRV', 'CCD1RV', 'CCD2RV', 'NOTJUNK','READSPED','SNRSC548', 'SCIMPATH']
        self.df = myTS.db.dataframe_from_db(start_date=start_date, end_date=end_date, columns=cols)
        self.df = self.prepare_table()

    def apply_drift(self, method):
        self.method = method

        is_solar = self.l1_obj.header['PRIMARY']['SCI-OBJ'].startswith('SoCal')
        if is_solar:
            self.log.warning(f'Drift correction not implemented for SoCal data')
            return self.l1_obj
        
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
        df = df[df['OBJECT'].str.contains(r'etalon-all|slewcal', na=False)]
        current_drp_tag = self.drptag
        df = df[(df['DRPTAGL1'] == current_drp_tag)]

        df = df[(df['READSPED'] == self.readmode)]
        snr_low_lim548  = 500 # needs double checked
        snr_high_lim548 = 2500 # needs double checked
        df = df[(df['SNRSC548'] > snr_low_lim548)]
        df = df[(df['SNRSC548'] <= snr_high_lim548)]
        # Extract date from DATE-MID, remove dashes without treating pattern as regex

        df['date_code_utctime'] = df['DATE-MID'].astype(str).str.slice(0, 10).str.replace('-', '', regex=False)
        df['date_code_wls_file'] = df['WLSFILE'].str.extract(r'(\d{8})', expand=False)
        df['date_code_wls_file2'] = df['WLSFILE2'].str.extract(r'(\d{8})', expand=False)

        df['etalon_mask_date'] = df['SCIMPATH'].str.split('/').str[-1].str.slice(0, 8) # SCIMPATH for etalon is the etalon mask date.
        df = df[df['date_code_wls_file'] == df['etalon_mask_date']] #This forces the WLS and Mask to be from the same date but this should never happen.

        df['datetime'] = pd.to_datetime(df['DATE-MID'])
        df = df[(df['ObsID'] != self.l1_obj.filename.split('_')[0])]         # don't allow it to choose itself as the drift correction

        return df


    def adjust_wls(self, drift_rv):
        is_science = self.l1_obj.header['PRIMARY']['SCI-OBJ'].startswith('Target')
        if not is_science:
            obj = self.l1_obj.header['PRIMARY']['OBJECT']
            self.log.warning(f'No drift correction for OBJECT {obj}')
        else:
            for ext in self.l1_obj.extensions:
                if 'WAVE' in ext.upper() and 'CA_HK' not in ext.upper():
                    self.l1_obj[ext] = self.l1_obj[ext] * (1 - drift_rv/c.to('km/s').value)



    def add_keywords(self, drift_rv):
        header = self.l1_obj.header['PRIMARY']
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
        self.l1_obj.header['PRIMARY']['DRFTOBS'] = best_row['ObsID']
        time_delta_hours = best_row['time_delta'] / 3600  # Convert seconds to hours
        self.l1_obj.header['PRIMARY']['DRFTDEL'] = time_delta_hours

        # Apply drift correction in RV space
        # self.adjust_wls(drift_rv) # Do not modify WLS
        self.add_keywords(drift_rv)

        return self.l1_obj


    def nearest_interpolation(self):
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df['time_delta'] = (self.df['datetime'] - self.dt).dt.total_seconds()
        after = self.df.query('time_delta >= 0')
        before = self.df.query('time_delta <= 0')
        if len(after) == 0 or len(before) == 0:
            self.log.warning('Could not find bracketing files for nearest interpolation method. Defaulting to nearest_neighbor method.')
            return self.nearest_neighbor()

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
        self.l1_obj.header['PRIMARY']['DRFTOBS'] = before.iloc[best_before]['ObsID']
        self.l1_obj.header['PRIMARY']['DRFTOBS2'] = after.iloc[best_after]['ObsID']
        before_time_delta_hours = before_dt / 3600  # Convert seconds to hours
        after_time_delta_hours = after_dt / 3600

        self.l1_obj.header['PRIMARY']['DRFTDEL'] = before_time_delta_hours
        self.l1_obj.header['PRIMARY']['DRFTDEL2'] = after_time_delta_hours
        # self.adjust_wls(drift_rv) # DO not change th wavelength solution itself.
        self.add_keywords(drift_rv)
        return self.l1_obj
