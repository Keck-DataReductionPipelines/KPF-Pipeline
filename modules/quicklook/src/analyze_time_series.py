import os
import ast
import time
import glob
import copy
import json
import sqlite3
import calendar
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
from astropy.table import Table
from astropy.io import fits
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from modules.Utils.utils import DummyLogger
from modules.Utils.kpf_parse import get_datecode
from matplotlib.dates import HourLocator, DayLocator, MonthLocator, YearLocator, AutoDateLocator, DateFormatter

class AnalyzeTimeSeries:

    """
    Description:
        This class contains a set of methods to create a database of data associated 
        with KPF observations, as well as methods to ingest data, query the database, 
        print data, and made time series plots.  An elaborate set of standard time series 
        plots can be made over intervals of days/months/years/decades spanning a date 
        range.  
        
        The ingested data comes from L0/2D/L1/L2 keywords and the TELEMETRY extension 
        in L0 files.  With the current version of this code, all TELEMETRY keywords are 
        added to the database an a small subset of the L0/2D/L1/L2 keywords are added. 
        These lists can be expanded, but will require re-ingesting the data (which takes 
        about half a day for all KPF observations).  RVs are currently not ingested, but 
        that capability should be added.

    Arguments:
        db_path (string) - path to database file
        base_dir (string) - L0 directory
        drop (boolean) - if true, the database at db_path is dropped at startup
        logger (logger object) - a logger object can be passed, or one will be created

    Attributes:
        L0_keyword_types   (dictionary) - specifies data types for L0 header keywords
        D2_keyword_types   (dictionary) - specifies data types for 2D header keywords
        L1_keyword_types   (dictionary) - specifies data types for L1 header keywords
        L2_keyword_types   (dictionary) - specifies data types for L2 header keywords
        L0_telemetry_types (dictionary) - specifies data types for L0 telemetry keywords
        L2_RV_header_keyword_types (dictionary) - specifies data types for L2 RV header keywords

    Related Commandline Scripts:
        'ingest_dates_kpf_tsdb.py' - ingest from a range of dates
        'ingest_watch_kpf_tsdb.py' - ingest by watching a set of directories
        'generate_time_series_plots.py' - creates standard time series plots
        
    To-do:
        * Add database for masters (separate from ObsIDs?)
        * Method to return the avg, std., etc. for a DB column over a time range, with conditions (e.g., fast-read mode only)
        * Check if the plot doesn't have data and don't generate if so
        * Make plots of temperature vs. RV for various types of RVs
        * Add standard plots of flux vs. time for cals (all types?), stars, and solar -- highlight Junked files
        * Check for proper data types (float vs. str) before plotting
        * Add "Last N Days" and implement N=10 on Jump
        * Add separate junk test from list of junked files
        * Add methods to print the schema
        * Augment statistics in legends (median and stddev upon request)
        * Add histogram plots, e.g. for DRPTAG
        * Add the capability of using Jump queries to find files for ingestion or plotting
        * Determine earliest observation with a TELEMETRY extension and act accordingly
        * Ingest information from masters, especially WLS masters
    """

    def __init__(self, db_path='kpf_ts.db', base_dir='/data/L0', logger=None, drop=False):
       
        self.logger = logger if logger is not None else DummyLogger()
        self.logger.info('Starting AnalyzeTimeSeries')
        if self.is_notebook():
            self.tqdm = tqdm_notebook
            self.logger.info('Jupyter Notebook environment detected.')
        else:
            self.tqdm = tqdm
        self.db_path = db_path
        self.logger.info('Path of database file: ' + os.path.abspath(self.db_path))
        self.base_dir = base_dir
        self.logger.info('Base data directory: ' + self.base_dir)
        self.L0_header_keyword_types     = self.get_keyword_types(level='L0')
        self.D2_header_keyword_types     = self.get_keyword_types(level='2D')
        self.L1_header_keyword_types     = self.get_keyword_types(level='L1')
        self.L2_header_keyword_types     = self.get_keyword_types(level='L2')
        self.L2_RV_header_keyword_types  = self.get_keyword_types(level='L2_RV_header')
        self.L0_telemetry_types          = self.get_keyword_types(level='L0_telemetry')
        
        if drop:
            self.drop_table()
            self.logger.info('Dropping KPF database ' + str(self.db_path))

        # the line below might be modified so that if the database exists, then the columns are read from it
        self.create_database()
        self.print_db_status()


    def drop_table(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS kpfdb")
        conn.commit()
        conn.close()


    def create_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint")
        cursor.execute("PRAGMA cache_size = -2000000;")
    
        # Define columns for each file type
        L0_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L0_header_keyword_types.items()]
        D2_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.D2_header_keyword_types.items()]
        L1_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L1_header_keyword_types.items()]
        L2_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L2_header_keyword_types.items()]
        L0_telemetry_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L0_telemetry_types.items()]
        L2_RV_header_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L2_RV_header_keyword_types.items()]
        columns = L0_columns + D2_columns + L1_columns + L2_columns + L0_telemetry_columns + L2_RV_header_columns
        columns += ['"datecode" TEXT', '"ObsID" TEXT']
        columns += ['"L0_filename" TEXT', '"D2_filename" TEXT', '"L1_filename" TEXT', '"L2_filename" TEXT', ]
        columns += ['"L0_header_read_time" TEXT', '"D2_header_read_time" TEXT', '"L1_header_read_time" TEXT', '"L2_header_read_time" TEXT', ]
        columns += ['"Source" TEXT']
        create_table_query = f'CREATE TABLE IF NOT EXISTS kpfdb ({", ".join(columns)}, UNIQUE(ObsID))'
        cursor.execute(create_table_query)
        
        # Define indexed columns
        index_commands = [
            ('CREATE UNIQUE INDEX idx_ObsID       ON kpfdb ("ObsID");',       'idx_ObsID'),
            ('CREATE UNIQUE INDEX idx_L0_filename ON kpfdb ("L0_filename");', 'idx_L0_filename'),
            ('CREATE UNIQUE INDEX idx_D2_filename ON kpfdb ("D2_filename");', 'idx_D2_filename'),
            ('CREATE UNIQUE INDEX idx_L1_filename ON kpfdb ("L1_filename");', 'idx_L1_filename'),
            ('CREATE UNIQUE INDEX idx_L2_filename ON kpfdb ("L2_filename");', 'idx_L2_filename'),
            ('CREATE INDEX idx_FIUMODE ON kpfdb ("FIUMODE");', 'idx_FIUMODE'),
            ('CREATE INDEX idx_OBJECT ON kpfdb ("OBJECT");', 'idx_OBJECT'),
            ('CREATE INDEX idx_DATE_MID ON kpfdb ("DATE-MID");', 'idx_DATE_MID'),
        ]
        
        # Iterate and create indexes if they don't exist
        for command, index_name in index_commands:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='index' AND name='{index_name}';")
            if cursor.fetchone() is None:
                cursor.execute(command)
                
        conn.commit()
        conn.close()


    def ingest_dates_to_db(self, start_date_str, end_date_str, batch_size=50):
        """
        Ingest KPF data for the date range start_date to end_date, inclusive.
        batch_size refers to the number of observations per DB insertion.
        To-do: scan for observations that have already been ingested at a higher level.
        """
        self.logger.info("Adding to database between " + start_date_str + " to " + end_date_str)
        dir_paths = glob.glob(f"{self.base_dir}/????????")
        sorted_dir_paths = sorted(dir_paths, key=lambda x: int(os.path.basename(x)), reverse=start_date_str > end_date_str)
        filtered_dir_paths = [
            dir_path for dir_path in sorted_dir_paths
            if start_date_str <= os.path.basename(dir_path) <= end_date_str
        ]
        t1 = self.tqdm(filtered_dir_paths, desc=(filtered_dir_paths[0]).split('/')[-1])
        for dir_path in t1:
            t1.set_description(dir_path.split('/')[-1])
            t1.refresh() 
            t2 = self.tqdm(os.listdir(dir_path), desc=f'Files', leave=False)
            batch = []
            for L0_filename in t2:
                if L0_filename.endswith(".fits"):
                    file_path = os.path.join(dir_path, L0_filename)
                    batch.append(file_path)
                    if len(batch) >= batch_size:
                        self.ingest_batch_observation(batch)
                        batch = []
            if batch:
                self.ingest_batch_observation(batch)


    def add_ObsID_list_to_db(self, ObsID_filename, reverse=False):
        """
        Read a CSV file with ObsID values in the first column and ingest those files
        into the database.  If reverse=True, then they will be ingested in reverse
        chronological order.
        """
        if os.path.isfile(ObsID_filename):
            try:
                df = pd.read_csv(ObsID_filename)
            except Exception as e:
                self.logger.info(f'Problem reading {ObsID_filename}: ' + e)
        else:
            self.logger.info('File missing: ObsID_filename')
        
        ObsID_pattern = r'KP\.20\d{6}\.\d{5}\.\d{2}'
        first_column = df.iloc[:, 0]
        filtered_column = first_column[first_column.str.match(ObsID_pattern)]
        df = filtered_column.to_frame()
        column_name = df.columns[0]
        df.rename(columns={column_name: 'ObsID'}, inplace=True)
        if reverse:
            df = df.sort_values(by='ObsID', ascending=False)
        else:
            df = df.sort_values(by='ObsID', ascending=True)

        self.logger.info('{ObsID_filename} read with ' + str(len(df)) + ' properly formatted ObsIDs.')

        #t = tqdm_notebook(df.iloc[:, 0].tolist(), desc=f'ObsIDs', leave=True)
        t = self.tqdm(df.iloc[:, 0].tolist(), desc=f'ObsIDs', leave=True)
        for ObsID in t:
            L0_filename = ObsID + '.fits'
            dir_path = self.base_dir + '/' + get_datecode(ObsID) + '/'
            file_path = os.path.join(dir_path, L0_filename)
            base_filename = L0_filename.split('.fits')[0]
            t.set_description(base_filename)
            t.refresh() 
            try:
                self.ingest_one_observation(dir_path, L0_filename) 
            except Exception as e:
                self.logger.error(e)


    def ingest_one_observation(self, dir_path, L0_filename):
        """
        Ingest a single observation into the database.
        """
        base_filename = L0_filename.split('.fits')[0]
        L0_file_path = f"{dir_path}/{base_filename}.fits"
        D2_file_path = f"{dir_path.replace('L0', '2D')}/{base_filename}_2D.fits"
        L1_file_path = f"{dir_path.replace('L0', 'L1')}/{base_filename}_L1.fits"
        L2_file_path = f"{dir_path.replace('L0', 'L2')}/{base_filename}_L2.fits"
        D2_filename  = f"{L0_filename.replace('L0', '2D')}"
        L1_filename  = f"{L0_filename.replace('L0', 'L1')}"
        L2_filename  = f"{L0_filename.replace('L0', 'L2')}"

        # update the DB if necessary
        if self.is_any_file_updated(L0_file_path):
        
            L0_header_data    = self.extract_kwd(L0_file_path, self.L0_header_keyword_types) 
            D2_header_data    = self.extract_kwd(D2_file_path, self.D2_header_keyword_types) 
            L1_header_data    = self.extract_kwd(L1_file_path, self.L1_header_keyword_types) 
            L2_header_data    = self.extract_kwd(L2_file_path, self.L2_header_keyword_types) 
            L2_RV_header_data = self.extract_kwd(L2_file_path, self.L2_RV_header_keyword_types) 
            L0_telemetry      = self.extract_telemetry(L0_file_path, self.L0_telemetry_types)

            header_data = {**L0_header_data, 
                           **D2_header_data, 
                           **L1_header_data, 
                           **L2_header_data, 
                           **L2_RV_header_data, 
                           **L0_telemetry
                          }
            header_data['ObsID'] = (L0_filename.split('.fits')[0])
            header_data['datecode'] = get_datecode(L0_filename)  
            header_data['L0_filename'] = L0_filename
            header_data['D2_filename'] = f"{base_filename}_2D.fits"
            header_data['L1_filename'] = f"{base_filename}_L1.fits"
            header_data['L2_filename'] = f"{base_filename}_L2.fits"
            header_data['L0_header_read_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header_data['D2_header_read_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header_data['L1_header_read_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header_data['L2_header_read_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # To-do: Data quality checks: DATE-MID not None
            #                             ObsID matches DATE-MID (a few observations have bad times)
        
            # Insert into database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA cache_size = -2000000;")
            columns = ', '.join([f'"{key}"' for key in header_data.keys()])
            placeholders = ', '.join(['?'] * len(header_data))
            insert_query = f'INSERT OR REPLACE INTO kpfdb ({columns}) VALUES ({placeholders})'
            cursor.execute(insert_query, tuple(header_data.values()))
            conn.commit()
            conn.close()


    def ingest_batch_observation(self, batch):
        """
        Ingest a set of observations into the database.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        batch_data = []
        for file_path in batch:
            base_filename = os.path.basename(file_path).split('.fits')[0]
            L0_filename = base_filename.split('.fits')[0]
            L0_filename = L0_filename.split('/')[-1]
            D2_filename  = f"{L0_filename.replace('L0', '2D')}"
            L1_filename  = f"{L0_filename.replace('L0', 'L1')}"
            L2_filename  = f"{L0_filename.replace('L0', 'L2')}"

            L0_file_path = file_path
            D2_file_path = file_path.replace('L0', '2D').replace('.fits', '_2D.fits')
            L1_file_path = file_path.replace('L0', 'L1').replace('.fits', '_L1.fits')
            L2_file_path = file_path.replace('L0', 'L2').replace('.fits', '_L2.fits')

            # If any associated file has been updated, proceed
            if self.is_any_file_updated(L0_file_path):
                L0_header_data = self.extract_kwd(L0_file_path,       self.L0_header_keyword_types, extension='PRIMARY')   
                D2_header_data = self.extract_kwd(D2_file_path,       self.D2_header_keyword_types, extension='PRIMARY')   
                L1_header_data = self.extract_kwd(L1_file_path,       self.L1_header_keyword_types, extension='PRIMARY')   
                L2_header_data = self.extract_kwd(L2_file_path,       self.L2_header_keyword_types, extension='PRIMARY')   
                L2_header_data = self.extract_kwd(L2_file_path,       self.L2_RV_header_keyword_types, extension='RV')   
                L0_telemetry   = self.extract_telemetry(L0_file_path, self.L0_telemetry_types) 

                header_data = {**L0_header_data, **D2_header_data, **L1_header_data, **L2_header_data, **L0_telemetry}
                header_data['ObsID'] = base_filename
                header_data['datecode'] = get_datecode(base_filename)
                header_data['L0_filename'] = os.path.basename(L0_file_path)
                header_data['D2_filename'] = os.path.basename(D2_file_path)
                header_data['L1_filename'] = os.path.basename(L1_file_path)
                header_data['L2_filename'] = os.path.basename(L2_file_path)
                header_data['L0_header_read_time'] = now_str
                header_data['D2_header_read_time'] = now_str
                header_data['L1_header_read_time'] = now_str
                header_data['L2_header_read_time'] = now_str
                header_data['Source'] = self.get_source(L0_header_data)
    
                batch_data.append(header_data)

        # Perform batch insertion/update in the database
        if batch_data != []:
            columns = ', '.join([f'"{key}"' for key in batch_data[0].keys()])
            placeholders = ', '.join(['?'] * len(batch_data[0]))
            insert_query = f'INSERT OR REPLACE INTO kpfdb ({columns}) VALUES ({placeholders})'
            data_tuples = [tuple(data.values()) for data in batch_data]
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA cache_size = -2000000;")
            cursor.executemany(insert_query, data_tuples)
            conn.commit()
            conn.close()


    def get_source(self, L0_dict):
        """
        Returns the name of the source in a spectrum.  For stellar observations, this 
        it returns 'Star'.  For calibration spectra, this is the lamp name 
        (ThAr, UNe, LFC, etalon) or bias/dark.  
        Flats using KPF's regular fibers are distinguished from wide flats.                           

        Returns:
            the source/image name
            possible values: 'Bias', 'Dark', 'Flat', 'Wide Flat', 
                             'LFC', 'Etalon', 'ThAr', 'UNe',
                             'Sun', 'Star'
        """

        try: 
            if (('ELAPSED' in L0_dict) and 
                ((L0_dict['IMTYPE'] == 'Bias') or (L0_dict['ELAPSED'] == 0))):
                    return 'Bias'
            elif L0_dict['IMTYPE'] == 'Dark':
                return 'Dark' 
            elif L0_dict['FFFB'].strip().lower() == 'yes':
                    return 'Wide Flat' # Flatfield Fiber (wide flats)
            elif L0_dict['IMTYPE'].strip().lower() == 'flatlamp':
                 if 'brdband' in L0_dict['OCTAGON'].strip().lower():
                    return 'Flat' # Flat through regular fibers
            elif L0_dict['IMTYPE'].strip().lower() == 'arclamp':
                if 'lfc' in L0_dict['OCTAGON'].strip().lower():
                    return 'LFC'
                if 'etalon' in L0_dict['OCTAGON'].strip().lower():
                    return 'Etalon'
                if 'th_' in L0_dict['OCTAGON'].strip().lower():
                    return 'ThAr'
                if 'u_' in L0_dict['OCTAGON'].strip().lower():
                    return 'UNe'
            elif ((L0_dict['TARGNAME'].strip().lower() == 'sun') or 
                  (L0_dict['TARGNAME'].strip().lower() == 'socal')):
                return 'Sun' # SoCal
            if ('OBJECT' in L0_dict) and ('FIUMODE' in L0_dict):
                if (L0_dict['FIUMODE'] == 'Observing'):
                    return 'Star'
        except:
            return 'Unknown'
        
    
    def extract_kwd(self, file_path, keyword_types, extension='PRIMARY'):
        """
        Extract keywords from keyword_types.keys from a L0/2D/L1/L2 file.
        """
        header_data = {key: None for key in keyword_types.keys()}
        if os.path.isfile(file_path):
            try:
                with fits.open(file_path, memmap=True) as hdul:
                    header = hdul[extension].header
                    # Use set intersection to find common keys
                    common_keys = set(header.keys()) & header_data.keys()
                    for key in common_keys:
                        header_data[key] = header[key]
            except:
            	self.logger.info("Bad file: " + file_path)
        return header_data


    def extract_telemetry(self, file_path, keyword_types):
        """
        Extract telemetry from the 'TELEMETRY' extension in an KPF L0 file.
        """
        try:
            df_telemetry = Table.read(file_path, format='fits', hdu='TELEMETRY').to_pandas()
            df_telemetry = df_telemetry[['keyword', 'average']]
        except:
            self.logger.info('Bad TELEMETRY extension in: ' + file_path)
            telemetry_dict = {key: None
                              for key in keyword_types}
            return telemetry_dict
        df_telemetry = df_telemetry.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        df_telemetry.replace({'-nan': np.nan, 'nan': np.nan, -999: np.nan}, inplace=True)
        df_telemetry.set_index("keyword", inplace=True)
        telemetry_dict = {key: float(df_telemetry.at[key, 'average']) if key in df_telemetry.index else None 
                          for key in keyword_types}
        return telemetry_dict


    def clean_df(self, df):
        """
        Remove known outliers from a dataframe.
        """
        # Hallway temperature
        if 'kpfmet.TEMP' in df.columns:
            df = df.loc[df['kpfmet.TEMP'] > 15]
        # Fiber temperatures
        kwrds = ['kpfmet.SIMCAL_FIBER_STG', 'kpfmet.SIMCAL_FIBER_STG']
        for key in kwrds:
            if key in df.columns:
                df = df.loc[df[key] > 0]
        # Dark Current
        kwrds = ['FLXCOLLG', 'FLXECHG', 'FLXREG1G', 'FLXREG2G', 'FLXREG3G', 'FLXREG4G', 
                 'FLXREG5G', 'FLXREG6G', 'FLXCOLLR', 'FLXECHR', 'FLXREG1R', 'FLXREG2R', 
                 'FLXREG3R', 'FLXREG4R', 'FLXREG5R', 'FLXREG6R']
        for key in kwrds:
            if key in df.columns:
                df = df.loc[df[key] < 10000]
        return df


    def is_notebook(self):
        """
        Determine if the code is being executed in a Jupyter Notebook.  
        This is useful for tqdm.
        """
        try:
            from IPython import get_ipython
            if 'IPKernelApp' not in get_ipython().config:  # Notebook not running
                return False
        except (ImportError, AttributeError):
            return False  # IPython not installed
        return True


    def is_any_file_updated(self, L0_file_path):
        """
        Determines if any file from the L0/2D/L1/L2 set has been updated since the last 
        noted modification in the database.  Returns True if is has been modified.
        """
        L0_filename = L0_file_path.split('/')[-1]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA cache_size = -2000000;")
        query = f'SELECT L0_header_read_time, D2_header_read_time, L1_header_read_time, L2_header_read_time FROM kpfdb WHERE L0_filename = "{L0_filename}"'
        cursor.execute(query)
        result = cursor.fetchone()
        conn.close()
    
        if not result:  
            return True # no record in database

        try:
            L0_file_mod_time = datetime.fromtimestamp(os.path.getmtime(L0_file_path)).strftime("%Y-%m-%d %H:%M:%S")
        except FileNotFoundError:
            L0_file_mod_time = '1000-01-01 01:01'
        if L0_file_mod_time > result[0]:
            return True # L0 file was modified

        D2_file_path = f"{L0_file_path.replace('L0', '2D')}"
        D2_file_path = f"{D2_file_path.replace('.fits', '_2D.fits')}"
        try:
            D2_file_mod_time = datetime.fromtimestamp(os.path.getmtime(D2_file_path)).strftime("%Y-%m-%d %H:%M:%S")
        except FileNotFoundError:
            D2_file_mod_time = '1000-01-01 01:01'
        if D2_file_mod_time > result[0]:
            return True # 2D file was modified

        L1_file_path = f"{D2_file_path.replace('2D', 'L1')}"
        try:
            L1_file_mod_time = datetime.fromtimestamp(os.path.getmtime(L1_file_path)).strftime("%Y-%m-%d %H:%M:%S")
        except FileNotFoundError:
            L1_file_mod_time = '1000-01-01 01:01'
        if L1_file_mod_time > result[0]:
            return True # L1 file was modified

        L2_file_path = f"{L0_file_path.replace('L1', 'L2')}"
        try:
            L2_file_mod_time = datetime.fromtimestamp(os.path.getmtime(L2_file_path)).strftime("%Y-%m-%d %H:%M:%S")
        except FileNotFoundError:
            L2_file_mod_time = '1000-01-01 01:01'
        if L2_file_mod_time > result[0]:
            return True # L2 file was modified
                    
        return False # DB modification times are all more recent than file modification times
           

    def print_db_status(self):
        """
        Prints a brief summary of the database status.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM kpfdb')
        nrows = cursor.fetchone()[0]
        cursor.execute('PRAGMA table_info(kpfdb)')
        ncolumns = len(cursor.fetchall())
        cursor.execute('SELECT MAX(MAX(L0_header_read_time),MAX(L1_header_read_time)) FROM kpfdb')
        most_recent_read_time = cursor.fetchone()[0]
        cursor.execute('SELECT MIN(datecode) FROM kpfdb')
        earliest_datecode = cursor.fetchone()[0]
        cursor.execute('SELECT MAX(datecode) FROM kpfdb')
        latest_datecode = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(DISTINCT datecode) FROM kpfdb')
        unique_datecodes_count = cursor.fetchone()[0]
        conn.close()
        self.logger.info(f"Summary: {nrows} obs x {ncolumns} cols over {unique_datecodes_count} days in {earliest_datecode}-{latest_datecode}; updated {most_recent_read_time}")


    def display_dataframe_from_db(self, columns, only_object=None, object_like=None, 
                                  on_sky=None, start_date=None, end_date=None):
        """
        TO-DO: should this method just call display_dataframe_from_db()?
        
        Prints a pandas dataframe of attributes (specified by column names) for all 
        observations in the DB. The query can be restricted to observations matching a 
        particular object name(s).  The query can also be restricted to observations 
        that are on-sky/off-sky and after start_date and/or before end_date. 

        Args:
            columns (string or list of strings) - database columns to query
            only_object (string or list of strings) - object names to include in query
            object_like (string or list of strings) - partial object names to search for
            on_sky (True, False, None) - using FIUMODE, select observations that are on-sky (True), off-sky (False), or don't care (None)
            start_date (datetime object) - only return observations after start_date
            end_date (datetime object) - only return observations after end_date
            false (boolean) - if True, prints the SQL query

        Returns:
            A printed dataframe of the specified columns matching the constraints.
        """
        conn = sqlite3.connect(self.db_path)
        
        # Enclose column names in double quotes
        quoted_columns = [f'"{column}"' for column in columns]
        query = f"SELECT {', '.join(quoted_columns)} FROM kpfdb"

        # Append WHERE clauses
        where_queries = []
        if only_object is not None:
            only_object = [f"OBJECT = '{only_object}'"]
            or_objects = ' OR '.join(only_object)
            where_queries.append(f'({or_objects})')
        if object_like is not None:
            object_like = [f"OBJECT LIKE '%{obj}%'" for obj in object_like]
            or_objects = ' OR '.join(object_like)
            where_queries.append(f'({or_objects})')
        if on_sky is not None:
            if on_sky == True:
                where_queries.append(f"FIUMODE = 'Observing'")
            if on_sky == False:
                where_queries.append(f"FIUMODE = 'Calibration'")
        if start_date is not None:
            start_date_txt = start_date.strftime('%Y-%m-%d %H:%M:%S')
            where_queries.append(f' ("DATE-MID" > "{start_date_txt}")')
        if end_date is not None:
            end_date_txt = end_date.strftime('%Y-%m-%d %H:%M:%S')
            where_queries.append(f' ("DATE-MID" < "{end_date_txt}")')
        if where_queries != []:
            query += " WHERE " + ' AND '.join(where_queries)
    
        # Execute query
        df = pd.read_sql_query(query, conn, params=(only_object,) if only_object is not None else None)
        conn.close()
        print(df)

   
    def dataframe_from_db(self, columns, 
                          start_date=None, end_date=None, 
                          only_object=None, object_like=None, 
                          on_sky=None, not_junk=None, 
                          verbose=False):
        """
        Returns a pandas dataframe of attributes (specified by column names) for all 
        observations in the DB. The query can be restricted to observations matching a 
        particular object name(s).  The query can also be restricted to observations 
        that are on-sky/off-sky and after start_date and/or before end_date. 

        Args:
            columns (string or list of strings) - database columns to query
            only_object (string) - object name to include in query
            object_like (string) - partial object name to search for
            on_sky (True, False, None) - using FIUMODE, select observations that are on-sky (True), off-sky (False), or don't care (None)
            start_date (datetime object) - only return observations after start_date
            end_date (datetime object) - only return observations after end_date
            false (boolean) - if True, prints the SQL query

        Returns:
            Pandas dataframe of the specified columns matching the constraints.
        """
        
        conn = sqlite3.connect(self.db_path)
        
        # Enclose column names in double quotes
        quoted_columns = [f'"{column}"' for column in columns]
        query = f"SELECT {', '.join(quoted_columns)} FROM kpfdb"

        # Append WHERE clauses
        where_queries = []
        if only_object is not None:
            only_object = convert_to_list_if_array(only_object)
            if isinstance(only_object, str):
                only_object = [only_object]
            object_queries = [f"OBJECT = '{obj}'" for obj in only_object]
            or_objects = ' OR '.join(object_queries)
            where_queries.append(f'({or_objects})')
        if object_like is not None: 
            object_like = [f"OBJECT LIKE '%{object_like}%'"]
            or_objects = ' OR '.join(object_like)
            where_queries.append(f'({or_objects})')
        if not_junk is not None:
            if not_junk == True:
                where_queries.append(f"NOTJUNK = 1")
            if not_junk == False:
                where_queries.append(f"NOTJUNK = 0")
        if on_sky is not None:
            if on_sky == True:
                where_queries.append(f"FIUMODE = 'Observing'")
            if on_sky == False:
                where_queries.append(f"FIUMODE = 'Calibration'")
        if start_date is not None:
            start_date_txt = start_date.strftime('%Y-%m-%d %H:%M:%S')
            where_queries.append(f' ("DATE-MID" > "{start_date_txt}")')
        if end_date is not None:
            end_date_txt = end_date.strftime('%Y-%m-%d %H:%M:%S')
            where_queries.append(f' ("DATE-MID" < "{end_date_txt}")')
        if where_queries != []:
            query += " WHERE " + ' AND '.join(where_queries)

        if verbose:
            print('query = ' + query)

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df


    def map_data_type_to_sql(self, dtype):
        """
        Function to map the data types specified in get_keyword_types to sqlite3
        data types.
        """
        return {
            'int': 'INTEGER',
            'float': 'REAL',
            'bool': 'BOOLEAN',
            'datetime': 'TEXT',  # SQLite does not have a native datetime type
            'string': 'TEXT'
        }.get(dtype, 'TEXT')


    def get_keyword_types(self, level):
        """
        Returns a dictionary of the data types for keywords at the L0/2D/L1/L2 or 
        L0_telemetry level.
        """
        
        # L0 PRIMARY header    
        if level == 'L0':
            keywords_csv='/code/KPF-Pipeline/static/tsdb_keywords/l0_primary_keywords.csv'
            df_keywords = pd.read_csv(keywords_csv, delimiter='|', dtype=str)
            keyword_types = dict(zip(df_keywords['keyword'], df_keywords['datatype']))
             
        # 2D PRIMARY header    
        elif level == '2D':
            keywords_csv='/code/KPF-Pipeline/static/tsdb_keywords/d2_primary_keywords.csv'
            df_keywords = pd.read_csv(keywords_csv, delimiter='|', dtype=str)
            keyword_types = dict(zip(df_keywords['keyword'], df_keywords['datatype']))

        # L1 PRIMARY header    
        elif level == 'L1':
            keywords_csv='/code/KPF-Pipeline/static/tsdb_keywords/l1_primary_keywords.csv'
            df_keywords = pd.read_csv(keywords_csv, delimiter='|', dtype=str)
            keyword_types = dict(zip(df_keywords['keyword'], df_keywords['datatype']))
        
        # L2 PRIMARY header    
        elif level == 'L2':
            keywords_csv='/code/KPF-Pipeline/static/tsdb_keywords/l2_primary_keywords.csv'
            df_keywords = pd.read_csv(keywords_csv, delimiter='|', dtype=str)
            keyword_types = dict(zip(df_keywords['keyword'], df_keywords['datatype']))

        # L0 TELEMETRY extension
        elif level == 'L0_telemetry':
            keywords_csv='/code/KPF-Pipeline/static/tsdb_keywords/l0_telemetry_keywords.csv'
            df_keywords = pd.read_csv(keywords_csv, delimiter='|', dtype=str)
            keyword_types = dict(zip(df_keywords['keyword'], df_keywords['datatype']))

#        # L2 RV extension
#        elif level == 'L2_RV':
#            keyword_types = {
#                'ABCD1234': 'string', #placeholder for now
#            }

        # L2 RV extension    
        elif level == 'L2_RV_header':
            keywords_csv='/code/KPF-Pipeline/static/tsdb_keywords/l2_rv_keywords.csv'
            df_keywords = pd.read_csv(keywords_csv, delimiter='|', dtype=str)
            keyword_types = dict(zip(df_keywords['keyword'], df_keywords['datatype']))

        else:
            keyword_types = {}
        
        return keyword_types


    def plot_nobs_histogram(self, interval='full', date=None, exclude_junk=False, 
                            plot_junk=False, plot_source=False, 
                            fig_path=None, show_plot=False):
        """
        Plot a histogram of the number of observations per day or hour, optionally color-coded by 'NOTJUNK' or 'Source'.
    
        Args:
            interval (string)      - time interval over which plot is made
                                     default: 'full',
                                     possible values: 'full', 'decade', 'year', 'month', 'day'
            date (string)          - one date in the interval (format: 'YYYYMMDD' or 'YYYY-MM-DD')
            exclude_junk (boolean) - if True, observations with NOTJUNK=False are removed
            plot_junk (boolean)    - if True, will color-code based on 'NOTJUNK' column
            plot_source (boolean)  - if True, will color-code based on 'Source' column
            fig_path (string)      - set to the path for the file to be generated
            show_plot (boolean)    - show the plot in the current environment
            
        Returns:
            PNG plot in fig_path or shows the plot in the current environment
            (e.g., in a Jupyter Notebook).
        
        To-do: 
        	Add plotting of QC tests
        """

        # Define the source categories and their colors
        source_order = ['Bias', 'Dark', 'Flat', 'Wide Flat', 'LFC', 'Etalon', 'ThAr', 'UNe', 'Sun', 'Star']
        source_colors = {
            'Bias': 'gray',
            'Dark': 'black',
            'Flat': 'gainsboro',
            'Wide Flat': 'silver',
            'LFC': 'gold',
            'Etalon': 'chocolate',
            'ThAr': 'orange',
            'UNe': 'forestgreen',
            'Sun': 'cornflowerblue',
            'Star': 'royalblue'
        }
    
        # Load data
        df = self.dataframe_from_db(['DATE-BEG', 'DATE-END', 'NOTJUNK', 'Source'])
        df['DATE-BEG'] = pd.to_datetime(df['DATE-BEG'], errors='coerce')
        df['DATE-END'] = pd.to_datetime(df['DATE-END'], errors='coerce')
        df = df.dropna(subset=['DATE-BEG'])
        df = df.dropna(subset=['DATE-END'])

        if exclude_junk:
            df = df[df['NOTJUNK'] == 1.0]      
    
        # Parse the date string into a timestamp
        if date is not None:
            date = pd.to_datetime(date, format='%Y%m%d', errors='coerce')  # Handle YYYYMMDD format
            if pd.isna(date):
                date = pd.to_datetime(date, errors='coerce')  # Handle other formats like YYYY-MM-DD
            if pd.isna(date):
                raise ValueError(f"Invalid date format: {date}")
    
        # Filter data based on interval
        if interval == 'decade':
            start_date = pd.Timestamp(f"{date.year // 10 * 10}-01-01")
            end_date = pd.Timestamp(f"{date.year // 10 * 10 + 9}-12-31")
            df = df[(df['DATE-BEG'] >= start_date) & (df['DATE-BEG'] <= end_date)]
            df['DATE'] = df['DATE-BEG'].dt.date
            entry_counts = df['DATE'].value_counts().sort_index()
    
            major_locator = YearLocator()
            major_formatter = DateFormatter("%Y")
            minor_locator = None
            column_to_count = 'DATE'
            plot_title = f"Observations (Decade: {start_date.year}-{end_date.year})"
    
        elif interval == 'year':
            start_date = pd.Timestamp(f"{date.year}-01-01")
            end_date = pd.Timestamp(f"{date.year}-12-31")
            df = df[(df['DATE-BEG'] >= start_date) & (df['DATE-BEG'] <= end_date)]
            df['DATE'] = df['DATE-BEG'].dt.date
            entry_counts = df['DATE'].value_counts().sort_index()
    
            major_locator = MonthLocator()
            major_formatter = DateFormatter("%b")  # Format ticks as month names (Jan, Feb, etc.)
            minor_locator = None
            column_to_count = 'DATE'
            plot_title = f"Observations (Year: {date.year})"
    
        elif interval == 'month':
            start_date = pd.Timestamp(f"{date.year}-{date.month:02d}-01")
            end_date = (start_date + pd.offsets.MonthEnd(0) + timedelta(days=1) - timedelta(seconds=0.1))
            df = df[(df['DATE-BEG'] >= start_date) & (df['DATE-BEG'] <= end_date)]
            df['DAY'] = df['DATE-BEG'].dt.day
            entry_counts = df['DAY'].value_counts().sort_index()
    
            major_locator = DayLocator()
            major_formatter = lambda x, _: f"{int(x)}" if 1 <= x <= end_date.day else ""
            minor_locator = None
            column_to_count = 'DAY'
            plot_title = f"Observations (Month: {date.year}-{date.month:02d})"
    
        elif interval == 'day':
            start_date = pd.Timestamp(f"{date.year}-{date.month:02d}-{date.day:02d}")
            end_date = start_date + timedelta(days=1)# - timedelta(seconds=1)
            df = df[(df['DATE-BEG'] >= start_date) & (df['DATE-BEG'] <= end_date)]
            df['HOUR'] = df['DATE-BEG'].dt.hour
            entry_counts = df['HOUR'].value_counts().sort_index()
    
            # Reindex for full hourly range
            hourly_range = pd.Index(range(24))  # 0 through 23 hours
            entry_counts = entry_counts.reindex(hourly_range, fill_value=0)
    
            major_locator = plt.MultipleLocator(1)  # Tick every hour
            major_formatter = lambda x, _: f"{int(x):02d}:00" if 0 <= x <= 23 else ""
            minor_locator = None
            column_to_count = 'HOUR'
            plot_title = f"Observations (Day: {date.year}-{date.month:02d}-{date.day:02d})"
        else:
            # Default: 'full' interval
            df['DATE'] = df['DATE-BEG'].dt.date
            entry_counts = df['DATE'].value_counts().sort_index()
    
            major_locator = AutoDateLocator()
            major_formatter = DateFormatter("%Y-%m")
            minor_locator = None
            column_to_count = 'DATE'
            start_date = df['DATE'].min()
            end_date = df['DATE'].max()
            plot_title = f"Observations (Full Range: {start_date.year}-{start_date.month:02d}-{start_date.day:02d} - {end_date.year}-{end_date.month:02d}-{end_date.day:02d})"
    
            # Ensure full date range is displayed
            full_range = pd.date_range(start=start_date, end=end_date, freq='D')
            entry_counts = entry_counts.reindex(full_range, fill_value=0)
    
        # Ensure all index values are datetime.date for consistent processing
        if isinstance(entry_counts.index, pd.DatetimeIndex):
            entry_counts.index = entry_counts.index.map(lambda x: x.date())
    
        # Adjust bar positions and plot edges for proper alignment
        bar_positions = entry_counts.index.map(
            lambda x: x.toordinal() if type(x) == type(datetime(2024, 1, 1, 1, 1, 1)) else x
        )
        if interval == 'decade':
            x_min = datetime(bar_positions[0].year // 10 * 10, 1, 1)
            x_max = datetime(bar_positions[0].year // 10 * 10 + 10, 1, 1)
        elif interval == 'year':
            x_min = datetime(bar_positions[0].year, 1, 1)
            x_max = datetime(bar_positions[0].year+1, 1, 1)
        elif interval == 'month':
            x_min = 0.5
            x_max = (datetime(date.year, date.month % 12 + 1, 1) - datetime(date.year, date.month, 1)).days + 0.5
        elif interval == 'day':
            x_min = 0 
            x_max = 24
        else: # Default: 'full' interval
            x_min = bar_positions.min() 
            x_max = bar_positions.max() 

        if plot_source and interval == 'day':
            plt.figure(figsize=(12, 4))
        else:
            plt.figure(figsize=(15, 4))
    
        # Plot stacked source data
        if plot_source:
            bottom_values = [0] * len(bar_positions)
            legend_labels = []  # Store labels with counts for the legend
            for source in source_order:
                source_counts = df[df['Source'] == source][column_to_count].value_counts().sort_index()
                source_counts = source_counts.reindex(entry_counts.index, fill_value=0)
    
                if interval == 'day':
                    plt.bar(bar_positions, source_counts.values, width=1, align='edge',
                            color=source_colors[source], label=source, bottom=bottom_values, zorder=3)
                else: 
                    plt.bar(bar_positions, source_counts.values, width=1, align='center',
                            color=source_colors[source], label=source, bottom=bottom_values, zorder=3)
                bottom_values = [b + s for b, s in zip(bottom_values, source_counts.values)]

                # Add source label with count for 'day' interval
                if interval in ['month', 'day']:
                    total_source_count = source_counts.sum()
                    legend_labels.append(f"{source} ({int(total_source_count)})")
                else:
                    legend_labels.append(source)

            # Place legend outside of the plot on the right
            handles, _ = plt.gca().get_legend_handles_labels()
            handles = handles[::-1]  # Reverse handles to match legend_labels order
            legend_labels = legend_labels[::-1]  # Reverse labels for proper order

            plt.legend(
                handles, legend_labels,  # Use updated labels
                title="Sources",
                loc='center left',
                bbox_to_anchor=(1.01, 0.5),  # Adjust legend position (to the right of the plot)
                fontsize=10
            )
            plt.gcf().set_size_inches(15, 4)  # Increase the figure width
        elif plot_junk:
            notjunk_counts = df[df['NOTJUNK'] == True][column_to_count].value_counts().sort_index()
            junk_counts    = df[df['NOTJUNK'] == False][column_to_count].value_counts().sort_index()
            notjunk_counts = notjunk_counts.reindex(entry_counts.index, fill_value=0)
            junk_counts    = junk_counts.reindex(entry_counts.index, fill_value=0)

            if interval == 'day':
                plt.bar(bar_positions, notjunk_counts.values, width=1, align='edge', color='green', label='Not Junk', zorder=3)
                plt.bar(bar_positions, junk_counts.values,    width=1, align='edge', color='red',   label='Junk',     zorder=3, bottom=notjunk_counts.values)
            else: 
                plt.bar(bar_positions, notjunk_counts.values, width=1, align='center', color='green', label='Not Junk', zorder=3)
                plt.bar(bar_positions, junk_counts.values,    width=1, align='center', color='red',   label='Junk',     zorder=3, bottom=notjunk_counts.values)

            plt.legend()
        else:
            if interval == 'day':
                plt.bar(bar_positions, entry_counts.values, width=1, align='edge', zorder=3)
            else: 
                plt.bar(bar_positions, entry_counts.values, width=1, align='center', zorder=3)

        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Number of Observations", fontsize=14)
        plt.title(plot_title, fontsize=14)
    
        ax = plt.gca()
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_formatter)
        ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)
        if minor_locator:
            ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(visible=True, which='major', axis='both', linestyle='--', color='lightgray', zorder=1)
        ax.set_axisbelow(True)
        ax.set_xlim(x_min, x_max)
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust plot area to leave space for the legend
    
        # Add black box around the axes
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
            spine.set_zorder(4)
            spine.set_visible(True)
    
        # Save or show the plot
        if fig_path is not None:
            plt.savefig(fig_path, dpi=300, facecolor='w')
        if show_plot:
            plt.show()
        plt.close('all')


    def plot_time_series_multipanel(self, panel_arr, start_date=None, end_date=None, 
                                    clean=False, fig_path=None, show_plot=False, 
                                    log_savefig_timing=True):
        """
        Generate a multi-panel plot of data in a KPF DB.  The data to be plotted and 
        attributes are stored in an array of dictionaries called 'panel_arr'.  
        The method plot_standard_time_series() provides several examples of using 
        this method.

        Args:
            panel_dict (array of dictionaries) - each dictionary in the array has keys:
                panelvars: a dictionary of matplotlib attributes including:
                    ylabel - text for y-axis label
                paneldict: a dictionary containing:
                    col: name of DB column to plot
                    plot_type: 'plot' (points with connecting lines), 
                               'scatter' (points), 
                               'step' (steps), 
                               'state' (for non-floats, like DRPTAG)
                    plot_attr: a dictionary containing plot attributes for a scatter plot, 
                        including 'label', 'marker', 'color'
                    not_junk: if set to 'True', only files with NOTJUNK=1 are included; 
                              if set to 'False', only files with NOTJUNK=0 are included
                    on_sky: if set to 'True', only on-sky observations will be included; 
                            if set to 'False', only calibrations will be included
                    only_object (not implemented yet): if set, only object names in the keyword's value will be queried
                    object_like (not implemented yet): if set, partial object names matching the keyword's value will be queried
            start_date (datetime object) - start date for plot
            end_date (datetime object) - end date for plot
            fig_path (string) - set to the path for the file to be generated
            show_plot (boolean) - show the plot in the current environment
            These are now part of the dictionaries:
                only_object (string or list of strings) - object names to include in query
                object_like (string or list of strings) - partial object names to search for
                on_sky (True, False, None) - using FIUMODE, select observations that are on-sky (True), off-sky (False), or don't care (None)

        Returns:
            PNG plot in fig_path or shows the plot it the current environment
            (e.g., in a Jupyter Notebook).
        """

        def num_fmt(n: float, sf: int = 3) -> str:
            """
            Returns number as a formatted string with specified number of significant figures
            :param n: number to format
            :param sf: number of sig figs in output
            """
            r = f'{n:.{sf}}'  # use existing formatter to get to right number of sig figs
            if 'e' in r:
                exp = int(r.split('e')[1])
                base = r.split('e')[0]
                r = base + '0' * (exp - sf + 2)
            return r
    
        def format_func(value, tick_number):
            """ For formatting of log plots """
            return num_fmt(value, sf=2)

        if start_date == None:
            start_date = min(df['DATE-MID'])
        if end_date == None:
            end_date = max(df['DATE-MID'])
        npanels = len(panel_arr)
        unique_cols = set()
        unique_cols.add('DATE-MID')
        unique_cols.add('FIUMODE')
        unique_cols.add('OBJECT')
        for panel in panel_arr:
            for d in panel['panelvars']:
                col_value = d['col']
                unique_cols.add(col_value)
        # add this logic
        #if 'only_object' in thispanel['paneldict']:
        #if 'object_like' in thispanel['paneldict']:

        fig, axs = plt.subplots(npanels, 1, sharex=True, figsize=(15, npanels*2.5+1), tight_layout=True)
        if npanels == 1:
            axs = [axs]  # Make axs iterable even when there's only one panel
        if npanels > 1:
            plt.subplots_adjust(hspace=0)
        #plt.tight_layout() # this caused a core dump in scripts/generate_time_series_plots.py

        for p in np.arange(npanels):
            thispanel = panel_arr[p]            
            not_junk = None
            if 'not_junk' in thispanel['paneldict']:
                if (thispanel['paneldict']['not_junk']).lower() == 'true':
                    not_junk = True
                elif (thispanel['paneldict']['not_junk']).lower() == 'false':
                    not_junk = False
            only_object = None
            if 'only_object' in thispanel['paneldict']:
                only_object = thispanel['paneldict']['only_object']
            object_like = None
#            if 'object_like' in thispanel['paneldict']:
#                if (thispanel['paneldict']['object_like']).lower() == 'true':
#                    object_like = True
#                elif (thispanel['paneldict']['object_like']).lower() == 'false':
#                    object_like = False
            df = self.dataframe_from_db(unique_cols, 
                                        start_date=start_date, 
                                        end_date=end_date, 
                                        not_junk=not_junk, 
                                        only_object=only_object, 
                                        object_like=object_like,
                                        verbose=False)
            df['DATE-MID'] = pd.to_datetime(df['DATE-MID']) # move this to dataframe_from_db ?
            df = df.sort_values(by='DATE-MID')
            if clean:
                df = self.clean_df(df)

            if 'on_sky' in thispanel['paneldict']:
                if (thispanel['paneldict']['on_sky']).lower() == 'true':
                    df = df[df['FIUMODE'] == 'Observing']
                elif (thispanel['paneldict']['on_sky']).lower() == 'false':
                    df = df[df['FIUMODE'] == 'Calibration']

            thistitle = ''
            if abs((end_date - start_date).days) <= 1.2:
                t = [(date - start_date).total_seconds() / 3600 for date in df['DATE-MID']]
                xtitle = 'Hours since ' + start_date.strftime('%Y-%m-%d %H:%M') + ' UT'
                if 'title' in thispanel['paneldict']:
                    thistitle = str(thispanel['paneldict']['title']) + ": " + start_date.strftime('%Y-%m-%d %H:%M') + " to " + end_date.strftime('%Y-%m-%d %H:%M')
                axs[p].set_xlim(0, (end_date - start_date).total_seconds() / 3600)
                if 'narrow_xlim_daily' in thispanel['paneldict']:
                    if thispanel['paneldict']['narrow_xlim_daily'] == 'true':
                        if len(t) > 1:
                            axs[p].set_xlim(min(t), max(t))
                axs[p].xaxis.set_major_locator(ticker.MaxNLocator(nbins=12, min_n_ticks=4, prune=None))
            elif abs((end_date - start_date).days) <= 3:
                t = [(date - start_date).total_seconds() / 86400 for date in df['DATE-MID']]
                xtitle = 'Days since ' + start_date.strftime('%Y-%m-%d %H:%M') + ' UT'
                if 'title' in thispanel['paneldict']:
                    thistitle = thispanel['paneldict']['title'] + ": " + start_date.strftime('%Y-%m-%d %H:%M') + " to " + end_date.strftime('%Y-%m-%d %H:%M')
                axs[p].set_xlim(0, (end_date - start_date).total_seconds() / 86400)
                axs[p].xaxis.set_major_locator(ticker.MaxNLocator(nbins=12, min_n_ticks=4, prune=None))
            elif abs((end_date - start_date).days) < 32:
                t = [(date - start_date).total_seconds() / 86400 for date in df['DATE-MID']]
                xtitle = 'Days since ' + start_date.strftime('%Y-%m-%d %H:%M') + ' UT'
                if 'title' in thispanel['paneldict']:
                    thistitle = thispanel['paneldict']['title'] + ": " + start_date.strftime('%Y-%m-%d') + " to " + end_date.strftime('%Y-%m-%d')
                axs[p].set_xlim(0, (end_date - start_date).total_seconds() / 86400)
                axs[p].xaxis.set_major_locator(ticker.MaxNLocator(nbins=12, min_n_ticks=3, prune=None))
            else:
                t = df['DATE-MID'] # dates
                xtitle = 'Date'
                if 'title' in thispanel['paneldict']:
                    thistitle = thispanel['paneldict']['title'] + ": " + start_date.strftime('%Y-%m-%d') + " to " + end_date.strftime('%Y-%m-%d')
                axs[p].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                axs[p].xaxis.set_major_locator(ticker.MaxNLocator(7, prune=None))
            if p == npanels-1: 
                axs[p].set_xlabel(xtitle, fontsize=14)
                axs[0].set_title(thistitle, fontsize=14)
            if 'ylabel' in thispanel['paneldict']:
                axs[p].set_ylabel(thispanel['paneldict']['ylabel'], fontsize=14)
            axs[p].grid(color='lightgray')        
            if 'yscale' in thispanel['paneldict']:
                if thispanel['paneldict']['yscale'] == 'log':
                    formatter = FuncFormatter(format_func)  # this doesn't seem to be working
                    axs[p].minorticks_on()
                    axs[p].grid(which='major', axis='x', color='darkgray',  linestyle='-', linewidth=0.5)
                    axs[p].grid(which='both',  axis='y', color='lightgray', linestyle='-', linewidth=0.5)
                    axs[p].set_yscale('log')
                    axs[p].yaxis.set_minor_locator(plt.AutoLocator())
                    axs[p].yaxis.set_major_formatter(formatter)
            else:
                axs[p].grid(color='lightgray')        
            ylim=False
            if 'ylim' in thispanel['paneldict']:
                if type(ast.literal_eval(thispanel['paneldict']['ylim'])) == type((1,2)):
                    ylim = ast.literal_eval(thispanel['paneldict']['ylim'])
            makelegend = True
            if 'nolegend' in thispanel['paneldict']:
                if (thispanel['paneldict']['nolegend']).lower() == 'true':
                    makelegend = False
            subtractmedian = False
            if 'subtractmedian' in thispanel['paneldict']:
                if (thispanel['paneldict']['subtractmedian']).lower() == 'true':
                    subtractmedian = True
            nvars = len(thispanel['panelvars'])
            for i in np.arange(nvars):
                if 'plot_type' in thispanel['panelvars'][i]:
                    plot_type = thispanel['panelvars'][i]['plot_type']
                else:
                    plot_type = 'scatter'
                col_data = df[thispanel['panelvars'][i]['col']]
                col_data_replaced = col_data.replace('NaN', np.nan)
                col_data_replaced = col_data.replace('null', np.nan)
                if plot_type == 'state':
                    states = np.array(col_data_replaced)
                else:
                    data = np.array(col_data_replaced, dtype='float')
                plot_attributes = {}
                if plot_type != 'state':
                    if np.count_nonzero(~np.isnan(data)) > 0:
                        if subtractmedian:
                            data -= np.nanmedian(data)
                        if 'plot_attr' in thispanel['panelvars'][i]:
                            if 'label' in thispanel['panelvars'][i]['plot_attr']:
                                label = thispanel['panelvars'][i]['plot_attr']['label']
                                try:
                                    if makelegend:
                                        if len(~np.isnan(data)) > 0:
                                            median = np.nanmedian(data)
                                        else:
                                            median = 0.
                                        if len(~np.isnan(data)) > 2:
                                            std_dev = np.nanstd(data)
                                            if std_dev != 0 and not np.isnan(std_dev):
                                                decimal_places = max(1, 2 - int(np.floor(np.log10(abs(std_dev)))) - 1)
                                            else:
                                                decimal_places = 1
                                        else:
                                            decimal_places = 1
                                            std_dev = 0.
                                        formatted_median = f"{median:.{decimal_places}f}"
                                        #label += '\n' + formatted_median 
                                        if len(~np.isnan(data)) > 2:
                                            formatted_std_dev = f"{std_dev:.{decimal_places}f}"
                                            label += ' (' + formatted_std_dev 
                                            if 'unit' in thispanel['panelvars'][i]:
                                                label += ' ' + str(thispanel['panelvars'][i]['unit'])
                                            label += ' rms)'
                                except Exception as e:
                                    self.logger.error(e)
                            plot_attributes = thispanel['panelvars'][i]['plot_attr']
                            if 'label' in plot_attributes:
                                plot_attributes['label'] = label
                        else:
                           plot_attributes = {}
                if plot_type == 'scatter':
                    axs[p].scatter(t, data, **plot_attributes)
                if plot_type == 'plot':
                    axs[p].plot(t, data, **plot_attributes)
                if plot_type == 'step':
                    axs[p].step(t, data, **plot_attributes)
                if plot_type == 'state':
                    # Map states (e.g., DRP version number) to a numerical scale
                    states = np.array(['None' if s is None or s == 'NaN' else s for s in states])
                    states = [x for x in states if not (isinstance(x, (int, float, complex)) and np.isnan(x))] # remove NaN values
                    unique_states = sorted(set(states))  # Remove duplicates and sort
                    state_to_num = {state: i for i, state in enumerate(unique_states)}
                    mapped_states = [state_to_num[state] for state in states]
                    colors = plt.cm.jet(np.linspace(0, 1, len(unique_states)))
                    for state, color in zip(unique_states, colors):
                        indices = [i for i, s in enumerate(states) if s == state]
                        axs[p].scatter([t[i] for i in indices], [mapped_states[i] for i in indices], color=color, label=state)
                    axs[p].set_yticks(range(len(unique_states)))
                    axs[p].set_yticklabels(unique_states)
                axs[p].xaxis.set_tick_params(labelsize=10)
                axs[p].yaxis.set_tick_params(labelsize=10)
                if 'axhspan' in thispanel['paneldict']:
                    for key, axh in thispanel['paneldict']['axhspan'].items():
                        ymin = axh['ymin']
                        ymax = axh['ymax']
                        clr  = axh['color']
                        alp  = axh['alpha']
                        axs[p].axhspan(ymin, ymax, color=clr, alpha=alp)
                if makelegend:
                    if 'legend_frac_size' in thispanel['paneldict']:
                        legend_frac_size = thispanel['paneldict']['legend_frac_size']
                    else:
                        legend_frac_size = 0.20
                    axs[p].legend(loc='upper right', bbox_to_anchor=(1+legend_frac_size, 1))
                if ylim:
                    axs[p].set_ylim(ylim)
            axs[p].grid(color='lightgray')

        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    #xytext=(100, -32), 
                    xytext=(0, -32), 
                    textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)     

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=300, facecolor='w')
            if log_savefig_timing:
                self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_standard_time_series(self, plot_name, start_date=None, end_date=None, 
                                  clean=False, fig_path=None, show_plot=False):
        """
        Generate one of several standard time-series plots of KPF data.

        Args:
            plot_name (string): chamber_temp - 4-panel plot showing KPF chamber temperatures
                                abc - ...
            start_date (datetime object) - start date for plot
            end_date (datetime object) - end date for plot
            fig_path (string) - set to the path for a SNR vs. wavelength file
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it the current environment
            (e.g., in a Jupyter Notebook).
        """

        if plot_name == 'hallway_temp':
            dict1 = {'col': 'kpfmet.TEMP', 'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label':  'Hallway', 'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1]
            thispaneldict = {'ylabel': 'Hallway\n' + r' Temperature ($^{\circ}$C)',
                             'title': 'KPF Hallway Temperature', 
                             'legend_frac_size': 0.3}
            halltemppanel = {'panelvars': thispanelvars,
                             'paneldict': thispaneldict}
            panel_arr = [halltemppanel]
        
        elif plot_name == 'chamber_temp':
            dict1 = {'col': 'kpfmet.TEMP',              'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label':  'Hallway',              'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'kpfmet.GREEN_LN2_FLANGE',  'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': r'Green LN$_2$ Flng',    'marker': '.', 'linewidth': 0.5, 'color': 'darkgreen'}}
            dict3 = {'col': 'kpfmet.RED_LN2_FLANGE',    'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': r'Red LN$_2$ Flng',      'marker': '.', 'linewidth': 0.5, 'color': 'darkred'}}
            dict4 = {'col': 'kpfmet.CHAMBER_EXT_BOTTOM','plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': r'Chamber Ext Bot',      'marker': '.', 'linewidth': 0.5}}
            dict5 = {'col': 'kpfmet.CHAMBER_EXT_TOP',   'plot_type': 'plot',    'unit': 'K', 'plot_attr': {'label': r'Chamber Exterior Top', 'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1]
            thispaneldict = {'ylabel': 'Hallway\n' + r' Temperature ($^{\circ}$C)',
                             'legend_frac_size': 0.3}
            halltemppanel = {'panelvars': thispanelvars,
                             'paneldict': thispaneldict}

            thispanelvars2 = [dict2, dict3, dict4]
            thispaneldict2 = {'ylabel': 'Exterior\n' + r' Temperatures ($^{\circ}$C)',
                             'legend_frac_size': 0.3}
            halltemppanel2 = {'panelvars': thispanelvars2,
                              'paneldict': thispaneldict2}
            
            thispanelvars3 = [dict2, dict3, dict4]
            thispaneldict3 = {'ylabel': 'Exterior\n' + r'$\Delta$Temperature (K)',
                             'title': 'KPF Hallway Temperatures', 
                             'legend_frac_size': 0.3}
            halltemppanel3 = {'panelvars': thispanelvars3,
                              'paneldict': thispaneldict3}
            
            dict1 = {'col': 'kpfmet.BENCH_BOTTOM_BETWEEN_CAMERAS', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Bench$\downarrow$ Cams',   'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'kpfmet.BENCH_BOTTOM_COLLIMATOR',      'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Bench$\downarrow$ Coll.',  'marker': '.', 'linewidth': 0.5}}
            dict3 = {'col': 'kpfmet.BENCH_BOTTOM_DCUT',            'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Bench$\downarrow$ D-Cut',  'marker': '.', 'linewidth': 0.5}}
            dict4 = {'col': 'kpfmet.BENCH_BOTTOM_ECHELLE',         'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Bench$\downarrow$ Echelle','marker': '.', 'linewidth': 0.5}}
            dict5 = {'col': 'kpfmet.BENCH_TOP_BETWEEN_CAMERAS',    'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Bench Cams',               'marker': '.', 'linewidth': 0.5}}
            dict6 = {'col': 'kpfmet.BENCH_TOP_COLL',               'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Bench Coll',               'marker': '.', 'linewidth': 0.5}}
            dict7 = {'col': 'kpfmet.BENCH_TOP_DCUT',               'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Bench D-Cut',              'marker': '.', 'linewidth': 0.5}}
            dict8 = {'col': 'kpfmet.BENCH_TOP_ECHELLE_CAM',        'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Bench Ech-Cam',            'marker': '.', 'linewidth': 0.5}}
            dict9 = {'col': 'kpfmet.ECHELLE_BOTTOM',               'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Echelle$\downarrow$',      'marker': '.', 'linewidth': 0.5}}
            dict10= {'col': 'kpfmet.ECHELLE_TOP',                  'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Echelle$\uparrow$',        'marker': '.', 'linewidth': 0.5}}
            dict11= {'col': 'kpfmet.GREEN_CAMERA_BOTTOM',          'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Green Cam$\downarrow$',    'marker': '.', 'linewidth': 0.5}}
            dict12= {'col': 'kpfmet.GREEN_CAMERA_COLLIMATOR',      'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Green Cam Coll',           'marker': '.', 'linewidth': 0.5}}
            dict13= {'col': 'kpfmet.GREEN_CAMERA_ECHELLE',         'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Green Cam Ech',            'marker': '.', 'linewidth': 0.5}}
            dict14= {'col': 'kpfmet.GREEN_CAMERA_TOP',             'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Green Cam$\uparrow$',      'marker': '.', 'linewidth': 0.5}}
            dict15= {'col': 'kpfmet.GREEN_GRISM_TOP',              'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Green Grism$\uparrow$',    'marker': '.', 'linewidth': 0.5}}
            dict16= {'col': 'kpfmet.PRIMARY_COLLIMATOR_TOP',       'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Primary Coll$\uparrow$',   'marker': '.', 'linewidth': 0.5}}
            dict17= {'col': 'kpfmet.RED_CAMERA_BOTTOM',            'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Red Cam$\downarrow$',      'marker': '.', 'linewidth': 0.5}}
            dict18= {'col': 'kpfmet.RED_CAMERA_COLLIMATOR',        'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Red Cam Coll',             'marker': '.', 'linewidth': 0.5}}
            dict19= {'col': 'kpfmet.RED_CAMERA_ECHELLE',           'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Red Cam Ech',              'marker': '.', 'linewidth': 0.5}}
            dict20= {'col': 'kpfmet.RED_CAMERA_TOP',               'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Red Cam$\uparrow$',        'marker': '.', 'linewidth': 0.5}}
            dict21= {'col': 'kpfmet.RED_GRISM_TOP',                'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Red Grism$\uparrow$',      'marker': '.', 'linewidth': 0.5}}
            dict22= {'col': 'kpfmet.REFORMATTER',                  'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Reformatter',              'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1, dict5, dict10, dict14, dict20, dict15, dict21, dict22]
            thispaneldict = {'ylabel': 'Spectrometer\nTemperature' + ' ($^{\circ}$C)',
                             'nolegend': 'false',
                             'legend_frac_size': 0.3}
            chambertemppanel = {'panelvars': thispanelvars,
                                'paneldict': thispaneldict}
            
            thispaneldict = {'ylabel': 'Spectrometer\n' + r'$\Delta$Temperature (K)',
                             'title': 'KPF Spectrometer Temperatures', 
                             # Not working yet
                             #'axhspan': {
                             #           1: {'ymin':  0.01, 'ymax':  100, 'color': 'red', 'alpha': 0.2},
                             #           2: {'ymin': -0.01, 'ymax': -100, 'color': 'red', 'alpha': 0.2},
                             #           },
                             'nolegend': 'false', 
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.3}
            chambertemppanel2 = {'panelvars': thispanelvars,
                                 'paneldict': thispaneldict}
            panel_arr = [halltemppanel, halltemppanel2, copy.deepcopy(halltemppanel3), chambertemppanel, copy.deepcopy(chambertemppanel2)]

        elif plot_name=='chamber_temp_detail':
            dict1 = {'col': 'kpfmet.BENCH_BOTTOM_BETWEEN_CAMERAS', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Bench$\downarrow$ Cams',   'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'kpfmet.BENCH_BOTTOM_COLLIMATOR',      'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Bench$\downarrow$ Coll.',  'marker': '.', 'linewidth': 0.5}}
            dict3 = {'col': 'kpfmet.BENCH_BOTTOM_DCUT',            'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Bench$\downarrow$ D-Cut',  'marker': '.', 'linewidth': 0.5}}
            dict4 = {'col': 'kpfmet.BENCH_BOTTOM_ECHELLE',         'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Bench$\downarrow$ Echelle','marker': '.', 'linewidth': 0.5}}
            dict5 = {'col': 'kpfmet.BENCH_TOP_BETWEEN_CAMERAS',    'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Bench Cams',               'marker': '.', 'linewidth': 0.5}}
            dict6 = {'col': 'kpfmet.BENCH_TOP_COLL',               'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Bench Coll',               'marker': '.', 'linewidth': 0.5}}
            dict7 = {'col': 'kpfmet.BENCH_TOP_DCUT',               'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Bench D-Cut',              'marker': '.', 'linewidth': 0.5}}
            dict8 = {'col': 'kpfmet.BENCH_TOP_ECHELLE_CAM',        'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Bench Ech-Cam',            'marker': '.', 'linewidth': 0.5}}
            dict9 = {'col': 'kpfmet.ECHELLE_BOTTOM',               'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Echelle$\downarrow$',      'marker': '.', 'linewidth': 0.5}}
            dict10= {'col': 'kpfmet.ECHELLE_TOP',                  'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Echelle$\uparrow$',        'marker': '.', 'linewidth': 0.5}}
            dict11= {'col': 'kpfmet.GREEN_CAMERA_BOTTOM',          'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Green Cam$\downarrow$',    'marker': '.', 'linewidth': 0.5}}
            dict12= {'col': 'kpfmet.GREEN_CAMERA_COLLIMATOR',      'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Green Cam Coll',           'marker': '.', 'linewidth': 0.5}}
            dict13= {'col': 'kpfmet.GREEN_CAMERA_ECHELLE',         'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Green Cam Ech',            'marker': '.', 'linewidth': 0.5}}
            dict14= {'col': 'kpfmet.GREEN_CAMERA_TOP',             'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Green Cam$\uparrow$',      'marker': '.', 'linewidth': 0.5}}
            dict15= {'col': 'kpfmet.GREEN_GRISM_TOP',              'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Green Grism$\uparrow$',    'marker': '.', 'linewidth': 0.5}}
            dict16= {'col': 'kpfmet.PRIMARY_COLLIMATOR_TOP',       'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Primary Coll$\uparrow$',   'marker': '.', 'linewidth': 0.5}}
            dict17= {'col': 'kpfmet.RED_CAMERA_BOTTOM',            'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Red Cam$\downarrow$',      'marker': '.', 'linewidth': 0.5}}
            dict18= {'col': 'kpfmet.RED_CAMERA_COLLIMATOR',        'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Red Cam Coll',             'marker': '.', 'linewidth': 0.5}}
            dict19= {'col': 'kpfmet.RED_CAMERA_ECHELLE',           'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Red Cam Ech',              'marker': '.', 'linewidth': 0.5}}
            dict20= {'col': 'kpfmet.RED_CAMERA_TOP',               'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Red Cam$\uparrow$',        'marker': '.', 'linewidth': 0.5}}
            dict21= {'col': 'kpfmet.RED_GRISM_TOP',                'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Red Grism$\uparrow$',      'marker': '.', 'linewidth': 0.5}}
            dict22= {'col': 'kpfmet.REFORMATTER',                  'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': r'Reformatter',              'marker': '.', 'linewidth': 0.5}}
                
            thispanelvars = [dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, ]
            thispaneldict = {'ylabel': 'Bench\n' + r'$\Delta$Temperature (K)',
                             'nolegend': 'false', 
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.3}
            chambertemppanel1 = {'panelvars': thispanelvars,
                                 'paneldict': thispaneldict}
            
            thispanelvars = [dict15, dict14, dict11, dict12, dict13, ]
            thispaneldict = {'ylabel': 'Green Camera\n' + r'$\Delta$Temperature (K)',
                             'nolegend': 'false', 
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.3}
            chambertemppanel2 = {'panelvars': thispanelvars,
                                 'paneldict': thispaneldict}
            
            thispanelvars = [dict21, dict20, dict17, dict18, dict19, ]
            thispaneldict = {'ylabel': 'Red Camera\n' + r'$\Delta$Temperature (K)',
                             'nolegend': 'false', 
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.3}
            chambertemppanel3 = {'panelvars': thispanelvars,
                                 'paneldict': thispaneldict}
                
            thispanelvars = [dict10, dict9, ]
            thispaneldict = {'ylabel': 'Echelle Grating\n' + r'$\Delta$Temperature (K)',
                             'nolegend': 'false', 
                             'title': 'KPF Spectrometer Temperatures', 
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.3}
            chambertemppanel4 = {'panelvars': thispanelvars,
                                 'paneldict': thispaneldict}
            panel_arr = [copy.deepcopy(chambertemppanel1), copy.deepcopy(chambertemppanel2), copy.deepcopy(chambertemppanel3), copy.deepcopy(chambertemppanel4)]

        elif plot_name=='fiber_temp':
            dict1 = {'col': 'kpfmet.SCIENCE_CAL_FIBER_STG',  'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': 'Sci Cal Fiber Stg',    'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'kpfmet.SCISKY_SCMBLR_CHMBR_EN', 'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': 'Sci/Sky Scrmb. Chmbr', 'marker': '.', 'linewidth': 0.5}}
            dict3 = {'col': 'kpfmet.SCISKY_SCMBLR_FIBER_EN', 'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': 'Sci/Sky Scrmb. Fiber', 'marker': '.', 'linewidth': 0.5}}
            dict4 = {'col': 'kpfmet.SIMCAL_FIBER_STG',       'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': 'SimulCal Fiber Stg',   'marker': '.', 'linewidth': 0.5}}
            dict5 = {'col': 'kpfmet.SKYCAL_FIBER_STG',       'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': 'SkyCal Fiber Stg',     'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1, dict2, dict3, dict4, dict5]
            thispaneldict = {'ylabel': 'Temperature' + ' ($^{\circ}$C)',
                             'title': 'Fiber Temperatures',
                             'legend_frac_size': 0.30}
            fibertempspanel = {'panelvars': thispanelvars,
                               'paneldict': thispaneldict}
            panel_arr = [fibertempspanel]

        elif plot_name=='ccd_readspeed':
            dict1 = {'col': 'GREENTRT', 'plot_type': 'plot', 'unit': 'e-', 'plot_attr': {'label': 'Green CCD', 'marker': '.', 'linewidth': 0.5, 'color': 'darkgreen'}}
            dict2 = {'col': 'REDTRT',   'plot_type': 'plot', 'unit': 'e-', 'plot_attr': {'label': 'Red CCD',   'marker': '.', 'linewidth': 0.5, 'color': 'darkred'}}
            thispanelvars = [dict1, dict2]
            thispaneldict = {'ylabel': 'Read Speed [sec]',
                             'title': 'CCD Read Speed',
                             'not_junk': 'true',
                             'legend_frac_size': 0.25}
            readspeedpanel = {'panelvars': thispanelvars,
                              'paneldict': thispaneldict}
            panel_arr = [readspeedpanel]

        elif plot_name=='ccd_readnoise':
            dict1 = {'col': 'RNGREEN1', 'plot_type': 'plot', 'unit': 'e-', 'plot_attr': {'label': 'Green CCD 1', 'marker': '.', 'linewidth': 0.5, 'color': 'darkgreen'}}
            dict2 = {'col': 'RNGREEN2', 'plot_type': 'plot', 'unit': 'e-', 'plot_attr': {'label': 'Green CCD 2', 'marker': '.', 'linewidth': 0.5, 'color': 'forestgreen'}}
            dict1b= {'col': 'RNGREEN3', 'plot_type': 'plot', 'unit': 'e-', 'plot_attr': {'label': 'Green CCD 3', 'marker': '.', 'linewidth': 0.5, 'color': 'limegreen'}}
            dict2b= {'col': 'RNGREEN4', 'plot_type': 'plot', 'unit': 'e-', 'plot_attr': {'label': 'Green CCD 4', 'marker': '.', 'linewidth': 0.5, 'color': 'lime'}}
            dict3 = {'col': 'RNRED1',   'plot_type': 'plot', 'unit': 'e-', 'plot_attr': {'label': 'RED CCD 1',   'marker': '.', 'linewidth': 0.5, 'color': 'darkred'}}
            dict4 = {'col': 'RNRED2',   'plot_type': 'plot', 'unit': 'e-', 'plot_attr': {'label': 'RED CCD 2',   'marker': '.', 'linewidth': 0.5, 'color': 'firebrick'}}
            dict3b= {'col': 'RNRED3',   'plot_type': 'plot', 'unit': 'e-', 'plot_attr': {'label': 'RED CCD 3',   'marker': '.', 'linewidth': 0.5, 'color': 'indianred'}}
            dict4b= {'col': 'RNRED4',   'plot_type': 'plot', 'unit': 'e-', 'plot_attr': {'label': 'RED CCD 4',   'marker': '.', 'linewidth': 0.5, 'color': 'lightcoral'}}
            thispanelvars = [dict1, dict2, dict1b, dict2b]
            thispaneldict = {'ylabel': 'Green CCD\nRead Noise [e-]',
                             'not_junk': 'true',
                             'legend_frac_size': 0.25}
            readnoisepanel1 = {'panelvars': thispanelvars,
                               'paneldict': thispaneldict}
            thispanelvars = [dict3, dict4, dict3b, dict4b]
            thispaneldict = {'ylabel': 'Red CCD\nRead Noise [e-]',
                             'title': 'CCD Read Noise',
                             'not_junk': 'true',
                             'legend_frac_size': 0.25}
            readnoisepanel2 = {'panelvars': thispanelvars,
                               'paneldict': thispaneldict}
            panel_arr = [readnoisepanel1, readnoisepanel2]
        
        elif plot_name=='ccd_dark_current':
            # Green CCD panel - Dark current
            dict1 = {'col': 'FLXCOLLG', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Collimator-side', 'marker': '.', 'linewidth': 0.5, 'color': 'darkgreen'}}
            dict2 = {'col': 'FLXECHG',  'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Echelle-side',    'marker': '.', 'linewidth': 0.5, 'color': 'forestgreen'}}
            dict3 = {'col': 'FLXREG1G', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 1',        'marker': '.', 'linewidth': 0.5, 'color': 'lightgreen'}}
            dict4 = {'col': 'FLXREG2G', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 2',        'marker': '.', 'linewidth': 0.5, 'color': 'lightgreen'}}
            dict5 = {'col': 'FLXREG3G', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 3',        'marker': '.', 'linewidth': 0.5, 'color': 'lightgreen'}}
            dict6 = {'col': 'FLXREG4G', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 4',        'marker': '.', 'linewidth': 0.5, 'color': 'lightgreen'}}
            dict7 = {'col': 'FLXREG5G', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 5',        'marker': '.', 'linewidth': 0.5, 'color': 'lightgreen'}}
            dict8 = {'col': 'FLXREG6G', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 6',        'marker': '.', 'linewidth': 0.5, 'color': 'lightgreen'}}
            thispanelvars = [dict3, dict4, dict1, dict2, ]
            thispaneldict = {'ylabel': 'Green CCD\nDark Current [e-/hr]',
                             'not_junk': 'true',
                             'legend_frac_size': 0.35}
            greenpanel = {'panelvars': thispanelvars,
                          'paneldict': thispaneldict}
            
            # Red CCD panel - Dark current
            dict1 = {'col': 'FLXCOLLR', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Coll-side', 'marker': '.', 'linewidth': 0.5, 'color': 'darkred'}}
            dict2 = {'col': 'FLXECHR',  'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Ech-side',  'marker': '.', 'linewidth': 0.5, 'color': 'firebrick'}}
            dict3 = {'col': 'FLXREG1R', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 1',  'marker': '.', 'linewidth': 0.5, 'color': 'lightcoral'}}
            dict4 = {'col': 'FLXREG2R', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 2',        'marker': '.', 'linewidth': 0.5, 'color': 'lightcoral'}}
            dict5 = {'col': 'FLXREG3R', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 3',        'marker': '.', 'linewidth': 0.5, 'color': 'lightcoral'}}
            dict6 = {'col': 'FLXREG4R', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 4',        'marker': '.', 'linewidth': 0.5, 'color': 'lightcoral'}}
            dict7 = {'col': 'FLXREG5R', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 5',        'marker': '.', 'linewidth': 0.5, 'color': 'lightcoral'}}
            dict8 = {'col': 'FLXREG6R', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 6',        'marker': '.', 'linewidth': 0.5, 'color': 'lightcoral'}}
            thispanelvars = [dict3, dict4, dict1, dict2, ]
            thispaneldict = {'ylabel': 'Red CCD\nDark Current [e-/hr]',
                             'not_junk': 'true',
                             'legend_frac_size': 0.35}
            redpanel = {'panelvars': thispanelvars,
                        'paneldict': thispaneldict}
            
            # Green CCD panel - ion pump current
            dict1 = {'col': 'kpfgreen.COL_CURR', 'plot_type': 'plot', 'unit': 'A', 'plot_attr': {'label': 'Coll-side', 'marker': '.', 'linewidth': 0.5, 'color': 'darkgreen'}}
            dict2 = {'col': 'kpfgreen.ECH_CURR', 'plot_type': 'plot', 'unit': 'A', 'plot_attr': {'label': 'Ech-side',    'marker': '.', 'linewidth': 0.5, 'color': 'forestgreen'}}
            thispanelvars = [dict1]
            thispaneldict = {'ylabel': 'Green CCD\nIon Pump Current [A]',
                             'yscale': 'log',
                             'not_junk': 'true',
                             'legend_frac_size': 0.35}
            greenpanel_ionpump = {'panelvars': thispanelvars,
                                  'paneldict': thispaneldict}
            thispanelvars = [dict2]
            thispaneldict = {'ylabel': 'Green CCD\nIon Pump Current [A]',
                             'yscale': 'log',
                             'not_junk': 'true',
                             'legend_frac_size': 0.35}
            greenpanel_ionpump2 = {'panelvars': thispanelvars,
                                   'paneldict': thispaneldict}
            
            # Red CCD panel - ion pump current
            dict1 = {'col': 'kpfred.COL_CURR', 'plot_type': 'plot', 'unit': 'A', 'plot_attr': {'label': 'Coll-side', 'marker': '.', 'linewidth': 0.5, 'color': 'darkred'}}
            dict2 = {'col': 'kpfred.ECH_CURR', 'plot_type': 'plot', 'unit': 'A', 'plot_attr': {'label': 'Ech-side',    'marker': '.', 'linewidth': 0.5, 'color': 'firebrick'}}
            thispanelvars = [dict1]
            thispaneldict = {'ylabel': 'Red CCD\nIon Pump Current [A]',
                             'yscale': 'log',
                             'not_junk': 'true',
                             'legend_frac_size': 0.35}
            redpanel_ionpump = {'panelvars': thispanelvars,
                                'paneldict': thispaneldict}
            thispanelvars = [dict2]
            thispaneldict = {'ylabel': 'Red CCD\nIon Pump Current [A]',
                             'yscale': 'log',
                             'not_junk': 'true',
                             'legend_frac_size': 0.35}
            redpanel_ionpump2 = {'panelvars': thispanelvars,
                                'paneldict': thispaneldict}
            # to do: add kpfred.COL_PRESS (green, too)
            #            kpfred.ECH_PRESS

            # Amplifier glow panel
            dict1 = {'col': 'FLXAMP1G', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Green Amp Reg 1', 'marker': '.', 'linewidth': 0.5, 'color': 'darkgreen'}}
            dict2 = {'col': 'FLXAMP2G', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Green Amp Reg 2', 'marker': '.', 'linewidth': 0.5, 'color': 'forestgreen'}}
            dict3 = {'col': 'FLXAMP1R', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Red Amp Reg 1',   'marker': '.', 'linewidth': 0.5, 'color': 'darkred'}}
            dict4 = {'col': 'FLXAMP2R', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Red Amp Reg 2',   'marker': '.', 'linewidth': 0.5, 'color': 'firebrick'}}
            thispanelvars = [dict3, dict4, dict1, dict2, ]
            thispaneldict = {'ylabel': 'CCD Amplifier\nDark Current [e-/hr]',
                             'title': 'CCD Dark Current',
                             'legend_frac_size': 0.35}
            amppanel = {'panelvars': thispanelvars,
                        'paneldict': thispaneldict}
            panel_arr = [greenpanel, redpanel, greenpanel_ionpump, greenpanel_ionpump2, redpanel_ionpump, redpanel_ionpump2, amppanel]

        elif plot_name=='ccd_temp':
            # CCD Temperatures
            dict1 = {'col': 'kpfgreen.STA_CCD_T', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'STA Sensor', 'marker': '.', 'linewidth': 0.5, 'color': 'darkgreen'}}
            dict2 = {'col': 'kpfgreen.KPF_CCD_T', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'SSL Sensor', 'marker': '.', 'linewidth': 0.5, 'color': 'forestgreen'}}
            thispanelvars = [dict2, dict1, ]
            thispaneldict = {'ylabel': 'Green CCD\nTemperature (C)',
                             'legend_frac_size': 0.25}
            green_ccd = {'panelvars': thispanelvars,
                         'paneldict': thispaneldict}

            dict1 = {'col': 'kpfred.STA_CCD_T', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'STA Sensor', 'marker': '.', 'linewidth': 0.5, 'color': 'darkred'}}
            dict2 = {'col': 'kpfred.KPF_CCD_T', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'SSL Sensor', 'marker': '.', 'linewidth': 0.5, 'color': 'firebrick'}}
            thispanelvars2 = [dict2, dict1, ]
            thispaneldict2 = {'ylabel': 'Red CCD\nTemperature (C)',
                             'legend_frac_size': 0.25}
            red_ccd = {'panelvars': thispanelvars2,
                       'paneldict': thispaneldict2}

            dict1 = {'col': 'kpfgreen.STA_CCD_T', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'STA Sensor', 'marker': '.', 'linewidth': 0.5, 'color': 'darkgreen'}}
            dict2 = {'col': 'kpfgreen.KPF_CCD_T', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'SSL Sensor', 'marker': '.', 'linewidth': 0.5, 'color': 'forestgreen'}}
            thispanelvars3 = [dict2, dict1, ]
            thispaneldict3 = {'ylabel': 'Green CCD\n' + r'$\Delta$Temperature (K)',
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.25}
            green_ccd2 = {'panelvars': thispanelvars3,
                          'paneldict': thispaneldict3}

            dict1 = {'col': 'kpfred.STA_CCD_T', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'STA Sensor', 'marker': '.', 'linewidth': 0.5, 'color': 'darkred'}}
            dict2 = {'col': 'kpfred.KPF_CCD_T', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'SSL Sensor', 'marker': '.', 'linewidth': 0.5, 'color': 'firebrick'}}
            thispanelvars4 = [dict2, dict1, ]
            thispaneldict4 = {'ylabel': 'Red CCD\n' + r'$\Delta$Temperature (K)',
                             'title': 'CCD Temperatures',
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.25}
            red_ccd2 = {'panelvars': thispanelvars4,
                        'paneldict': thispaneldict4}

            panel_arr = [green_ccd, red_ccd, green_ccd2, red_ccd2]

# Additional keywords to add:
#                'kpfred.CRYOBODY_T':                   'float',  # degC    Cryo Body Temperature c- double degC {%.3f}
#                'kpfred.CRYOBODY_TRG':                 'float',  # degC    Cryo body heater 7B, target temp c2 double deg...
#                'kpfred.CURRTEMP':                     'float',  # degC    Current cold head temperature c- double degC {...
#                'kpfred.STA_CCD_TRG':                  'float',  # degC    Detector heater 7A, target temp c2 double degC...
#                'kpfred.TEMPSET':                      'float',  # degC    Set point for the cold head temperature c2 dou...

#                'kpfred.CF_BASE_2WT':                  'float',  # degC    tip cold finger (2 wire) c- double degC {%.3f}
#                'kpfred.CF_BASE_T':                    'float',  # degC    base cold finger 2wire temp c- double degC {%.3f}
#                'kpfred.CF_BASE_TRG':                  'float',  # degC    base cold finger heater 1A, target temp c2 dou...
#                'kpfred.CF_TIP_T':                     'float',  # degC    tip cold finger c- double degC {%.3f}
#                'kpfred.CF_TIP_TRG':                   'float',  # degC    tip cold finger heater 1B, target temp c2 doub...

        elif plot_name=='ccd_controller':
            dict1 = {'col': 'kpfred.BPLANE_TEMP',     'plot_type': 'plot', 'unit': 'C', 'plot_attr': {'label': 'Backplane',          'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'kpfred.BRD10_DRVR_T',    'plot_type': 'plot', 'unit': 'C', 'plot_attr': {'label': 'Board 10 (Driver)',  'marker': '.', 'linewidth': 0.5}}
            dict3 = {'col': 'kpfred.BRD11_DRVR_T',    'plot_type': 'plot', 'unit': 'C', 'plot_attr': {'label': 'Board 11 (Driver)',  'marker': '.', 'linewidth': 0.5}}
            dict4 = {'col': 'kpfred.BRD12_LVXBIAS_T', 'plot_type': 'plot', 'unit': 'C', 'plot_attr': {'label': 'Board 12 (LVxBias)', 'marker': '.', 'linewidth': 0.5}}
            dict5 = {'col': 'kpfred.BRD1_HTRX_T',     'plot_type': 'plot', 'unit': 'C', 'plot_attr': {'label': 'Board 1 (HeaterX)',  'marker': '.', 'linewidth': 0.5}}
            dict6 = {'col': 'kpfred.BRD2_XVBIAS_T',   'plot_type': 'plot', 'unit': 'C', 'plot_attr': {'label': 'Board 2 (XV Bias)',  'marker': '.', 'linewidth': 0.5}}
            dict7 = {'col': 'kpfred.BRD3_LVDS_T',     'plot_type': 'plot', 'unit': 'C', 'plot_attr': {'label': 'Board 3 (LVDS)',     'marker': '.', 'linewidth': 0.5}}
            dict8 = {'col': 'kpfred.BRD4_DRVR_T',     'plot_type': 'plot', 'unit': 'C', 'plot_attr': {'label': 'Board 4 (Driver)',   'marker': '.', 'linewidth': 0.5}}
            dict9 = {'col': 'kpfred.BRD5_AD_T',       'plot_type': 'plot', 'unit': 'C', 'plot_attr': {'label': 'Board 5 (AD)',       'marker': '.', 'linewidth': 0.5}}
            dict10= {'col': 'kpfred.BRD7_HTRX_T',     'plot_type': 'plot', 'unit': 'C', 'plot_attr': {'label': 'Board 7 (HeaterX)',  'marker': '.', 'linewidth': 0.5}}
            dict11= {'col': 'kpfred.BRD9_HVXBIAS_T',  'plot_type': 'plot', 'unit': 'C', 'plot_attr': {'label': 'Board 9 (HVxBias)',  'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9, dict10, dict11, ]
            thispaneldict = {'ylabel': 'Temperatures (C)',
                             'title': 'CCD Controllers',
                             'legend_frac_size': 0.30}
            controller1 = {'panelvars': thispanelvars,
                           'paneldict': thispaneldict}

            thispanelvars2 = [dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9, dict10, dict11, ]
            thispaneldict2 = {'ylabel': r'$\Delta$Temperature (K)',
                             'title': 'CCD Controllers',
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.30}
            controller2 = {'panelvars': thispanelvars2,
                           'paneldict': thispaneldict2}
            panel_arr = [copy.deepcopy(controller1), copy.deepcopy(controller2)]

        elif plot_name=='lfc':
            dict1 = {'col': 'kpfcal.IRFLUX',  'plot_type': 'scatter', 'unit': 'counts', 'plot_attr': {'label': 'Fiberlock IR',  'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1]
            thispaneldict1 = {'ylabel': 'Intensity (counts)',
                              'legend_frac_size': 0.25}
            lfcpanel1 = {'panelvars': thispanelvars,
                         'paneldict': thispaneldict1}
            dict1 = {'col': 'kpfcal.VISFLUX', 'plot_type': 'scatter', 'unit': 'counts', 'plot_attr': {'label': 'Fiberlock Vis', 'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1]
            thispaneldict2 = {'ylabel': 'Intensity (counts)',
                              'legend_frac_size': 0.25}
            lfcpanel2 = {'panelvars': thispanelvars,
                         'paneldict': thispaneldict2}

            dict1 = {'col': 'kpfcal.BLUECUTIACT', 'plot_type': 'scatter', 'unit': 'A', 'plot_attr': {'label': 'Blue Cut Amp.',  'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1]
            thispaneldict3 = {'ylabel': 'Current (A)',
                              'title': 'LFC Diagnostics',
                              'legend_frac_size': 0.25}
            lfcpanel3 = {'panelvars': thispanelvars,
                         'paneldict': thispaneldict3}
            panel_arr = [lfcpanel1, lfcpanel2, lfcpanel3]

        elif plot_name=='etalon':
            dict1 = {'col': 'ETAV1C1T',  'plot_type': 'plot', 'unit': 'C', 'plot_attr': {'label': 'Vescent 1 Ch 1',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            dict2 = {'col': 'ETAV1C2T',  'plot_type': 'plot', 'unit': 'C', 'plot_attr': {'label': 'Vescent 1 Ch 2',  'marker': '.', 'linewidth': 0.5, 'color': 'blue'}}
            dict3 = {'col': 'ETAV1C3T',  'plot_type': 'plot', 'unit': 'C', 'plot_attr': {'label': 'Vescent 1 Ch 3',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict4 = {'col': 'ETAV1C4T',  'plot_type': 'plot', 'unit': 'C', 'plot_attr': {'label': 'Vescent 1 Ch 4',  'marker': '.', 'linewidth': 0.5, 'color': 'orange'}}
            dict5 = {'col': 'ETAV2C3T',  'plot_type': 'plot', 'unit': 'C', 'plot_attr': {'label': 'Vescent 2 Ch 3',  'marker': '.', 'linewidth': 0.5, 'color': 'purple'}}
            thispanelvars = [dict1, dict2, dict3, dict4, dict5]
            thispaneldict = {'ylabel': 'Temperature (C)',
                             'legend_frac_size': 0.25}
            thispaneldict2 = {'ylabel': r'$\Delta$Temperature (K)',
                             'title': 'Etalon Temperatures',
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.25}
            etalonpanel = {'panelvars': thispanelvars,
                           'paneldict': thispaneldict}
            etalonpanel2 = {'panelvars': [dict1],
                           'paneldict': thispaneldict2}
            etalonpanel3 = {'panelvars': [dict2],
                           'paneldict': thispaneldict2}
            etalonpanel4 = {'panelvars': [dict3],
                           'paneldict': thispaneldict2}
            etalonpanel5 = {'panelvars': [dict4],
                           'paneldict': thispaneldict2}
            etalonpanel6 = {'panelvars': [dict5],
                           'paneldict': thispaneldict2}
            panel_arr = [copy.deepcopy(etalonpanel), copy.deepcopy(etalonpanel2), copy.deepcopy(etalonpanel3), copy.deepcopy(etalonpanel4), copy.deepcopy(etalonpanel5), copy.deepcopy(etalonpanel6)]

        elif plot_name=='hcl':
            dict1 = {'col': 'kpfmet.TEMP',     'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': 'Hallway',      'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'kpfmet.TH_DAILY', 'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': 'Th-Ar Daily',  'marker': '.', 'linewidth': 0.5}}
            dict3 = {'col': 'kpfmet.TH_GOLD',  'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': 'Th-Ar Gold',   'marker': '.', 'linewidth': 0.5}}
            dict4 = {'col': 'kpfmet.U_DAILY',  'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': 'U-Ar Daily',   'marker': '.', 'linewidth': 0.5}}
            dict5 = {'col': 'kpfmet.U_GOLD',   'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': 'U-Ar Gold',    'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1]
            thispaneldict = {'ylabel': 'Temperature (C)',
                             'legend_frac_size': 0.35}
            hclpanel = {'panelvars': thispanelvars,
                        'paneldict': thispaneldict}
            thispanelvars = [dict2, dict3, dict4, dict5]
            thispaneldict = {'ylabel': 'Temperature (C)',
                             'title': 'Hollow-Cathode Lamp Temperatures',
                             'legend_frac_size': 0.35}
            hclpanel2 = {'panelvars': thispanelvars,
                         'paneldict': thispaneldict}
            panel_arr = [copy.deepcopy(hclpanel), copy.deepcopy(hclpanel2)]
            
        elif plot_name=='hk_temp':
            dict1 = {'col': 'kpfexpose.BENCH_C',     'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'HK BENCH_C',     'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'kpfexpose.CAMBARREL_C', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'HK CAMBARREL_C', 'marker': '.', 'linewidth': 0.5}}
            dict3 = {'col': 'kpfexpose.DET_XTRN_C',  'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'HK DET_XTRN_C',  'marker': '.', 'linewidth': 0.5}}
            dict4 = {'col': 'kpfexpose.ECHELLE_C',   'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'HK ECHELLE_C',   'marker': '.', 'linewidth': 0.5}}
            dict5 = {'col': 'kpfexpose.ENCLOSURE_C', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'HK ENCLOSURE_C', 'marker': '.', 'linewidth': 0.5}}
            dict6 = {'col': 'kpfexpose.RACK_AIR_C',  'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'HK RACK_AIR_C',  'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1, dict2, dict3, dict5, dict6, dict4]
            thispaneldict = {'ylabel': 'Spectrometer\nTemperature (K)',
                             'legend_frac_size': 0.30}
            hkpanel1 = {'panelvars': thispanelvars,
                        'paneldict': thispaneldict}

            thispanelvars2 = [dict1, dict2, dict3, dict5, dict6, dict4]
            thispaneldict2 = {'ylabel': 'Spectrometer\n' + '$\Delta$Temperature (K)',
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.30}
            hkpanel2 = {'panelvars': thispanelvars2,
                        'paneldict': thispaneldict2}

            dict1 = {'col': 'kpf_hk.COOLTARG', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'Detector Target Temp.', 'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'kpf_hk.CURRTEMP', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'Detector Temp.',        'marker': '.', 'linewidth': 0.5}}
            thispanelvars3 = [dict1, dict2] 
            thispaneldict3 = {'ylabel': 'Detector\nTemperature (K)',
                              'legend_frac_size': 0.30}
            hkpanel3 = {'panelvars': thispanelvars3,
                        'paneldict': thispaneldict3}

            thispanelvars4 = [dict1, dict2]
            thispaneldict4 = {'ylabel': 'Detector\n' + '$\Delta$Temperature (K)',
                             'title': 'Ca H&K Spectrometer Temperatures',
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.30}
            hkpanel4 = {'panelvars': thispanelvars4,
                        'paneldict': thispaneldict4}

            panel_arr = [copy.deepcopy(hkpanel1), copy.deepcopy(hkpanel2), copy.deepcopy(hkpanel3), copy.deepcopy(hkpanel4)]

            
        elif plot_name=='agitator':
            dict1 = {'col': 'kpfmot.AGITSPD', 'plot_type': 'scatter', 'unit': 'counts/sec', 'plot_attr': {'label': 'Agitator Speed', 'marker': '.', 'linewidth': 0.5}}
            thispanelvars1 = [dict1]
            thispaneldict1 = {'ylabel': 'Agitator Speed\n(counts/sec)',
                              'not_junk': 'true',
                             'legend_frac_size': 0.25}
            agitatorpanel1 = {'panelvars': thispanelvars1,
                              'paneldict': thispaneldict1}
            dict2 = {'col': 'kpfmot.AGITTOR', 'plot_type': 'scatter', 'unit': 'V', 'plot_attr': {'label': 'Agitator Motor Torque', 'marker': '.', 'linewidth': 0.5}}
            thispanelvars2 = [dict2]
            thispaneldict2 = {'ylabel': 'Motor Torque (V)',
                              'not_junk': 'true',
                              'legend_frac_size': 0.25}
            agitatorpanel2 = {'panelvars': thispanelvars2,
                              'paneldict': thispaneldict2}
            dict3 = {'col': 'kpfmot.AGITAMBI_T', 'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': 'Ambient Temp.', 'marker': '.', 'linewidth': 0.5}}
            dict4 = {'col': 'kpfmot.AGITMOT_T',  'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': 'Motor Temp.',   'marker': '.', 'linewidth': 0.5}}
            thispanelvars3 = [dict3, dict4]
            thispaneldict3 = {'ylabel': 'Temperature (C)',
                              'not_junk': 'true',
                              'legend_frac_size': 0.25}
            agitatorpanel3 = {'panelvars': thispanelvars3,
                              'paneldict': thispaneldict3}
            dict5 = {'col': 'kpfmot.AGITAMBI_T', 'plot_type': 'scatter', 'unit': 'mA', 'plot_attr': {'label': 'Outlet A1 Power', 'marker': '.', 'linewidth': 0.5}}
            thispanelvars4 = [dict5]
            thispaneldict4 = {'ylabel': 'Outlet A1 Power\n(mA)',
                              'title': r'KPF Agitator',
                              'not_junk': 'true',
                              'legend_frac_size': 0.25}
            agitatorpanel4 = {'panelvars': thispanelvars4,
                              'paneldict': thispaneldict4}
            panel_arr = [agitatorpanel1, agitatorpanel2, agitatorpanel3, agitatorpanel4]

        elif plot_name=='guiding':
            dict1 = {'col': 'GDRXRMS',  'plot_type': 'scatter', 'unit': 'mas', 'plot_attr': {'label': 'Error (X)', 'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'GDRYRMS',  'plot_type': 'scatter', 'unit': 'mas', 'plot_attr': {'label': 'Error (Y)', 'marker': '.', 'linewidth': 0.5}}
            dict3 = {'col': 'GDRXBIAS', 'plot_type': 'scatter', 'unit': 'mas', 'plot_attr': {'label': 'Bias (X)',  'marker': '.', 'linewidth': 0.5}}
            dict4 = {'col': 'GDRYBIAS', 'plot_type': 'scatter', 'unit': 'mas', 'plot_attr': {'label': 'Bias (Y)',  'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1, dict2]
            thispaneldict = {'ylabel': 'RMS Guiding Errors (mas)',
                             'narrow_xlim_daily': 'true',
                             'not_junk': 'true',
                             'on_sky': 'true', 
                             'legend_frac_size': 0.20}
            guidingpanel1 = {'panelvars': thispanelvars,
                             'paneldict': thispaneldict}

            thispanelvars2 = [dict3, dict4]
            thispaneldict2 = {'ylabel': 'RMS Guiding Bias (mas)',
                             'narrow_xlim_daily': 'true',
                             'title': 'Guiding',
                             'not_junk': 'true',
                             'on_sky': 'true', 
                             'legend_frac_size': 0.20}
            guidingpanel2 = {'panelvars': thispanelvars2,
                             'paneldict': thispaneldict2}
            panel_arr = [guidingpanel1, guidingpanel2]

        elif plot_name=='seeing':
            dict1 = {'col': 'GDRSEEJZ', 'plot_type': 'scatter', 'unit': 'as', 'plot_attr': {'label': 'Seeing in J+Z band', 'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'GDRSEEV',  'plot_type': 'scatter', 'unit': 'as', 'plot_attr': {'label': 'Seeing in V band',   'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1, dict2]
            thispaneldict = {'ylabel': 'Seeing (arcsec)',
                             'yscale': 'log',
                             'narrow_xlim_daily': 'true',
                             'title': 'Seeing',
                             'not_junk': 'true',
                             'on_sky': 'true', 
                             'legend_frac_size': 0.30}
            seeingpanel = {'panelvars': thispanelvars,
                           'paneldict': thispaneldict}
            panel_arr = [seeingpanel]

        elif plot_name=='sun_moon':
            dict1 = {'col': 'MOONSEP', 'plot_type': 'scatter', 'unit': 'deg', 'plot_attr': {'label': 'Moon-star separation', 'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'SUNALT',  'plot_type': 'scatter', 'unit': 'deg', 'plot_attr': {'label': 'Altitude of Sun',      'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1]
            thispaneldict = {'ylabel': 'Angle (deg)',
                             'narrow_xlim_daily': 'true',
                             'ylim': '(0,180)',
                             'axhspan': {
                                        1: {'ymin':  0, 'ymax': 30, 'color': 'red', 'alpha': 0.2},
                                        },
                             'not_junk': 'true',
                             'on_sky': 'true', 
                             'legend_frac_size': 0.30}
            sunpanel = {'panelvars': thispanelvars,
                        'paneldict': thispaneldict}
            thispanelvars = [dict2]
            thispaneldict = {'ylabel': 'Angle (deg)',
                             'title': 'Separation of Sun and Moon from Target',
                             'narrow_xlim_daily': 'true',
                             'ylim': '(-90,0)',
                             'axhspan': {
                                        1: {'ymin':  0, 'ymax':  -6, 'color': 'red',    'alpha': 0.2},
                                        2: {'ymin': -6, 'ymax': -12, 'color': 'orange', 'alpha': 0.2}
                                        },
                             'not_junk': 'true',
                             'on_sky': 'true', 
                             'legend_frac_size': 0.30}
            moonpanel = {'panelvars': thispanelvars,
                         'paneldict': thispaneldict}
            panel_arr = [sunpanel, moonpanel]

        elif plot_name=='drptag':
            dict1 = {'col': 'DRPTAG', 'plot_type': 'state', 'plot_attr': {'label': 'Version Number', 'marker': '.'}}
            thispanelvars = [dict1]
            thispaneldict = {'ylabel': 'DRP Version Number',
                             'title': 'KPF-Pipeline Version Number',
                             'not_junk': 'true',
                             'legend_frac_size': 0.10}
            drptagpanel = {'panelvars': thispanelvars,
                           'paneldict': thispaneldict}
            panel_arr = [drptagpanel]

        elif plot_name=='drphash':
            dict1 = {'col': 'DRPHASH', 'plot_type': 'state', 'plot_attr': {'label': 'Commit Hash', 'marker': '.'}}
            thispanelvars = [dict1]
            thispaneldict = {'ylabel': 'DRP Commit Hash',
                             'title': 'KPF-Pipeline Commit Hash String',
                             'not_junk': 'true',
                             'nolegend': 'true',
                             'legend_frac_size': 0.00}
            drphashpanel = {'panelvars': thispanelvars,
                            'paneldict': thispaneldict}
            panel_arr = [drphashpanel]

        elif plot_name=='junk_status':
            dict1 = {'col': 'NOTJUNK', 'plot_type': 'state', 'plot_attr': {'label': 'Junk State', 'marker': '.'}}
            thispanelvars = [dict1]
            thispaneldict = {'ylabel': 'Junk Status (1 = not junk)',
                             'title': 'Junk Status',
                             'legend_frac_size': 0.10}
            junkpanel = {'panelvars': thispanelvars,
                         'paneldict': thispaneldict}
            panel_arr = [junkpanel]

        # to-do: add 2D, L1, L2 QC keywords to the two panels below when those keywords are made
        elif plot_name=='qc_data_keywords_present':
            dict1 = {'col': 'DATAPRL0', 'plot_type': 'state', 'plot_attr': {'label': 'L0 Data Present', 'marker': '.'}}
            dict2 = {'col': 'KWRDPRL0', 'plot_type': 'state', 'plot_attr': {'label': 'L0 Keywords Present', 'marker': '.'}}
            thispanelvars = [dict1]
            thispaneldict = {'ylabel': 'L0 Data Present\n(1=True)',
                             'legend_frac_size': 0.10}
            data_present_panel = {'panelvars': thispanelvars,
                                  'paneldict': thispaneldict}
            thispanelvars = [dict2]
            thispaneldict = {'ylabel': 'L0 Keywords Present\n(1=True)',
                             'title': 'Quality Control - L0 Data and Keywords Products Present',
                             'legend_frac_size': 0.10}
            keywords_present_panel = {'panelvars': thispanelvars,
                                      'paneldict': thispaneldict}
            panel_arr = [data_present_panel, keywords_present_panel]

        elif plot_name=='qc_time_check':
            dict1 = {'col': 'TIMCHKL0', 'plot_type': 'state', 'plot_attr': {'label': 'L0 Time Check', 'marker': '.'}}
            dict2 = {'col': 'TIMCHKL2', 'plot_type': 'state', 'plot_attr': {'label': 'L2 Time Check', 'marker': '.'}}
            thispanelvars = [dict1]
            thispaneldict = {'ylabel': 'L0 Time Check\n(1=True)',
                             'legend_frac_size': 0.10}
            time_check_l0_panel = {'panelvars': thispanelvars,
                                   'paneldict': thispaneldict}
            thispanelvars = [dict2]
            thispaneldict = {'ylabel': 'L2 Time Check\n(1=True)',
                             'title': 'Quality Control - L0 and L2 Times Consistent',
                             'legend_frac_size': 0.10}
            time_check_l2_panel = {'panelvars': thispanelvars,
                                   'paneldict': thispaneldict}
            panel_arr = [time_check_l0_panel, time_check_l2_panel]

        elif plot_name=='qc_em':
            dict1 = {'col': 'EMSAT', 'plot_type': 'state', 'plot_attr': {'label': 'EM Not Saturated', 'marker': '.'}}
            dict2 = {'col': 'EMNEG', 'plot_type': 'state', 'plot_attr': {'label': 'EM Not Netative Flux', 'marker': '.'}}
            thispanelvars = [dict1]
            thispaneldict = {'ylabel': 'EM Not Saturated\n(1=True)',
                             'legend_frac_size': 0.10}
            emsat_panel = {'panelvars': thispanelvars,
                           'paneldict': thispaneldict}
            thispanelvars = [dict2]
            thispaneldict = {'ylabel': 'EM Not Netative Flux\n(1=True)',
                             'title': 'Quality Control - Exposure Meter',
                             'legend_frac_size': 0.10}
            emneg_panel = {'panelvars': thispanelvars,
                           'paneldict': thispaneldict}
            panel_arr = [emsat_panel, emneg_panel]

        elif plot_name=='autocal-flat_snr':
            dict1 = {'col': 'SNRSC452',  'plot_type': 'scatter', 'plot_attr': {'label': 'SNR (452 nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'darkviolet'}}
            dict2 = {'col': 'SNRSC548',  'plot_type': 'scatter', 'plot_attr': {'label': 'SNR (548 nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'blue'}}
            dict3 = {'col': 'SNRSC652',  'plot_type': 'scatter', 'plot_attr': {'label': 'SNR (652 nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict4 = {'col': 'SNRSC747',  'plot_type': 'scatter', 'plot_attr': {'label': 'SNR (747 nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'orange'}}
            dict5 = {'col': 'SNRCL852',  'plot_type': 'scatter', 'plot_attr': {'label': 'SNR (852 nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            thispanelvars = [dict1, dict2, dict3, dict4, dict5]
            thispaneldict = {'ylabel': 'SNR (SCI1+SCI2+SCI3)',
                             'only_object': 'autocal-flat-all',
                             'not_junk': 'true',
                             'legend_frac_size': 0.30}
            flat_snr_panel = {'panelvars': thispanelvars,
                              'paneldict': thispaneldict}
            dict1 = {'col': 'FR452652',  'plot_type': 'scatter', 'plot_attr': {'label': 'Flux Ratio (452/652nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'darkviolet'}}
            dict2 = {'col': 'FR548652',  'plot_type': 'scatter', 'plot_attr': {'label': 'Flux Ratio (548/652nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'blue'}}
            dict3 = {'col': 'FR747652',  'plot_type': 'scatter', 'plot_attr': {'label': 'Flux Ratio (747/652nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'orange'}}
            dict4 = {'col': 'FR852652',  'plot_type': 'scatter', 'plot_attr': {'label': 'Flux Ratio (852/652nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            thispanelvars = [dict1, dict2, dict3, dict4]
            thispaneldict = {'ylabel': 'Flux Ratio (SCI2)',
                             'title': 'autocal-flat-all SNR & Flux Ratio',
                             'only_object': 'autocal-flat-all',
                             'not_junk': 'true',
                             'legend_frac_size': 0.30}
            flat_fr_panel = {'panelvars': thispanelvars,
                             'paneldict': thispaneldict}
            panel_arr = [flat_snr_panel, flat_fr_panel]

        elif plot_name=='socal_snr':
            dict1 = {'col': 'SNRSC452',  'plot_type': 'scatter', 'plot_attr': {'label': 'SNR (452 nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'darkviolet'}}
            dict2 = {'col': 'SNRSC548',  'plot_type': 'scatter', 'plot_attr': {'label': 'SNR (548 nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'blue'}}
            dict3 = {'col': 'SNRSC652',  'plot_type': 'scatter', 'plot_attr': {'label': 'SNR (652 nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict4 = {'col': 'SNRSC747',  'plot_type': 'scatter', 'plot_attr': {'label': 'SNR (747 nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'orange'}}
            dict5 = {'col': 'SNRCL852',  'plot_type': 'scatter', 'plot_attr': {'label': 'SNR (852 nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            thispanelvars = [dict1, dict2, dict3, dict4, dict5]
            thispaneldict = {'ylabel': 'SNR (SCI1+SCI2+SCI3)',
                             'only_object': 'SoCal',
                             'narrow_xlim_daily': 'true',
                             'not_junk': 'true',
                             'legend_frac_size': 0.30}
            socal_snr_panel = {'panelvars': thispanelvars,
                               'paneldict': thispaneldict}
            dict1 = {'col': 'FR452652',  'plot_type': 'scatter', 'plot_attr': {'label': 'Flux Ratio (452/652nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'darkviolet'}}
            dict2 = {'col': 'FR548652',  'plot_type': 'scatter', 'plot_attr': {'label': 'Flux Ratio (548/652nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'blue'}}
            dict3 = {'col': 'FR747652',  'plot_type': 'scatter', 'plot_attr': {'label': 'Flux Ratio (747/652nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'orange'}}
            dict4 = {'col': 'FR852652',  'plot_type': 'scatter', 'plot_attr': {'label': 'Flux Ratio (852/652nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            thispanelvars = [dict1, dict2, dict3, dict4]
            thispaneldict = {'ylabel': 'Flux Ratio (SCI2)',
                             'title': 'SoCal SNR & Flux Ratio',
                             'only_object': 'SoCal',
                             'narrow_xlim_daily': 'true',
                             'not_junk': 'true',
                             'legend_frac_size': 0.30}
            socal_fr_panel = {'panelvars': thispanelvars,
                              'paneldict': thispaneldict}
            panel_arr = [socal_snr_panel, socal_fr_panel]

        elif plot_name=='observing_snr':
            dict1 = {'col': 'SNRSC452',  'plot_type': 'scatter', 'plot_attr': {'label': 'SNR (452 nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'darkviolet'}}
            dict2 = {'col': 'SNRSC548',  'plot_type': 'scatter', 'plot_attr': {'label': 'SNR (548 nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'blue'}}
            dict3 = {'col': 'SNRSC652',  'plot_type': 'scatter', 'plot_attr': {'label': 'SNR (652 nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict4 = {'col': 'SNRSC747',  'plot_type': 'scatter', 'plot_attr': {'label': 'SNR (747 nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'orange'}}
            dict5 = {'col': 'SNRCL852',  'plot_type': 'scatter', 'plot_attr': {'label': 'SNR (852 nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            thispanelvars = [dict1, dict2, dict3, dict4, dict5]
            thispaneldict = {'ylabel': 'SNR (SCI1+SCI2+SCI3)',
                             'on_sky': 'true', 
                             'narrow_xlim_daily': 'true',
                             'not_junk': 'true',
                             'legend_frac_size': 0.30}
            observing_snr_panel = {'panelvars': thispanelvars,
                                   'paneldict': thispaneldict}
            dict1 = {'col': 'FR452652',  'plot_type': 'scatter', 'plot_attr': {'label': 'Flux Ratio (452/652nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'darkviolet'}}
            dict2 = {'col': 'FR548652',  'plot_type': 'scatter', 'plot_attr': {'label': 'Flux Ratio (548/652nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'blue'}}
            dict3 = {'col': 'FR747652',  'plot_type': 'scatter', 'plot_attr': {'label': 'Flux Ratio (747/652nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'orange'}}
            dict4 = {'col': 'FR852652',  'plot_type': 'scatter', 'plot_attr': {'label': 'Flux Ratio (852/652nm)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            thispanelvars = [dict1, dict2, dict3, dict4]
            thispaneldict = {'ylabel': 'Flux Ratio (SCI2)',
                             'title': 'SoCal SNR & Flux Ratio',
                             'on_sky': 'true', 
                             'narrow_xlim_daily': 'true',
                             'not_junk': 'true',
                             'legend_frac_size': 0.30}
            observing_fr_panel = {'panelvars': thispanelvars,
                                  'paneldict': thispaneldict}
            panel_arr = [observing_snr_panel, observing_fr_panel]

        elif plot_name=='autocal_rv':
            dict1 = {'col': 'CCD1RV1',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV1 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict2 = {'col': 'CCD1RV2',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV2 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict3 = {'col': 'CCD1RV3',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV3 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict4 = {'col': 'CCD1RVC',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV3 (km/s)',  'marker': 's', 'linewidth': 0.5, 'color': 'limegreen'}}
            dict5 = {'col': 'CCD2RV1',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RV1 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            dict6 = {'col': 'CCD2RV2',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RV2 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            dict7 = {'col': 'CCD2RV3',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RV3 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            dict8 = {'col': 'CCD2RVC',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RV3 (km/s)',  'marker': 's', 'linewidth': 0.5, 'color': 'indianred'}}
            thispanelvars = [dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8]
            thispaneldict = {
                             'ylabel': r'LFC RV (km/s)',
                             #'ylabel': r'LFC $\Delta$RV (km/s)',
                             #'subtractmedian': 'true',
                             'only_object': '["autocal-lfc-all-morn", "autocal-lfc-all-eve", "autocal-lfc-all-night", "cal-LFC", "cal-LFC-morn", "cal-LFC-eve", "LFC_all", "lfc_all", "LFC"]',
                             'not_junk': 'true',
                             'legend_frac_size': 0.30
                             }
            lfc_rv_panel = {'panelvars': thispanelvars,
                            'paneldict': thispaneldict}
            dict11 = {'col': 'CCD1RV1',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV1 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict12 = {'col': 'CCD1RV2',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV2 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict13 = {'col': 'CCD1RV3',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV3 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict14 = {'col': 'CCD1RVC',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RVC (km/s)',  'marker': 's', 'linewidth': 0.5, 'color': 'limegreen'}}
            dict15 = {'col': 'CCD2RV1',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RV1 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            dict16 = {'col': 'CCD2RV2',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RV2 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            dict17 = {'col': 'CCD2RV3',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RV3 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            dict18 = {'col': 'CCD2RVC',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RVC (km/s)',  'marker': 's', 'linewidth': 0.5, 'color': 'indianred'}}
            thispanelvars2 = [dict11, dict12, dict13, dict14, dict15, dict16, dict17, dict18]
            thispaneldict2 = {
                              'ylabel': r'ThAr RV (km/s)',
                              #'ylabel': r'Etalon $\Delta$RV (km/s)',
                              #'subtractmedian': 'true',
                              'only_object': '["autocal-thar-all-night", "autocal-thar-all-eve", "autocal-thar-all-morn"]',
                              'not_junk': 'true',
                              'legend_frac_size': 0.30
                              }
            thar_rv_panel = {'panelvars': thispanelvars2,
                             'paneldict': thispaneldict2}
            dict21 = {'col': 'CCD1RV1',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV1 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict22 = {'col': 'CCD1RV2',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV2 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict23 = {'col': 'CCD1RV3',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV3 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict24 = {'col': 'CCD1RVC',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RVC (km/s)',  'marker': 's', 'linewidth': 0.5, 'color': 'limegreen'}}
            dict25 = {'col': 'CCD2RV1',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RV1 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            dict26 = {'col': 'CCD2RV2',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RV2 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            dict27 = {'col': 'CCD2RV3',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RV3 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            dict28 = {'col': 'CCD2RVC',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RVC (km/s)',  'marker': 's', 'linewidth': 0.5, 'color': 'indianred'}}
            thispanelvars3 = [dict21, dict22, dict23, dict24, dict25, dict26, dict27, dict28]
            thispaneldict3 = {
                              'title': 'LFC, ThAr, & Etalon RVs',
                              'ylabel': r'Etalon RV (km/s)',
                              #'ylabel': r'Etalon $\Delta$RV (km/s)',
                              #'subtractmedian': 'true',
                              'only_object': '["autocal-etalon-all-night", "autocal-etalon-all-eve", "autocal-etalon-all-morn", "manualcal-etalon-all", "Etalon_cal", "etalon-sequence"]',
                              'not_junk': 'true',
                              'legend_frac_size': 0.30
                              }
            etalon_rv_panel = {'panelvars': thispanelvars3,
                               'paneldict': thispaneldict3}
            panel_arr = [lfc_rv_panel, thar_rv_panel, etalon_rv_panel]

        elif plot_name=='socal_rv':
            dict1 = {'col': 'CCD1RV1',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV1 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict2 = {'col': 'CCD1RV2',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV2 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict3 = {'col': 'CCD1RV3',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV3 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict4 = {'col': 'CCD1RVC',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV3 (km/s)',  'marker': 's', 'linewidth': 0.5, 'color': 'limegreen'}}
            dict5 = {'col': 'CCD2RV1',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RV1 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            dict6 = {'col': 'CCD2RV2',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RV2 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            dict7 = {'col': 'CCD2RV3',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RV3 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            dict8 = {'col': 'CCD2RVC',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RVC (km/s)',  'marker': 's', 'linewidth': 0.5, 'color': 'indianred'}}
            thispanelvars = [dict1, dict2, dict3, dict5, dict6, dict7]
            thispaneldict = {
                             'ylabel': r'SoCal RV (km/s)',
                             'title': 'SoCal RVs',
                             'only_object': '["SoCal"]',
                             'narrow_xlim_daily': 'true',
                             'not_junk': 'true',
                             'legend_frac_size': 0.28
                             }
            socal_rv_panel = {'panelvars': thispanelvars,
                              'paneldict': thispaneldict}
            dict11 = {'col': 'CCD1RV1',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV1 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict12 = {'col': 'CCD1RV2',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV2 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict13 = {'col': 'CCD1RV3',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV3 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'green'}}
            dict14 = {'col': 'CCD1RVC',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD1RV3 (km/s)',  'marker': 's', 'linewidth': 0.5, 'color': 'limegreen'}}
            dict15 = {'col': 'CCD2RV1',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RV1 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            dict16 = {'col': 'CCD2RV2',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RV2 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            dict17 = {'col': 'CCD2RV3',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RV3 (km/s)',  'marker': '.', 'linewidth': 0.5, 'color': 'red'}}
            dict18 = {'col': 'CCD2RVC',  'plot_type': 'plot', 'plot_attr': {'label': 'CCD2RVC (km/s)',  'marker': 's', 'linewidth': 0.5, 'color': 'indianred'}}
            thispanelvars = [dict11, dict12, dict13, dict15, dict16, dict17]
            thispaneldict = {
                             'ylabel': r'SoCal $\Delta$RV (km/s)',
                             'subtractmedian': 'true',
                             'title': 'SoCal RVs',
                             'only_object': '["SoCal"]',
                             'narrow_xlim_daily': 'true',
                             'not_junk': 'true',
                             'legend_frac_size': 0.28
                             }
            socal_rv_panel2 = {'panelvars': thispanelvars,
                               'paneldict': thispaneldict}
            panel_arr = [socal_rv_panel,socal_rv_panel2]

        else:
            self.logger.error('plot_name not specified')
            return
        
        self.plot_time_series_multipanel(panel_arr, start_date=start_date, end_date=end_date, 
                                         fig_path=fig_path, show_plot=show_plot, clean=clean, 
                                         log_savefig_timing=False)        


    def plot_all_quicklook(self, start_date=None, interval=None, clean=True, 
                                 last_n_days=None,
                                 fig_dir=None, show_plot=False, 
                                 print_plot_names=False):
        """
        Generate all of the standard time series plots for the quicklook.  
        Depending on the value of the input 'interval', the plots have time ranges 
        that are daily, weekly, yearly, or decadal.

        Args:
            start_date (datetime object) - start date for plot
            interval (string) - 'day', 'week', 'year', or 'decade'
            last_n_days (int) - overrides start_date and makes a plot over the last n days
            fig_path (string) - set to the path for the files to be generated.
            show_plot (boolean) - show the plot in the current environment.
            print_plot_names (boolean) - prints the names of possible plots and exits

        Returns:
            PNG plot in fig_path or shows the plots it the current environment
            (e.g., in a Jupyter Notebook).
        """
        plots = { 
            "p1a":  {"plot_name": "hallway_temp",             "subdir": "Chamber",   "desc": "Hallway temperature"},
            "p1b":  {"plot_name": "chamber_temp",             "subdir": "Chamber",   "desc": "Vacuum chamber temperatures"},
            "p1c":  {"plot_name": "chamber_temp_detail",      "subdir": "Chamber",   "desc": "Vacuum chamber temperatures (by optical element)"},
            "p1d":  {"plot_name": "fiber_temp",               "subdir": "Chamber",   "desc": "Fiber scrambler temperatures"},
            "p2a":  {"plot_name": "ccd_readnoise",            "subdir": "CCDs",      "desc": "CCD readnoise"},
            "p2b":  {"plot_name": "ccd_dark_current",         "subdir": "CCDs",      "desc": "CCD dark current"},
            "p2c":  {"plot_name": "ccd_readspeed",            "subdir": "CCDs",      "desc": "CCD read speed"},
            "p2d":  {"plot_name": "ccd_controller",           "subdir": "CCDs",      "desc": "CCD controller temperatures"},
            "p2e":  {"plot_name": "ccd_temp",                 "subdir": "CCDs",      "desc": "CCD temperatures"},
            "p3a":  {"plot_name": "lfc",                      "subdir": "Cal",       "desc": "LFC parameters"},
            "p3b":  {"plot_name": "etalon",                   "subdir": "Cal",       "desc": "Etalon temperatures"},
            "p3c":  {"plot_name": "hcl",                      "subdir": "Cal",       "desc": "Hollow-cathode lamp temperatures"},
            "p3d":  {"plot_name": "autocal-flat_snr",         "subdir": "Cal",       "desc": "SNR of flats"},
            "p4a":  {"plot_name": "hk_temp",                  "subdir": "Subsystems","desc": "Ca H&K Spectrometer temperatures"},
            "p4b":  {"plot_name": "agitator",                 "subdir": "Subsystems","desc": "Agatitator temperatures"},
            "p5a":  {"plot_name": "guiding",                  "subdir": "Observing", "desc": "FIU Guiding performance of"},
            "p5b":  {"plot_name": "seeing",                   "subdir": "Observing", "desc": "Seeing measurements for stars"},
            "p5c":  {"plot_name": "sun_moon",                 "subdir": "Observing", "desc": "Target separation to Sun and Moon"},
            "p5c":  {"plot_name": "observing_snr",            "subdir": "Observing", "desc": "SNR of stellar spectra"},
            "p6a":  {"plot_name": "socal_snr",                "subdir": "SoCal",     "desc": "SNR of SoCal spectra"},
            "p6b":  {"plot_name": "socal_rv",                 "subdir": "RV",        "desc": "RVs from SoCal spectra"}, 
            "p7a":  {"plot_name": "drptag",                   "subdir": "DRP",       "desc": "DRP Tag"},   
            "p7b":  {"plot_name": "drphash",                  "subdir": "DRP",       "desc": "DRP Hash"},   
            "p8a":  {"plot_name": "junk_status",              "subdir": "QC",        "desc": "Quality control: junk status"}, 
            "p8b":  {"plot_name": "qc_data_keywords_present", "subdir": "QC",        "desc": "Quality Control: keywords present"}, 
            "p8c":  {"plot_name": "qc_time_check",            "subdir": "QC",        "desc": "Quality Control: time checks"}, 
            "p8d":  {"plot_name": "qc_em",                    "subdir": "QC",        "desc": "Quality Control: Exposure Meter"}, 
            "p9a":  {"plot_name": "autocal_rv",               "subdir": "RV",        "desc": "RVs from LFC, ThAr, and etalon spectra"}, 
        }
        if print_plot_names:
            print("Plots available in AnalyzeTimeSeries.plot_standard_time_series():")
            for p in plots:
                print("    '" + plots[p]["plot_name"] + "': " + plots[p]["desc"])
            return

        if (last_n_days != None) and (type(last_n_days) == type(1)):
            now = datetime.now()
            if last_n_days > 3:
                end_date = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                end_date = now
            start_date = end_date - timedelta(days=last_n_days)

        if not isinstance(start_date, datetime):
            self.logger.error("'start_date' must be a datetime object.")
            return        
        
        for p in plots:
            plot_name = plots[p]["plot_name"]
            if interval == 'day':
                end_date = start_date + timedelta(days=1)
                filename = 'kpf_' + start_date.strftime("%Y%m%d") + '_telemetry_' + plot_name + '.png' 
            elif interval == 'month':
                end_date = add_one_month(start_date)
                filename = 'kpf_' + start_date.strftime("%Y%m") + '_telemetry_' + plot_name + '.png' 
            elif interval == 'year':
                end_date = datetime(start_date.year+1, start_date.month, start_date.day)
                filename = 'kpf_' + start_date.strftime("%Y") + '_telemetry_' + plot_name + '.png' 
            elif interval == 'decade':
                end_date = datetime(start_date.year+10, start_date.month, start_date.day)
                filename = 'kpf_' + start_date.strftime("%Y")[0:3] + '0_telemetry_' + plot_name + '.png' 
            elif (last_n_days != None) and (type(last_n_days) == type(1)):
                filename = 'kpf_last' + str(last_n_days) + 'days_telemetry_' + plot_name + '.png'                 
            else:
                self.logger.error("The input 'interval' must be 'daily', 'weekly', 'yearly', or 'decadal'.")
                return

            if fig_dir != None:
                if not fig_dir.endswith('/'):
                    fig_dir += '/'
                savedir = fig_dir + plots[p]["subdir"] + '/'
                os.makedirs(savedir, exist_ok=True) # make directories if needed
                fig_path = savedir + filename
                self.logger.info('Making QL time series plot ' + fig_path)
            else:
                fig_path = None
            self.plot_standard_time_series(plot_name, start_date=start_date, end_date=end_date, 
                                           fig_path=fig_path, show_plot=show_plot, clean=clean)


    def plot_all_quicklook_daterange(self, start_date=None, end_date=None, 
                                     time_range_type = 'all', clean=True, 
                                     base_dir='/data/QLP/', show_plot=False):
        """
        Generate all of the standard time series plots for the quicklook for a date 
        range.  Every unique day, month, year, and decade between start_date and end_date 
        will have a full set of plots produced using plot_all_quicklook().
        The set of date range types ('day', 'month', 'year', 'decade', 'all')
        is set by the time_range_type parameter.

        Args:
            start_date (datetime object) - start date for plot
            end_date (datetime object) - start date for plot
            time_range_type (string)- one of: 'day', 'month', 'year', 'decade', 'all'
            base_dir (string) - set to the path for the files to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plots in the output director or shows the plots it the current 
            environment (e.g., in a Jupyter Notebook).
        """
        time_range_type = time_range_type.lower()
        if time_range_type not in ['day', 'month', 'year', 'decade', 'all']:
            time_range_type = 'all'

        days = []
        months = []
        years = []
        decades = []
        current_date = start_date
        while current_date <= end_date:
            days.append(current_date)
            months.append(datetime(current_date.year,current_date.month,1))
            years.append(datetime(current_date.year,1,1))
            decades.append(datetime(int(str(current_date.year)[0:3])*10,1,1))
            current_date += timedelta(days=1)
        days    = sorted(set(days),    reverse=True)
        months  = sorted(set(months),  reverse=True)
        years   = sorted(set(years),   reverse=True)
        decades = sorted(set(decades), reverse=True)

        if time_range_type in ['day', 'all']:
            self.logger.info('Making time series plots for ' + str(len(days)) + ' day(s)')
            for day in days:
                try:
                    if base_dir != None:
                        savedir = base_dir + day.strftime("%Y%m%d") + '/Masters/'
                    else:
                        savedir = None
                    self.plot_all_quicklook(day, interval='day', fig_dir=savedir, show_plot=show_plot)
                except Exception as e:
                    self.logger.error(e)
    
        if time_range_type in ['month', 'all']:
            self.logger.info('Making time series plots for ' + str(len(months)) + ' month(s)')
            for month in months:
                try:
                    if base_dir != None:
                        savedir = base_dir + month.strftime("%Y%m") + '00/Masters/'
                    else:
                        savedir = None
                    self.plot_all_quicklook(month, interval='month', fig_dir=savedir)
                except Exception as e:
                    self.logger.error(e)
    
        if time_range_type in ['year', 'all']:
            self.logger.info('Making time series plots for ' + str(len(years)) + ' year(s)')
            for year in years:
                try:
                    if base_dir != None:
                        savedir = base_dir + year.strftime("%Y") + '0000/Masters/'
                    else:
                        savedir = None
                    self.plot_all_quicklook(year, interval='year', fig_dir=savedir)
                except Exception as e:
                    self.logger.error(e)
    
        if time_range_type in ['decade', 'all']:
            self.logger.info('Making time series plots for ' + str(len(decades)) + ' decade(s)')
            for decade in decades:
                try:
                    if base_dir != None:
                        savedir = base_dir + decade.strftime("%Y")[0:3] + '00000/Masters/' 
                    else:
                        savedir = None
                    self.plot_all_quicklook(decade, interval='decade', fig_dir=savedir)
                except Exception as e:
                    self.logger.error(e)


def add_one_month(inputdate):
    """
    Add one month to a datetime object, accounting for the number of days per month.
    """
    year, month, day = inputdate.year, inputdate.month, inputdate.day
    if month == 12:
        month = 1
        year += 1
    else:
        month += 1
    if month in [4, 6, 9, 11] and day > 30:
        day = 30
    elif month == 2:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            if day > 29:
                day = 29
        else:
            if day > 28:
                day = 28
    
    outputdate = datetime(year, month, day)
    return outputdate

def convert_to_list_if_array(string):
    """
    Convert a string like '["autocal-lfc-all-morn", "autocal-lfc-all-eve"]' to an array.
    """
    # Check if the string starts with '[' and ends with ']'
    if string.startswith('[') and string.endswith(']'):
        try:
            # Attempt to parse the string as JSON
            return json.loads(string)
        except json.JSONDecodeError:
            # The string is not a valid JSON array
            return string
    else:
        # The string does not look like a JSON array
        return string
