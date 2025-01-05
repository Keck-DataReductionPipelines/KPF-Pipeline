import os
import ast
import time
import glob
import copy
import json
import yaml
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
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from matplotlib.dates import HourLocator, DayLocator, MonthLocator, YearLocator, AutoDateLocator, DateFormatter

import sys
if sys.version_info >= (3, 9):
    from importlib import resources
else:
    import importlib_resources as resources

import cProfile
import pstats
from io import StringIO

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
        L0_keyword_types (dictionary) - specifies data types for L0 header keywords
        D2_keyword_types (dictionary) - specifies data types for 2D header keywords
        L1_keyword_types (dictionary) - specifies data types for L1 header keywords
        L2_keyword_types (dictionary) - specifies data types for L2 header keywords
        L0_telemetry_types (dictionary) - specifies data types for L0 telemetry keywords
        L2_RV_header_keyword_types (dictionary) - specifies data types for L2 RV header keywords
        L2_RV_ccf_keyword_types (dictionary) - specifies data types for L2 CCF header keywords

    Related Commandline Scripts:
        'ingest_dates_kpf_tsdb.py' - ingest from a range of dates
        'ingest_watch_kpf_tsdb.py' - ingest by watching a set of directories
        'generate_time_series_plots.py' - creates standard time series plots
        
    To-do:
        * Add database for masters (separate from ObsIDs?)
        * Method to return the avg, std., etc. for a DB column over a time range, with conditions (e.g., fast-read mode only)
        * Make plots of temperature vs. RV for various types of RVs
        * Add standard plots of flux vs. time for cals (all types?), stars, and solar -- highlight Junked files
        * Add methods to print the schema
        * Augment statistics in legends (median and stddev upon request)
        * Add the capability of using one DB for ingestion into another or plotting
        * check mod times before issuing parallel threads
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
        self.L0_telemetry_types          = self.get_keyword_types(level='L0_telemetry')
        self.D2_header_keyword_types     = self.get_keyword_types(level='2D')
        self.L1_header_keyword_types     = self.get_keyword_types(level='L1')
        self.L2_header_keyword_types     = self.get_keyword_types(level='L2')
        self.L2_CCF_header_keyword_types = self.get_keyword_types(level='L2_CCF_header')
        self.L2_RV_header_keyword_types  = self.get_keyword_types(level='L2_RV_header')
        self.L2_RV_keyword_types         = self.get_keyword_types(level='L2_RV')
        
        if drop:
            self.drop_table()
            self.logger.info('Dropping KPF database ' + str(self.db_path))

        # the line below might be modified so that if the database exists, then the columns are read from it
        self.create_database()
        self.print_db_status()


    def create_database(self):
        """
        Create SQLite3 database using the standard KPF scheme.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint")
        cursor.execute("PRAGMA cache_size = -2000000;")
    
        # Define columns for each file type
        L0_header_cols     = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L0_header_keyword_types.items()]
        L0_telemetry_cols  = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L0_telemetry_types.items()]
        D2_header_cols     = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.D2_header_keyword_types.items()]
        L1_header_cols     = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L1_header_keyword_types.items()]
        L2_header_cols     = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L2_header_keyword_types.items()]
        L2_CCF_header_cols = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L2_CCF_header_keyword_types.items()]
        L2_RV_header_cols  = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L2_RV_header_keyword_types.items()]
        L2_RV_cols         = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L2_RV_keyword_types.items()]
        cols = L0_header_cols + L0_telemetry_cols + D2_header_cols + L1_header_cols + L2_header_cols + L2_CCF_header_cols + L2_RV_header_cols + L2_RV_cols
        cols += ['"datecode" TEXT', '"ObsID" TEXT']
        cols += ['"L0_filename" TEXT', '"D2_filename" TEXT', '"L1_filename" TEXT', '"L2_filename" TEXT', ]
        cols += ['"L0_header_read_time" TEXT', '"D2_header_read_time" TEXT', '"L1_header_read_time" TEXT', '"L2_header_read_time" TEXT', ]
        cols += ['"Source" TEXT']
        create_table_query = f'CREATE TABLE IF NOT EXISTS kpfdb ({", ".join(cols)}, UNIQUE(ObsID))'
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


    def drop_table(self):
        """
        Start over on the database by dropping the main table.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS kpfdb")
        conn.commit()
        conn.close()


    def unlock_db(self):
        """
        Remove the -wal and -shm lock files, 
        e.g. /data/time_series/kpf_ts.db-wal and /data/time_series/kpf_ts.db-shm
        
        Use this method sparingly.
        """
        wal_file = f"{self.db_path}-wal"
        shm_file = f"{self.db_path}-shm"
    
        if os.path.exists(wal_file):
            os.remove(wal_file)
        if os.path.exists(shm_file):
            os.remove(shm_file)


    def ingest_dates_to_db(self, start_date_str, end_date_str, batch_size=100, reverse=False, quiet=False):
        """
        Ingest KPF data for the date range start_date to end_date, inclusive.
        batch_size refers to the number of observations per DB insertion.
        """
        if not quiet:
            self.logger.info("Adding to database between " + start_date_str + " and " + end_date_str)
        dir_paths = glob.glob(f"{self.base_dir}/????????")
        sorted_dir_paths = sorted(dir_paths, key=lambda x: int(os.path.basename(x)), reverse=start_date_str > end_date_str)
        filtered_dir_paths = [
            dir_path for dir_path in sorted_dir_paths
            if start_date_str <= os.path.basename(dir_path) <= end_date_str
        ]
        if len(filtered_dir_paths) > 0:
            # Reverse dates if the reverse flag is set
            if reverse:
                filtered_dir_paths.reverse()
            
            # Iterate over date directories
            t1 = self.tqdm(filtered_dir_paths, desc=(filtered_dir_paths[0]).split('/')[-1], disable=quiet)
            for dir_path in t1:
                t1.set_description(dir_path.split('/')[-1])
                t1.refresh() 
                t2 = self.tqdm(os.listdir(dir_path), desc=f'Files', leave=False, disable=quiet)
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
        if not quiet:
            self.logger.info(f"Files for {len(filtered_dir_paths)} days ingested/checked")


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

        self.logger.info(f'{ObsID_filename} read with {str(len(df))} properly formatted ObsIDs.')

        #t = tqdm_notebook(df.iloc[:, 0].tolist(), desc=f'ObsIDs', leave=True)
        t = self.tqdm(df.iloc[:, 0].tolist(), desc=f'ObsIDs', leave=True)
        for ObsID in t:
            dir_path = self.base_dir + '/' + get_datecode(ObsID) + '/'
            filename = ObsID + '.fits'
            file_path = os.path.join(dir_path, filename)
            base_filename = filename.split('.fits')[0]
            t.set_description(base_filename)
            t.refresh() 
            try:
                if os.path.exists(ObsID_filename):
                    self.ingest_one_observation(dir_path, filename) 
            except Exception as e:
                self.logger.error(e)


    def add_ObsIDs_to_db(self, ObsID_list):
        """
        Ingest files into the database from a list of strings 'ObsID_list'.  
        """
        t = self.tqdm(ObsID_list, desc=f'ObsIDs', leave=True)
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

        # update the DB if necessary
        if self.is_any_file_updated(L0_file_path):
        
            D2_file_path = f"{dir_path.replace('L0', '2D')}/{base_filename}_2D.fits"
            L1_file_path = f"{dir_path.replace('L0', 'L1')}/{base_filename}_L1.fits"
            L2_file_path = f"{dir_path.replace('L0', 'L2')}/{base_filename}_L2.fits"
            D2_filename  = f"{L0_filename.replace('L0', '2D')}"
            L1_filename  = f"{L0_filename.replace('L0', 'L1')}"
            L2_filename  = f"{L0_filename.replace('L0', 'L2')}"

            L0_header_data    = self.extract_kwd(L0_file_path, self.L0_header_keyword_types, extension='PRIMARY') 
            L0_telemetry      = self.extract_telemetry(L0_file_path, self.L0_telemetry_types)
            D2_header_data    = self.extract_kwd(D2_file_path, self.D2_header_keyword_types, extension='PRIMARY') 
            L1_header_data    = self.extract_kwd(L1_file_path, self.L1_header_keyword_types, extension='PRIMARY') 
            L2_header_data    = self.extract_kwd(L2_file_path, self.L2_header_keyword_types, extension='PRIMARY') 
            L2_RV_header_data = self.extract_kwd(L2_file_path, self.L2_RV_header_keyword_types, extension='RV') 
            L2_CCF_header_data= self.extract_kwd(L2_file_path, self.L2_CCF_header_keyword_types, extension='GREEN_CCF') 
            L2_RV_header_data = self.extract_kwd(L2_file_path, self.L2_RV_header_keyword_types, extension='RV') 
            L2_RV_data        = self.extract_rvs(L2_file_path) 

            header_data = {**L0_header_data, 
                           **L0_telemetry,
                           **D2_header_data, 
                           **L1_header_data, 
                           **L2_header_data, 
                           **L2_CCF_header_data, 
                           **L2_RV_header_data, 
                           **L2_RV_data, 
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
        Ingest a batch of observations into the database in parallel using 
        ProcessPoolExecutor, but check if each file has been updated before 
        parallel processing, to reduce overhead.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
        # === 1) Check for updated files in main thread ===
        updated_batch = []
        for file_path in batch:
            if self.is_any_file_updated(file_path):
                updated_batch.append(file_path)
    
        # If nothing to do, exit quickly
        if not updated_batch:
            return
    
        # === 2) Prepare arguments for parallel execution ===
        args = {
            'now_str': now_str,
            'L0_header_keyword_types': self.L0_header_keyword_types,
            'L0_telemetry_types': self.L0_telemetry_types,
            'D2_header_keyword_types': self.D2_header_keyword_types,
            'L1_header_keyword_types': self.L1_header_keyword_types,
            'L2_header_keyword_types': self.L2_header_keyword_types,
            'L2_CCF_header_keyword_types': self.L2_CCF_header_keyword_types,
            'L2_RV_header_keyword_types': self.L2_RV_header_keyword_types,
            'extract_kwd_func': self.extract_kwd,
            'extract_telemetry_func': self.extract_telemetry,
            'extract_rvs_func': self.extract_rvs,
#            'is_any_file_updated_func': self.is_any_file_updated,
            'get_source_func': self.get_source,
            'get_datecode_func': get_datecode  # Assuming get_datecode is a standalone function
        }
    
        partial_process_file = partial(process_file, **args)
    
        # === 3) Run extraction in parallel ONLY for updated files ===
        max_workers = min([len(updated_batch), 20, os.cpu_count()])
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(partial_process_file, updated_batch))
    
        # Filter out None results (though now we expect fewer None’s, 
        # because we already did the update check in the main thread)
        batch_data = [res for res in results if res is not None]
    
        # === 4) Perform bulk insert ===
        if batch_data:
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
        Extract keywords from keyword_types.keys from an extension in a L0/2D/L1/L2 file.
        """
        # Initialize the result dictionary with None for all keywords
        header_data = {key: None for key in keyword_types.keys()}
    
        # Check if the file exists before proceeding
        if not os.path.isfile(file_path):
            return header_data
    
        try:
            # Open the FITS file and read the specified header
            with fits.open(file_path, memmap=True) as hdul:
                header = hdul[extension].header
    
                # Use dictionary comprehension to populate header_data
                header_data = {key: header.get(key, None) for key in keyword_types.keys()}
        except Exception as e:
            # Log any issues with the file
            self.logger.info(f"Bad file: {file_path}. Error: {e}")
    
        return header_data


    def extract_telemetry(self, file_path, keyword_types):
        """
        Extract telemetry from the 'TELEMETRY' extension in a KPF L0 file.
        """
        try:
            # Use astropy's Table to load only necessary data
            telemetry_table = Table.read(file_path, format='fits', hdu='TELEMETRY')
            keywords = telemetry_table['keyword']
            averages = telemetry_table['average']
        except Exception as e:
            self.logger.info(f"Bad TELEMETRY extension in: {file_path}. Error: {e}")
            return {key: None for key in keyword_types}
    
        try:
            # Decode and sanitize 'keyword' column
            keywords = [k.decode('utf-8') if isinstance(k, bytes) else k for k in keywords]
    
            # Replace invalid values efficiently using NumPy
            averages = np.array(averages, dtype=object)  # Convert to object to allow mixed types
            mask_invalid = np.isin(averages, ['-nan', 'nan', -999]) | np.isnan(pd.to_numeric(averages, errors='coerce'))
            averages[mask_invalid] = np.nan
            averages = averages.astype(float)  # Convert valid data to float
    
            # Create the telemetry dictionary
            telemetry_data = dict(zip(keywords, averages))
    
            # Build the output dictionary for the requested keywords
            telemetry_dict = {key: float(telemetry_data.get(key, np.nan)) for key in keyword_types}
        except Exception as e:
            self.logger.info(f"Error processing TELEMETRY data in: {file_path}. Error: {e}")
            telemetry_dict = {key: None for key in keyword_types}
    
        return telemetry_dict


    def extract_rvs(self, file_path):
        """
        Extract RVs from the 'RV' extension in a KPF L2 file.
        """
        mapping = {
            'orderlet1':   'RV1{}',
            'orderlet2':   'RV2{}',
            'orderlet3':   'RV3{}',
            'RV':          'RVS{}',
            'RV error':    'ERVS{}',
            'CAL RV':      'RVC{}',
            'CAL error':   'ERVC{}',
            'SKY RV':      'RVY{}',
            'SKY error':   'ERVY{}',
            'CCFBJD':      'CCFBJD{}',
            'Bary_RVC':    'BCRV{}',
            'CCF Weights': 'CCFW{}',
        }
    
        cols = ['orderlet1', 'orderlet2', 'orderlet3', 'RV', 'RV error', 
                'CAL RV', 'CAL error', 'SKY RV', 'SKY error', 'CCFBJD', 
                'Bary_RVC', 'CCF Weights']
    
        expected_count = 67 * len(cols)
    
        def make_dummy_dict():
            keys = []
            for i in range(0, 67):
                NN = f"{i:02d}"  # two-digit row number, from 00 to 66
                for pattern in mapping.values():
                    keys.append(pattern.format(NN))
            return {key: None for key in keys}
        
        try:
            df_rv = Table.read(file_path, format='fits', hdu='RV').to_pandas()
            df_rv = df_rv[cols]
        except Exception as e:
            # If we can't read RVs, return a dict with None values for all expected keys
            rv_dict = make_dummy_dict()
            return rv_dict
    
        df_filtered = df_rv[list(mapping.keys())]
        stacked = df_filtered.stack()
        keyed = stacked.reset_index()
        keyed.columns = ['row_idx', 'col', 'val']
        keyed['NN'] = keyed['row_idx'].apply(lambda x: f"{x:02d}")  # zero-based indexing
        keyed['key'] = keyed['col'].map(mapping)
        keyed['key'] = keyed['key'].str[:-2] + keyed['NN']
        rv_dict = dict(zip(keyed['key'], keyed['val']))
    
        # Check the count - if data wasn't computed for Green or Red, the database will give an error on insertion
        if len(rv_dict) != expected_count:
            # If count doesn't match, return the dummy dictionary of None values
            rv_dict = make_dummy_dict()
            
        return rv_dict        

        
    def clean_df(self, df):
        """
        Remove known outliers from a dataframe.
        """
        # CCD Read Noise
        cols = ['RNGREEN1', 'RNGREEN2', 'RNGREEN3', 'RNGREEN4', 'RNRED1', 'RNRED2', 'RNRED3', 'RNRED4']
        for col in cols:
            if col in df.columns:
                df = df.loc[df[col] < 500]
        
        # Hallway temperature
        if 'kpfmet.TEMP' in df.columns:
            df = df.loc[df['kpfmet.TEMP'] > 15]
        
        # Fiber temperatures
        kwrds = ['kpfmet.SIMCAL_FIBER_STG', 'kpfmet.SIMCAL_FIBER_STG']
        for key in kwrds:
            if key in df.columns:
                df = df.loc[df[key] > 0]
                
        # CCD temperatures
#        kwrds = ['kpfgreen.STA_CCD_T', 'kpfred.STA_CCD_T']
#        for key in kwrds:
#            if key in df.columns:
#                df = df.loc[df[key] > -200]
        
        # Dark Current
        kwrds = ['FLXCOLLG', 'FLXECHG', 'FLXREG1G', 'FLXREG2G', 'FLXREG3G', 'FLXREG4G', 
                 'FLXREG5G', 'FLXREG6G', 'FLXCOLLR', 'FLXECHR', 'FLXREG1R', 'FLXREG2R', 
                 'FLXREG3R', 'FLXREG4R', 'FLXREG5R', 'FLXREG6R']

        return df


    def get_first_last_dates(self):
        """
        Returns a tuple of datetime objects containing the first and last dates 
        in the database.  DATE-MID is used for the date.
        """

        conn = sqlite3.connect(self.db_path)
    
        # Query for the minimum and maximum dates in the 'DATE-MID' column
        query = """
            SELECT MIN("DATE-MID") AS min_date, MAX("DATE-MID") AS max_date
            FROM kpfdb
        """
        result = pd.read_sql_query(query, conn)
        conn.close()
    
        # Extract dates from the result and convert them to datetime objects
        min_date_str = result['min_date'][0]
        max_date_str = result['max_date'][0]
    
        # Convert strings to datetime objects, handling None values gracefully
        date_format = '%Y-%m-%dT%H:%M:%S.%f'
        first_date = datetime.strptime(min_date_str, date_format) if min_date_str else None
        last_date = datetime.strptime(max_date_str, date_format) if max_date_str else None
    
        return first_date, last_date

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
        if D2_file_mod_time > result[1]:
            return True # 2D file was modified

        L1_file_path = f"{D2_file_path.replace('2D', 'L1')}"
        try:
            L1_file_mod_time = datetime.fromtimestamp(os.path.getmtime(L1_file_path)).strftime("%Y-%m-%d %H:%M:%S")
        except FileNotFoundError:
            L1_file_mod_time = '1000-01-01 01:01'
        if L1_file_mod_time > result[2]:
            return True # L1 file was modified

        L2_file_path = f"{L0_file_path.replace('L1', 'L2')}"
        try:
            L2_file_mod_time = datetime.fromtimestamp(os.path.getmtime(L2_file_path)).strftime("%Y-%m-%d %H:%M:%S")
        except FileNotFoundError:
            L2_file_mod_time = '1000-01-01 01:01'
        if L2_file_mod_time > result[3]:
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
        TO-DO: should this method just call dataframe_from_db()?
        
        Prints a pandas dataframe of attributes (specified by column names) for all 
        observations in the DB. The query can be restricted to observations matching a 
        particular object name(s).  The query can also be restricted to observations 
        that are on-sky/off-sky and after start_date and/or before end_date. 

        Args:
            columns (string, list of strings, or '*' for all) - database columns to query
            only_object (string or list of strings) - object names to include in query
            object_like (string or list of strings) - partial object names to search for
            on_sky (True, False, None) - using FIUMODE, select observations that are on-sky (True), off-sky (False), or don't care (None)
            start_date (datetime object) - only return observations after start_date
            end_date (datetime object) - only return observations before end_date
            false (boolean) - if True, prints the SQL query

        Returns:
            A printed dataframe of the specified columns matching the constraints.
        """
        conn = sqlite3.connect(self.db_path)
        
        # Enclose column names in double quotes
        if columns == '*':
            quoted_columns = '*'
        else:
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


    def dataframe_from_db(self, columns=None, 
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
            columns (string or list of strings, optional) - 
               database columns to query. 
               If None, all columns are retrieved.
               Retrieving all columns can be time consuming.  
               With two years of observations in the database, 
               retrieving 1, 10, 100, 1000 days takes 0.13, 0.75, 2.05, 44 seconds.
            only_object (string) - object name to include in query
            object_like (string) - partial object name to search for
            on_sky (True, False, None) - using FIUMODE, select observations that are on-sky (True), off-sky (False), or don't care (None)
            not_junk (True, False, None) using NOTJUNK, select observations that are not Junk (True), Junk (False), or don't care (None)
            start_date (datetime object) - only return observations after start_date
            end_date (datetime object) - only return observations after end_date
            verbose (boolean) - if True, prints the SQL query
        """
        
        conn = sqlite3.connect(self.db_path)
    
        # Get all column names if columns are not specified
        if columns is None:
            query_get_columns = "PRAGMA table_info(kpfdb)"
            all_columns_info = pd.read_sql_query(query_get_columns, conn)
            columns = all_columns_info['name'].tolist()
    
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
            self.logger.info('query = ' + query)
    
        df = pd.read_sql_query(query, conn)
        conn.close()
    
        return df

    def ObsIDlist_from_db(self, object_name, start_date=None, end_date=None, not_junk=None):
        """
        Returns a list of ObsIDs for the observations of object_name.

        Args:
            object_name (string) - name of object (e.g., '4614')
            not_junk (True, False, None) using NOTJUNK, select observations that are not Junk (True), Junk (False), or don't care (None)
            start_date (datetime object) - only return observations after start_date
            end_date (datetime object) - only return observations after end_date

        Returns:
            Pandas dataframe of the specified columns matching the constraints.
        """
        # to-do: check if object_name is in the database before trying to create the df
        df = self.dataframe_from_db(['ObsID'], object_like=object_name, 
                                    start_date=start_date, end_date=end_date, 
                                    not_junk=not_junk)
        
        return df['ObsID'].tolist()
        

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

        # L2 RV extension    
        elif level == 'L2_RV_header':
            keywords_csv='/code/KPF-Pipeline/static/tsdb_keywords/l2_rv_keywords.csv'
            df_keywords = pd.read_csv(keywords_csv, delimiter='|', dtype=str)
            keyword_types = dict(zip(df_keywords['keyword'], df_keywords['datatype']))

        # L2 CCF extension    
        elif level == 'L2_CCF_header':
            keywords_csv='/code/KPF-Pipeline/static/tsdb_keywords/l2_green_ccf_keywords.csv'
            df_keywords = pd.read_csv(keywords_csv, delimiter='|', dtype=str)
            keyword_types = dict(zip(df_keywords['keyword'], df_keywords['datatype']))

        # L2 RV data    
        elif level == 'L2_RV':
            prefixes = ['RV1', 'RV2', 'RV3', 'RVS', 'ERVS', 'RVC', 'ERVC', 'RVY', 'ERVY', 'CCFBJD', 'BCRV', 'CCFW']
            nums = [f"{i:02d}" for i in range(67)]
            keyword_types = {f"{prefix}{num}": 'REAL' for num in nums for prefix in prefixes}

        else:
            keyword_types = {}

        return keyword_types


    def plot_nobs_histogram(self, plot_dict=None, 
                            interval='full', date=None, exclude_junk=False, 
                            only_sources=['all'], only_autocal=False,
                            plot_junk=False, plot_source=False, 
                            fig_path=None, show_plot=False):
        """
        Plot a histogram of the number of observations per day or hour, 
        optionally color-coded by 'NOTJUNK' or 'Source'.
    
        Args:
            interval (string) - time interval over which plot is made
                                default: 'full',
                                possible values: 'full', 'decade', 'year', 'month', 'day'
            date (string) - one date in the interval (format: 'YYYYMMDD' or 'YYYY-MM-DD')
            only_sources (array of strings) - only observations whose Source name matches an element of only_strings are used
                                              possible sources = 'Bias', 'Dark', 'Flat', 'Wide Flat', 'LFC', 'Etalon', 'ThAr', 'UNe', 'Sun', 'Star'
            only_autocal - only observations OBJECT name includes 'autocal' are used
            exclude_junk (boolean) - if True, observations with NOTJUNK=False are removed
            plot_junk (boolean) - if True, will color-code based on 'NOTJUNK' column
            plot_source (boolean) - if True, will color-code based on 'Source' column
            fig_path (string) - set to the path for the file to be generated
            show_plot (boolean) - show the plot in the current environment
            
        Returns:
            PNG plot in fig_path or shows the plot in the current environment
            (e.g., in a Jupyter Notebook).
        
        To-do: 
        	Add highlighting of QC tests
        """
        
        # Use plotting dictionary, if provided
        # (inspired by dictionaries for plot_time_series_multipanel)
        
        dict_ylabel = ''
        if plot_dict != None:
            panel_arr = plot_dict['panel_arr']
            if 'ylabel' in ['paneldict']:
                 dict_ylabel = panel_arr[0]['paneldict']['ylabel']
            if 'not_junk' in panel_arr[0]['paneldict']:
                 plot_junk = not bool(panel_arr[0]['paneldict']['not_junk'])
            if 'only_sources' in panel_arr[0]['paneldict']:
                 only_sources = panel_arr[0]['paneldict']['only_sources']
            if 'plot_source' in panel_arr[0]['paneldict']:
                 plot_source = panel_arr[0]['paneldict']['plot_source']

        # Define the source categories and their colors
        source_order = ['Bias', 'Dark', 'Flat', 'Wide Flat', 'LFC', 'Etalon', 'ThAr', 'UNe', 'Sun', 'Star']
        source_colors = {
            'Bias':      'gray',
            'Dark':      'black',
            'Flat':      'gainsboro',
            'Wide Flat': 'silver',
            'LFC':       'gold',
            'Etalon':    'chocolate',
            'ThAr':      'orange',
            'UNe':       'forestgreen',
            'Sun':       'cornflowerblue',
            'Star':      'royalblue'
        }
    
        # Load data
        columns = ['DATE-BEG', 'NOTJUNK', 'Source', 'OBJECT']
        df = self.dataframe_from_db(columns)
        df['DATE-BEG'] = pd.to_datetime(df['DATE-BEG'], errors='coerce')
        #df['DATE-END'] = pd.to_datetime(df['DATE-END'], errors='coerce')
        df = df.dropna(subset=['DATE-BEG'])
        #df = df.dropna(subset=['DATE-END'])
        start_date = df['DATE-BEG'].dt.date.min()
        end_date   = df['DATE-BEG'].dt.date.max()

        if exclude_junk:
            df = df[df['NOTJUNK'] == 1.0]      
    
        if not ('all' in only_sources):
            df = df[df['Source'].isin(only_sources)]
            
        if only_autocal:
            df = df[df['OBJECT'].str.contains('autocal', na=False)]
        
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
            full_range = pd.date_range(start=f'{start_date.year}-{start_date.month}-{start_date.day}', end=f'{end_date.year}-{end_date.month}-{end_date.day}', freq='D')
            entry_counts = df['DATE'].value_counts().sort_index()
            entry_counts = entry_counts.reindex(full_range, fill_value=0)
            major_locator = YearLocator()
            major_formatter = DateFormatter("%Y")
            minor_locator = None
            column_to_count = 'DATE'
            plot_title = f"Observations (Decade: {start_date.year}-{end_date.year}"
    
        elif interval == 'year':
            start_date = pd.Timestamp(f"{date.year}-01-01")
            end_date = pd.Timestamp(f"{date.year}-12-31")
            df = df[(df['DATE-BEG'] >= start_date) & (df['DATE-BEG'] <= end_date)]
            df['DATE'] = df['DATE-BEG'].dt.date
            full_range = pd.date_range(start=f'{start_date.year}-{start_date.month}-{start_date.day}', end=f'{end_date.year}-{end_date.month}-{end_date.day}', freq='D')
            entry_counts = df['DATE'].value_counts().sort_index()
            entry_counts = entry_counts.reindex(full_range, fill_value=0)
            major_locator = MonthLocator()
            major_formatter = DateFormatter("%b")  # Format ticks as month names (Jan, Feb, etc.)
            minor_locator = None
            column_to_count = 'DATE'
            plot_title = f"Observations (Year: {date.year}"
    
        elif interval == 'month':
            start_date = pd.Timestamp(f"{date.year}-{date.month:02d}-01")
            end_date = (start_date + pd.offsets.MonthEnd(0) + timedelta(days=1) - timedelta(seconds=0.1))
            df = df[(df['DATE-BEG'] >= start_date) & (df['DATE-BEG'] <= end_date)]
            df['DAY'] = df['DATE-BEG'].dt.day    
            #full_range = pd.date_range(start=f'{start_date.year}-{start_date.month}-{start_date.day}', end=f'{end_date.year}-{end_date.month}-{end_date.day}', freq='D')
            full_range = range(1, end_date.day + 1) 
            entry_counts = df['DAY'].value_counts().sort_index()
            entry_counts = entry_counts.reindex(full_range, fill_value=0)
            major_locator = DayLocator()
            major_formatter = lambda x, _: f"{int(x)}" if 1 <= x <= end_date.day else ""
            minor_locator = None
            column_to_count = 'DAY'
            plot_title = f"Observations (Month: {date.year}-{date.month:02d}"
    
        elif interval == 'day':
            start_date = pd.Timestamp(f"{date.year}-{date.month:02d}-{date.day:02d}")
            end_date = start_date + timedelta(days=1)# - timedelta(seconds=1)
            df = df[(df['DATE-BEG'] >= start_date) & (df['DATE-BEG'] <= end_date)]
            df['HOUR'] = df['DATE-BEG'].dt.hour
            entry_counts = df['HOUR'].value_counts().sort_index()
            hourly_range = pd.Index(range(24))  # 0 through 23 hours
            entry_counts = entry_counts.reindex(hourly_range, fill_value=0)
            major_locator = plt.MultipleLocator(1)  # Tick every hour
            major_formatter = lambda x, _: f"{int(x):02d}:00" if 0 <= x <= 23 else ""
            minor_locator = None
            column_to_count = 'HOUR'
            plot_title = f"Observations (Day: {date.year}-{date.month:02d}-{date.day:02d}"
        else: # Default: 'full' interval
            df['DATE'] = df['DATE-BEG'].dt.date
            full_range = pd.date_range(start=f'{start_date.year}-{start_date.month}-{start_date.day}', end=f'{end_date.year}-{end_date.month}-{end_date.day}', freq='D')
            entry_counts = df['DATE'].value_counts().sort_index()
            entry_counts = entry_counts.reindex(full_range, fill_value=0)
            major_locator = AutoDateLocator()
            major_formatter = DateFormatter("%Y-%m")
            minor_locator = None
            column_to_count = 'DATE'
            plot_title = f"Observations (Full Range: {start_date.year}-{start_date.month:02d}-{start_date.day:02d} - {end_date.year}-{end_date.month:02d}-{end_date.day:02d}"
            
            # Ensure full date range is displayed
            full_range = pd.date_range(start=start_date, end=end_date, freq='D')
            entry_counts = entry_counts.reindex(full_range, fill_value=0)

        if not ('all' in only_sources):
            plot_title = plot_title + " - " + ', '.join(only_sources)
        if only_autocal:
            plot_title = plot_title + " - only autocal"
        if exclude_junk:
            plot_title = plot_title + " - junk excluded"
        plot_title = plot_title + ")"
        
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
        if interval == 'day':
            plt.xlabel("Hour", fontsize=14)
        if interval == 'month':
            plt.xlabel("Day", fontsize=14)
        else:
            plt.xlabel("Date", fontsize=14)
        if dict_ylabel != '':
            plt.ylabel(dict_ylabel, fontsize=14)
        else:
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


    def plot_time_series_multipanel(self, plotdict, 
                                    start_date=None, end_date=None, 
                                    clean=False, 
                                    fig_path=None, show_plot=False, 
                                    log_savefig_timing=False):
        """
        Generate a multi-panel plot of data in a KPF DB.  The data to be 
        plotted and attributes are stored in an array of dictionaries, which 
        can be read from YAML configuration files.  

        Args:
            panel_dict makes panel_arr ...
            panel_arr (array of dictionaries) - each dictionary in the array has keys:
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
            
        To do:
            * Make a standard plot type that excludes outliers using ranges set 
              to, say, +/- 4-sigma where sigma is determined by aggressive outlier
              rejection.  This should be in Delta values.
            * Make standard correlation plots.
            * Make standard phased plots (by day)
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

        # Retrieve the appropriate standard plot dictionary
        if type(plotdict) == type('str'):
            plotdict_str = plotdict
            import static.tsdb_plot_configs
            all_yaml = static.tsdb_plot_configs.all_yaml # an attribute from static/tsdb_plot_configs/__init__.py        
            base_filenames = [os.path.basename(y) for y in all_yaml]
            base_filenames = [str.split(f,'.')[0] for f in base_filenames]
            try:
                ind = base_filenames.index(plotdict_str)
                plotdict = self.yaml_to_dict(all_yaml[ind])
                self.logger.info(f'Plotting from config: {all_yaml[ind]}')
            except Exception as e:
                self.logger.info(f"Couldn't find the file {plotdict_str}.  Error message: {e}")
                return
        
        panel_arr = plotdict['panel_arr']
        
        npanels = len(panel_arr)
        unique_cols = set()
        unique_cols.add('DATE-MID')
        unique_cols.add('FIUMODE')
        unique_cols.add('OBJECT')
        unique_cols.add('NOTJUNK')
        for panel in panel_arr:
            for d in panel['panelvars']:
                unique_cols.add(d['col'])
                if 'col_err' in d:
                    unique_cols.add(d['col_err'])
                if 'col_subtract' in d:
                    unique_cols.add(d['col_subtract'])
        # add this logVERTEX - Visible Experiment for Rapid Transient EXplorationic?
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
                if str(thispanel['paneldict']['not_junk']).lower() == 'true':
                    not_junk = True
                elif str(thispanel['paneldict']['not_junk']).lower() == 'false':
                    not_junk = False
            only_object = None
            if 'only_object' in thispanel['paneldict']:
                only_object = thispanel['paneldict']['only_object']
            object_like = None
#            if 'object_like' in thispanel['paneldict']:
#                if str(thispanel['paneldict']['object_like']).lower() == 'true':
#                    object_like = True
#                elif str(thispanel['paneldict']['object_like']).lower() == 'false':
#                    object_like = False

            if start_date == None:
                start_date = datetime(2020, 1,  1)
                start_date_was_none = True
            else:
                start_date_was_none = False
            if end_date == None:
                end_date = datetime(2300, 1,  1)
                end_date_was_none = True
            else:
                end_date_was_none = False

            # Get data from database
            df = self.dataframe_from_db(unique_cols, 
                                        start_date=start_date, 
                                        end_date=end_date, 
                                        not_junk=not_junk, 
                                        only_object=only_object, 
                                        object_like=object_like,
                                        verbose=False)
            df['DATE-MID'] = pd.to_datetime(df['DATE-MID']) # move this to dataframe_from_db ?
            if start_date_was_none == True:
                start_date = min(df['DATE-MID'])
            if end_date_was_none == True:
                end_date = max(df['DATE-MID'])
            df = df.sort_values(by='DATE-MID')

            # Remove outliers
            if clean:
                df = self.clean_df(df)

            # Filter using on_sky criterion
            if 'on_sky' in thispanel['paneldict']:
                if str(thispanel['paneldict']['on_sky']).lower() == 'true':
                    df = df[df['FIUMODE'] == 'Observing']
                elif str(thispanel['paneldict']['on_sky']).lower() == 'false':
                    df = df[df['FIUMODE'] == 'Calibration']
                    
            # Apply multiplier, if needed

            thistitle = ''
            if abs((end_date - start_date).days) <= 1.2:
                t = [(date - start_date).total_seconds() / 3600 for date in df['DATE-MID']]
                xtitle = 'Hours since ' + start_date.strftime('%Y-%m-%d %H:%M') + ' UT'
                if 'title' in thispanel['paneldict']:
                    thistitle = str(thispanel['paneldict']['title']) + ": " + start_date.strftime('%Y-%m-%d %H:%M') + " to " + end_date.strftime('%Y-%m-%d %H:%M')
                axs[p].set_xlim(0, (end_date - start_date).total_seconds() / 3600)
                if 'narrow_xlim_daily' in thispanel['paneldict']:
                    if str(thispanel['paneldict']['narrow_xlim_daily']).lower() == 'true':
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
                if str(thispanel['paneldict']['nolegend']).lower() == 'true':
                    makelegend = False

            subtractmedian = False
            if 'subtractmedian' in thispanel['paneldict']:
                if str(thispanel['paneldict']['subtractmedian']).lower() == 'true':
                    subtractmedian = True

            nvars = len(thispanel['panelvars'])
            df_initial = df
            for i in np.arange(nvars):
                df = df_initial # start fresh for each panel in case NaN values were removed.
                if 'plot_type' in thispanel['panelvars'][i]:
                    plot_type = thispanel['panelvars'][i]['plot_type']
                else:
                    plot_type = 'scatter'
                
                # Extract data from df and manipulate
                col_name = thispanel['panelvars'][i]['col']
                # Filter out invalid values in col_name
                df = df[~df[col_name].isin(['NaN', 'null', 'nan', 'None', None, np.nan])]
                col_data = df[col_name]
                col_data_replaced = col_data  # default, no subtraction
                
                if 'col_subtract' in thispanel['panelvars'][i]:
                    col_subtract_name = thispanel['panelvars'][i]['col_subtract']
                    # Now filter out invalid values in col_subtract_name,
                    # and also re-filter col_name because removing rows re-indexes the DataFrame.
                    df = df[~df[col_subtract_name].isin(['NaN', 'null', 'nan', 'None', None, np.nan])]
                    # Re-grab the series after dropping rows
                    col_data = df[col_name]
                    col_subtract_data = df[col_subtract_name]
                    col_data_replaced = col_data - col_subtract_data

                if 'col_multiply' in thispanel['panelvars'][i]:
                    col_data_replaced = pd.to_numeric(col_data_replaced, errors='coerce') * thispanel['panelvars'][i]['col_multiply']

                if 'col_offset' in thispanel['panelvars'][i]:
                    col_data_replaced = pd.to_numeric(col_data_replaced, errors='coerce') + thispanel['panelvars'][i]['col_offset']

                if 'col_err' in thispanel['panelvars'][i]:
                    col_data_err = df[thispanel['panelvars'][i]['col_err']]
                    col_data_err_replaced = col_data_err.replace('NaN',  np.nan)
                    col_data_err_replaced = col_data_err.replace('null', np.nan)
                    if 'col_multiply' in thispanel['panelvars'][i]:
                        col_data_err_replaced = pd.to_numeric(col_data_err_replaced, errors='coerce') * thispanel['panelvars'][i]['col_multiply']
                
                if plot_type == 'state':
                    states = np.array(col_data_replaced)
                else:
                    data = np.array(col_data_replaced, dtype='float')
                    if plot_type == 'errorbar':
                        data_err = np.array(col_data_err_replaced, dtype='float')

                if abs((end_date - start_date).days) <= 1.2:
                    t = [(date - start_date).total_seconds() / 3600 for date in df['DATE-MID']]
                elif abs((end_date - start_date).days) <= 3:
                    t = [(date - start_date).total_seconds() / 86400 for date in df['DATE-MID']]
                elif abs((end_date - start_date).days) < 32:
                    t = [(date - start_date).total_seconds() / 86400 for date in df['DATE-MID']]
                else:
                    t = df['DATE-MID'] # dates

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
                
                if plot_type == 'errorbar':
                    axs[p].errorbar(t, data, yerr=data_err, **plot_attributes)
                
                if plot_type == 'plot':
                    axs[p].plot(t, data, **plot_attributes)
                
                if plot_type == 'step':
                    axs[p].step(t, data, **plot_attributes)
                
                if plot_type == 'state':
                    # Plot states (e.g., DRP version number or QC result)
                    # Convert states to a consistent type for comparison
                    states = [float(s) if is_numeric(s) else s for s in states]
                    # Separate numeric and non-numeric states for sorting
                    numeric_states = sorted(s for s in states if isinstance(s, float))
                    non_numeric_states = sorted(s for s in states if isinstance(s, str))
                    unique_states = sorted(set(states), key=lambda x: (not isinstance(x, float), x))
                    unique_states = list(set(unique_states))
                    # Check if unique_states contains only 0, 1, and None - QC test
                    if set(unique_states).issubset({0.0, 1.0, 'None'}):
                        states = ['Pass' if s == 1.0 else 'Fail' if s == 0.0 else s for s in states]
                        unique_states = sorted(set(states), key=lambda x: (not isinstance(x, float), x))
                        unique_states = list(set(unique_states))
                        if (unique_states == ['Pass', 'Fail']) or (unique_states == ['Pass']) or (unique_states == ['Fail']):
                             unique_states = ['Fail', 'Pass']  # put Pass on the top of the plot
                        state_to_color = {'Fail': 'indianred', 'Pass': 'forestgreen', 'None': 'cornflowerblue'}
                        if thispanel['paneldict']['ylabel'] == 'Junk Status':
                            states = ['Not Junk' if s == 'Pass' else 'Junk' if s == 'Fail' else s for s in states]
                            unique_states = ['Junk', 'Not Junk']
                            state_to_color = {'Junk': 'indianred', 'Not Junk': 'forestgreen', 'None': 'cornflowerblue'}
                        mapped_states = [unique_states.index(state) if state in unique_states else None for state in states]
                        colors = [state_to_color[state] if state in state_to_color else 'black' for state in states]
                        color_map = {state: state_to_color[state] for state in unique_states if state in state_to_color}
                    else:
                        state_to_num = {state: i for i, state in enumerate(unique_states)}
                        mapped_states = [state_to_num[state] for state in states]
                        colors = plt.cm.jet(np.linspace(0, 1, len(unique_states)))
                        color_map = {state: colors[i] for i, state in enumerate(unique_states)}
                    for state in unique_states:
                        color = color_map[state]
                        indices = [i for i, s in enumerate(states) if s == state]
                        axs[p].scatter([t[i] for i in indices], [mapped_states[i] for i in indices], color=color, label=state)
                    axs[p].set_yticks(range(len(unique_states)))
                    axs[p].set_yticklabels(unique_states)
                
                if len(t) < 1:
                    axs[p].text(0.5, 0.5, 'No Data', 
                                horizontalalignment='center', verticalalignment='center', 
                                fontsize=24, transform=axs[p].transAxes)

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
                    if len(t) > 0:
                        if 'legend_frac_size' in thispanel['paneldict']:
                            legend_frac_size = thispanel['paneldict']['legend_frac_size']
                        else:
                            legend_frac_size = 0.20
                        handles, labels = axs[p].get_legend_handles_labels()
                        sorted_pairs = sorted(zip(handles, labels), key=lambda x: x[1], reverse=True)
                        handles, labels = zip(*sorted_pairs)
                        axs[p].legend(handles, labels, loc='upper right', bbox_to_anchor=(1+legend_frac_size, 1))
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


    def yaml_to_dict(self, yaml_or_path):
        """
        Read a plotting configuration from either a YAML file or a YAML string.
        
        1) If `yaml_or_path` is a valid file path, open that file and parse.
        2) Otherwise, treat `yaml_or_path` as a YAML string and parse it directly.
        """
        if os.path.isfile(yaml_or_path):
            # It's an actual file path on disk
            with open(yaml_or_path, 'r') as f:
                plotdict = yaml.safe_load(f)
        else:
            # It's a (multi-line) YAML string
            plotdict = yaml.safe_load(yaml_or_path)
    
        return plotdict
    
    
    def plot_all_quicklook(self, start_date=None, interval=None, clean=True, 
                           last_n_days=None, 
                           fig_dir=None, show_plot=False, 
                           print_plot_names=False, verbose=False):
        """
        Generate all of the standard time series plots for the quicklook.  
        Depending on the value of the input 'interval', the plots have time ranges 
        that are daily, weekly, yearly, or decadal.

        Args:
            start_date (datetime object) - start date for plot
            interval (string) - 'day', 'month', 'year', or 'decade'
            last_n_days (int) - overrides start_date and makes a plot over the last n days
            fig_path (string) - set to the path for the files to be generated.
            show_plot (boolean) - show the plot in the current environment.
            print_plot_names (boolean) - prints the names of possible plots and exits

        Returns:
            PNG plot in fig_path or shows the plots it the current environment
            (e.g., in a Jupyter Notebook).
        """

        plots = {}
        
        import static.tsdb_plot_configs
        all_yaml = static.tsdb_plot_configs.all_yaml # an attribute from static/tsdb_plot_configs/__init__.py        
        for this_yaml_path in all_yaml:
            thisplotconfigdict = self.yaml_to_dict(this_yaml_path)
            plot_name = str.split(str.split(this_yaml_path,'/')[-1], '.')[0]
            subdir = str.split(os.path.dirname(this_yaml_path),'/')[-1]
            tempdict = {
                "plot_name": plot_name,
                "plot_type": thisplotconfigdict.get("plot_type", ""),
                "subdir": subdir,
                "description": thisplotconfigdict.get("description", ""),
                "panel_arr": thisplotconfigdict["panel_arr"],
            }
            plots[plot_name] = tempdict

        if print_plot_names:
            print("Plots available:")
            for p in plots:
                print("    '" + plots[p]["plot_name"] + "': " + plots[p]["description"])
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
            if verbose:
                self.logger.info(f"AnalyzeTimeSeries.plot_all_quicklook: making {plot_name}")

            # Set filename 
            if plots[p]['plot_type'] == 'time_series_multipanel':
                type_string = '_ts_'
            elif plots[p]['plot_type'] == 'nobs_histogram':
                type_string = '_nobs_'

            if interval == 'day':
                end_date = start_date + timedelta(days=1)
                filename = 'kpf_' + start_date.strftime("%Y%m%d") + type_string + plot_name + '.png' 
            elif interval == 'month':
                end_date = add_one_month(start_date)
                filename = 'kpf_' + start_date.strftime("%Y%m") + type_string + plot_name + '.png' 
            elif interval == 'year':
                end_date = datetime(start_date.year+1, start_date.month, start_date.day)
                filename = 'kpf_' + start_date.strftime("%Y") + type_string + plot_name + '.png' 
            elif interval == 'decade':
                end_date = datetime(start_date.year+10, start_date.month, start_date.day)
                filename = 'kpf_' + start_date.strftime("%Y")[0:3] + '0' + type_string + plot_name + '.png' 
            elif (last_n_days != None) and (type(last_n_days) == type(1)):
                filename = 'kpf_last' + str(last_n_days) + 'days' + type_string + plot_name + '.png'                 
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

            # Make Plot
            plot_dict = plots[p]
            if plot_dict['plot_type'] == 'time_series_multipanel':
                self.plot_time_series_multipanel(plot_dict, 
                                                 start_date=start_date, 
                                                 end_date=end_date, 
                                                 fig_path=fig_path, 
                                                 show_plot=show_plot, 
                                                 clean=clean)
            elif plot_dict['plot_type'] == 'nobs_histogram':        
                self.plot_nobs_histogram(plot_dict=plot_dict, 
                                         date=start_date.strftime('%Y%m%d'), 
                                         interval=interval,
                                         fig_path=fig_path, 
                                         show_plot=show_plot)


    def plot_all_quicklook_daterange(self, start_date=None, end_date=None, 
                                     time_range_type = 'all', clean=True, 
                                     base_dir='/data/QLP/', show_plot=False):
        """
        Generate all of the standard time series plots for the quicklook for a 
        date range.  Every unique day, month, year, and decade between 
        start_date and end_date will have a full set of plots produced using 
        plot_all_quicklook(). The set of date range types ('day', 'month', 
        'year', 'decade', 'all') is set by the time_range_type parameter.

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
        if start_date == None or end_date == None:
            dates = self.get_first_last_dates()
            if start_date == None:
                start_date = dates[0]
            if end_date == None:
                end_date = dates[1]
        
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
                        savedir = base_dir + day.strftime("%Y%m%d") + '/Time_Series/'
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
                        savedir = base_dir + month.strftime("%Y%m") + '00/Time_Series/'
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
                        savedir = base_dir + year.strftime("%Y") + '0000/Time_Series/'
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
                        savedir = base_dir + decade.strftime("%Y")[0:3] + '00000/Time_Series/' 
                    else:
                        savedir = None
                    self.plot_all_quicklook(decade, interval='decade', fig_dir=savedir)
                except Exception as e:
                    self.logger.error(e)


def process_file(file_path, now_str,
                 L0_header_keyword_types, L0_telemetry_types, D2_header_keyword_types,
                 L1_header_keyword_types, L2_header_keyword_types, L2_CCF_header_keyword_types, L2_RV_header_keyword_types,
                 extract_kwd_func, extract_telemetry_func, extract_rvs_func, 
                 #is_any_file_updated_func, 
                 get_source_func, get_datecode_func):
    """
    This method runs in a worker process. It returns the extracted header data for one file.
    """
    base_filename = os.path.basename(file_path).split('.fits')[0]
    L0_filename = base_filename.split('.fits')[0].split('/')[-1]
    L0_file_path = file_path
    

    # Check if updated
#    if not is_any_file_updated_func(L0_file_path):
#        return None

    D2_filename  = f"{L0_filename.replace('L0', '2D')}"
    L1_filename  = f"{L0_filename.replace('L0', 'L1')}"
    L2_filename  = f"{L0_filename.replace('L0', 'L2')}"
    D2_file_path = file_path.replace('L0', '2D').replace('.fits', '_2D.fits')
    L1_file_path = file_path.replace('L0', 'L1').replace('.fits', '_L1.fits')
    L2_file_path = file_path.replace('L0', 'L2').replace('.fits', '_L2.fits')

    # Extract headers and telemetry
    L0_header_data     = extract_kwd_func(L0_file_path, L0_header_keyword_types, extension='PRIMARY')   
    L0_telemetry       = extract_telemetry_func(L0_file_path, L0_telemetry_types) 
    D2_header_data     = extract_kwd_func(D2_file_path, D2_header_keyword_types, extension='PRIMARY')   
    L1_header_data     = extract_kwd_func(L1_file_path, L1_header_keyword_types, extension='PRIMARY')   
    L2_header_data     = extract_kwd_func(L2_file_path, L2_header_keyword_types, extension='PRIMARY')   
    L2_CCF_header_data = extract_kwd_func(L2_file_path, L2_CCF_header_keyword_types, extension='GREEN_CCF')   
    L2_RV_header_data  = extract_kwd_func(L2_file_path, L2_RV_header_keyword_types, extension='RV')   
    L2_RV_data         = extract_rvs_func(L2_file_path)   

    header_data = {
        **L0_header_data,
        **L0_telemetry,
        **D2_header_data,
        **L1_header_data,
        **L2_header_data,
        **L2_CCF_header_data,
        **L2_RV_header_data,
        **L2_RV_data
    }

    header_data['ObsID'] = base_filename
    header_data['datecode'] = get_datecode_func(base_filename)
    header_data['L0_filename'] = os.path.basename(L0_file_path)
    header_data['D2_filename'] = os.path.basename(D2_file_path)
    header_data['L1_filename'] = os.path.basename(L1_file_path)
    header_data['L2_filename'] = os.path.basename(L2_file_path)
    header_data['L0_header_read_time'] = now_str
    header_data['D2_header_read_time'] = now_str
    header_data['L1_header_read_time'] = now_str
    header_data['L2_header_read_time'] = now_str
    header_data['Source'] = get_source_func(L0_header_data)

    return header_data


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
    if type(string) == 'str':
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

def is_numeric(value):
    if value is None:  # Explicitly handle NoneType
        return False
    try:
        float(value)  # Attempt to convert to float
        return True
    except (ValueError, TypeError):
        return False
