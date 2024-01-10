import os
import time
import glob
import copy
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
from modules.Utils.utils import DummyLogger
from modules.Utils.kpf_parse import get_datecode

class AnalyzeTimeSeries:

    """
    Description:
        This class contains ....

    Arguments:
        TBD

    Attributes:
        TBD
        
    To-do:
        * add date range to title
        * documentation
        * make plots using only_object and object_like
        * augment statistics in legends (median and stddev upon request)
        * optimize ingestion efficiency
        * determine file modification times with a single call, if possible
        * check that updated rows overwrite old results
        * Add the capability of using Jump queries to find files for ingestion
        * All for other plot types, e.g. histograms of DRPTAG
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
        self.logger.info('Full path of database file: ' + os.path.abspath(self.db_path))
        self.base_dir = base_dir
        self.logger.info('Base directory: ' + self.base_dir)
        self.L0_keyword_types   = self.get_keyword_types(level='L0')
        self.D2_keyword_types   = self.get_keyword_types(level='2D')
        self.L1_keyword_types   = self.get_keyword_types(level='L1')
        self.L2_keyword_types   = self.get_keyword_types(level='L2')
        self.L0_telemetry_types = self.get_keyword_types(level='L0_telemetry')
        
        if drop:
            self.drop_table()
            self.logger.info('Dropping KPF database ' + str(self.db_path))

        # the line below might be modified so that if the database exists, then the columns are read from it
        self.create_database()
        self.logger.info('Initialization complete')
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
    
        # Define columns for each file type
        L0_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L0_keyword_types.items()]
        D2_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.D2_keyword_types.items()]
        L1_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L1_keyword_types.items()]
        L2_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L2_keyword_types.items()]
        L0_telemetry_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L0_telemetry_types.items()]
        columns = L0_columns + D2_columns + L1_columns + L2_columns + L0_telemetry_columns
        columns += ['"datecode" TEXT', '"ObsID" TEXT']
        columns += ['"L0_filename" TEXT', '"D2_filename" TEXT', '"L1_filename" TEXT', '"L2_filename" TEXT', ]
        columns += ['"L0_header_read_time" TEXT', '"D2_header_read_time" TEXT', '"L1_header_read_time" TEXT', '"L2_header_read_time" TEXT', ]
        create_table_query = f'CREATE TABLE IF NOT EXISTS kpfdb ({", ".join(columns)}, UNIQUE(ObsID))'
        cursor.execute(create_table_query)
        conn.commit()
        conn.close()


    def ingest_dates_to_db(self, start_date, end_date, batch_size=25):
        """
        Ingest KPF data for the date range start_date to end_date, inclusive.
        batch_size refers to the number of observations per DB insertion.
        To-do: scan for observations that have already been ingested at a higher level.
        """
        self.logger.info("Adding to database between " + start_date + " to " + end_date)
        dir_paths = glob.glob(f"{self.base_dir}/????????")
        sorted_dir_paths = sorted(dir_paths, key=lambda x: int(os.path.basename(x)), reverse=start_date > end_date)
        filtered_dir_paths = [
            dir_path for dir_path in sorted_dir_paths
            if start_date <= os.path.basename(dir_path) <= end_date
        ]
        #t1 = tqdm_notebook(filtered_dir_paths, desc=(filtered_dir_paths[0]).split('/')[-1])
        t1 = self.tqdm(filtered_dir_paths, desc=(filtered_dir_paths[0]).split('/')[-1])
        for dir_path in t1:
            t1.set_description(dir_path.split('/')[-1])
            t1.refresh() 
            #t2 = tqdm_notebook(os.listdir(dir_path), desc=f'Files', leave=False)
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
    
        L0_exists = os.path.isfile(L0_file_path)
        D2_exists = os.path.isfile(D2_file_path)
        L1_exists = os.path.isfile(L1_file_path)
        L2_exists = os.path.isfile(L2_file_path)

        # determine if any associated file has been updated - more efficient logic could be written that only accesses filesystem when needed
        L0_updated = False
        D2_updated = False
        L1_updated = False
        L2_updated = False
        if L0_exists:
            if self.is_file_updated(L0_file_path, L0_filename, 'L0'):
                L0_updated = True
        if D2_exists:
            if self.is_file_updated(D2_file_path, D2_filename, '2D'):
                D2_updated = True
        if L1_exists:
            if self.is_file_updated(L1_file_path, L1_filename, 'L1'):
                L1_updated = True
        if L2_exists:
            if self.is_file_updated(L2_file_path, L2_filename, 'L2'):
                L2_updated = True

        # update the DB if necessary
        if L0_updated or D2_updated or L1_updated or L2_updated:
        
            L0_header_data = self.extract_kwd(L0_file_path, self.L0_keyword_types) 
            D2_header_data = self.extract_kwd(D2_file_path, self.D2_keyword_types) 
            L1_header_data = self.extract_kwd(L1_file_path, self.L1_keyword_types) 
            L2_header_data = self.extract_kwd(L2_file_path, self.L2_keyword_types) 
            L0_telemetry   = self.extract_telemetry(L0_file_path, self.L0_telemetry_types) if L0_exists else {}

            header_data = {**L0_header_data, **D2_header_data, **L1_header_data, **L2_header_data, **L0_telemetry}
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        batch_data = []
    
        for file_path in batch:
            base_filename = os.path.basename(file_path).split('.fits')[0]
            L0_filename = base_filename.split('.fits')[0]
            L0_filename = L0_filename.split('/')[-1]
            
            L0_file_path = file_path
            D2_file_path = file_path.replace('L0', '2D').replace('.fits', '_2D.fits')
            L1_file_path = file_path.replace('L0', 'L1').replace('.fits', '_L1.fits')
            L2_file_path = file_path.replace('L0', 'L2').replace('.fits', '_L2.fits')
    
            D2_filename  = f"{L0_filename.replace('L0', '2D')}"
            L1_filename  = f"{L0_filename.replace('L0', 'L1')}"
            L2_filename  = f"{L0_filename.replace('L0', 'L2')}"
        
            L0_exists = os.path.isfile(L0_file_path)
            D2_exists = os.path.isfile(D2_file_path)
            L1_exists = os.path.isfile(L1_file_path)
            L2_exists = os.path.isfile(L2_file_path)
    
            # determine if any associated file has been updated - more efficient logic could be written that only accesses filesystem when needed
            L0_updated = False
            D2_updated = False
            L1_updated = False
            L2_updated = False
            if L0_exists:
                if self.is_file_updated(L0_file_path, L0_filename, 'L0'):
                    L0_updated = True
            if D2_exists:
                if self.is_file_updated(D2_file_path, D2_filename, '2D'):
                    D2_updated = True
            if L1_exists:
                if self.is_file_updated(L1_file_path, L1_filename, 'L1'):
                    L1_updated = True
            if L2_exists:
                if self.is_file_updated(L2_file_path, L2_filename, 'L2'):
                    L2_updated = True
    
            # If any associated file has been updated, proceed
            if L0_updated or D2_updated or L1_updated or L2_updated:
                L0_header_data = self.extract_kwd(L0_file_path,       self.L0_keyword_types)   
                D2_header_data = self.extract_kwd(D2_file_path,       self.D2_keyword_types)   
                L1_header_data = self.extract_kwd(L1_file_path,       self.L1_keyword_types)   
                L2_header_data = self.extract_kwd(L2_file_path,       self.L2_keyword_types)   
                L0_telemetry   = self.extract_telemetry(L0_file_path, self.L0_telemetry_types) 

                header_data = {**L0_header_data, **D2_header_data, **L1_header_data, **L2_header_data, **L0_telemetry}
                header_data['ObsID'] = base_filename
                header_data['datecode'] = get_datecode(base_filename)
                header_data['L0_filename'] = os.path.basename(L0_file_path)
                header_data['D2_filename'] = os.path.basename(D2_file_path)
                header_data['L1_filename'] = os.path.basename(L1_file_path)
                header_data['L2_filename'] = os.path.basename(L2_file_path)
                header_data['L0_header_read_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                header_data['D2_header_read_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                header_data['L1_header_read_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                header_data['L2_header_read_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
                batch_data.append(header_data)
    
        # Perform batch insertion/update in the database
        if batch_data:
            columns = ', '.join([f'"{key}"' for key in batch_data[0].keys()])
            placeholders = ', '.join(['?'] * len(batch_data[0]))
            insert_query = f'INSERT OR REPLACE INTO kpfdb ({columns}) VALUES ({placeholders})'
            data_tuples = [tuple(data.values()) for data in batch_data]
            cursor.executemany(insert_query, data_tuples)
            conn.commit()
    
        conn.close()


    def extract_kwd(self, file_path, keyword_types):
        """
        Extract keywords from keyword_types.keys from a L0/2D/L1/L2 file.
        """
        header_data = {}
        if os.path.isfile(file_path):
            with fits.open(file_path, memmap=True) as hdul: # memmap=True minimizes RAM usage
                header = hdul[0].header
                for key in keyword_types.keys():
                    if key in header:
                        header_data[key] = header[key]
                    else:
                        header_data[key] = None 
        else:
            for key in keyword_types.keys():
                header_data[key] = None 
        return header_data


    def extract_telemetry(self, file_path, keyword_types):
        """
        Extract telemetry from the 'TELEMETRY' extension in KPF L0 files.
        """
        df_telemetry = Table.read(file_path, format='fits', hdu='TELEMETRY').to_pandas()
        num_columns = ['average', 'stddev', 'min', 'max']
        for column in df_telemetry:
            df_telemetry[column] = df_telemetry[column].str.decode('utf-8')
            #df_telemetry = df_telemetry.replace('-nan', 0)# replace nan with 0
            df_telemetry = df_telemetry.replace('-nan', np.nan)
            df_telemetry = df_telemetry.replace('nan', np.nan)
            df_telemetry = df_telemetry.replace(-999, np.nan)
            if column in num_columns:
                df_telemetry[column] = pd.to_numeric(df_telemetry[column], downcast="float")
            else:
                df_telemetry[column] = df_telemetry[column].astype(str)
        df_telemetry.set_index("keyword", inplace=True)
        telemetry_dict = {}
        for key in keyword_types.keys():
            if key in df_telemetry.index:
                telemetry_dict[key] = float(df_telemetry.loc[key, 'average'])
            else:
                telemetry_dict[key] = None 
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



    def is_file_updated(self, file_path, filename, level):
        """
        Determines if an L0/2D/L1/L2 has been updated since the last noted modification
        in the database.  Returns True if is has been modified.
        """
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if level == 'L0':
            query = f'SELECT L0_header_read_time FROM kpfdb WHERE L0_filename = "{filename}"'
        if level == '2D':
            query = f'SELECT D2_header_read_time FROM kpfdb WHERE D2_filename = "{filename}"'
        if level == 'L1':
            query = f'SELECT L1_header_read_time FROM kpfdb WHERE L1_filename = "{filename}"'
        if level == 'L2':
            query = f'SELECT L2_header_read_time FROM kpfdb WHERE L2_filename = "{filename}"'
        cursor.execute(query)
        result = cursor.fetchone()
        conn.close()
    
        if result:
            stored_mod_time = datetime.strptime(result[0], "%Y-%m-%d %H:%M:%S")
            current_mod_time = datetime.strptime(file_mod_time, "%Y-%m-%d %H:%M:%S")
            return current_mod_time > stored_mod_time
        
        return True  # Process if file is not in the database
           

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
            only_object = [f"OBJECT = '{obj}'" for obj in only_object]
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

   
    def dataframe_from_db(self, columns, only_object=None, object_like=None, on_sky=None, 
                          start_date=None, end_date=None, verbose=False):
        """
        Returns a pandas dataframe of attributes (specified by column names) for all 
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
            Pandas dataframe of the specified columns matching the constraints.
        """
        
        conn = sqlite3.connect(self.db_path)
        
        # Enclose column names in double quotes
        quoted_columns = [f'"{column}"' for column in columns]
        query = f"SELECT {', '.join(quoted_columns)} FROM kpfdb"

        # Append WHERE clauses
        where_queries = []
        if only_object is not None:
            only_object = [f"OBJECT = '{obj}'" for obj in only_object]
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
        if level == 'L0':
            keyword_types = {
                'DATE-MID': 'datetime',
                'MJD-OBS':  'float',
                'EXPTIME':  'float',
                'ELAPSED':  'float',
                'FRAMENO':  'int',
                'PROGNAME': 'string',
                'TARGRA':   'string',
                'TARGDEC':  'string',
                'EL':       'float',
                'AZ':       'float',
                'OBJECT':   'string',
                'GAIAMAG':  'float',
                '2MASSMAG': 'float',
                'AIRMASS':  'float',
                'IMTYPE':   'string',
                'GREEN':    'string',
                'RED':      'string',
                'GREEN':    'string',
                'CA_HK':    'string',
                'EXPMETER': 'string',
                'GUIDE':    'string',
                'SKY-OBJ':  'string',
                'SCI-OBJ':  'string',
                'AGITSTA':  'string',
                'FIUMODE':  'string', # FIU operating mode - 'Observing' = on-sky
                'ETAV1C1T': 'float', # Etalon Vescent 1 Channel 1 temperature
                'ETAV1C2T': 'float', # Etalon Vescent 1 Channel 2 temperature
                'ETAV1C3T': 'float', # Etalon Vescent 1 Channel 3 temperature
                'ETAV1C4T': 'float', # Etalon Vescent 1 Channel 4 temperature
                'ETAV2C3T': 'float', # Etalon Vescent 2 Channel 3 temperature
                'TOTCORR':  'string', # need to correct this to split  '498.12 604.38 710.62 816.88' / Wavelength of EM bins in nm
                'USTHRSH':  'string', 
                'THRSHLD': 'float',
                'THRSBIN': 'float',
            }
     
        elif level == '2D':
            keyword_types = {
                'DRPTAG':   'string', # Git version number of KPF-Pipeline used for processing
                'RNRED1':   'float',  # Read noise for RED_AMP1 [e-] (first amplifier region on Red CCD)
                'RNRED2':   'float',  # Read noise for RED_AMP2 [e-] (second amplifier region on Red CCD)
                'RNGREEN1': 'float',  # Read noise for GREEN_AMP1 [e-] (first amplifier region on Green CCD)
                'RNGREEN2': 'float',  # Read noise for GREEN_AMP2 [e-] (second amplifier region on Green CCD)
                'GREENTRT': 'float',  # Green CCD read time [sec]
                'REDTRT':   'float',  # Red CCD read time [sec]
                'READSPED': 'string', # Categorization of CCD read speed ('regular' or 'fast')
                'FLXREG1G': 'float',  # Dark current [e-/hr] - Green CCD region 1 - coords = [1690:1990,1690:1990]
                'FLXREG2G': 'float',  # Dark current [e-/hr] - Green CCD region 2 - coords = [1690:1990,2090:2390]
                'FLXREG3G': 'float',  # Dark current [e-/hr] - Green CCD region 3 - coords = [2090:2390,1690:1990]
                'FLXREG4G': 'float',  # Dark current [e-/hr] - Green CCD region 4 - coords = [2090:2390,2090:2390]
                'FLXREG5G': 'float',  # Dark current [e-/hr] - Green CCD region 5 - coords = [80:380,3080:3380]
                'FLXREG6G': 'float',  # Dark current [e-/hr] - Green CCD region 6 - coords = [1690:1990,1690:1990]
                'FLXAMP1G': 'float',  # Dark current [e-/hr] - Green CCD amplifier region 1 - coords = [3700:4000,700:1000]
                'FLXAMP2G': 'float',  # Dark current [e-/hr] - Green CCD amplifier region 2 - coords = [3700:4000,3080:3380]
                'FLXCOLLG': 'float',  # Dark current [e-/hr] - Green CCD collimator-side region = [3700:4000,700:1000]
                'FLXECHG':  'float',  # Dark current [e-/hr] - Green CCD echelle-side region = [3700:4000,700:1000]
                'FLXREG1R': 'float',  # Dark current [e-/hr] - Red CCD region 1 - coords = [1690:1990,1690:1990]
                'FLXREG2R': 'float',  # Dark current [e-/hr] - Red CCD region 2 - coords = [1690:1990,2090:2390]
                'FLXREG3R': 'float',  # Dark current [e-/hr] - Red CCD region 3 - coords = [2090:2390,1690:1990]
                'FLXREG4R': 'float',  # Dark current [e-/hr] - Red CCD region 4 - coords = [2090:2390,2090:2390]
                'FLXREG5R': 'float',  # Dark current [e-/hr] - Red CCD region 5 - coords = [80:380,3080:3380]
                'FLXREG6R': 'float',  # Dark current [e-/hr] - Red CCD region 6 - coords = [1690:1990,1690:1990]
                'FLXAMP1R': 'float',  # Dark current [e-/hr] - Red CCD amplifier region 1 = [3700:4000,700:1000]
                'FLXAMP2R': 'float',  # Dark current [e-/hr] - Red CCD amplifier region 2 = [3700:4000,3080:3380]
                'FLXCOLLR': 'float',  # Dark current [e-/hr] - Red CCD collimator-side region = [3700:4000,700:1000]
                'FLXECHR':  'float',  # Dark current [e-/hr] - Red CCD echelle-side region = [3700:4000,700:1000]
                'GDRXRMS':  'float',  # x-coordinate RMS guiding error in milliarcsec (mas)
                'GDRYRMS':  'float',  # y-coordinate RMS guiding error in milliarcsec (mas)
                'GDRRRMS':  'float',  # r-coordinate RMS guiding error in milliarcsec (mas)
                'GDRXBIAS': 'float',  # x-coordinate bias guiding error in milliarcsec (mas)
                'GDRYBIAS': 'float',  # y-coordinate bias guiding error in milliarcsec (mas)
                'GDRSEEJZ': 'float',  # Seeing (arcsec) in J+Z-band from Moffat func fit
                'GDRSEEV':  'float',  # Scaled seeing (arcsec) in V-band from J+Z-band
                'MOONSEP':  'float',  # Separation between Moon and target star (deg)
                'SUNALT':   'float',  # Altitude of Sun (deg); negative = below horizon
                'SKYSCIMS': 'float',  # SKY/SCI flux ratio in main spectrometer scaled from EM data. 
                'EMSCCT48': 'float',  # cumulative EM counts [ADU] in SCI in 445-870 nm
                'EMSCCT45': 'float',  # cumulative EM counts [ADU] in SCI in 445-551 nm
                'EMSCCT56': 'float',  # cumulative EM counts [ADU] in SCI in 551-658 nm
                'EMSCCT67': 'float',  # cumulative EM counts [ADU] in SCI in 658-764 nm
                'EMSCCT78': 'float',  # cumulative EM counts [ADU] in SCI in 764-870 nm
                'EMSKCT48': 'float',  # cumulative EM counts [ADU] in SKY in 445-870 nm
                'EMSKCT45': 'float',  # cumulative EM counts [ADU] in SKY in 445-551 nm
                'EMSKCT56': 'float',  # cumulative EM counts [ADU] in SKY in 551-658 nm
                'EMSKCT67': 'float',  # cumulative EM counts [ADU] in SKY in 658-764 nm
                'EMSKCT78': 'float',  # cumulative EM counts [ADU] in SKY in 764-870 nm
            }
        elif level == 'L1':
            keyword_types = {
                'MONOTWLS': 'bool',
                'SNRSC452': 'float', # SNR of L1 SCI spectrum (SCI1+SCI2+SCI3; 95th %ile) near 452 nm (second bluest order); on Green CCD
                'SNRSK452': 'float', # SNR of L1 SKY spectrum (95th %ile) near 452 nm (second bluest order); on Green CCD
                'SNRCL452': 'float', # SNR of L1 CAL spectrum (95th %ile) near 452 nm (second bluest order); on Green CCD
                'SNRSC548': 'float', # SNR of L1 SCI spectrum (SCI1+SCI2+SCI3; 95th %ile) near 548 nm; on Green CCD
                'SNRSK548': 'float', # SNR of L1 SKY spectrum (95th %ile) near 548 nm; on Green CCD
                'SNRCL548': 'float', # SNR of L1 CAL spectrum (95th %ile) near 548 nm; on Green CCD
                'SNRSC652': 'float', # SNR of L1 SCI spectrum (SCI1+SCI2+SCI3; 95th %ile) near 652 nm; on Red CCD
                'SNRSK652': 'float', # SNR of L1 SKY spectrum (95th %ile) near 652 nm; on Red CCD
                'SNRCL652': 'float', # SNR of L1 CAL spectrum (95th %ile) near 652 nm; on Red CCD
                'SNRSC747': 'float', # SNR of L1 SCI spectrum (SCI1+SCI2+SCI3; 95th %ile) near 747 nm; on Red CCD
                'SNRSK747': 'float', # SNR of L1 SKY spectrum (95th %ile) near 747 nm; on Red CCD
                'SNRCL747': 'float', # SNR of L1 CAL spectrum (95th %ile) near 747 nm; on Red CCD
                'SNRSC852': 'float', # SNR of L1 SCI (SCI1+SCI2+SCI3; 95th %ile) near 852 nm (second reddest order); on Red CCD
                'SNRSK852': 'float', # SNR of L1 SKY spectrum (95th %ile) near 852 nm (second reddest order); on Red CCD
                'SNRCL852': 'float', # SNR of L1 CAL spectrum (95th %ile) near 852 nm (second reddest order); on Red CCD
                'FR452652': 'float', # Peak flux ratio between orders (452nm/652nm) using SCI2
                'FR548652': 'float', # Peak flux ratio between orders (548nm/652nm) using SCI2
                'FR747652': 'float', # Peak flux ratio between orders (747nm/652nm) using SCI2
                'FR852652': 'float', # Peak flux ratio between orders (852nm/652nm) using SCI2
                'FR12M452': 'float', # median(SCI1/SCI2) flux ratio near 452 nm; on Green CCD
                'FR12U452': 'float', # uncertainty on the median(SCI1/SCI2) flux ratio near 452 nm; on Green CCD
                'FR32M452': 'float', # median(SCI3/SCI2) flux ratio near 452 nm; on Green CCD
                'FR32U452': 'float', # uncertainty on the median(SCI1/SCI2) flux ratio near 452 nm; on Green CCD
                'FRS2M452': 'float', # median(SKY/SCI2) flux ratio near 452 nm; on Green CCD
                'FRS2U452': 'float', # uncertainty on the median(SKY/SCI2) flux ratio near 452 nm; on Green CCD
                'FRC2M452': 'float', # median(CAL/SCI2) flux ratio near 452 nm; on Green CCD
                'FRC2U452': 'float', # uncertainty on the median(CAL/SCI2) flux ratio near 452 nm; on Green CCD
                'FR12M548': 'float', # median(SCI1/SCI2) flux ratio near 548 nm; on Green CCD
                'FR12U548': 'float', # uncertainty on the median(SCI1/SCI2) flux ratio near 548 nm; on Green CCD
                'FR32M548': 'float', # median(SCI3/SCI2) flux ratio near 548 nm; on Green CCD
                'FR32U548': 'float', # uncertainty on the median(SCI1/SCI2) flux ratio near 548 nm; on Green CCD
                'FRS2M548': 'float', # median(SKY/SCI2) flux ratio near 548 nm; on Green CCD
                'FRS2U548': 'float', # uncertainty on the median(SKY/SCI2) flux ratio near 548 nm; on Green CCD
                'FRC2M548': 'float', # median(CAL/SCI2) flux ratio near 548 nm; on Green CCD
                'FRC2U548': 'float', # uncertainty on the median(CAL/SCI2) flux ratio near 548 nm; on Green CCD
                'FR12M652': 'float', # median(SCI1/SCI2) flux ratio near 652 nm; on Red CCD
                'FR12U652': 'float', # uncertainty on the median(SCI1/SCI2) flux ratio near 652 nm; on Red CCD
                'FR32M652': 'float', # median(SCI3/SCI2) flux ratio near 652 nm; on Red CCD
                'FR32U652': 'float', # uncertainty on the median(SCI1/SCI2) flux ratio near 652 nm; on Red CCD
                'FRS2M652': 'float', # median(SKY/SCI2) flux ratio near 652 nm; on Red CCD
                'FRS2U652': 'float', # uncertainty on the median(SKY/SCI2) flux ratio near 652 nm; on Red CCD
                'FRC2M652': 'float', # median(CAL/SCI2) flux ratio near 652 nm; on Red CCD
                'FRC2U652': 'float', # uncertainty on the median(CAL/SCI2) flux ratio near 652 nm; on Red CCD
                'FR12M747': 'float', # median(SCI1/SCI2) flux ratio near 747 nm; on Red CCD
                'FR12U747': 'float', # uncertainty on the median(SCI1/SCI2) flux ratio near 747 nm; on Red CCD
                'FR32M747': 'float', # median(SCI3/SCI2) flux ratio near 747 nm; on Red CCD
                'FR32U747': 'float', # uncertainty on the median(SCI1/SCI2) flux ratio near 747 nm; on Red CCD
                'FRS2M747': 'float', # median(SKY/SCI2) flux ratio near 747 nm; on Red CCD
                'FRS2U747': 'float', # uncertainty on the median(SKY/SCI2) flux ratio near 747 nm; on Red CCD
                'FRC2M747': 'float', # median(CAL/SCI2) flux ratio near 747 nm; on Red CCD
                'FRC2U747': 'float', # uncertainty on the median(CAL/SCI2) flux ratio near 747 nm; on Red CCD
                'FR12M852': 'float', # median(SCI1/SCI2) flux ratio near 852 nm; on Red CCD
                'FR12U852': 'float', # uncertainty on the median(SCI1/SCI2) flux ratio near 852 nm; on Red CCD
                'FR32M852': 'float', # median(SCI3/SCI2) flux ratio near 852 nm; on Red CCD
                'FR32U852': 'float', # uncertainty on the median(SCI1/SCI2) flux ratio near 852 nm; on Red CCD
                'FRS2M852': 'float', # median(SKY/SCI2) flux ratio near 852 nm; on Red CCD
                'FRS2U852': 'float', # uncertainty on the median(SKY/SCI2) flux ratio near 852 nm; on Red CCD
                'FRC2M852': 'float', # median(CAL/SCI2) flux ratio near 852 nm; on Red CCD
                'FRC2U852': 'float', # uncertainty on the median(CAL/SCI2) flux ratio near 852 nm; on Red CCD
            }
        elif level == 'L2':
            keyword_types = {
                'ABCDEFGH': 'string', #placeholder for now
            }
        elif level == 'L0_telemetry':
            keyword_types = {
                'kpfmet.BENCH_BOTTOM_BETWEEN_CAMERAS': 'float',  # degC    Bench Bottom Between Cameras C2 c- double degC...
                'kpfmet.BENCH_BOTTOM_COLLIMATOR':      'float',  # degC    Bench Bottom Coll C3 c- double degC {%.3f}
                'kpfmet.BENCH_BOTTOM_DCUT':            'float',  # degC    Bench Bottom D-cut C4 c- double degC {%.3f}
                'kpfmet.BENCH_BOTTOM_ECHELLE':         'float',  # degC    Bench Bottom Echelle Cam B c- double degC {%.3f}
                'kpfmet.BENCH_TOP_BETWEEN_CAMERAS':    'float',  # degC    Bench Top Between Cameras D4 c- double degC {%...
                'kpfmet.BENCH_TOP_COLL':               'float',  # degC    Bench Top Coll D5 c- double degC {%.3f}
                'kpfmet.BENCH_TOP_DCUT':               'float',  # degC    Bench Top D-cut D3 c- double degC {%.3f}
                'kpfmet.BENCH_TOP_ECHELLE_CAM':        'float',  # degC    Bench Top Echelle Cam D1 c- double degC {%.3f}
                'kpfmet.CALEM_SCMBLR_CHMBR_END':       'float',  # degC    Cal EM Scrammbler Chamber End C1 c- double deg...
                'kpfmet.CALEM_SCMBLR_FIBER_END':       'float',  # degC    Cal EM Scrambler Fiber End D1 c- double degC {...
                'kpfmet.CAL_BENCH':                    'float',  # degC    Cal_Bench temperature c- double degC {%.1f}
                'kpfmet.CAL_BENCH_BB_SRC':             'float',  # degC    CAL_Bench_BB_Src temperature c- double degC {%...
                'kpfmet.CAL_BENCH_BOT':                'float',  # degC    Cal_Bench_Bot temperature c- double degC {%.1f}
                'kpfmet.CAL_BENCH_ENCL_AIR':           'float',  # degC    Cal_Bench_Encl_Air temperature c- double degC ...
                'kpfmet.CAL_BENCH_OCT_MOT':            'float',  # degC    Cal_Bench_Oct_Mot temperature c- double degC {...
                'kpfmet.CAL_BENCH_TRANS_STG_MOT':      'float',  # degC    Cal_Bench_Trans_Stg_Mot temperature c- double ...
                'kpfmet.CAL_RACK_TOP':                 'float',  # degC    Cal_Rack_Top temperature c- double degC {%.1f}
                'kpfmet.CHAMBER_EXT_BOTTOM':           'float',  # degC    Chamber Exterior Bottom B c- double degC {%.3f}
                'kpfmet.CHAMBER_EXT_TOP':              'float',  # degC    Chamber Exterior Top C1 c- double degC {%.3f}
                'kpfmet.CRYOSTAT_G1':                  'float',  # degC    Within cryostat green D2 c- double degC {%.3f}
                'kpfmet.CRYOSTAT_G2':                  'float',  # degC    Within cryostat green D3 c- double degC {%.3f}
                'kpfmet.CRYOSTAT_G3':                  'float',  # degC    Within cryostat green D4 c- double degC {%.3f}
                'kpfmet.CRYOSTAT_R1':                  'float',  # degC    Within Cryostat red D2 c- double degC {%.3f}
                'kpfmet.CRYOSTAT_R2':                  'float',  # degC    Within Cryostat red D3 c- double degC {%.3f}
                'kpfmet.CRYOSTAT_R3':                  'float',  # degC    Within Cryostat red D4 c- double degC {%.3f}
                'kpfmet.ECHELLE_BOTTOM':               'float',  # degC    Echelle Bottom D1 c- double degC {%.3f}
                'kpfmet.ECHELLE_TOP':                  'float',  # degC    Echelle Top C1 c- double degC {%.3f}
                'kpfmet.FF_SRC':                       'float',  # degC    FF_Src temperature c- double degC {%.1f}
                'kpfmet.GREEN_CAMERA_BOTTOM':          'float',  # degC    Green Camera Bottom C3 c- double degC {%.3f}
                'kpfmet.GREEN_CAMERA_COLLIMATOR':      'float',  # degC    Green Camera Collimator C4 c- double degC {%.3f}
                'kpfmet.GREEN_CAMERA_ECHELLE':         'float',  # degC    Green Camera Echelle D5 c- double degC {%.3f}
                'kpfmet.GREEN_CAMERA_TOP':             'float',  # degC    Green Camera Top C2 c- double degC {%.3f}
                'kpfmet.GREEN_GRISM_TOP':              'float',  # degC    Green Grism Top C5 c- double degC {%.3f}
                'kpfmet.GREEN_LN2_FLANGE':             'float',  # degC    Green LN2 Flange A c- double degC {%.3f}
                'kpfmet.PRIMARY_COLLIMATOR_TOP':       'float',  # degC    Primary Col Top D2 c- double degC {%.3f}
                'kpfmet.RED_CAMERA_BOTTOM':            'float',  # degC    Red Camera Bottom D5 c- double degC {%.3f}
                'kpfmet.RED_CAMERA_COLLIMATOR':        'float',  # degC    Red Camera Coll C3 c- double degC {%.3f}
                'kpfmet.RED_CAMERA_ECHELLE':           'float',  # degC    Red Camera Ech C4 c- double degC {%.3f}
                'kpfmet.RED_CAMERA_TOP':               'float',  # degC    Red Camera Top C5 c- double degC {%.3f}
                'kpfmet.RED_GRISM_TOP':                'float',  # degC    Red Grism Top C2 c- double degC {%.3f}
                'kpfmet.RED_LN2_FLANGE':               'float',  # degC    Red LN2 Flange D1 c- double degC {%.3f}
                'kpfmet.REFORMATTER':                  'float',  # degC    Reformatter A c- double degC {%.3f}
                'kpfmet.SCIENCE_CAL_FIBER_STG':        'float',  # degC    Science_Cal_Fiber_Stg temperature c- double de...
                'kpfmet.SCISKY_SCMBLR_CHMBR_EN':       'float',  # degC    SciSky Scrambler Chamber End A c- double degC ...
                'kpfmet.SCISKY_SCMBLR_FIBER_EN':       'float',  # degC    SciSky Scrammbler Fiber End B c- double degC {...
                'kpfmet.SIMCAL_FIBER_STG':             'float',  # degC    SimCal_Fiber_Stg temperature c- double degC {%...
                'kpfmet.SKYCAL_FIBER_STG':             'float',  # degC    SkyCal_Fiber_Stg temperature c- double degC {%...
                'kpfmet.TEMP':                         'float',  # degC    Vaisala Temperature c- double degC {%.3f}
                'kpfmet.TH_DAILY':                     'float',  # degC    Th_daily temperature c- double degC {%.1f}
                'kpfmet.TH_GOLD':                      'float',  # degC    Th_gold temperature c- double degC {%.1f}
                'kpfmet.U_DAILY':                      'float',  # degC    U_daily temperature c- double degC {%.1f}
                'kpfmet.U_GOLD':                       'float',  # degC    U_gold temperature c- double degC {%.1f}
                'kpfgreen.BPLANE_TEMP':                'float',  # degC    Backplane temperature c- double degC {%.3f}
                'kpfgreen.BRD10_DRVR_T':               'float',  # degC    Board 10 (Driver) temperature c- double degC {...
                'kpfgreen.BRD11_DRVR_T':               'float',  # degC    Board 11 (Driver) temperature c- double degC {...
                'kpfgreen.BRD12_LVXBIAS_T':            'float',  # degC    Board 12 (LVxBias) temperature c- double degC ...
                'kpfgreen.BRD1_HTRX_T':                'float',  # degC    Board 1 (HeaterX) temperature c- double degC {...
                'kpfgreen.BRD2_XVBIAS_T':              'float',  # degC    Board 2 (XV Bias) temperature c- double degC {...
                'kpfgreen.BRD3_LVDS_T':                'float',  # degC    Board 3 (LVDS) temperature c- double degC {%.3f}
                'kpfgreen.BRD4_DRVR_T':                'float',  # degC    Board 4 (Driver) temperature c- double degC {%...
                'kpfgreen.BRD5_AD_T':                  'float',  # degC    Board 5 (AD) temperature c- double degC {%.3f}
                'kpfgreen.BRD7_HTRX_T':                'float',  # degC    Board 7 (HeaterX) temperature c- double degC {...
                'kpfgreen.BRD9_HVXBIAS_T':             'float',  # degC    Board 9 (HVxBias) temperature c- double degC {...
                'kpfgreen.CF_BASE_2WT':                'float',  # degC    tip cold finger (2 wire) c- double degC {%.3f}
                'kpfgreen.CF_BASE_T':                  'float',  # degC    base cold finger 2wire temp c- double degC {%.3f}
                'kpfgreen.CF_BASE_TRG':                'float',  # degC    base cold finger heater 1A, target temp c2 dou...
                'kpfgreen.CF_TIP_T':                   'float',  # degC    tip cold finger c- double degC {%.3f}
                'kpfgreen.CF_TIP_TRG':                 'float',  # degC    tip cold finger heater 1B, target temp c2 doub...
                'kpfgreen.COL_PRESS':                  'float',  # Torr    Current ion pump pressure c- double Torr {%.3e}
                'kpfgreen.CRYOBODY_T':                 'float',  # degC    Cryo Body Temperature c- double degC {%.3f}
                'kpfgreen.CRYOBODY_TRG':               'float',  # degC    Cryo body heater 7B, target temp c2 double deg...
                'kpfgreen.CURRTEMP':                   'float',  # degC    Current cold head temperature c- double degC {...
                'kpfgreen.ECH_PRESS':                  'float',  # Torr    Current ion pump pressure c- double Torr {%.3e}
                'kpfgreen.KPF_CCD_T':                  'float',  # degC    SSL Detector temperature c- double degC {%.3f}
                'kpfgreen.STA_CCD_T':                  'float',  # degC    STA Detector temperature c- double degC {%.3f}
                'kpfgreen.STA_CCD_TRG':                'float',  # degC    Detector heater 7A, target temp c2 double degC...
                'kpfgreen.TEMPSET':                    'float',  # degC    Set point for the cold head temperature c2 dou...
                'kpfred.BPLANE_TEMP':                  'float',  # degC    Backplane temperature c- double degC {%.3f}
                'kpfred.BRD10_DRVR_T':                 'float',  # degC    Board 10 (Driver) temperature c- double degC {...
                'kpfred.BRD11_DRVR_T':                 'float',  # degC    Board 11 (Driver) temperature c- double degC {...
                'kpfred.BRD12_LVXBIAS_T':              'float',  # degC    Board 12 (LVxBias) temperature c- double degC ...
                'kpfred.BRD1_HTRX_T':                  'float',  # degC    Board 1 (HeaterX) temperature c- double degC {...
                'kpfred.BRD2_XVBIAS_T':                'float',  # degC    Board 2 (XV Bias) temperature c- double degC {...
                'kpfred.BRD3_LVDS_T':                  'float',  # degC    Board 3 (LVDS) temperature c- double degC {%.3f}
                'kpfred.BRD4_DRVR_T':                  'float',  # degC    Board 4 (Driver) temperature c- double degC {%...
                'kpfred.BRD5_AD_T':                    'float',  # degC    Board 5 (AD) temperature c- double degC {%.3f}
                'kpfred.BRD7_HTRX_T':                  'float',  # degC    Board 7 (HeaterX) temperature c- double degC {...
                'kpfred.BRD9_HVXBIAS_T':               'float',  # degC    Board 9 (HVxBias) temperature c- double degC {...
                'kpfred.CF_BASE_2WT':                  'float',  # degC    tip cold finger (2 wire) c- double degC {%.3f}
                'kpfred.CF_BASE_T':                    'float',  # degC    base cold finger 2wire temp c- double degC {%.3f}
                'kpfred.CF_BASE_TRG':                  'float',  # degC    base cold finger heater 1A, target temp c2 dou...
                'kpfred.CF_TIP_T':                     'float',  # degC    tip cold finger c- double degC {%.3f}
                'kpfred.CF_TIP_TRG':                   'float',  # degC    tip cold finger heater 1B, target temp c2 doub...
                'kpfred.COL_PRESS':                    'float',  # Torr    Current ion pump pressure c- double Torr {%.3e}
                'kpfred.CRYOBODY_T':                   'float',  # degC    Cryo Body Temperature c- double degC {%.3f}
                'kpfred.CRYOBODY_TRG':                 'float',  # degC    Cryo body heater 7B, target temp c2 double deg...
                'kpfred.CURRTEMP':                     'float',  # degC    Current cold head temperature c- double degC {...
                'kpfred.ECH_PRESS':                    'float',  # Torr    Current ion pump pressure c- double Torr {%.3e}
                'kpfred.KPF_CCD_T':                    'float',  # degC    SSL Detector temperature c- double degC {%.3f}
                'kpfred.STA_CCD_T':                    'float',  # degC    STA Detector temperature c- double degC {%.3f}
                'kpfred.STA_CCD_TRG':                  'float',  # degC    Detector heater 7A, target temp c2 double degC...
                'kpfred.TEMPSET':                      'float',  # degC    Set point for the cold head temperature c2 dou...
                'kpfexpose.BENCH_C':                   'float',  # degC    rtd bench c- double degC {%.1f} { -100.0 .. 10...
                'kpfexpose.CAMBARREL_C':               'float',  # degC    rtd camera barrel c- double degC {%.1f} { -100...
                'kpfexpose.DET_XTRN_C':                'float',  # degC    rtd detector extermal c- double degC {%.1f} { ...
                'kpfexpose.ECHELLE_C':                 'float',  # degC    rtd echelle c- double degC {%.1f} { -100.0 .. ...
                'kpfexpose.ENCLOSURE_C':               'float',  # degC    rtd enclosure c- double degC {%.1f} { -100.0 ....
                'kpfexpose.RACK_AIR_C':                'float',  # degC    rtd rack air c- double degC {%.1f} { -100.0 .....
                'kpfvac.PUMP_TEMP':                    'float',  # degC    Motor temperature c- double degC {%.2f}
                'kpf_hk.COOLTARG':                     'float',  # degC    temperature target c2 int degC
                'kpf_hk.CURRTEMP':                     'float',  # degC    current temperature c- double degC {%.2f}
                'kpfgreen.COL_CURR':                   'float',  # A       Current ion pump current c- double A {%.3e}
                'kpfgreen.ECH_CURR':                   'float',  # A       Current ion pump current c- double A {%.3e}
                'kpfred.COL_CURR':                     'float',  # A       Current ion pump current c- double A {%.3e}
                'kpfred.ECH_CURR':                     'float',  # A       Current ion pump current c- double A {%.3e}
                'kpfcal.IRFLUX':                       'float',  # Counts  LFC Fiberlock IR Intensity c- int Counts {%d}
                'kpfcal.VISFLUX':                      'float',  # Counts  LFC Fiberlock Vis Intensity c- int Counts {%d}
                'kpfcal.BLUECUTIACT':                  'float',  # A       Blue cut amplifier 0 measured current c- doubl...
                'kpfmot.AGITSPD':                      'float',  # motor_counts/s agit raw velocity c2 int motor counts/s { -750...
                'kpfmot.AGITTOR':                      'float',  # V       agit motor torque c- double V {%.3f}
                'kpfmot.AGITAMBI_T':                   'float',  # degC    Agitator ambient temperature c- double degC {%...
                'kpfmot.AGITMOT_T':                    'float',  # degC    Agitator motor temperature c- double degC {%.2...
                'kpfpower.OUTLET_A1_Amps':             'float',  # milliamps Outlet A1 current amperage c- int milliamps
            }

        else:
            keyword_types = {}
        
        return keyword_types
       
 
    def plot_time_series_multipanel(self, panel_arr, start_date=None, end_date=None, 
                                    clean=False, fig_path=None, show_plot=False):
        """
        Generate a multi-panel plot of data in a KPF DB.  The data to be plotted and 
        attributes are stored in an array of dictionaries called 'panel_arr'.

        Args:
            panel_dict (array of dictionaries) - each dictionary in the array has keys:
                panelvars: a dictionary of matplotlib attributes including:
                    ylabel - text for y-axis label
                paneldict: a dictionary containing:
                    col: name of DB column to plot
                    plot_type: 
                    plot_attr: a dictionary containing plot attributes for a scatter plot, 
                        including 'label', 'marker', 'color'
                    on_sky: if set to 'True', only on-sky observations will be included; if set to 'False', only calibrations will be included
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
        
        Example:
            myTS = AnalyzeTimeSeries()
            
            # Green CCD panel
            dict1 = {'col': 'FLXCOLLG', 'plot_type' 'scatter', 'plot_attr': {'label': 'Collimator-side', 'marker': '.', 'color': 'darkgreen'}}
            dict2 = {'col': 'FLXECHG',  'plot_type' 'scatter', 'plot_attr': {'label': 'Echelle-side',    'marker': '.', 'color': 'forestgreen'}}
            dict3 = {'col': 'FLXREG1G', 'plot_type' 'scatter', 'plot_attr': {'label': 'Region 1',        'marker': '.', 'color': 'lightgreen'}}
            dict4 = {'col': 'FLXREG2G', 'plot_type' 'scatter', 'plot_attr': {'label': 'Region 2',        'marker': '.', 'color': 'lightgreen'}}
            thispanelvars = [dict3, dict4, dict1, dict2, ]
            thispaneldict = {'ylabel': 'Green CCD\nDark current [e-/hr]'}
            greenpanel = {'panelvars': thispanelvars,
                          'paneldict': thispaneldict}
            # Red CCD panel
            dict1 = {'col': 'FLXCOLLR', 'plot_type': 'scatter', 'plot_attr': {'label': 'Collimator-side', 'marker': '.', 'color': 'darkred'}}
            dict2 = {'col': 'FLXECHR',  'plot_type': 'scatter', 'plot_attr': {'label': 'Echelle-side',    'marker': '.', 'color': 'firebrick'}}
            dict3 = {'col': 'FLXREG1R', 'plot_type': 'scatter', 'plot_attr': {'label': 'Region 1',        'marker': '.', 'color': 'lightcoral'}}
            dict4 = {'col': 'FLXREG2R', 'plot_type': 'scatter', 'plot_attr': {'label': 'Region 2',        'marker': '.', 'color': 'lightcoral'}}
            thispanelvars = [dict3, dict4, dict1, dict2]
            thispaneldict = {'ylabel': 'Red CCD\nDark current [e-/hr]'}
            redpanel = {'panelvars': thispanelvars,
                        'paneldict': thispaneldict}
            panel_arr = [greenpanel, redpanel]
            start_date = datetime(2023,11, 1)
            end_date   = datetime(2023,12, 1)
            myTS.plot_time_series_multipanel(panel_arr, start_date=start_date, end_date=end_date, show_plot=True)        
        """

        if start_date == None:
            start_date = max(df['DATE-MID'])
        if end_date == None:
            end_date = max(df['DATE-MID'])
        npanels = len(panel_arr)
        unique_cols = set()
        unique_cols.add('DATE-MID')
        unique_cols.add('FIUMODE')
        for panel in panel_arr:
            for d in panel['panelvars']:
                col_value = d['col']
                unique_cols.add(col_value)
        df = self.dataframe_from_db(unique_cols, start_date=start_date, end_date=end_date, verbose=False)
        df['DATE-MID'] = pd.to_datetime(df['DATE-MID']) # move this to dataframe_from_db ?
        df = df.sort_values(by='DATE-MID')
        if clean:
            df = self.clean_df(df)
        
        fig, axs = plt.subplots(npanels, 1, sharex=True, figsize=(15, npanels*2.5), tight_layout=True)
        if npanels == 1:
            axs = [axs]  # Make axs iterable even when there's only one panel
        if npanels > 1:
            plt.subplots_adjust(hspace=0)
        plt.tight_layout()

        for p in np.arange(npanels):
            thispanel = panel_arr[p]
            this_df = df.copy(deep=True)
            if 'on_sky' in thispanel['paneldict']:
                if (thispanel['paneldict']['on_sky']).lower() == 'true':
                    this_df = this_df[this_df['FIUMODE'] == 'Observing']
                elif (thispanel['paneldict']['on_sky']).lower() == 'false':
                    this_df = this_df[this_df['FIUMODE'] == 'Calibration']
            # add this logic
            #if 'only_object' in thispanel['paneldict']:
            #if 'object_like' in thispanel['paneldict']:
            if abs((end_date - start_date).days) <= 1.2:
                t = [(date - start_date).total_seconds() /  3600 for date in this_df['DATE-MID']]
                xtitle = 'Hours since ' + start_date.strftime('%Y-%m-%d %H:%M') + ' UT'
                if 'title' in thispanel['paneldict']:
                    thistitle = thispanel['paneldict']['title'] + ": " + start_date.strftime('%Y-%m-%d %H:%M') + " to " + end_date.strftime('%Y-%m-%d %H:%M')
                axs[p].set_xlim(0, (end_date - start_date).total_seconds() /  3600)
                axs[p].xaxis.set_major_locator(ticker.MaxNLocator(nbins=12, min_n_ticks=4, prune=None))
            elif abs((end_date - start_date).days) <= 3:
                t = [(date - start_date).total_seconds() / 86400 for date in this_df['DATE-MID']]
                xtitle = 'Days since ' + start_date.strftime('%Y-%m-%d %H:%M') + ' UT'
                if 'title' in thispanel['paneldict']:
                    thistitle = thispanel['paneldict']['title'] + ": " + start_date.strftime('%Y-%m-%d %H:%M') + " to " + end_date.strftime('%Y-%m-%d %H:%M')
                axs[p].set_xlim(0, (end_date - start_date).total_seconds() /  86400)
                axs[p].xaxis.set_major_locator(ticker.MaxNLocator(nbins=12, min_n_ticks=4, prune=None))
            elif abs((end_date - start_date).days) < 32:
                t = [(date - start_date).total_seconds() / 86400 for date in this_df['DATE-MID']]
                xtitle = 'Days since ' + start_date.strftime('%Y-%m-%d %H:%M') + ' UT'
                if 'title' in thispanel['paneldict']:
                    thistitle = thispanel['paneldict']['title'] + ": " + start_date.strftime('%Y-%m-%d') + " to " + end_date.strftime('%Y-%m-%d')
                axs[p].set_xlim(0, (end_date - start_date).total_seconds() /  86400)
                axs[p].xaxis.set_major_locator(ticker.MaxNLocator(nbins=12, min_n_ticks=3, prune=None))
            else:
                t = this_df['DATE-MID'] # dates
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
            if 'yscale' in thispanel['paneldict']:
                if thispanel['paneldict']['yscale'] == 'log':
                    axs[p].set_yscale('log')
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
                col_data = this_df[thispanel['panelvars'][i]['col']]
                col_data_replaced = col_data.replace('NaN', np.nan)
                col_data_replaced = col_data.replace('null', np.nan)
                data = np.array(col_data_replaced, dtype='float')
                plot_attributes = {}
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
                                            label += ' ' + thispanel['panelvars'][i]['unit']
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
                axs[p].xaxis.set_tick_params(labelsize=10)
                axs[p].yaxis.set_tick_params(labelsize=10)
                if makelegend:
                    if 'legend_frac_size' in thispanel['paneldict']:
                        legend_frac_size = thispanel['paneldict']['legend_frac_size']
                    else:
                        legend_frac_size = 0.20
                    axs[p].legend(loc='upper right', bbox_to_anchor=(1+legend_frac_size, 1))
            axs[p].grid(color='lightgray')

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=300, facecolor='w')
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
        
        if plot_name == 'chamber_temp':
            dict1 = {'col': 'kpfmet.TEMP',              'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label':  'Hallway',              'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'kpfmet.GREEN_LN2_FLANGE',  'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': r'Green LN$_2$ Flng',    'marker': '.', 'linewidth': 0.5, 'color': 'darkgreen'}}
            dict3 = {'col': 'kpfmet.RED_LN2_FLANGE',    'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': r'Red LN$_2$ Flng',      'marker': '.', 'linewidth': 0.5, 'color': 'darkred'}}
            dict4 = {'col': 'kpfmet.CHAMBER_EXT_BOTTOM','plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': r'Chamber Ext Bot',      'marker': '.', 'linewidth': 0.5}}
            dict5 = {'col': 'kpfmet.CHAMBER_EXT_TOP',   'plot_type': 'plot',    'unit': 'K', 'plot_attr': {'label': r'Chamber Exterior Top', 'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1]
            thispaneldict = {'ylabel': 'Hallway\n' + r' Temperature ($^{\circ}$C)',
                             'title': 'KPF Temperatures',
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
                             'title': 'KPF Temperatures',
                             'subtractmedian': 'true',
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
            thispaneldict = {'ylabel': 'Spectrometer\nTemperatures' + ' ($^{\circ}$C)',
                             'nolegend': 'false',
                             'legend_frac_size': 0.3}
            chambertemppanel = {'panelvars': thispanelvars,
                                'paneldict': thispaneldict}
            
            thispaneldict = {'ylabel': 'Spectrometer\n' + r'$\Delta$Temperature (K)',
                             'title': 'KPF Spectrometer Temperatures', 
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
                             'title': 'KPF Spectrometer Temperatures', 
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
            thispaneldict = {'ylabel': 'Fiber \n Temperatures' + ' ($^{\circ}$C)',
                             'legend_frac_size': 0.25}
            fibertempspanel = {'panelvars': thispanelvars,
                               'paneldict': thispaneldict}
            panel_arr = [fibertempspanel]

        elif plot_name=='ccd_readspeed':
            dict1 = {'col': 'GREENTRT', 'plot_type': 'plot', 'unit': 'e-', 'plot_attr': {'label': 'Green CCD', 'marker': '.', 'linewidth': 0.5, 'color': 'darkgreen'}}
            dict2 = {'col': 'REDTRT',   'plot_type': 'plot', 'unit': 'e-', 'plot_attr': {'label': 'Red CCD',   'marker': '.', 'linewidth': 0.5, 'color': 'darkred'}}
            thispanelvars = [dict1, dict2]
            thispaneldict = {'ylabel': 'Read Speed [sec]',
                             'title': 'CCD Read Speed',
                             'legend_frac_size': 0.25}
            readspeedpanel = {'panelvars': thispanelvars,
                              'paneldict': thispaneldict}
            panel_arr = [readspeedpanel]

        elif plot_name=='ccd_readnoise':
            dict1 = {'col': 'RNGREEN1', 'plot_type': 'plot', 'unit': 'e-', 'plot_attr': {'label': 'Green CCD 1', 'marker': '.', 'linewidth': 0.5, 'color': 'darkgreen'}}
            dict2 = {'col': 'RNGREEN2', 'plot_type': 'plot', 'unit': 'e-', 'plot_attr': {'label': 'Green CCD 2', 'marker': '.', 'linewidth': 0.5, 'color': 'forestgreen'}}
            dict3 = {'col': 'RNRED1',   'plot_type': 'plot', 'unit': 'e-', 'plot_attr': {'label': 'RED CCD 1',   'marker': '.', 'linewidth': 0.5, 'color': 'darkred'}}
            dict4 = {'col': 'RNRED2',   'plot_type': 'plot', 'unit': 'e-', 'plot_attr': {'label': 'RED CCD 2',   'marker': '.', 'linewidth': 0.5, 'color': 'firebrick'}}
            thispanelvars = [dict1, dict2, dict3, dict4]
            thispaneldict = {'ylabel': 'Read Noise [e-]',
                             'title': 'Read Noise',
                             'legend_frac_size': 0.25}
            readnoisepanel = {'panelvars': thispanelvars,
                              'paneldict': thispaneldict}
            panel_arr = [readnoisepanel]
        
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
            thispaneldict = {'ylabel': 'Green CCD\nDark current [e-/hr]',
                             'title': 'Dark Current',
                             'legend_frac_size': 0.30}
            greenpanel = {'panelvars': thispanelvars,
                          'paneldict': thispaneldict}
            
            # Red CCD panel - Dark current
            dict1 = {'col': 'FLXCOLLR', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Collimator-side', 'marker': '.', 'linewidth': 0.5, 'color': 'darkred'}}
            dict2 = {'col': 'FLXECHR',  'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Echelle-side',    'marker': '.', 'linewidth': 0.5, 'color': 'firebrick'}}
            dict3 = {'col': 'FLXREG1R', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 1',        'marker': '.', 'linewidth': 0.5, 'color': 'lightcoral'}}
            dict4 = {'col': 'FLXREG2R', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 2',        'marker': '.', 'linewidth': 0.5, 'color': 'lightcoral'}}
            dict5 = {'col': 'FLXREG3R', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 3',        'marker': '.', 'linewidth': 0.5, 'color': 'lightcoral'}}
            dict6 = {'col': 'FLXREG4R', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 4',        'marker': '.', 'linewidth': 0.5, 'color': 'lightcoral'}}
            dict7 = {'col': 'FLXREG5R', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 5',        'marker': '.', 'linewidth': 0.5, 'color': 'lightcoral'}}
            dict8 = {'col': 'FLXREG6R', 'plot_type': 'plot', 'unit': 'e-/hr', 'plot_attr': {'label': 'Region 6',        'marker': '.', 'linewidth': 0.5, 'color': 'lightcoral'}}
            thispanelvars = [dict3, dict4, dict1, dict2, ]
            thispaneldict = {'ylabel': 'Red CCD\nDark current [e-/hr]',
                             'legend_frac_size': 0.30}
            redpanel = {'panelvars': thispanelvars,
                        'paneldict': thispaneldict}
            
            # Green CCD panel - ion pump current
            dict1 = {'col': 'kpfgreen.COL_CURR', 'plot_type': 'plot', 'unit': 'A', 'plot_attr': {'label': 'Collimator-side', 'marker': '.', 'linewidth': 0.5, 'color': 'darkgreen'}}
            dict2 = {'col': 'kpfgreen.ECH_CURR', 'plot_type': 'plot', 'unit': 'A', 'plot_attr': {'label': 'Echelle-side',    'marker': '.', 'linewidth': 0.5, 'color': 'forestgreen'}}
            thispanelvars = [dict1]
            thispaneldict = {'ylabel': 'Green CCD\nIon Pump current [A]',
                             'yscale': 'log',
                             'legend_frac_size': 0.30}
            greenpanel_ionpump = {'panelvars': thispanelvars,
                                  'paneldict': thispaneldict}
            thispanelvars = [dict2]
            thispaneldict = {'ylabel': 'Green CCD\nIon Pump current [A]',
                             'yscale': 'log',
                             'legend_frac_size': 0.30}
            greenpanel_ionpump2 = {'panelvars': thispanelvars,
                                   'paneldict': thispaneldict}
            
            # Red CCD panel - ion pump current
            dict1 = {'col': 'kpfred.COL_CURR', 'plot_type': 'plot', 'unit': 'A', 'plot_attr': {'label': 'Collimator-side', 'marker': '.', 'linewidth': 0.5, 'color': 'darkred'}}
            dict2 = {'col': 'kpfred.ECH_CURR', 'plot_type': 'plot', 'unit': 'A', 'plot_attr': {'label': 'Echelle-side',    'marker': '.', 'linewidth': 0.5, 'color': 'firebrick'}}
            thispanelvars = [dict1]
            thispaneldict = {'ylabel': 'Red CCD\nIon Pump current [A]',
                             'yscale': 'log',
                             'legend_frac_size': 0.30}
            redpanel_ionpump = {'panelvars': thispanelvars,
                                'paneldict': thispaneldict}
            thispanelvars = [dict2]
            thispaneldict = {'ylabel': 'Red CCD\nIon Pump current [A]',
                             'yscale': 'log',
                             'legend_frac_size': 0.30}
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
            thispaneldict = {'ylabel': 'CCD Amplifier\nDark current [e-/hr]',
                             'legend_frac_size': 0.30}
            amppanel = {'panelvars': thispanelvars,
                        'paneldict': thispaneldict}
            panel_arr = [greenpanel, redpanel, greenpanel_ionpump, greenpanel_ionpump2, redpanel_ionpump, redpanel_ionpump2, amppanel]

        elif plot_name=='ccd_temp':
            # CCD Temperatures
            dict1 = {'col': 'kpfgreen.STA_CCD_T', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'STA Sensor', 'marker': '.', 'linewidth': 0.5, 'color': 'darkgreen'}}
            dict2 = {'col': 'kpfgreen.KPF_CCD_T', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'SSL Sensor', 'marker': '.', 'linewidth': 0.5, 'color': 'forestgreen'}}
            thispanelvars = [dict2, dict1, ]
            thispaneldict = {'ylabel': 'Green CCD\nTemperature (C)',
                             'title': 'CCD Temperatures',
                             'legend_frac_size': 0.30}
            green_ccd = {'panelvars': thispanelvars,
                         'paneldict': thispaneldict}

            dict1 = {'col': 'kpfred.STA_CCD_T', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'STA Sensor', 'marker': '.', 'linewidth': 0.5, 'color': 'darkred'}}
            dict2 = {'col': 'kpfred.KPF_CCD_T', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'SSL Sensor', 'marker': '.', 'linewidth': 0.5, 'color': 'firebrick'}}
            thispanelvars2 = [dict2, dict1, ]
            thispaneldict2 = {'ylabel': 'Red CCD\nTemperature (C)',
                             'legend_frac_size': 0.30}
            red_ccd = {'panelvars': thispanelvars2,
                       'paneldict': thispaneldict2}

            dict1 = {'col': 'kpfgreen.STA_CCD_T', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'STA Sensor', 'marker': '.', 'linewidth': 0.5, 'color': 'darkgreen'}}
            dict2 = {'col': 'kpfgreen.KPF_CCD_T', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'SSL Sensor', 'marker': '.', 'linewidth': 0.5, 'color': 'forestgreen'}}
            thispanelvars3 = [dict2, dict1, ]
            thispaneldict3 = {'ylabel': 'Green CCD\n' + r'$\Delta$Temperature (K)',
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.30}
            green_ccd2 = {'panelvars': thispanelvars3,
                          'paneldict': thispaneldict3}

            dict1 = {'col': 'kpfred.STA_CCD_T', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'STA Sensor', 'marker': '.', 'linewidth': 0.5, 'color': 'darkred'}}
            dict2 = {'col': 'kpfred.KPF_CCD_T', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'SSL Sensor', 'marker': '.', 'linewidth': 0.5, 'color': 'firebrick'}}
            thispanelvars4 = [dict2, dict1, ]
            thispaneldict4 = {'ylabel': 'Red CCD\n' + r'$\Delta$Temperature (K)',
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.30}
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
                             'legend_frac_size': 0.35}
            controller1 = {'panelvars': thispanelvars,
                           'paneldict': thispaneldict}

            thispanelvars2 = [dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9, dict10, dict11, ]
            thispaneldict2 = {'ylabel': r'$\Delta$Temperature (K)',
                             'title': 'CCD Controllers',
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.35}
            controller2 = {'panelvars': thispanelvars2,
                           'paneldict': thispaneldict2}
            panel_arr = [copy.deepcopy(controller1), copy.deepcopy(controller2)]

        elif plot_name=='lfc':
            dict1 = {'col': 'kpfcal.IRFLUX',  'plot_type': 'scatter', 'unit': 'counts', 'plot_attr': {'label': 'Fiberlock IR',  'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1]
            thispaneldict1 = {'ylabel': 'Intensity (counts)',
                              'title': 'LFC Diagnostics',
                              'legend_frac_size': 0.35}
            lfcpanel1 = {'panelvars': thispanelvars,
                         'paneldict': thispaneldict1}
            dict1 = {'col': 'kpfcal.VISFLUX', 'plot_type': 'scatter', 'unit': 'counts', 'plot_attr': {'label': 'Fiberlock Vis', 'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1]
            thispaneldict2 = {'ylabel': 'Intensity (counts)',
                              'legend_frac_size': 0.35}
            lfcpanel2 = {'panelvars': thispanelvars,
                         'paneldict': thispaneldict2}

            dict1 = {'col': 'kpfcal.BLUECUTIACT', 'plot_type': 'scatter', 'unit': 'A', 'plot_attr': {'label': 'Blue Cut Amp. Current',  'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1]
            thispaneldict3 = {'ylabel': 'Current (A)',
                              'title': 'LFC Diagnostics',
                              'legend_frac_size': 0.35}
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
                             'title': 'Etalon Temperatures',
                             'legend_frac_size': 0.35}
            thispaneldict2 = {'ylabel': r'$\Delta$Temperature (K)',
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.35}
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
            dict1 = {'col': 'kpfmet.TEMP',     'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'Hallway',      'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'kpfmet.TH_DAILY', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'Th-Ar Daily',  'marker': '.', 'linewidth': 0.5}}
            dict3 = {'col': 'kpfmet.TH_GOLD',  'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'Th-Ar Gold',   'marker': '.', 'linewidth': 0.5}}
            dict4 = {'col': 'kpfmet.U_DAILY',  'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'U-Ar Daily',   'marker': '.', 'linewidth': 0.5}}
            dict5 = {'col': 'kpfmet.U_GOLD',   'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'U-Ar Gold',    'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1]
            thispaneldict = {'ylabel': 'Temperature (C)',
                             'title': 'Hollow-Cathode Lamp Temperatures',
                             'legend_frac_size': 0.35}
            hclpanel = {'panelvars': thispanelvars,
                        'paneldict': thispaneldict}
            thispanelvars = [dict2, dict3, dict4, dict5]
            thispaneldict = {'ylabel': 'Temperature (C)',
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
            thispaneldict = {'ylabel': 'Spectrometer\nTemperatures (K)',
                             'title': 'Ca H&K Spectrometer Temperatures',
                             'legend_frac_size': 0.35}
            hkpanel1 = {'panelvars': thispanelvars,
                        'paneldict': thispaneldict}

            thispanelvars2 = [dict1, dict2, dict3, dict5, dict6, dict4]
            thispaneldict2 = {'ylabel': 'Spectrometer\n' + '$\Delta$Temperature (K)',
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.35}
            hkpanel2 = {'panelvars': thispanelvars2,
                        'paneldict': thispaneldict2}

            dict1 = {'col': 'kpf_hk.COOLTARG', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'Detector Target Temp.', 'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'kpf_hk.CURRTEMP', 'plot_type': 'plot', 'unit': 'K', 'plot_attr': {'label': 'Detector Temp.',        'marker': '.', 'linewidth': 0.5}}
            thispanelvars3 = [dict1, dict2] 
            thispaneldict3 = {'ylabel': 'Detector\nTemperatures (K)',
                              'legend_frac_size': 0.35}
            hkpanel3 = {'panelvars': thispanelvars3,
                        'paneldict': thispaneldict3}

            thispanelvars4 = [dict1, dict2]
            thispaneldict4 = {'ylabel': 'Detector\n' + '$\Delta$Temperature (K)',
                             'subtractmedian': 'true',
                             'legend_frac_size': 0.35}
            hkpanel4 = {'panelvars': thispanelvars4,
                        'paneldict': thispaneldict4}

            panel_arr = [copy.deepcopy(hkpanel1), copy.deepcopy(hkpanel2), copy.deepcopy(hkpanel3), copy.deepcopy(hkpanel4)]

            
        elif plot_name=='agitator':
            dict1 = {'col': 'kpfmot.AGITSPD', 'plot_type': 'scatter', 'unit': 'counts/sec', 'plot_attr': {'label': 'Agitator Speed', 'marker': '.', 'linewidth': 0.5}}
            thispanelvars1 = [dict1]
            thispaneldict1 = {'ylabel': 'Agitator Speed\n(counts/sec)',
                              'title': r'KPF Agitator',
                              'legend_frac_size': 0.35}
            agitatorpanel1 = {'panelvars': thispanelvars1,
                              'paneldict': thispaneldict1}
            dict2 = {'col': 'kpfmot.AGITTOR', 'plot_type': 'scatter', 'unit': 'V', 'plot_attr': {'label': 'Agitator Motor Torque', 'marker': '.', 'linewidth': 0.5}}
            thispanelvars2 = [dict2]
            thispaneldict2 = {'ylabel': 'Motor Torque (V)',
                              'legend_frac_size': 0.35}
            agitatorpanel2 = {'panelvars': thispanelvars2,
                              'paneldict': thispaneldict2}
            dict3 = {'col': 'kpfmot.AGITAMBI_T', 'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': 'Ambient Temperature', 'marker': '.', 'linewidth': 0.5}}
            dict4 = {'col': 'kpfmot.AGITMOT_T',  'plot_type': 'scatter', 'unit': 'K', 'plot_attr': {'label': 'Motor Temperature',   'marker': '.', 'linewidth': 0.5}}
            thispanelvars3 = [dict3, dict4]
            thispaneldict3 = {'ylabel': 'Temperature (C)',
                              'legend_frac_size': 0.35}
            agitatorpanel3 = {'panelvars': thispanelvars3,
                              'paneldict': thispaneldict3}
            dict5 = {'col': 'kpfmot.AGITAMBI_T', 'plot_type': 'scatter', 'unit': 'mA', 'plot_attr': {'label': 'Outlet A1 Power', 'marker': '.', 'linewidth': 0.5}}
            thispanelvars4 = [dict5]
            thispaneldict4 = {'ylabel': 'Outlet A1 Power (mA)',
                              'legend_frac_size': 0.35}
            agitatorpanel4 = {'panelvars': thispanelvars4,
                              'paneldict': thispaneldict4}
            panel_arr = [agitatorpanel1, agitatorpanel2, agitatorpanel3, agitatorpanel4]

        elif plot_name=='guiding':
            dict1 = {'col': 'GDRXRMS',  'plot_type': 'plot', 'unit': 'mas', 'plot_attr': {'label': 'RMS Guiding Error (X)', 'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'GDRYRMS',  'plot_type': 'plot', 'unit': 'mas', 'plot_attr': {'label': 'RMS Guiding Error (Y)', 'marker': '.', 'linewidth': 0.5}}
            dict3 = {'col': 'GDRXBIAS', 'plot_type': 'plot', 'unit': 'mas', 'plot_attr': {'label': 'RMS Guiding Bias (X)',  'marker': '.', 'linewidth': 0.5}}
            dict4 = {'col': 'GDRYBIAS', 'plot_type': 'plot', 'unit': 'mas', 'plot_attr': {'label': 'RMS Guiding Bias (Y)',  'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1, dict2]
            thispaneldict = {'ylabel': 'Guiding Errors (mas)',
                             'title': 'Guiding',
                             'on_sky': 'true', 
                             'legend_frac_size': 0.35}
            guidingpanel1 = {'panelvars': thispanelvars,
                             'paneldict': thispaneldict}

            thispanelvars2 = [dict3, dict4]
            thispaneldict2 = {'ylabel': 'Guiding Bias (mas)',
                             'title': 'Guiding',
                             'on_sky': 'true', 
                             'legend_frac_size': 0.35}
            guidingpanel2 = {'panelvars': thispanelvars,
                             'paneldict': thispaneldict}
            panel_arr = [guidingpanel1, guidingpanel2]

        elif plot_name=='seeing':
            dict1 = {'col': 'GDRSEEJZ', 'plot_type': 'scatter', 'unit': 'as', 'plot_attr': {'label': 'Seeing in J+Z band', 'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'GDRSEEV',  'plot_type': 'scatter', 'unit': 'as', 'plot_attr': {'label': 'Seeing in V band',   'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1, dict2]
            thispaneldict = {'ylabel': 'Seeing (arcsec)',
                             'title': 'Seeing',
                             'on_sky': 'true', 
                             'legend_frac_size': 0.35}
            seeingpanel = {'panelvars': thispanelvars,
                           'paneldict': thispaneldict}
            panel_arr = [seeingpanel]

        elif plot_name=='sun_moon':
            dict1 = {'col': 'MOONSEP', 'plot_type': 'scatter', 'unit': 'deg', 'plot_attr': {'label': 'Moon-target separation', 'marker': '.', 'linewidth': 0.5}}
            dict2 = {'col': 'SUNALT',  'plot_type': 'scatter', 'unit': 'deg', 'plot_attr': {'label': 'Alt. of Sun',            'marker': '.', 'linewidth': 0.5}}
            thispanelvars = [dict1, dict2]
            thispaneldict = {'ylabel': 'Angle (deg)',
                             'title': 'Separation of Sun and Moon from Target',
                             'on_sky': 'true', 
                             'legend_frac_size': 0.35}
            seeingpanel = {'panelvars': thispanelvars,
                           'paneldict': thispaneldict}
            panel_arr = [seeingpanel]

        else:
            self.logger.error('plot_name not specified')
            return
        
        self.plot_time_series_multipanel(panel_arr, start_date=start_date, end_date=end_date, 
                                         fig_path=fig_path, show_plot=show_plot, clean=clean)        


    def plot_all_quicklook(self, start_date=None, interval='day', clean=True, 
                                 fig_dir=None, show_plot=False):
        """
        Generate all of the standard time series plots for the quicklook.  
        Depending on the value of the input 'interval', the plots have time ranges 
        that are daily, weekly, yearly, or decadal.

        Args:
            start_date (datetime object) - start date for plot
            interval (string) - 'day', 'week', 'year', or 'decade'
            fig_path (string) - set to the path for the files to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plots it the current environment
            (e.g., in a Jupyter Notebook).
        """
        
        if not isinstance(start_date, datetime):
            self.logger.error("'start_date' must be a datetime object.")
            return        
        
        plots = { 
            "p1a":  {"plot_name": "chamber_temp",        "subdir": "Chamber",   },
            "p1b":  {"plot_name": "chamber_temp_detail", "subdir": "Chamber",   },
            "p1c":  {"plot_name": "fiber_temp",          "subdir": "Chamber",   },
            "p2a":  {"plot_name": "ccd_readnoise",       "subdir": "CCDs",      },
            "p2b":  {"plot_name": "ccd_dark_current",    "subdir": "CCDs",      },
            "p2c":  {"plot_name": "ccd_readspeed",       "subdir": "CCDs",      },
            "p2d":  {"plot_name": "ccd_controller",      "subdir": "CCDs",      },
            "p2e":  {"plot_name": "ccd_temp",            "subdir": "CCDs",      },
            "p3a":  {"plot_name": "lfc",                 "subdir": "Cal",       },
            "p3b":  {"plot_name": "etalon",              "subdir": "Cal",       },
            "p4":   {"plot_name": "hk_temp",             "subdir": "Subsystems",},
            "p5a":  {"plot_name": "guiding",             "subdir": "Observing", },
            "p5b":  {"plot_name": "seeing",              "subdir": "Observing", },
            "p5c":  {"plot_name": "sun_moon",            "subdir": "Observing", },
        }
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
            else:
                self.logger.error("The input 'interval' must be 'daily', 'weekly', 'yearly', or 'decadal'.")
                return

            if fig_dir != None:
                if not fig_dir.endswith('/'):
                    fig_dir += '/'
                savedir = fig_dir + plots[p]["subdir"] + '/'
                os.makedirs(savedir, exist_ok=True) # make directories if needed
                fig_path = savedir + filename
            else:
                fig_path = None
            self.logger.info('Making QL time series plot ' + fig_path)
            self.plot_standard_time_series(plot_name, start_date=start_date, end_date=end_date, 
                                           fig_path=fig_path, show_plot=show_plot, clean=clean)


    def plot_all_quicklook_daterange(self, start_date=None, end_date=None, clean=True, 
                                     base_dir=None, show_plot=False):
        """
        Generate all of the standard time series plots for the quicklook for a date 
        range.  Every unique day, month, year, and decade between start_date and end_date 
        will have a full set of plots produced using plot_all_quicklook().

        Args:
            start_date (datetime object) - start date for plot
            end_date (datetime object) - start date for plot
            fig_path (string) - set to the path for the files to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plots in fig_path or shows the plots it the current environment
            (e.g., in a Jupyter Notebook).
        """

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

        self.logger.info('Making time series plots for ' + str(len(days)) + ' day(s)')
        for day in days:
            try:
                savedir = base_dir + day.strftime("%Y%m%d") + '/Masters/'
                self.plot_all_quicklook(day, interval='day', fig_dir=savedir)
            except Exception as e:
                self.logger.error(e)

        self.logger.info('Making time series plots for ' + str(len(months)) + ' month(s)')
        for month in months:
            try:
                savedir = base_dir + month.strftime("%Y%m") + '00/Masters/'
                self.plot_all_quicklook(month, interval='month', fig_dir=savedir)
            except Exception as e:
                self.logger.error(e)

        self.logger.info('Making time series plots for ' + str(len(years)) + ' year(s)')
        for year in years:
            try:
                savedir = base_dir + year.strftime("%Y") + '0000/Masters/'
                self.plot_all_quicklook(year, interval='year', fig_dir=savedir)
            except Exception as e:
                self.logger.error(e)

        self.logger.info('Making time series plots for ' + str(len(decades)) + ' decade(s)')
        for decade in decades:
            try:
                savedir = base_dir + decade.strftime("%Y")[0:3] + '00000/Masters/' 
                self.plot_all_quicklook(decade, interval='decade', fig_dir=savedir)
            except Exception as e:
                self.logger.error(e)


def add_one_month(inputdate):
    """
    Add one month to a datetime object, accounting for the different number of days per month.
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
