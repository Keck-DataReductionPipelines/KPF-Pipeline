import os
import re
import glob
import sqlite3
import hashlib
import psycopg2
import pandas as pd
import numpy as np
import time
from astropy.time import Time
from astropy.table import Table
from astropy.io import fits
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
from datetime import datetime
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from kpfpipe.models.level1 import KPF1
from kpfpipe.logger import start_logger
from modules.Utils.utils import DummyLogger
from modules.Utils.kpf_parse import get_datecode

DEFAULT_CFG_PATH = 'database/modules/utils/tsdb.cfg'

class TSDB:
    """
    Description:
        Class to facilitate execution of queries in the KPF time series database.
        This class is designed to work with SQLITE3 and PostgreSQL databases.
        Separate methods are defined for each query.

        This class contains a set of methods to create a database of data associated 
        with KPF observations, as well as methods to ingest data, query the database, 
        and print data.  
        
        The ingested data comes from L0/2D/L1/L2 primary header keywords, 
        header keywords from the RV extension in L2 files, data from the 
        table in the RV extension in L2 files, and data from the TELEMETRY 
        extension in L0 files.  All TELEMETRY keywords are added to the 
        database an a subset of the L0/2D/L1/L2 keywords are added. 
        These lists of ingested keywords can be expanded by modifying 
        configuration files.
        
    Arguments:
        base_dir (string) - L0 directory
        db_path (string) - path to database file (for SQLITE3)
        drop (boolean) - if true, the database at db_path is dropped at startup
                         (for SQLITE3)
        logger (logger object) - a logger object can be passed, or a 
                                 DummyLogger will be created to make formatted 
                                 print statements

    Attributes:
        L0_keyword_types (dictionary) - specifies data types for L0 header keywords
        D2_keyword_types (dictionary) - specifies data types for 2D header keywords
        L1_keyword_types (dictionary) - specifies data types for L1 header keywords
        L2_keyword_types (dictionary) - specifies data types for L2 header keywords
        L0_telemetry_types (dictionary) - specifies data types for L0 telemetry keywords
        L2_RV_header_keyword_types (dictionary) - specifies data types for L2 RV header keywords
        L2_RV_ccf_keyword_types (dictionary) - specifies data types for L2 CCF header keywords

    To-do:
        * Add temperature derivatives as columns; they will need to be computed.
        * Add database for masters (separate from ObsIDs?)

#    For PostgreSQL, it returns exitcode:
#         0 = Normal
#         2 = Exception raised closing database connection
#        64 = Cannot connect to database
#        65 = Input file does not exist
#        66 = File checksum does not match database checksum
#        67 = Could not execute query
#        68 = Failed to compute checksum
    """

    def __init__(self, db_type='sqlite3', db_path='kpf_ts.db', base_dir='/data/L0', logger=None, drop=False, verbose=False):
        """
        Todo: add docstring, including explanation of db_type = 'sqlite3' or 'postgres'
        """
        
        self.db_type = db_type # sqlite3 or postgresql
        self.verbose = verbose
        self.logger = logger if logger is not None else DummyLogger()
        self.logger.info('Starting KPF_TSDB')

        if self.is_notebook():
            self.tqdm = tqdm_notebook
            self.logger.info('Jupyter Notebook environment detected.')
        else:
            self.tqdm = tqdm

# ADJUST this for sqlite3 only
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
            self.drop_tables()
            self.logger.info('Dropping KPF database ' + str(self.db_path))

        self.conn = None

        if db_type == 'sqlite3':
            pass

        elif db_type == 'postgresql':
            # Get database connection parameters from environment.
            dbport = os.getenv('DBPORT')
            dbname = os.getenv('DBNAME')
            dbuser = os.getenv('DBUSER')
            dbpass = os.getenv('DBPASS')
            dbserver = os.getenv('DBSERVER')
        
            self.exit_code = 0
            #self.cId = None
            #self.db_level = None

            # Connect to database
            db_fail = True
            n_attempts = 3
            for i in range(n_attempts):
                try:
                    self.conn = psycopg2.connect(host=dbserver,database=dbname,port=dbport,user=dbuser,password=dbpass)
                    db_fail = False
                    break
                except:
                    self.logger.info("Could not connect to database, retrying...")
                    db_fail = True
                    time.sleep(10)
    
            if db_fail:
                self.logger.error(f"Could not connect to database after {n_attempts} attempts...")
                self.exit_code = 64
                return
    
            # Open database cursor.
            self.cur = self.conn.cursor()
    
            # Select database version.
            q1 = 'SELECT version();'
            if self.verbose:
                self.logger.info('q1 = {}'.format(q1))
            self.cur.execute(q1)
            db_version = self.cur.fetchone()
            if self.verbose:
                self.logger.info('PostgreSQL database version = {}'.format(db_version))
    
            # Check database current_user.
            q2 = 'SELECT current_user;'
            if self.verbose:
                print('q2 = {}'.format(q2))
            self.cur.execute(q2)
            for record in self.cur:
                if self.verbose:
                    print('record = {}'.format(record))
        
        if drop:
            self.drop_tables()
            self.logger.info('Dropping KPF database ' + str(self.db_path))

        # Create tables if needed
        if not self.check_if_table_exsits(tablename='tsdb'):
            self.create_database()
        else:
            self.logger.info("Primary table 'tsdb' already exists.")
        if not self.check_if_table_exsits(tablename='tsdb_metadata'):
            self.create_metadata_table()
        else:
            self.logger.info("Metadata table 'tsdb_metadata' already exists.")
        self.print_db_status()


    def close(self):
        """
        Close database cursor and then connection.
        """
        try:
            self.cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            self.exit_code = 2
        finally:
            if self.conn is not None:
                self.conn.close()
                print('Database connection closed.')


    def drop_tables(self):
        """
        Start over on the database by dropping the main table.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS tsdb")
        cursor.execute("DROP TABLE IF EXISTS tsdb_metadata")
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

        
    def print_db_status(self):
        """
        Prints a brief summary of the database status.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM tsdb')
        nrows = cursor.fetchone()[0]
        cursor.execute('PRAGMA table_info(tsdb)')
        ncolumns = len(cursor.fetchall())
        cursor.execute('SELECT MAX(MAX(L0_header_read_time),MAX(L1_header_read_time)) FROM tsdb')
        most_recent_read_time = cursor.fetchone()[0]
        cursor.execute('SELECT MIN(datecode) FROM tsdb')
        earliest_datecode = cursor.fetchone()[0]
        cursor.execute('SELECT MAX(datecode) FROM tsdb')
        latest_datecode = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(DISTINCT datecode) FROM tsdb')
        unique_datecodes_count = cursor.fetchone()[0]
        conn.close()
        self.logger.info(f"Summary: {nrows} obs x {ncolumns} cols over {unique_datecodes_count} days in {earliest_datecode}-{latest_datecode}; updated {most_recent_read_time}")
        

    def create_metadata_table(self):
        """
        Create a separate table 'kpfdb_metadata' to store column/keyword 
        descriptions and units. Then load data from multiple CSV sources plus 
        custom RV prefixes and insert into the database table.
        """
        # Connect to the database and create the metadata table if it doesn't exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
    
        create_meta_table_query = """
            CREATE TABLE IF NOT EXISTS kpfdb_metadata (
                keyword     TEXT NOT NULL PRIMARY KEY,
                datatype    TEXT,
                description TEXT,
                units       TEXT,
                source      TEXT
            )
        """
        cursor.execute(create_meta_table_query)
        conn.commit()

        def load_keyword_csv(csv_path, source_label):
            """
            Helper function to read a CSV file and return a DataFrame with columns:
               keyword | datatype | unit | description | source
            """
            df = pd.read_csv(csv_path, delimiter='|', dtype=str)
            df['source'] = source_label
            df = df[['keyword', 'datatype', 'unit', 'description', 'source']]
            return df

        # Load keywords from CSV files for multiple levels
        df_der   = load_keyword_csv('/code/KPF-Pipeline/static/tsdb_keywords/derived_keywords.csv',      source_label='Derived Keywords')
        df_l0    = load_keyword_csv('/code/KPF-Pipeline/static/tsdb_keywords/l0_primary_keywords.csv',   source_label='L0 PRIMARY Header')
        df_2d    = load_keyword_csv('/code/KPF-Pipeline/static/tsdb_keywords/d2_primary_keywords.csv',   source_label='2D PRIMARY Header')
        df_l1    = load_keyword_csv('/code/KPF-Pipeline/static/tsdb_keywords/l1_primary_keywords.csv',   source_label='L1 PRIMARY Header')
        df_l2    = load_keyword_csv('/code/KPF-Pipeline/static/tsdb_keywords/l2_primary_keywords.csv',   source_label='L2 PRIMARY Header')
        df_l0t   = load_keyword_csv('/code/KPF-Pipeline/static/tsdb_keywords/l0_telemetry_keywords.csv', source_label='L0 TELEMETRY Extension')
        df_l2rv  = load_keyword_csv('/code/KPF-Pipeline/static/tsdb_keywords/l2_rv_keywords.csv',        source_label='L2 RV Header')
        df_l2ccf = load_keyword_csv('/code/KPF-Pipeline/static/tsdb_keywords/l2_green_ccf_keywords.csv', source_label='L2 CCF Header')

        # Build the RV prefix keywords (the 8th source)
        prefixes = ['RV1','RV2','RV3','RVS','ERVS','RVC','ERVC','RVY','ERVY','CCFBJD','BCRV','CCFW']
        units    = ['km/s','km/s','km/s','km/s','km/s','km/s','km/s','km/s','km/s','days','km/s','None']
        descs    = ['RV for SCI1 order ', 'RV for SCI2 order ', 'RV for SCI3 order ','RV for SCI order ', 'Error in RV for SCI order ', 'RV for CAL order ','Error in RV for CAL order ', 'RV for SKY order ',  'Error in RV for SKY order ','BJD for order ','Barycentric RV for order ','CCF weight for order ']
        nums = [f"{i:02d}" for i in range(67)]

        prefix_unit_map = dict(zip(prefixes, units))
        prefix_desc_map = dict(zip(prefixes, descs))

        rv_entries = []
        for prefix in prefixes:
            for num in nums:
                kw   = f"{prefix}{num}"
                dtyp = "REAL"
                desc = f"{prefix_desc_map[prefix]}{num}"
                unt  = prefix_unit_map[prefix]
                rv_entries.append([kw, dtyp, unt, desc, 'L2 RV Extension'])

        df_rv = pd.DataFrame(rv_entries, columns=['keyword','datatype','unit','description','source'])

        # Combine all dataframes and remove duplicate keywords
        df_all = pd.concat([
            df_der, df_l0, df_2d, df_l1, df_l2, df_l0t, df_l2rv, df_l2ccf, df_rv
        ], ignore_index=True)
        df_all.drop_duplicates(subset='keyword', keep='first', inplace=True)

        # Insert (or replace) into kpfdb_metadata
        insert_query = """
            INSERT OR REPLACE INTO kpfdb_metadata
            (keyword, datatype, description, units, source)
            VALUES (?, ?, ?, ?, ?)
        """

        for _, row in df_all.iterrows():
            cursor.execute(
                insert_query,
                (row['keyword'], row['datatype'], row['description'], row['unit'], row['source'])
            )

        conn.commit()
        conn.close()

        self.logger.info("Metadata table 'kpfdb_metadata' created.")


    def check_if_table_exsits(self, tablename=None):
        """
        Return True if the named table exists.
        """
        
        # To-do: check if SQLITE3 is being used

        result = False
        try: 
            if tablename != None:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?;
                """, (tablename,))
                tables = cursor.fetchone() 
                result = False
                if tables != None:
                    if isinstance(tables, tuple):
                        if tablename in tables:
                            result = True
            else:
            	self.logger.info('check_if_table_exsits: tablename not specified.')
        except:
            self.logger.info('check_if_table_exsits: problem with query')

        return result


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
        create_table_query = f'CREATE TABLE IF NOT EXISTS tsdb ({", ".join(cols)}, UNIQUE(ObsID))'
        cursor.execute(create_table_query)
        
        # Define indexed columns
        index_commands = [
            ('CREATE UNIQUE INDEX idx_ObsID       ON tsdb ("ObsID");',       'idx_ObsID'),
            ('CREATE UNIQUE INDEX idx_L0_filename ON tsdb ("L0_filename");', 'idx_L0_filename'),
            ('CREATE UNIQUE INDEX idx_D2_filename ON tsdb ("D2_filename");', 'idx_D2_filename'),
            ('CREATE UNIQUE INDEX idx_L1_filename ON tsdb ("L1_filename");', 'idx_L1_filename'),
            ('CREATE UNIQUE INDEX idx_L2_filename ON tsdb ("L2_filename");', 'idx_L2_filename'),
            ('CREATE INDEX idx_FIUMODE ON tsdb ("FIUMODE");', 'idx_FIUMODE'),
            ('CREATE INDEX idx_OBJECT ON tsdb ("OBJECT");', 'idx_OBJECT'),
            ('CREATE INDEX idx_DATE_MID ON tsdb ("DATE-MID");', 'idx_DATE_MID'),
        ]
        
        # Iterate and create indexes if they don't exist
        for command, index_name in index_commands:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='index' AND name='{index_name}';")
            if cursor.fetchone() is None:
                cursor.execute(command)
                
        conn.commit()
        conn.close()
        self.logger.info("Primary table 'tsdb' created.")


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
            header_data['Source'] = self.get_source(L0_header_data)
            header_data['L0_filename'] = L0_filename
            header_data['D2_filename'] = f"{base_filename}_2D.fits"
            header_data['L1_filename'] = f"{base_filename}_L1.fits"
            header_data['L2_filename'] = f"{base_filename}_L2.fits"
            header_data['L0_header_read_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header_data['D2_header_read_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header_data['L1_header_read_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header_data['L2_header_read_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
            # Insert into database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA cache_size = -2000000;")
            columns = ', '.join([f'"{key}"' for key in header_data.keys()])
            placeholders = ', '.join(['?'] * len(header_data))
            insert_query = f'INSERT OR REPLACE INTO tsdb ({columns}) VALUES ({placeholders})'
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
#            'is_any_file_updated_func': self.is_any_file_updated,  # this isn't needed because it's checked eaerlier
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
            insert_query = f'INSERT OR REPLACE INTO tsdb ({columns}) VALUES ({placeholders})'
            data_tuples = [tuple(data.values()) for data in batch_data]
    
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA cache_size = -2000000;")
            cursor.executemany(insert_query, data_tuples)
            conn.commit()
            conn.close()


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
        Additionally, if DRPTAG is valid, populate DRPTAG2D, DRPTAGL1, and DRPTAGL2 with its value.
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
    
                # Populate header_data from header
                header_data = {key: header.get(key, None) for key in keyword_types.keys()}
    
                # If DRPTAG is valid, propagate its value to appropriate data level
                drptag_value = header.get('DRPTAG', None)
                if drptag_value is not None:
                    for target_key in ['DRPTAG2D', 'DRPTAGL1', 'DRPTAGL2']:
                        if target_key in header_data:
                            header_data[target_key] = drptag_value
    
                # If DRPHASH is valid, propagate its value to appropriate data level
                drphash_value = header.get('DRPHASH', None)
                if drphash_value is not None:
                    for target_key in ['DRPHSH2D', 'DRPHSHL1', 'DRPHSHL2']:
                        if target_key in header_data:
                            header_data[target_key] = drphash_value
    
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

        # Spectrometer and other temperatures from kpfmet
        kwrds = ['kpfmet.BENCH_BOTTOM_BETWEEN_CAMERAS', 'kpfmet.BENCH_BOTTOM_COLLIMATOR', 'kpfmet.BENCH_BOTTOM_DCUT', 'kpfmet.BENCH_BOTTOM_ECHELLE', 'kpfmet.BENCH_TOP_BETWEEN_CAMERAS', 'kpfmet.BENCH_TOP_COLL', 'kpfmet.BENCH_TOP_DCUT', 'kpfmet.BENCH_TOP_ECHELLE_CAM', 'kpfmet.CALEM_SCMBLR_CHMBR_END', 'kpfmet.CALEM_SCMBLR_FIBER_END', 'kpfmet.CAL_BENCH', 'kpfmet.CAL_BENCH_BB_SRC', 'kpfmet.CAL_BENCH_BOT', 'kpfmet.CAL_BENCH_ENCL_AIR', 'kpfmet.CAL_BENCH_OCT_MOT', 'kpfmet.CAL_BENCH_TRANS_STG_MOT', 'kpfmet.CAL_RACK_TOP|float', 'kpfmet.CHAMBER_EXT_BOTTOM', 'kpfmet.CHAMBER_EXT_TOP', 'kpfmet.CRYOSTAT_G1', 'kpfmet.CRYOSTAT_G2', 'kpfmet.CRYOSTAT_G3', 'kpfmet.CRYOSTAT_R1', 'kpfmet.CRYOSTAT_R2', 'kpfmet.CRYOSTAT_R3', 'kpfmet.ECHELLE_BOTTOM', 'kpfmet.ECHELLE_TOP', 'kpfmet.FF_SRC', 'kpfmet.GREEN_CAMERA_BOTTOM', 'kpfmet.GREEN_CAMERA_COLLIMATOR', 'kpfmet.GREEN_CAMERA_ECHELLE', 'kpfmet.GREEN_CAMERA_TOP', 'kpfmet.GREEN_GRISM_TOP', 'kpfmet.GREEN_LN2_FLANGE', 'kpfmet.PRIMARY_COLLIMATOR_TOP', 'kpfmet.RED_CAMERA_BOTTOM', 'kpfmet.RED_CAMERA_COLLIMATOR', 'kpfmet.RED_CAMERA_ECHELLE', 'kpfmet.RED_CAMERA_TOP', 'kpfmet.RED_GRISM_TOP', 'kpfmet.RED_LN2_FLANGE', 'kpfmet.REFORMATTER', 'kpfmet.SCIENCE_CAL_FIBER_STG', 'kpfmet.SCISKY_SCMBLR_CHMBR_EN', 'kpfmet.SCISKY_SCMBLR_FIBER_EN', 'kpfmet.SIMCAL_FIBER_STG', 'kpfmet.SKYCAL_FIBER_STG', 'kpfmet.TEMP', 'kpfmet.TH_DAILY', 'kpfmet.TH_GOLD', 'kpfmet.U_DAILY', 'kpfmet.U_GOLD',]
        for key in kwrds:
            if key in df.columns:
                pass
                # need to check on why these values occur
                #df = df.loc[df[key] > -200]
                #df = df.loc[df[key] <  400]

        return df


    def is_any_file_updated(self, L0_file_path):
        """
        Determines if any file from the L0/2D/L1/L2 set has been updated since the last 
        noted modification in the database.  Returns True if is has been modified.
        """
        L0_filename = L0_file_path.split('/')[-1]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA cache_size = -2000000;")
        query = f'SELECT L0_header_read_time, D2_header_read_time, L1_header_read_time, L2_header_read_time FROM tsdb WHERE L0_filename = "{L0_filename}"'
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
           

    def ingest_dates_to_db(self, start_date_str, end_date_str, batch_size=1000, reverse=False, quiet=False):
        """
        Ingest KPF data for the date range start_date to end_date, inclusive.
        batch_size refers to the number of observations per DB insertion.
        """

        # Convert input dates to strings if necessary
        if isinstance(start_date_str, datetime):
            start_date_str = start_date_str.strftime("%Y%m%d")
        if isinstance(end_date_str, datetime):
            end_date_str = end_date_str.strftime("%Y%m%d")
        
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

        self.print_db_status()


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


    def print_metadata_table(self):
        """
        Read the tsdb_metadata table, group by 'source', and print out rows
        in fixed-width columns in the custom order below, without printing
        the 'source' column.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
    
        # Define your custom order of sources
        custom_order = [
            "Derived Keywords",
            "L0 PRIMARY Header",
            "2D PRIMARY Header",
            "L1 PRIMARY Header",
            "L2 PRIMARY Header",
            "L0 TELEMETRY Extension",
            "L2 RV Header",
            "L2 RV Extension"
        ]
    
        col_width_keyword  = 35
        col_width_datatype = 9
        col_width_units    = 9
        col_width_desc     = 90
    
        for src in custom_order:
            cursor.execute(
                """
                SELECT keyword, datatype, units, description
                FROM tsdb_metadata
                WHERE source = ?
                ORDER BY keyword;
                """,
                (src,)
            )
            rows = cursor.fetchall()
    
            if not rows:
                continue
    
            print(f"{src}:")
            print("-" * 90)
            print(
                f"{'Keyword':<{col_width_keyword}} "
                f"{'Datatype':<{col_width_datatype}} "
                f"{'Units':<{col_width_units}} "
                f"{'Description':<{col_width_desc}}"
            )
            print("-" * 90)
    
            for keyword, datatype, units, description in rows:
                # Convert None to an empty string to avoid formatting errors
                keyword_str   = keyword if keyword else ""
                datatype_str  = datatype if datatype else ""
                units_str     = units if units else ""
                desc_str      = description if description else ""
    
                print(
                    f"{keyword_str:<{col_width_keyword}} "
                    f"{datatype_str:<{col_width_datatype}} "
                    f"{units_str:<{col_width_units}} "
                    f"{desc_str:<{col_width_desc}}"
                )
            print()  
    
        conn.close()


    def metadata_table_to_df(self):
        """
        Return a dataframe of the metadata table with columns:
            keyword | datatype | unit | description | source
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = f"SELECT keyword, datatype, units, description, source FROM tsdb_metadata"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df


    def select_query(self, query):
        """
        Query the database with the query 'query'.  Any query can be used, 
        but no 'commit' statement is used do database changes won't be saved.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        tables = cursor.fetchall()
        conn.close()
        
        return tables
        

    def query_to_pandas(self, query):
        try:
            results = pd.read_sql_query(query, self.conn)
        except:
            self.log.warning(f"Error running database query:\n{query}\n")
            results = pd.DataFrame([])

        return results


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


    def get_first_last_dates(self):
        """
        Returns a tuple of datetime objects containing the first and last dates 
        in the database.  DATE-MID is used for the date.
        """

        conn = sqlite3.connect(self.db_path)
    
        # Query for the minimum and maximum dates in the 'DATE-MID' column
        query = """
            SELECT MIN("DATE-MID") AS min_date, MAX("DATE-MID") AS max_date
            FROM tsdb
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


    def display_dataframe_from_db(self, columns, only_object=None, object_like=None, only_source=None, 
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
            only_source (string or list of strings) - source names to include in query (e.g., 'Star')
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
        query = f"SELECT {', '.join(quoted_columns)} FROM tsdb"

        # Append WHERE clauses
        where_queries = []
        if only_object is not None:
            only_object = [f"OBJECT = '{only_object}'"]
            or_objects = ' OR '.join(only_object)
            where_queries.append(f'({or_objects})')
        # is object_like working?
        if object_like is not None:
            object_like = [f"OBJECT LIKE '%{obj}%'" for obj in object_like]
            or_objects = ' OR '.join(object_like)
            where_queries.append(f'({or_objects})')
        if only_source is not None:
            only_source = [f"SOURCE = '{only_source}'"]
            or_sources = ' OR '.join(only_source)
            where_queries.append(f'({or_sources})')
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
                          only_object=None, only_source=None, object_like=None, 
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
            start_date (datetime object) - only return observations after start_date
            end_date (datetime object) - only return observations after end_date
            only_object (string) - object name to include in query
            only_source (string or list of strings) - source names to include in query (e.g., 'Star')
            object_like (string) - partial object name to search for
            on_sky (True, False, None) - using FIUMODE, select observations that are on-sky (True), off-sky (False), or don't care (None)
            not_junk (True, False, None) using NOTJUNK, select observations that are not Junk (True), Junk (False), or don't care (None)
            verbose (boolean) - if True, prints the SQL query
        """
        
        conn = sqlite3.connect(self.db_path)
    
        # Get all column names if columns are not specified
        if columns is None:
            query_get_columns = "PRAGMA table_info(tsdb)"
            all_columns_info = pd.read_sql_query(query_get_columns, conn)
            columns = all_columns_info['name'].tolist()
    
        # Enclose column names in double quotes
        quoted_columns = [f'"{column}"' for column in columns]
        query = f"SELECT {', '.join(quoted_columns)} FROM tsdb"
    
        # Append WHERE clauses
        where_queries = []
        if only_object is not None:
            only_object = convert_to_list_if_array(only_object)
            if isinstance(only_object, str):
                only_object = [only_object]
            object_queries = [f"OBJECT = '{obj}'" for obj in only_object]
            or_objects = ' OR '.join(object_queries)
            where_queries.append(f'({or_objects})')
        # does object_like work?
        if object_like is not None: 
            object_like = [f"OBJECT LIKE '%{object_like}%'"]
            or_objects = ' OR '.join(object_like)
            where_queries.append(f'({or_objects})')
        if only_source is not None:
            only_source = convert_to_list_if_array(only_source)
            if isinstance(only_source, str):
                only_source = [only_source]
            source_queries = [f"SOURCE = '{src}'" for src in only_source]
            or_sources = ' OR '.join(source_queries)
            where_queries.append(f'({or_sources})')
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
            start_date_txt = start_date.strftime('%Y-%m-%dT%H:%M:%S')
            where_queries.append(f' ("DATE-MID" > "{start_date_txt}")')
        if end_date is not None:
            end_date_txt = end_date.strftime('%Y-%m-%dT%H:%M:%S')
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
        

def process_file(file_path, now_str,
                 L0_header_keyword_types, L0_telemetry_types, D2_header_keyword_types,
                 L1_header_keyword_types, L2_header_keyword_types, L2_CCF_header_keyword_types, L2_RV_header_keyword_types,
                 extract_kwd_func, extract_telemetry_func, extract_rvs_func, 
                 #is_any_file_updated_func, # this check was moved
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
    header_data['Source'] = get_source_func(L0_header_data)
    header_data['L0_filename'] = os.path.basename(L0_file_path)
    header_data['D2_filename'] = os.path.basename(D2_file_path)
    header_data['L1_filename'] = os.path.basename(L1_file_path)
    header_data['L2_filename'] = os.path.basename(L2_file_path)
    header_data['L0_header_read_time'] = now_str
    header_data['D2_header_read_time'] = now_str
    header_data['L1_header_read_time'] = now_str
    header_data['L2_header_read_time'] = now_str

    return header_data


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
