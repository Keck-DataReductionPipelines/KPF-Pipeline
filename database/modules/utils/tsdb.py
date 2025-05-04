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

    Related Commandline Scripts:
        'ingest_dates_kpf_tsdb.py' - ingest from a range of dates
        'ingest_watch_kpf_tsdb.py' - ingest by watching a set of directories

    To-do:
        * Add temperature derivatives as columns; they will need to be computed.
        * Add database for masters (separate from ObsIDs?)

    """

    def __init__(self, backend='sqlite', db_path='kpf_ts.db', base_dir='/data/L0', logger=None, drop=False, verbose=False):
        """
        Todo: add docstring, including explanation of backend = 'sqlite' or 'psql'
        """
        
        self.backend = backend # sqlite or psql
        self.verbose = verbose
        self.logger = logger if logger is not None else DummyLogger()
        self.logger.info('Starting KPF_TSDB')

        if self.is_notebook():
            self.tqdm = tqdm_notebook
            self.logger.info('Jupyter Notebook environment detected.')
        else:
            self.tqdm = tqdm
        
        self.tables = [
            'tsdb_base', 'tsdb_l0', 'tsdb_2d', 'tsdb_l1',
            'tsdb_l2', 'tsdb_l0t', 'tsdb_l2rv', 'tsdb_l2rvdata', 'tsdb_l2ccf',
            'tsdb_metadata'  # include metadata table explicitly
        ]
    
# ADJUST this for sqlite only
        self.db_path = db_path
        self.logger.info('Path of database file: ' + os.path.abspath(self.db_path))
        self.base_dir = base_dir
        self.logger.info('Base data directory: ' + self.base_dir)
        self.cursor = None

        # Initialize metadata entries first
        self.init_metadata_entries()

        # These may not be needed after multi-table update
        self.base_keyword_types          = self.get_keyword_types(level='base')
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

        if self.backend == 'sqlite':
            pass

        elif backend == 'psql':
            self.dbport   = os.getenv('DBPORT_TSDB')
            self.dbname   = os.getenv('DBNAME_TSDB')
            self.dbuser   = os.getenv('DBUSER_TSDB')
            self.dbpass   = os.getenv('DBPASS_TSDB')
            self.dbserver = os.getenv('DBSERVER_TSDB')
                
        if drop:
            self.drop_tables()
            self.logger.info('Dropping KPF database ' + str(self.db_path))

        # Always (re)create metadata table first
        self.create_metadata_table()
        
        # Then create the data tables using metadata
        primary_table = 'tsdb_base'
        if not self.check_if_table_exists(tablename=primary_table):
            self.create_database()
        else:
            self.logger.info("Primary tables already exist.")


    def _open_connection(self):
        if self.backend == 'sqlite':
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        elif self.backend == 'psql':
            try:
                db_fail = True
                n_attempts = 3
                for i in range(n_attempts):
                    try: 
                        self.conn = psycopg2.connect(
                            host=self.dbserver,
                            database=self.dbname,
                            port=self.dbport,
                            user=self.dbuser,
                            password=self.dbpass
                        )
                        db_fail = False
                        break
                    except:
                        self.logger.info("Could not connect to database, retrying...")
                        db_fail = True
                        time.sleep(10)
                self.cursor = self.conn.cursor()
            except Exception as e:
                self.logger.error(f"Failed to connect to PostgreSQL: {e}")
                raise
                
    
    def _close_connection(self):
        if self.conn:
            if self.backend == 'sqlite':
                self.conn.commit()
            elif self.backend == 'psql':
                self.conn.commit()
            self.cursor.close()
            self.conn.close()
            self.conn = None
            self.cursor = None

    
    def _execute_sql_command(self, command, params=None, fetch=False):
        if self.cursor is None:
            raise RuntimeError("Database connection is not open.")
    
        try:
            if params:
                if self.backend == 'sqlite':
                    self.cursor.execute(command, params)
                elif self.backend == 'psql':
                    self.cursor.execute(command.replace('?', '%s'), params)
            else:
                self.cursor.execute(command)
    
            if fetch:
                return self.cursor.fetchall()
        except Exception as e:
            self.logger.error(f"SQL Execution error: {e}\nCommand: {command}\nParams: {params}")
            raise


    def print_summary_all_tables(self):
        """
        Prints a summary of all tables in the database (not just the intended tables), 
        including the number of rows and columns.
        """
        self._open_connection()
        try:
            if self.backend == 'sqlite':
                self._execute_sql_command("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
                tables = [row[0] for row in self.cursor.fetchall()]
    
                print(f"{'Table Name':<20} {'Columns':>7} {'Rows':>10}")
                print("-" * 40)
    
                for tbl in tables:
                    self._execute_sql_command(f"PRAGMA table_info({tbl});")
                    columns = len(self.cursor.fetchall())
    
                    self._execute_sql_command(f"SELECT COUNT(*) FROM {tbl};")
                    rows = self.cursor.fetchone()[0]
    
                    print(f"{tbl:<20} {columns:>7} {rows:>10}")
    
            elif self.backend == 'psql':
                self._execute_sql_command("""
                    SELECT tablename FROM pg_catalog.pg_tables
                    WHERE schemaname='public' ORDER BY tablename;
                """)
                tables = [row[0] for row in self.cursor.fetchall()]
    
                print(f"{'Table Name':<20} {'Columns':>7} {'Rows':>10}")
                print("-" * 40)
    
                for tbl in tables:
                    self._execute_sql_command(f"""
                        SELECT COUNT(*) FROM information_schema.columns
                        WHERE table_name = '{tbl}';
                    """)
                    columns = self.cursor.fetchone()[0]
    
                    self._execute_sql_command(f"SELECT COUNT(*) FROM {tbl};")
                    rows = self.cursor.fetchone()[0]
    
                    print(f"{tbl:<20} {columns:>7} {rows:>10}")
        finally:
            self._close_connection()


    def drop_tables(self):
        """
        Start over on the database by dropping all data and metadata tables.
        """
        self._open_connection()
        try:
            for tbl in self.tables:
                self._execute_sql_command(f"DROP TABLE IF EXISTS {tbl}")
        finally:
            self._close_connection()


    def unlock_db(self):
        """
        Remove the -wal and -shm lock files, 
        e.g. /data/time_series/kpf_ts.db-wal and /data/time_series/kpf_ts.db-shm
        
        Use this method sparingly.
        """
        if self.backend == 'sqlite':
            wal_file = f"{self.db_path}-wal"
            shm_file = f"{self.db_path}-shm"
        
            if os.path.exists(wal_file):
                os.remove(wal_file)
                self.logger.info(f"File removed: {wal_file}")
            if os.path.exists(shm_file):
                os.remove(shm_file)
                self.logger.info(f"File removed: {shm_file}")


    def init_metadata_entries(self):
        """
        Load and combine all keyword metadata entries from CSV files into a single attribute (self.metadata_entries).
        """
        def load_keyword_csv(csv_path, source_label):
            df = pd.read_csv(csv_path, delimiter='|', dtype=str)
            df['source'] = source_label
            df.rename(columns={'unit': 'units'}, inplace=True)  # fix potential unit naming
            return df[['keyword', 'datatype', 'units', 'description', 'source']]
    
        df_base  = load_keyword_csv('/code/KPF-Pipeline/static/tsdb_keywords/base_keywords.csv',         'Base Keywords')
        df_l0    = load_keyword_csv('/code/KPF-Pipeline/static/tsdb_keywords/l0_primary_keywords.csv',   'L0 PRIMARY Header')
        df_2d    = load_keyword_csv('/code/KPF-Pipeline/static/tsdb_keywords/d2_primary_keywords.csv',   '2D PRIMARY Header')
        df_l1    = load_keyword_csv('/code/KPF-Pipeline/static/tsdb_keywords/l1_primary_keywords.csv',   'L1 PRIMARY Header')
        df_l2    = load_keyword_csv('/code/KPF-Pipeline/static/tsdb_keywords/l2_primary_keywords.csv',   'L2 PRIMARY Header')
        df_l0t   = load_keyword_csv('/code/KPF-Pipeline/static/tsdb_keywords/l0_telemetry_keywords.csv', 'L0 TELEMETRY Extension')
        df_l2rv  = load_keyword_csv('/code/KPF-Pipeline/static/tsdb_keywords/l2_rv_keywords.csv',        'L2 RV Header')
        df_l2ccf = load_keyword_csv('/code/KPF-Pipeline/static/tsdb_keywords/l2_green_ccf_keywords.csv', 'L2 CCF Header')
    
        # RV prefix keywords
        prefixes = ['RV1','RV2','RV3','RVS','ERVS','RVC','ERVC','RVY','ERVY','CCFBJD','BCRV','CCFW']
        units    = ['km/s','km/s','km/s','km/s','km/s','km/s','km/s','km/s','km/s','days','km/s','None']
        descs    = ['RV for SCI1 order ', 'RV for SCI2 order ', 'RV for SCI3 order ','RV for SCI order ', 'Error in RV for SCI order ', 'RV for CAL order ','Error in RV for CAL order ', 'RV for SKY order ',  'Error in RV for SKY order ','BJD for order ','Barycentric RV for order ','CCF weight for order ']
        nums = [f"{i:02d}" for i in range(67)]
    
        rv_entries = []
        for prefix, unit, desc in zip(prefixes, units, descs):
            for num in nums:
                kw = f"{prefix}{num}"
                rv_entries.append({
                    'keyword': kw,
                    'datatype': 'REAL',
                    'units': unit,
                    'description': f"{desc}{num}",
                    'source': 'L2 RV Extension'
                })
    
        df_rv = pd.DataFrame(rv_entries)
#        for index, row in df_rv.iterrows():
#            print(f'{row["keyword"]}|float|{row["description"]}|{row["units"]}')

        # Combine all into one DataFrame
        df_all = pd.concat([df_base, df_l0, df_2d, df_l1, df_l2, df_l0t, df_l2rv, df_l2ccf, df_rv], ignore_index=True)
    
        # Remove duplicates
        df_all.drop_duplicates(subset='keyword', inplace=True)
    
        # Store as a list of dictionaries
        self.metadata_entries = df_all.to_dict(orient='records')


    def print_db_status(self):
        """
        Prints a formatted summary table of the database status for each table.
        """
        tables = self.tables.copy()
        tables.remove('tsdb_metadata')
    
        self._open_connection()
    
        summary_data = []
    
        for table in tables:
            self._execute_sql_command(f'SELECT COUNT(*) FROM {table}')
            nrows = self.cursor.fetchone()[0]
    
            self._execute_sql_command(f'PRAGMA table_info({table})')
            ncolumns = len(self.cursor.fetchall())
    
            summary_data.append((table, ncolumns, nrows))
    
        self._execute_sql_command('SELECT MAX(L0_header_read_time), MAX(L1_header_read_time) FROM tsdb_base')
        most_recent_read_time = max(filter(None, self.cursor.fetchone()))
    
        self._execute_sql_command('SELECT MIN(datecode), MAX(datecode), COUNT(DISTINCT datecode) FROM tsdb_base')
        earliest_datecode, latest_datecode, unique_datecodes_count = self.cursor.fetchone()
    
        self._close_connection()
    
        # Print the summary table
        self.logger.info("Database Table Summary:")
        self.logger.info(f"{'Table':<15} {'Columns':>7} {'Rows':>10}")
        self.logger.info("-" * 35)
        for table, cols, rows in summary_data:
            self.logger.info(f"{table:<15} {cols:>7} {rows:>10}")
    
        # Print the additional stats
        self.logger.info(f"Dates: {unique_datecodes_count} days from {earliest_datecode} to {latest_datecode}")
        self.logger.info(f"Last update: {most_recent_read_time}")
           

# UPDATE THIS BY READING METADATA_TABLE
    def create_metadata_table(self):
        """
        Create the tsdb_metadata table with an added table_name column for 
        category mapping. Compatible with both SQLite and PostgreSQL.
        """
        self._open_connection()
        try:
            self._execute_sql_command("DROP TABLE IF EXISTS tsdb_metadata")
            create_sql = """
                CREATE TABLE tsdb_metadata (
                    keyword     TEXT PRIMARY KEY,
                    source      TEXT,
                    datatype    TEXT,
                    units       TEXT,
                    description TEXT,
                    table_name  TEXT
                );
            """
            self._execute_sql_command(create_sql)
    
            # Mapping from source to new table name
            source_to_table = {
                'Base Keywords':          'tsdb_base',
                'L0 PRIMARY Header':      'tsdb_l0',
                '2D PRIMARY Header':      'tsdb_2d',
                'L1 PRIMARY Header':      'tsdb_l1',
                'L2 PRIMARY Header':      'tsdb_l2',
                'L0 TELEMETRY Extension': 'tsdb_l0t',
                'L2 RV Header':           'tsdb_l2rv',
                'L2 RV Extension':        'tsdb_l2rvdata',
                'L2 CCF Header':          'tsdb_l2ccf'
            }
    
            # Backend-specific insert command
            if self.backend == 'sqlite':
                insert_sql = """
                    INSERT OR REPLACE INTO tsdb_metadata 
                    (keyword, source, datatype, units, description, table_name) 
                    VALUES (?, ?, ?, ?, ?, ?);
                """
            elif self.backend == 'psql':
                insert_sql = """
                    INSERT INTO tsdb_metadata 
                    (keyword, source, datatype, units, description, table_name)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (keyword) DO UPDATE SET
                        source=EXCLUDED.source,
                        datatype=EXCLUDED.datatype,
                        units=EXCLUDED.units,
                        description=EXCLUDED.description,
                        table_name=EXCLUDED.table_name;
                """
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
    
            # Insert metadata entries
            for entry in self.metadata_entries:
                keyword     = entry.get('keyword')
                source      = entry.get('source')
                datatype    = entry.get('datatype', 'TEXT')
                units       = entry.get('units', None)
                description = entry.get('description', None)
                table_name  = source_to_table.get(source, None)
                self._execute_sql_command(insert_sql, params=(keyword, source, datatype, units, description, table_name))
    
        finally:
            self._close_connection()


    def check_if_table_exists(self, tablename=None):
        """
        Return True if the named table exists.
        """
        if tablename is None:
            self.logger.info('check_if_table_exists: tablename not specified.')
            return False
    
        self._open_connection()
        try:
            self._execute_sql_command("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?;
            """, params=(tablename,))
            tables = self.cursor.fetchone()
            result = tables is not None and tablename in tables
        except Exception as e:
            self.error.info(f'check_if_table_exists: problem with query - {e}')
            result = False
        finally:
            self._close_connection()
    
        return result


    def create_database(self):
        """
        Create TSDB tables split by category with ObsID as primary key.
        """
        self._open_connection()
    
        tables = self.tables.copy()
        tables.remove('tsdb_metadata')
    
        # Drop existing tables if they exist
        for tbl in tables:
            self._execute_sql_command(f"DROP TABLE IF EXISTS {tbl}")
    
        # Fetch keyword, datatype, and table_name from metadata
        sql_metadata = "SELECT keyword, datatype, table_name FROM tsdb_metadata;"
        metadata_rows = self._execute_sql_command(sql_metadata, fetch=True)
    
        columns_by_table = {tbl: [] for tbl in tables}
        for keyword, dtype, table_name in metadata_rows:
            if table_name not in columns_by_table or table_name is None:
                continue
            if keyword.strip().lower() == 'obsid':
                continue  # ObsID is primary key
            sql_type = self.map_data_type_to_sql(dtype)
            columns_by_table[table_name].append((keyword, sql_type))
    
        # Create each table with ObsID as primary key and indexed
        for tbl, cols in columns_by_table.items():
            col_defs = ['"ObsID" TEXT PRIMARY KEY']
            col_defs += [f'"{kw}" {sql_type}' for kw, sql_type in cols]
            col_defs_sql = ", ".join(col_defs)
    
            create_table_sql = f"CREATE TABLE IF NOT EXISTS {tbl} ({col_defs_sql});"
            create_index_sql = f"CREATE INDEX IF NOT EXISTS idx_{tbl}_ObsID ON {tbl}(ObsID);"
    
            self._execute_sql_command(create_table_sql)
            self._execute_sql_command(create_index_sql)
    
        self._close_connection()
    
        self.logger.info("Tables successfully created.")
        

    def ingest_one_observation(self, dir_path, L0_filename):
        """
        Ingest a single observation into the multi-table database structure.
        """
        base_filename = L0_filename.split('.fits')[0]
        L0_file_path = f"{dir_path}/{base_filename}.fits"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
        if not self.is_any_file_updated(L0_file_path):
            return
    
        D2_file_path = f"{dir_path.replace('L0', '2D')}/{base_filename}_2D.fits"
        L1_file_path = f"{dir_path.replace('L0', 'L1')}/{base_filename}_L1.fits"
        L2_file_path = f"{dir_path.replace('L0', 'L2')}/{base_filename}_L2.fits"
    
        # Extract header data
        header_data = {
            **self.extract_kwd(L0_file_path, self.L0_header_keyword_types, extension='PRIMARY'),
            **self.extract_telemetry(L0_file_path, self.L0_telemetry_types),
            **self.extract_kwd(D2_file_path, self.D2_header_keyword_types, extension='PRIMARY'),
            **self.extract_kwd(L1_file_path, self.L1_header_keyword_types, extension='PRIMARY'),
            **self.extract_kwd(L2_file_path, self.L2_header_keyword_types, extension='PRIMARY'),
            **self.extract_kwd(L2_file_path, self.L2_CCF_header_keyword_types, extension='GREEN_CCF'),
            **self.extract_kwd(L2_file_path, self.L2_RV_header_keyword_types, extension='RV'),
            **self.extract_rvs(L2_file_path)
        }
    
        # Add base/common metadata
        header_data['ObsID'] = base_filename
        header_data['datecode'] = get_datecode(base_filename)
        header_data['Source'] = self.get_source(header_data)
        header_data['L0_filename'] = L0_filename
        header_data['D2_filename'] = f"{base_filename}_2D.fits"
        header_data['L1_filename'] = f"{base_filename}_L1.fits"
        header_data['L2_filename'] = f"{base_filename}_L2.fits"
        header_data['L0_header_read_time'] = now_str
        header_data['D2_header_read_time'] = now_str
        header_data['L1_header_read_time'] = now_str
        header_data['L2_header_read_time'] = now_str
    
        # Connect to DB
        self._open_connection()
    
        try:
            # Get keyword-to-table mapping from metadata
            metadata_rows = self._execute_sql_command("SELECT keyword, table_name FROM tsdb_metadata;", fetch=True)
            kw_to_table = dict(metadata_rows)
    
            # Prepare data for each table
            table_data = {}
            for kw, value in header_data.items():
                table_name = kw_to_table.get(kw)
                if table_name:
                    table_data.setdefault(table_name, {})[kw] = value
            # ObsID must be in all tables
            for tbl in table_data:
                table_data[tbl]['ObsID'] = base_filename
    
            # Insert data into each table separately
            for tbl, data in table_data.items():
                columns = ', '.join([f'"{col}"' for col in data])
                placeholders = ', '.join(['?'] * len(data))
                insert_query = f'INSERT OR REPLACE INTO {tbl} ({columns}) VALUES ({placeholders})'
                self._execute_sql_command(insert_query, params=tuple(data.values()))
        finally:
            self._close_connection()
    
        self.logger.info(f"Ingested observation: {base_filename}")
    

    def ingest_batch_observation(self, batch, force_ingest=False):
        """
        Ingest a batch of observations into the multi-table database in parallel,
        checking each file for updates beforehand.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
        # === 1) Check for updated files ===
        if force_ingest:
            updated_batch = batch
        else:
            updated_batch = [file_path for file_path in batch if self.is_any_file_updated(file_path)]
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
            'get_source_func': self.get_source,
            'get_datecode_func': get_datecode
        }
    
        partial_process_file = partial(process_file, **args)
    
        # === 3) Run extraction in parallel ===
        max_workers = min(len(updated_batch), 20, os.cpu_count())
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(partial_process_file, updated_batch))
    
        # Filter None results just in case
        valid_results = [res for res in results if res]
    
        if not valid_results:
            return
    
        self._open_connection()
        try:
            # === 4) Group extracted data by table ===
            metadata_rows = self._execute_sql_command("SELECT keyword, table_name FROM tsdb_metadata;", fetch=True)
            kw_to_table = dict(metadata_rows)
    
            # Organize data per table
            table_data = {}
            for obs_data in valid_results:
                obsid = obs_data['ObsID']
                data_by_table = {}
                for kw, value in obs_data.items():
                    table = kw_to_table.get(kw)
                    if table:
                        data_by_table.setdefault(table, {})[kw] = value
                # Ensure ObsID is in every table
                for table, data in data_by_table.items():
                    data['ObsID'] = obsid
                    table_data.setdefault(table, []).append(data)
    
            # === 5) Perform bulk insert for each table ===
            for table, rows in table_data.items():
                if not rows:
                    continue
                columns = list(rows[0].keys())
                placeholders = ', '.join(['?'] * len(columns))
                column_str = ', '.join([f'"{col}"' for col in columns])
                insert_query = f'INSERT OR REPLACE INTO {table} ({column_str}) VALUES ({placeholders})'
                data_tuples = [tuple(row[col] for col in columns) for row in rows]
    
                for data_tuple in data_tuples:
                    self._execute_sql_command(insert_query, params=data_tuple)
        finally:
            self._close_connection()
    

    def map_data_type_to_sql(self, dtype):
        """
        Function to map the data types specified in get_keyword_types to sqlite
        data types.
        """
        if self.backend == 'sqlite':
            return {
                'int': 'INTEGER',
                'float': 'REAL',
                'bool': 'BOOLEAN',
                'datetime': 'TEXT',  # SQLite does not have a native datetime type
                'string': 'TEXT'
            }.get(dtype, 'TEXT')
        elif self.backend == 'psql':
            return {
                'int': 'INTEGER',
                'float': 'DOUBLE PRECISION',
                'bool': 'BOOLEAN',
                'datetime': 'TIMESTAMP',
                'string': 'TEXT'
            }.get(dtype, 'TEXT')
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")


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

        # Base keywords    
        elif level == 'base':
            keywords_csv='/code/KPF-Pipeline/static/tsdb_keywords/base_keywords.csv'
            df_keywords = pd.read_csv(keywords_csv, delimiter='|', dtype=str)
            keyword_types = dict(zip(df_keywords['keyword'], df_keywords['datatype']))

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
            self.logger.error(f"Bad file: {file_path}. Error: {e}")
    
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
        noted modification in the database. Returns True if it has been modified.
        """
        L0_filename = os.path.basename(L0_file_path)
    
        self._open_connection()
        try:
            query = 'SELECT L0_header_read_time, D2_header_read_time, L1_header_read_time, L2_header_read_time FROM tsdb_base WHERE L0_filename = ?'
            self._execute_sql_command(query, params=(L0_filename,))
            result = self.cursor.fetchone()
        finally:
            self._close_connection()
    
        if not result:
            return True  # No record in the database
    
        file_paths = {
            'L0': L0_file_path,
            'D2': L0_file_path.replace('L0', '2D').replace('.fits', '_2D.fits'),
            'L1': L0_file_path.replace('L0', 'L1').replace('.fits', '_L1.fits'),
            'L2': L0_file_path.replace('L0', 'L2').replace('.fits', '_L2.fits'),
        }
    
        for idx, (key, path) in enumerate(file_paths.items()):
            try:
                mod_time = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
            except FileNotFoundError:
                mod_time = '1000-01-01 00:00:00'
    
            if mod_time > (result[idx] or '1000-01-01 00:00:00'):
                return True  # File was modified more recently than DB timestamp
    
        return False  # No updates found
           

    def ingest_dates_to_db(self, start_date_str, end_date_str, batch_size=10000, reverse=False, force_ingest=False, quiet=False):
        """
        Ingest KPF data for the date range start_date to end_date, inclusive.
        batch_size refers to the number of observations per DB insertion.
        If force_ingest=False, files are not reingested unless they have more recent modification dates than in DB.
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
                            self.ingest_batch_observation(batch, force_ingest=force_ingest)
                            batch = []
                if batch:
                    self.ingest_batch_observation(batch, force_ingest=force_ingest)

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
        # Define your custom order of sources
        custom_order = [
            "Base Keywords",
            "L0 PRIMARY Header",
            "2D PRIMARY Header",
            "L1 PRIMARY Header",
            "L2 PRIMARY Header",
            "L0 TELEMETRY Extension",
            "L2 RV Header",
            "L2 RV Extension",
            "L2 CCF Header"
        ]
    
        col_width_keyword = 35
        col_width_datatype = 9
        col_width_units = 9
        col_width_desc = 90
    
        self._open_connection()
        try:
            for src in custom_order:
                query = """
                    SELECT keyword, datatype, units, description
                    FROM tsdb_metadata
                    WHERE source = ?
                    ORDER BY keyword;
                """
                self._execute_sql_command(query, params=(src,))
                rows = self.cursor.fetchall()
    
                if not rows:
                    continue
    
                print(f"{src}:")
                print("-" * 150)
                print(
                    f"{'Keyword':<{col_width_keyword}} "
                    f"{'Datatype':<{col_width_datatype}} "
                    f"{'Units':<{col_width_units}} "
                    f"{'Description':<{col_width_desc}}"
                )
                print("-" * 150)
    
                for keyword, datatype, units, description in rows:
                    keyword_str = keyword or ""
                    datatype_str = datatype or ""
                    units_str = units or ""
                    desc_str = description or ""
    
                    print(
                        f"{keyword_str:<{col_width_keyword}} "
                        f"{datatype_str:<{col_width_datatype}} "
                        f"{units_str:<{col_width_units}} "
                        f"{desc_str:<{col_width_desc}}"
                    )
                print()
        finally:
            self._close_connection()


    def metadata_table_to_df(self):
        """
        Return a dataframe of the metadata table with columns:
            keyword | datatype | units | description | source
        """
        self._open_connection()
        try:
            query = "SELECT keyword, datatype, units, description, source FROM tsdb_metadata"
            self._execute_sql_command(query)
            rows = self.cursor.fetchall()
    
            df = pd.DataFrame(rows, columns=['keyword', 'datatype', 'units', 'description', 'source'])
        finally:
            self._close_connection()
    
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


    def get_first_last_dates(self):
        """
        Returns a tuple of datetime objects containing the first and last dates 
        in the database. DATE-MID is used for the date.
        """
        self._open_connection()
        try:
            query = """
                SELECT MIN("DATE-MID") AS min_date, MAX("DATE-MID") AS max_date
                FROM tsdb_l0
            """
            self._execute_sql_command(query)
            min_date_str, max_date_str = self.cursor.fetchone()
    
            # Convert strings to datetime objects, handling None values gracefully
            date_format = '%Y-%m-%dT%H:%M:%S.%f'
            first_date = datetime.strptime(min_date_str, date_format) if min_date_str else None
            last_date = datetime.strptime(max_date_str, date_format) if max_date_str else None
        finally:
            self._close_connection()
    
        return first_date, last_date


    def display_dataframe_from_db(self, columns, only_object=None, object_like=None, only_source=None, 
                                  on_sky=None, start_date=None, end_date=None):
        """
        Prints a pandas dataframe of attributes (specified by column names) for all 
        observations in the DB. The query can be restricted to observations matching a 
        particular object name(s), source(s), on-sky status, and a date range.
    
        Args:
            columns (string or list of strings, or '*' for all) - database columns to query
            only_object (string or list of strings) - object names to include in query
            object_like (string or list of strings) - partial object names to search for
            only_source (string or list of strings) - source names to include in query
            on_sky (True, False, None) - select on-sky/off-sky observations
            start_date (datetime) - only return observations after this date
            end_date (datetime) - only return observations before this date
    
        Returns:
            None. Prints the resulting dataframe.
        """
        df = self.dataframe_from_db(
            columns=columns,
            only_object=only_object,
            object_like=object_like,
            only_source=only_source,
            on_sky=on_sky,
            start_date=start_date,
            end_date=end_date
        )
        print(df)


    def dataframe_from_db(self, columns=None, 
                          start_date=None, end_date=None, 
                          only_object=None, only_source=None, object_like=None, 
                          on_sky=None, not_junk=None, 
                          verbose=False):
        """
        Returns a pandas dataframe of attributes (specified by column names) from
        multi-table database. Queries can be restricted by object, source, date, etc.
        """
        self._open_connection()
        try:
            if columns is None or columns == '*':
                sql = "SELECT keyword, table_name FROM tsdb_metadata;"
                self._execute_sql_command(sql)
                metadata_df = pd.DataFrame(self.cursor.fetchall(), columns=['keyword', 'table_name'])
                columns = metadata_df['keyword'].tolist()
            else:
                columns = [columns] if isinstance(columns, str) else list(columns)
                clean_columns = [str(col).strip() for col in columns if isinstance(col, str) or col is not None]
                placeholders = ','.join('?' for _ in clean_columns)
                sql = f"SELECT keyword, table_name FROM tsdb_metadata WHERE keyword IN ({placeholders});"
                self._execute_sql_command(sql, clean_columns)
                metadata_df = pd.DataFrame(self.cursor.fetchall(), columns=['keyword', 'table_name'])
                columns = clean_columns
    
            kw_table_map = dict(zip(metadata_df['keyword'], metadata_df['table_name']))
            tables_needed = set(metadata_df['table_name'].tolist())
    
            # Ensure tables for filters are included
            filter_columns = ['OBJECT', 'Source', 'NOTJUNK', 'FIUMODE', 'datecode']
            placeholders = ','.join('?' for _ in filter_columns)
            filter_sql = f"SELECT keyword, table_name FROM tsdb_metadata WHERE keyword IN ({placeholders});"
            self._execute_sql_command(filter_sql, filter_columns)
            filter_metadata = pd.DataFrame(self.cursor.fetchall(), columns=['keyword', 'table_name'])
            filter_kw_table_map = dict(zip(filter_metadata['keyword'], filter_metadata['table_name']))
            tables_needed.update(filter_metadata['table_name'])
    
            dataframes = {}
            for table in tables_needed:
                table_cols = metadata_df[metadata_df['table_name'] == table]['keyword'].tolist()
                table_cols += [col for col in filter_columns if filter_kw_table_map.get(col) == table and col not in table_cols]
                if 'ObsID' not in table_cols:
                    table_cols.append('ObsID')
    
                quoted_cols = [f'"{col}"' for col in table_cols]
                query = f"SELECT {', '.join(quoted_cols)} FROM {table}"
    
                conditions = []
                if only_object and filter_kw_table_map.get('OBJECT') == table:
                    only_object = [only_object] if isinstance(only_object, str) else only_object
                    conditions.append('(' + ' OR '.join([f'OBJECT = "{obj}"' for obj in only_object]) + ')')
                if object_like and filter_kw_table_map.get('OBJECT') == table:
                    conditions.append(f"OBJECT LIKE '%{object_like}%'")
                if only_source and filter_kw_table_map.get('Source') == table:
                    only_source = [only_source] if isinstance(only_source, str) else only_source
                    conditions.append('(' + ' OR '.join([f'Source = "{src}"' for src in only_source]) + ')')
                if not_junk is not None and filter_kw_table_map.get('NOTJUNK') == table:
                    conditions.append(f'NOTJUNK = {1 if not_junk else 0}')
                if on_sky is not None and filter_kw_table_map.get('FIUMODE') == table:
                    mode = 'Observing' if on_sky else 'Calibration'
                    conditions.append(f'FIUMODE = "{mode}"')
                if filter_kw_table_map.get('datecode') == table:
                    if start_date:
                        conditions.append(f'datecode >= "{start_date.strftime("%Y%m%d")}"')
                    if end_date:
                        conditions.append(f'datecode <= "{end_date.strftime("%Y%m%d")}"')
    
                if conditions:
                    query += " WHERE " + ' AND '.join(conditions)
    
                if verbose:
                    self.logger.info(f"Querying {table}: {query}")
    
                self._execute_sql_command(query)
                fetched_data = self.cursor.fetchall()
                col_names = [desc[0] for desc in self.cursor.description]
                dataframes[table] = pd.DataFrame(fetched_data, columns=col_names)
    
        finally:
            self._close_connection()
    
        base_table = 'tsdb_base'
        if base_table in dataframes:
            df_merged = dataframes.pop(base_table)
        else:
            df_merged = None
    
        for table, df in dataframes.items():
            if df_merged is None:
                df_merged = df
            else:
                df_merged = df_merged.merge(df, on='ObsID', how='left')
    
        if columns not in [None, '*']:
            if 'ObsID' not in columns:
                final_cols = ['ObsID'] + columns
            else:
                final_cols = columns
            df_merged = df_merged[[col for col in final_cols if col in df_merged.columns]]
    
        return df_merged


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
