import os
import time
import glob
import sqlite3
import numpy as np
import pandas as pd
#from tqdm import tqdm
from tqdm.notebook import tqdm_notebook

from astropy.io import fits
from datetime import datetime
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
    """

    def __init__(self, db_path='kpfdb.db', base_dir='/data/L0', logger=None, drop=False):
       
        self.logger = logger if logger is not None else DummyLogger()
        self.logger.info('Initializing database')
        self.db_path = db_path
        self.base_dir = base_dir
        self.L0_keyword_types = self.get_keyword_types(level='L0')
        self.D2_keyword_types = self.get_keyword_types(level='2D')
        self.L1_keyword_types = self.get_keyword_types(level='L1')
        self.L2_keyword_types = self.get_keyword_types(level='L2')

        if drop:
            self.drop_table()
            self.logger.info('Dropping KPF database ' + str(self.db_path))

        self.create_database()
        self.logger.info('Initialization complete')


    def drop_table(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS kpfdb")
        conn.commit()
        conn.close()


    def create_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
    
        # Define columns for each file type
        L0_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L0_keyword_types.items()]
        D2_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.D2_keyword_types.items()]
        L1_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L1_keyword_types.items()]
        L2_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L2_keyword_types.items()]
    
        columns = L0_columns + D2_columns + L1_columns + L2_columns
        columns += ['"L0_filename" TEXT', '"D2_filename" TEXT', '"L1_filename" TEXT', '"L2_filename" TEXT', ]
        columns += ['"datecode" TEXT', '"ObsID" TEXT']
        columns += ['"L0_header_read_time" TEXT', '"D2_header_read_time" TEXT', '"L1_header_read_time" TEXT', '"L2_header_read_time" TEXT', ]
        create_table_query = f'CREATE TABLE IF NOT EXISTS kpfdb ({", ".join(columns)}, UNIQUE(ObsID))'
        cursor.execute(create_table_query)
        conn.commit()
        conn.close()


    def add_dates_to_db(self, start_date, end_date):
        self.logger.info("Adding to database between " + start_date + " to " + end_date)
        dir_paths = glob.glob(f"{self.base_dir}/????????")
        sorted_dir_paths = sorted(dir_paths, key=lambda x: int(os.path.basename(x)), reverse=start_date > end_date)
        filtered_dir_paths = [
            dir_path for dir_path in sorted_dir_paths
            if start_date <= os.path.basename(dir_path) <= end_date
        ]
        t1 = tqdm_notebook(filtered_dir_paths, desc=(filtered_dir_paths[0]).split('/')[-1])
        for dir_path in t1:
            t1.set_description(dir_path.split('/')[-1])
            t1.refresh() 
            t2 = tqdm_notebook(os.listdir(dir_path), desc=f'Files', leave=False)
            for L0_filename in t2:
                if L0_filename.endswith(".fits"):
                    file_path = os.path.join(dir_path, L0_filename)
                    base_filename = L0_filename.split('.fits')[0]
                    t2.set_description(base_filename)
                    t2.refresh() 
                    self.ingest_one_observation(dir_path, L0_filename) 

    def add_ObsID_list_to_db(self, ObsID_filename):
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
        result_df = filtered_column.to_frame()
        self.logger.info('ObsID_filename read with ' + str(len(result_df)) + ' properly formatted ObsIDs.')

        t = tqdm_notebook(result_df.iloc[:, 0].tolist(), desc=f'ObsIDs', leave=True)
        for ObsID in t:
            L0_filename = ObsID + '.fits'
            dir_path = self.base_dir + '/' + get_datecode(ObsID) + '/'
            file_path = os.path.join(dir_path, L0_filename)
            base_filename = L0_filename.split('.fits')[0]
            t.set_description(base_filename)
            t.refresh() 
            #print('dir_path = ' + dir_path)
            #print('L0_filename = ' + L0_filename)
            self.ingest_one_observation(dir_path, L0_filename) 


    def ingest_one_observation(self, dir_path, L0_filename):        

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
        
            L0_header_data = self.extract_kwd(L0_file_path, self.L0_keyword_types) if L0_exists else {}
            D2_header_data = self.extract_kwd(D2_file_path, self.D2_keyword_types) if D2_exists else {}
            L1_header_data = self.extract_kwd(L1_file_path, self.L1_keyword_types) if L1_exists else {}
            L2_header_data = self.extract_kwd(L2_file_path, self.L2_keyword_types) if L2_exists else {}

            header_data = {**L0_header_data, **D2_header_data, **L1_header_data, **L2_header_data}
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
        
            # Insert into database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            columns = ', '.join([f'"{key}"' for key in header_data.keys()])
            placeholders = ', '.join(['?'] * len(header_data))
            insert_query = f'INSERT OR REPLACE INTO kpfdb ({columns}) VALUES ({placeholders})'
            cursor.execute(insert_query, tuple(header_data.values()))
            conn.commit()
            conn.close()


    def extract_kwd(self, file_path, keyword_types):
        with fits.open(file_path, memmap=True) as hdul: # memmap=True minimizes RAM usage
            header = hdul[0].header
            header_data = {}
            for key in keyword_types.keys():
                if key in header:
                    header_data[key] = header[key]
                else:
                    header_data[key] = None 
            return header_data


    def is_file_updated(self, file_path, filename, level):
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
            #print('current_mod_time  = ' + str(current_mod_time))
            #print('stored_mod_time  = ' + str(stored_mod_time))
            #print(current_mod_time > stored_mod_time)
            return current_mod_time > stored_mod_time
        
        return True  # Process if file is not in the database
           

    def print_db_status(self):
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
        self.logger.info(f"Summary: {nrows} obs x {ncolumns} colns over {unique_datecodes_count} days in {earliest_datecode}-{latest_datecode}; updated {most_recent_read_time}")


    def print_selected_columns(self, columns):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = f"SELECT {', '.join(columns)} FROM kpfdb"
        cursor.execute(query)
        rows = cursor.fetchall()
    
        # Print column headers
        print(' | '.join(columns))
        print('-' * (len(columns) * 10))  # Adjust the number for formatting
    
        # Print each row
        for row in rows:
            print(' | '.join(str(item) for item in row))
    
        conn.close()


    def display_dataframe_from_db(self, columns, only_object=None):
        conn = sqlite3.connect(self.db_path)
        
        # Enclose column names in double quotes
        quoted_columns = [f'"{column}"' for column in columns]
        query = f"SELECT {', '.join(quoted_columns)} FROM kpfdb"
        
        # Append WHERE clause if only_object is not None
        if only_object is not None:
            # Use parameterized queries to prevent SQL injection
            query += " WHERE OBJECT = ?"
    
        # Execute query
        df = pd.read_sql_query(query, conn, params=(only_object,) if only_object is not None else None)
        conn.close()
        print(df)
        
    def dataframe_from_db(self, columns, only_object=None):
        conn = sqlite3.connect(self.db_path)
        
        # Append WHERE clause if only_object is not None
        if only_object is not None:
            # Use parameterized queries to prevent SQL injection
            query += " WHERE OBJECT = ?"
        
        # Enclose column names in double quotes
        quoted_columns = [f'"{column}"' for column in columns]
        query = f"SELECT {', '.join(quoted_columns)} FROM kpfdb"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def map_data_type_to_sql(self, dtype):
        return {
            'int': 'INTEGER',
            'float': 'REAL',
            'bool': 'BOOLEAN',
            'datetime': 'TEXT',  # SQLite does not have a native datetime type
            'string': 'TEXT'
        }.get(dtype, 'TEXT')
        
    def get_keyword_types(self, level):
    
        if level == 'L0':
            keyword_types = {
                'DATE-MID': 'datetime',
                'EXPTIME':  'float',
                'ELAPSED':  'float',
                'PROGNAME': 'string',
                'TARGRA':   'string',
                'TARGDEC':  'string',
                'OBJECT':   'string',
                'GAIAMAG':  'float',
                '2MASSMAG': 'float',
                'AIRMASS':  'float',
                'IMTYPE':   'string',
                'CAL-OBJ':  'string',
                'SKY-OBJ':  'string',
                'SCI-OBJ':  'string',
                'AGITSTA':  'string',
            }
        elif level == '2D':
            keyword_types = {
                'DRPTAG':   'string',
                'RNRED1':   'float',
                'RNRED2':   'float',
                'RNGREEN1': 'float',
                'RNGREEN2': 'float',
                'READSPED': 'string',
                'FLXREG1G': 'float',
                'FLXREG2G': 'float',
                'FLXREG3G': 'float',
                'FLXREG4G': 'float',
                'FLXREG5G': 'float',
                'FLXREG6G': 'float',
                'FLXAMP1G': 'float',
                'FLXAMP2G': 'float',
                'FLXCOLLG': 'float',
                'FLXECHG':  'float',
                'FLXREG1R': 'float',
                'FLXREG2R': 'float',
                'FLXREG3R': 'float',
                'FLXREG4R': 'float',
                'FLXREG5R': 'float',
                'FLXREG6R': 'float',
                'FLXAMP1R': 'float',
                'FLXAMP2R': 'float',
                'FLXCOLLR': 'float',
                'FLXECHR':  'float',
                'GDRXRMS':  'float',
                'GDRYRMS':  'float',
                'GDRRRMS':  'float',
                'GDRXBIAS': 'float',
                'GDRYBIAS': 'float',
                'GDRSEEJZ': 'float',
                'GDRSEEV':  'float',
                'MOONSEP':  'float',
                'SUNALT':   'float',
                'SKYSCIMS': 'float',
                'EMSCCT48': 'float',
                'EMSCCT45': 'float',
                'EMSCCT56': 'float',
                'EMSCCT67': 'float',
                'EMSCCT78': 'float',
                'EMSKCT48': 'float',
                'EMSKCT45': 'float',
                'EMSKCT56': 'float',
                'EMSKCT67': 'float',
                'EMSKCT78': 'float',
            }
        elif level == 'L1':
            keyword_types = {
                'MONOTWLS': 'bool',
                'SNRSC452': 'float',
                'SNRSK452': 'float',
                'SNRCL452': 'float',
                'SNRSC548': 'float',
                'SNRSK548': 'float',
                'SNRCL548': 'float',
                'SNRSC652': 'float',
                'SNRSK652': 'float',
                'SNRCL652': 'float',
                'SNRSC747': 'float',
                'SNRSK747': 'float',
                'SNRCL747': 'float',
                'SNRSC852': 'float',
                'SNRSK852': 'float',
                'SNRCL852': 'float',
                'FR452652': 'float',
                'FR548652': 'float',
                'FR747652': 'float',
                'FR852652': 'float',
                'FR12M452': 'float',
                'FR12U452': 'float',
                'FR32M452': 'float',
                'FR32U452': 'float',
                'FRS2M452': 'float',
                'FRS2U452': 'float',
                'FRC2M452': 'float',
                'FRC2U452': 'float',
                'FR12M548': 'float',
                'FR12U548': 'float',
                'FR32M548': 'float',
                'FR32U548': 'float',
                'FRS2M548': 'float',
                'FRS2U548': 'float',
                'FRC2M548': 'float',
                'FRC2U548': 'float',
                'FR12M652': 'float',
                'FR12U652': 'float',
                'FR32M652': 'float',
                'FR32U652': 'float',
                'FRS2M652': 'float',
                'FRS2U652': 'float',
                'FRC2M652': 'float',
                'FRC2U652': 'float',
                'FR12M747': 'float',
                'FR12U747': 'float',
                'FR32M747': 'float',
                'FR32U747': 'float',
                'FRS2M747': 'float',
                'FRS2U747': 'float',
                'FRC2M747': 'float',
                'FRC2U747': 'float',
                'FR12M852': 'float',
                'FR12U852': 'float',
                'FR32M852': 'float',
                'FR32U852': 'float',
                'FRS2M852': 'float',
                'FRS2U852': 'float',
                'FRC2M852': 'float',
                'FRC2U852': 'float',
            }
        elif level == 'L2':
            keyword_types = {
                'ABCDEFGH': 'string', #placeholder for now
            }
        else:
            keyword_types = {}
        
        return keyword_types