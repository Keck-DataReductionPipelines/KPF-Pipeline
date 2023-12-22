import os
import glob
import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm

from tqdm.notebook import tqdm_notebook
import time

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

        self.create_database(self.db_path, self.L0_keyword_types, self.D2_keyword_types, self.L1_keyword_types, self.L2_keyword_types)
        self.logger.info('Initialization complete')


    def drop_table(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS kpfdb")
        conn.commit()
        conn.close()


    def create_database(self, db_path, L0_keyword_types, D2_keyword_types, L1_keyword_types, L2_keyword_types):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    
        # Define columns for each file type
        L0_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in L0_keyword_types.items()]
        D2_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in D2_keyword_types.items()]
        L1_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in L1_keyword_types.items()]
        L2_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in L2_keyword_types.items()]
    
        columns = L0_columns + D2_columns + L1_columns + L2_columns
        columns += ['"L0_filename" TEXT', '"D2_filename" TEXT', '"L1_filename" TEXT', '"L2_filename" TEXT', ]
        columns += ['"header_read_time" TEXT', '"datecode" TEXT', '"ObsID" TEXT']
        create_table_query = f'CREATE TABLE IF NOT EXISTS kpfdb ({", ".join(columns)}, UNIQUE(ObsID))'
        cursor.execute(create_table_query)
        conn.commit()
        conn.close()


    def add_to_db(self, start_date, end_date):
        self.logger.info("Adding to database between " + start_date + " to " + end_date)
        dir_paths = glob.glob(f"{self.base_dir}/????????")
        sorted_dir_paths = sorted(dir_paths, key=lambda x: int(os.path.basename(x)), reverse=start_date > end_date)
        filtered_dir_paths = [
            dir_path for dir_path in sorted_dir_paths
            if start_date <= os.path.basename(dir_path) <= end_date
        ]
        t = tqdm_notebook(filtered_dir_paths, desc=(filtered_dir_paths[0]).split('/')[-1])
        for dir_path in t:
            t.set_description(dir_path.split('/')[-1])
            t.refresh() 
            self.extract_and_store_fits_headers(dir_path, self.L0_keyword_types, self.D2_keyword_types, self.L1_keyword_types, self.L2_keyword_types, self.db_path)


    def extract_fits_header_keywords(self, file_path, keyword_types):
        with fits.open(file_path, memmap=True) as hdul:
            header = hdul[0].header
            header_data = {}
            for key in keyword_types.keys():
                if key in header:
                    header_data[key] = header[key]
                else:
                    header_data[key] = None  # Ensure all expected keys are present
            return header_data
    
    
    def extract_and_store_fits_headers(self, dir_path, L0_keyword_types, D2_keyword_types, L1_keyword_types, L2_keyword_types, db_path):

        t = tqdm_notebook(os.listdir(dir_path), desc=f'Files', leave=False)
        for filename in t:
            if filename.endswith(".fits"):
                file_path = os.path.join(dir_path, filename)
                base_filename = filename.split('.fits')[0]
                t.set_description(base_filename)
                t.refresh() 
                file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
    
                # Check if the file has been updated since the last recorded in the database
                if not self.is_file_updated(db_path, filename, file_mod_time):
                    continue            
                
                # Extract header data for each file type
                L0_file_path = f"{dir_path}/{base_filename}.fits"
                D2_file_path = f"{dir_path}/{base_filename}_2D.fits"
                L1_file_path = f"{dir_path}/{base_filename}_L1.fits"
                L2_file_path = f"{dir_path}/{base_filename}_L2.fits"
    
                L0_header_data = self.extract_fits_header_keywords(L0_file_path, L0_keyword_types) if os.path.exists(L0_file_path) else {}
                D2_header_data = self.extract_fits_header_keywords(D2_file_path, D2_keyword_types) if os.path.exists(D2_file_path) else {}
                L1_header_data = self.extract_fits_header_keywords(L1_file_path, L1_keyword_types) if os.path.exists(L1_file_path) else {}
                L2_header_data = self.extract_fits_header_keywords(L2_file_path, L2_keyword_types) if os.path.exists(L2_file_path) else {}
                
                header_data = {**L0_header_data, **D2_header_data, **L1_header_data, **L2_header_data}
    
                # Add common data
                header_data['L0_filename'] = filename
                header_data['D2_filename'] = f"{base_filename}_2D.fits"
                header_data['L1_filename'] = f"{base_filename}_L1.fits"
                header_data['L2_filename'] = f"{base_filename}_L2.fits"
                header_data['header_read_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                header_data['datecode'] = get_datecode(filename)  # Assuming get_datecode extracts datecode from filename
                header_data['ObsID'] = (filename.split('.fits')[0])
    
                # Insert into database
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                columns = ', '.join([f'"{key}"' for key in header_data.keys()])
                placeholders = ', '.join(['?'] * len(header_data))
                insert_query = f'INSERT OR REPLACE INTO kpfdb ({columns}) VALUES ({placeholders})'
                cursor.execute(insert_query, tuple(header_data.values()))
                conn.commit()
                conn.close()


    def is_file_updated(self, db_path, filename, file_mod_time):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = f'SELECT header_read_time FROM kpfdb WHERE L0_filename = "{filename}"'
        cursor.execute(query)
        result = cursor.fetchone()
        conn.close()
    
        if result:
            stored_mod_time = datetime.strptime(result[0], "%Y-%m-%d %H:%M:%S")
            current_mod_time = datetime.strptime(file_mod_time, "%Y-%m-%d %H:%M:%S")
            return current_mod_time > stored_mod_time
        return True  # Process if file is not in the database
           

    def print_db_status(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM kpfdb')
        nrows = cursor.fetchone()[0]
        cursor.execute('PRAGMA table_info(kpfdb)')
        ncolumns = len(cursor.fetchall())
        cursor.execute('SELECT MAX(header_read_time) FROM kpfdb')
        most_recent_read_time = cursor.fetchone()[0]
        cursor.execute('SELECT MIN(datecode) FROM kpfdb')
        earliest_datecode = cursor.fetchone()[0]
        cursor.execute('SELECT MAX(datecode) FROM kpfdb')
        latest_datecode = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(DISTINCT datecode) FROM kpfdb')
        unique_datecodes_count = cursor.fetchone()[0]
        conn.close()
        self.logger.info(f"Summary: {nrows} obs x {ncolumns} colns over {unique_datecodes_count} days in {earliest_datecode}-{latest_datecode}; updated {most_recent_read_time}")


    def print_selected_columns(self, db_path, columns):
        conn = sqlite3.connect(db_path)
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
                'EXPTIME': 'float',
                'ELAPSED': 'float',
                'PROGNAME': 'string',
                'OBJECT': 'string',
                'AIRMASS': 'float',
                'CAL-OBJ': 'string',
                'SKY-OBJ': 'string',
                'SCI-OBJ': 'string',
            }
        elif level == '2D':
            keyword_types = {
                'DRPTAG': 'string',
                'MOONSEP': 'float',
                'SUNALT': 'float',
                'SKYSCIMS': 'float',
            }
        elif level == 'L1':
            keyword_types = {
                'RNRED1': 'float',
                'RNRED2': 'float',
                'RNGREEN1': 'float',
                'RNGREEN2': 'float',
                'SNRSC452': 'string',
            }
        elif level == 'L2':
            keyword_types = {
                'SNRSC123': 'string', #placeholder for now
            }
        else:
            keyword_types = {}
        
        return keyword_types