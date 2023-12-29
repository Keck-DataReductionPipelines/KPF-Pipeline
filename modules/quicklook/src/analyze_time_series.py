import os
import time
import glob
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from tqdm.notebook import tqdm_notebook
from astropy.table import Table
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
        self.L0_keyword_types   = self.get_keyword_types(level='L0')
        self.D2_keyword_types   = self.get_keyword_types(level='2D')
        self.L1_keyword_types   = self.get_keyword_types(level='L1')
        self.L2_keyword_types   = self.get_keyword_types(level='L2')
        self.L0_telemetry_types = self.get_keyword_types(level='L0_telemetry')
        
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
        L0_telemetry_columns = [f'"{key}" {self.map_data_type_to_sql(dtype)}' for key, dtype in self.L0_telemetry_types.items()]
    
        columns = L0_columns + D2_columns + L1_columns + L2_columns + L0_telemetry_columns
        columns += ['"datecode" TEXT', '"ObsID" TEXT']
        columns += ['"L0_filename" TEXT', '"D2_filename" TEXT', '"L1_filename" TEXT', '"L2_filename" TEXT', ]
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
            L0_telemetry   = self.extract_kwd(L0_file_path, self.L0_telemetry_types) if L0_exists else {}

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

    def extract_telemetry(self, file_path, keyword_types):
        df = Table.read(L0_file, format='fits', hdu='TELEMETRY').to_pandas()
        num_columns = ['average', 'stddev', 'min', 'max']
        for column in df_telemetry:
            df_telemetry[column] = df_telemetry[column].str.decode('utf-8')
            df_telemetry = df_telemetry.replace('-nan', 0)# replace nan with 0
            if column in num_columns:
                df_telemetry[column] = pd.to_numeric(df_telemetry[column], downcast="float")
            else:
                df_telemetry[column] = df_telemetry[column].astype(str)
        df_telemetry.set_index("keyword", inplace=True)
        telemetry_data = {}
        for key in keyword_types.keys():
            if key in df_telemetry.index:
                telemetry_data[key] = df_telemetry.loc[key, 'average']
            else:
                telemetry_data[key] = None 
        
        return telemetry_data


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
        self.logger.info(f"Summary: {nrows} obs x {ncolumns} cols over {unique_datecodes_count} days in {earliest_datecode}-{latest_datecode}; updated {most_recent_read_time}")


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

   
    def dataframe_from_db(self, columns, only_object=None, object_like=None, 
                          start_date=None, end_date=None, verbose=False):
        '''
        Returns a pandas dataframe of attributes (specified by column names) for all 
        observations in the DB. The query can be restricted to observations matching a 
        particular object name(s).  The query can also be restricted to observations 
        after start_date and/or before end_date. 

        Args:
            columns (string or list of strings) - database columns to query
            only_object (string or list of strings) - object names to include in query
            object_like (string or list of strings) - partial object names to search for
            start_date (datetime object) - only return observations after start_date
            end_date (datetime object) - only return observations after end_date
            false (boolean) - if True, prints the SQL query

        Returns:
            Pandas dataframe of the specified columns matching the object name and 
            start_time/end_time constraints.
        '''
        
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
                'ETAV1C1T': 'float',
                'ETAV1C2T': 'float',
                'ETAV1C3T': 'float',
                'ETAV1C4T': 'float',
                'ETAV2C3T': 'float',
                'TOTCORR':  'string', # need to correct this to split into four bins
                'USTHRSH':  'string', 
                'THRSHLD': 'float',
                'THRSBIN': 'float',
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
        elif level == 'L0_telemetry':
            keyword_types = {
                'kpfmet.BENCH_BOTTOM_BETWEEN_CAMERAS': 'float',  # degC    Bench Bottom Between Cameras C2 c- double degC...
                'kpfmet.BENCH_BOTTOM_COLLIMATOR':      'float',  # degC    Bench Bottom Coll C3 c- double degC {%.3f}
                'kpfmet.BENCH_BOTTOM_DCUT':            'float',  # degC    Bench Bottom D-cut C4 c- double degC {%.3f}
                'kpfmet.BENCH_BOTTOM_ECHELLE ':        'float',  # degC    Bench Bottom Echelle Cam B c- double degC {%.3f}
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
                                    only_object=None, object_like=None, 
                                    fig_path=None, show_plot=False):
        """
        Generate a multi-panel plot of data in a KPF DB.  The data to be plotted and 
        attributes are stored in an array of dictionaries called 'panel_arr'.

        Args:
            panel_dict (array of dictionaries) - each dictionary in the array has keys:
                panelnum - panel index number
                panelvars: a dictionary of matplotlib attributes including:
                    ylabel - text for y-axis label
                paneldict: a dictionary containing:
                    col: name of DB column to plot
                    plot_attr: a dictionary containing plot attributes for a scatter plot, 
                        including 'label', 'marker', 'color'
            only_object (string or list of strings) - object names to include in query
            object_like (string or list of strings) - partial object names to search for
            start_date (datetime object) - start date for plot
            end_date (datetime object) - end date for plot
            fig_path (string) - set to the path for a SNR vs. wavelength file
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

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
            greenpanel = {'panelnum': 1, 
                          'panelvars': thispanelvars,
                          'paneldict': thispaneldict}
            # Red CCD panel
            dict1 = {'col': 'FLXCOLLR', 'plot_type': 'scatter', 'plot_attr': {'label': 'Collimator-side', 'marker': '.', 'color': 'darkred'}}
            dict2 = {'col': 'FLXECHR',  'plot_type': 'scatter', 'plot_attr': {'label': 'Echelle-side',    'marker': '.', 'color': 'firebrick'}}
            dict3 = {'col': 'FLXREG1R', 'plot_type': 'scatter', 'plot_attr': {'label': 'Region 1',        'marker': '.', 'color': 'lightcoral'}}
            dict4 = {'col': 'FLXREG2R', 'plot_type': 'scatter', 'plot_attr': {'label': 'Region 2',        'marker': '.', 'color': 'lightcoral'}}
            thispanelvars = [dict3, dict4, dict1, dict2]
            thispaneldict = {'ylabel': 'Red CCD\nDark current [e-/hr]'}
            redpanel = {'panelnum': 2, 
                        'panelvars': thispanelvars,
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
            
        start = start_date
        end = end_date
            
        npanels = len(panel_arr)
        unique_cols = set()
        unique_cols.add('DATE-MID')
        for panel in panel_arr:
            for d in panel['panelvars']:
                col_value = d['col']
                unique_cols.add(col_value)
        df = self.dataframe_from_db(unique_cols, object_like=object_like, only_object=only_object, start_date=start_date, end_date=end_date, verbose=False)
        df['DATE-MID'] = pd.to_datetime(df['DATE-MID']) # move this to dataframe_from_db ?

        #nrow = len(df)
        #self.logger.info('Plotting')
        fig, axs = plt.subplots(npanels, 1, sharex=True, figsize=(12, npanels*2.5), tight_layout=True)
        if npanels > 1:
            plt.subplots_adjust(hspace=0)
        plt.tight_layout()

        for p in np.arange(npanels):
            thispanel = panel_arr[p]
            time = df['DATE-MID']
            if p == 0: 
                axs[p].set_title('Dark Current Measurements', fontsize=14)
            if p == npanels-1: 
                axs[p].set_xlabel('Date', fontsize=14)
                if abs((end_date - start_date).days) > 3:
                    axs[p].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                else:
                    axs[p].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%Y-%m-%d'))
            if 'ylabel' in thispanel['paneldict']:
                axs[p].set_ylabel(thispanel['paneldict']['ylabel'], fontsize=14)
            nvars = len(thispanel['panelvars'])
            for i in np.arange(nvars):
                data = df[thispanel['panelvars'][i]['col']]
                if 'plot_type' in thispanel['panelvars'][i]:
                    plot_type = thispanel['panelvars'][i]['plot_type']
                else:
                    plot_type = 'scatter'
                if 'plot_attr' in thispanel['panelvars'][i]:
                    plot_attributes = thispanel['panelvars'][i]['plot_attr']
                else:
                   plot_attributes = {}
                if plot_type == 'scatter':
                    axs[p].scatter(time, data, **plot_attributes)
                if plot_type == 'plot':
                    axs[p].plot(time, data, **plot_attributes)
                if plot_type == 'step':
                    axs[p].step(time, data, **plot_attributes)
                axs[p].xaxis.set_tick_params(labelsize=10)
                axs[p].yaxis.set_tick_params(labelsize=10)
            axs[p].legend()
            axs[p].grid(color='lightgray')
            axs[p].set_xlim(start_date, end_date)

        # possibly add this to put set the lower limit of y to 0
        #ymin, ymax = ax1.get_ylim()
        #if ymin > 0:
        #    ax1.set_ylim(bottom=0)

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=300, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')
