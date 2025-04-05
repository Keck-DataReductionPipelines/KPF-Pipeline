import os
import psycopg2
import re
import hashlib
import pandas as pd
import numpy as np
import time
from astropy.time import Time

from kpfpipe.models.level1 import KPF1
from kpfpipe.logger import start_logger

DEFAULT_CFG_PATH = 'database/modules/utils/kpf_tsdb.cfg'

# Common methods.

def md5(fname):
    """
    Returns checksum = 68 if it fails to compute the MD5 checksum.
    """

    hash_md5 = hashlib.md5()

    try:
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        print("*** Error: Failed to compute checksum =",fname,"; quitting...")
        return 68


class KPF_TSDB:

    """
    Class to facilitate execution of queries in the KPF time series database.
    This class is designed to work with SQLITE3 and PostgreSQL databases.
    For each query a different method is defined.

    For PostgreSQL, it returns exitcode:
         0 = Normal
         2 = Exception raised closing database connection
        64 = Cannot connect to database
        65 = Input file does not exist
        66 = File checksum does not match database checksum
        67 = Could not execute query
        68 = Failed to compute checksum
    """

    def __init__(self, db_type='sqlite3', logger=None):
        """
        Todo: add docstring, including explanation of db_type = 'sqlite3' or 'postgres'
        """
        
        self.db_type = db_type
        
        if logger == None:
            self.log = start_logger('KPF_TSDB', DEFAULT_CFG_PATH)
        else:
            self.log = logger

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
                    print("Could not connect to database, retrying...")
                    db_fail = True
                    time.sleep(10)
    
            if db_fail:
                print(f"Could not connect to database after {n_attempts} attempts...")
                self.exit_code = 64
                return
    
            # Open database cursor.
            self.cur = self.conn.cursor()
    
            # Select database version.
            q1 = 'SELECT version();'
            print('q1 = {}'.format(q1))
            self.cur.execute(q1)
            db_version = self.cur.fetchone()
            print('PostgreSQL database version = {}'.format(db_version))
    
            # Check database current_user.
            q2 = 'SELECT current_user;'
            print('q2 = {}'.format(q2))
            self.cur.execute(q2)
            for record in self.cur:
                print('record = {}'.format(record))


    def close(self):

        '''
        Close database cursor and then connection.
        '''

        try:
            self.cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            self.exit_code = 2
        finally:
            if self.conn is not None:
                self.conn.close()
                print('Database connection closed.')


    def query_to_pandas(self, query):
        try:
            results = pd.read_sql_query(query, self.conn)
        except:
            self.log.warning(f"Error running database query:\n{query}\n")
            results = pd.DataFrame([])

        return results
