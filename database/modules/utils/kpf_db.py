import os
import psycopg2
import re
import hashlib
import pandas as pd
from astropy.time import Time

from kpfpipe.models.level1 import KPF1
from kpfpipe.logger import start_logger

DEFAULT_CFG_PATH = 'database/utils/configs/kpf_db.cfg'

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


class KPFDB:

    """
    Class to facilitate execution of queries in the KPF operations database.
    For each query a different method is defined.

    Returns exitcode:
         0 = Normal
         2 = Exception raised closing database connection
        64 = Cannot connect to database
        65 = Input file does not exist
        66 = File checksum does not match database checksum
        67 = Could not execute query
        68 = Failed to compute checksum
    """

    def __init__(self, logger=None):
        if logger == None:
            self.log = start_logger('KPFDB', DEFAULT_CFG_PATH)
        else:
            self.log = logger

        self.exit_code = 0
        self.cId = None
        self.db_level = None
        self.db_cal_type = None
        self.db_object = None
        self.infobits = None
        self.filename = None
        self.conn = None

        # Get database connection parameters from environment.

        dbport = os.getenv('DBPORT')
        dbname = os.getenv('DBNAME')
        dbuser = os.getenv('DBUSER')
        dbpass = os.getenv('DBPASS')
        dbserver = os.getenv('DBSERVER')

        # Connect to database

        try:
            self.conn = psycopg2.connect(host=dbserver,database=dbname,port=dbport,user=dbuser,password=dbpass)
        except:
            print("Could not connect to database...")
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

    def query_to_pandas(self, query):
        results = pd.read_sql_query(query, self.conn)

        return results

    def get_nearest_master(self, obs_date, cal_file_level, cal_type_pair, contentbitmask=3, max_cal_delta_time='1000 days'):
        """Get the master file closest in time to obs_date of the specified calibration type
        
        Args:
            obs_date (string): ISO formatted datetime string
            cal_file_level (int): data level (0, 1, 2)
            cal_type_pair (list): two-element list. First element is the calibration type (e.g. bias) and the second element is the object name (e.g. autocal-bias)
            contentbitmask (int): (optional) contentbitmask flag to match (default=3)
            max_cal_delta_time (string): (optional) maximum delta time between obs_date and the calibration file to consider (default = 14 days)

        Returns:
            string: path to calibration file
        
        """
        query_template = f"""
SELECT *,
(minmjd + maxmjd)/2 as meanmjd
FROM calfiles
WHERE CAST('{obs_date}' as date) BETWEEN (startdate - INTERVAL '{max_cal_delta_time}') AND (startdate + INTERVAL '{max_cal_delta_time}')
AND level = '{cal_file_level}'
AND contentbits = {contentbitmask}
AND caltype = '{cal_type_pair[0].lower()}'
AND object = '{cal_type_pair[1]}'
ORDER BY startdate;"""

        df = self.query_to_pandas(query_template)
        if len(df) == 0:
            return [1, None]

        obst = Time(obs_date)
        obs_jd = obst.mjd
        print(obs_jd)

        df['delta'] = (df['meanmjd'] - obs_jd).abs()
        best_match = df.loc[df['delta'].idxmin()]
        fname = os.path.join('/', best_match['filename'])

        self.verify_checksum(fname, best_match['checksum'])

        return [self.exit_code, fname]
    
    def get_bracketing_wls(self, obs_date, object_name, contentbitmask=3, max_cal_delta_time='3 days'):
        """Get the WLS files that bracket a given obs_date
        
        Args:
            obs_date (string): ISO formatted datetime string
            object_name (list): Partial object name to search for (e.g. autocal-lfc-all). Object name in database must contain this name but doesn't need to be a complete match.
            contentbitmask (int): (optional) contentbitmask flag to match (default=3)
            max_cal_delta_time (string): (optional) maximum delta time between obs_date and the calibration file to consider (default = 3 days)
        
        """

        query_template = f"""
SELECT *,
(minmjd + maxmjd)/2 as meanmjd
FROM calfiles
WHERE CAST('{obs_date}' as date) BETWEEN (startdate - INTERVAL '{max_cal_delta_time}') AND (startdate + INTERVAL '{max_cal_delta_time}')
and level = 1
and contentbits = {contentbitmask}
AND caltype = 'wls'
AND object like '%{object_name}%'
ORDER BY startdate;"""
        
        df = self.query_to_pandas(query_template)
        if len(df) == 0:
            return [1, None, 1, None]

        obst = Time(obs_date)
        obs_jd = obst.mjd

        mjds = []
        for i, row in df.iterrows():
            fname = '/' + row['filename']
            l1 = KPF1.from_fits(fname)
            dt = l1.header['PRIMARY']['DATE-MID']
            mjd = Time(dt).mjd
            mjds.append(mjd)

        df['meanmjd'] = mjds

        df['delta'] = (df['meanmjd'] - obs_jd).abs()
        before_df = df[df['meanmjd'] < obs_jd]
        after_df = df[df['meanmjd'] >= obs_jd]

        try:
            best_before = before_df.loc[before_df['delta'].idxmin()]
            fname_before = os.path.join('/', best_before['filename'])
            self.verify_checksum(fname_before, best_before['checksum'])
            before_code = 0
        except (TypeError, ValueError):
            fname_before = None
            before_code = 1
        try:
            best_after = after_df.loc[after_df['delta'].idxmin()]
            fname_after = os.path.join('/', best_after['filename'])
            self.verify_checksum(fname_after, best_after['checksum'])
            after_code = 0
        except (TypeError, ValueError):
            fname_after = None
            after_code = 1

        return [before_code, fname_before, after_code, fname_after]


    def get_nearest_master_file(self,obs_date,cal_file_level,contentbitmask,cal_type_pair,max_cal_file_age='1000 days'):

        '''
        Get nearest master file for the specified set of input parameters.

        obs_date is a YYYYMMDD (string or number)
        '''

        # Reinitialize.

        self.cId = None
        self.db_level = None
        self.db_cal_type = None
        self.db_object = None
        self.infobits = None
        self.filename = None
        self.exit_code = 0


        # Define query template.

        query_template =\
            "select * from getCalFile(" +\
            "cast(OBSDATE as date)," +\
            "cast(LEVEL as smallint)," +\
            "cast('CALTYPE' as character varying(32))," +\
            "cast('OBJECT' as character varying(32))," +\
            "cast(CONTENTBITMASK as integer), " +\
            "cast('MAXFILEAGE' as interval)) as " +\
            "(cId integer," +\
            " level smallint," +\
            " caltype varchar(32)," +\
            " object varchar(32)," +\
            " filename varchar(255)," +\
            " checksum varchar(32)," +\
            " infobits integer," +\
            " startDate date);"

        obs_date_str = str(obs_date)
        obsdate = "'" + obs_date_str[0:4] + "-" + obs_date_str[4:6] + "-" + obs_date_str[6:8] + "'"


        # Query database for all cal_types.

        print('----> cal_file_level = {}'.format(cal_file_level))
        print('----> contentbitmask = {}'.format(contentbitmask))
        print('----> cal_type_pair = {}'.format(cal_type_pair))

        levelstr = str(cal_file_level)
        cal_type = cal_type_pair[0]
        object = cal_type_pair[1]

        rep = {"OBSDATE": obsdate,
               "LEVEL": levelstr,
               "CALTYPE": cal_type,
               "OBJECT": object,
               "MAXFILEAGE": max_cal_file_age}

        rep["CONTENTBITMASK"] = str(contentbitmask)

        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        query = pattern.sub(lambda m: rep[re.escape(m.group(0))], query_template)

        print('query = {}'.format(query))


        # Execute query.

        try:
            self.cur.execute(query)

        except (Exception, psycopg2.DatabaseError) as error:
            print('*** Error executing query ({}); skipping...'.format(query))
            self.exit_code = 67
            return


        record = self.cur.fetchone()

        if record is not None:
            cId = record[0]
            db_level = record[1]
            db_cal_type = record[2]
            db_object = record[3]
            filename = '/' + record[4]         # docker run has -v /data/kpf/masters:/masters
            checksum = record[5]
            infobits = record[6]

            print('cId = {}'.format(cId))
            print('filename = {}'.format(filename))
            print('checksum = {}'.format(checksum))

            self.verify_checksum(filename, checksum)

            self.cId = cId
            self.db_level = db_level
            self.db_cal_type = db_cal_type
            self.db_object = db_object
            self.infobits = infobits


    def verify_checksum(self, filename, checksum):
        # See if file exists.
        isExist = os.path.exists(filename)
        print('File existence = {}'.format(isExist))

        if isExist is True:
            print("File exists...")
        else:
            print("*** Error: File does not exist; quitting...")
            self.exit_code = 65
            return


        # Compute checksum and compare with database value.

        cksum = md5(filename)
        print('cksum = {}'.format(cksum))

        if  cksum == 68:
            self.exit_code = 68
            return

        if cksum == checksum:
            print("File checksum is correct ({})...".format(filename))
            self.filename = filename
            self.exit_code = 0
        else:
            print("*** Error: File checksum is incorrect ({}); quitting...".format(filename))
            self.exit_code = 66
            return


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


    def get_nearest_master_file_before(self,obsdatetime,cal_file_level,contentbitmask,cal_type_pair,max_cal_file_age='1000 days'):

        '''
        Get nearest master file before for the specified set of input parameters.

        obsdatetime is an # ISO datetime string, generally from the DATE-MID FITS keyword.
        '''

        # Reinitialize.

        self.cId = None
        self.db_level = None
        self.db_cal_type = None
        self.db_object = None
        self.infobits = None
        self.filename = None
        self.exit_code = 0


        # Define query template.

        query_template =\
            "select * from getCalFileBefore(" +\
            "cast('OBSDATETIME' as timestamp)," +\
            "cast(LEVEL as smallint)," +\
            "cast('CALTYPE' as character varying(32))," +\
            "cast('OBJECT' as character varying(32))," +\
            "cast(CONTENTBITMASK as integer), " +\
            "cast('MAXFILEAGE' as interval)) as " +\
            "(cId integer," +\
            " level smallint," +\
            " caltype varchar(32)," +\
            " object varchar(32)," +\
            " filename varchar(255)," +\
            " checksum varchar(32)," +\
            " infobits integer," +\
            " startDate date);"


        # Query database for all cal_types.

        print('----> cal_file_level = {}'.format(cal_file_level))
        print('----> contentbitmask = {}'.format(contentbitmask))
        print('----> cal_type_pair = {}'.format(cal_type_pair))

        levelstr = str(cal_file_level)
        cal_type = cal_type_pair[0]
        object = cal_type_pair[1]

        rep = {"OBSDATETIME": obsdatetime,
               "LEVEL": levelstr,
               "CALTYPE": cal_type,
               "OBJECT": object,
               "MAXFILEAGE": max_cal_file_age}

        rep["CONTENTBITMASK"] = str(contentbitmask)

        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        query = pattern.sub(lambda m: rep[re.escape(m.group(0))], query_template)

        print('query = {}'.format(query))


        # Execute query.

        try:
            self.cur.execute(query)

        except (Exception, psycopg2.DatabaseError) as error:
            print('*** Error executing query ({}); skipping...'.format(query))
            self.exit_code = 67
            return


        record = self.cur.fetchone()

        if record is not None:
            cId = record[0]
            db_level = record[1]
            db_cal_type = record[2]
            db_object = record[3]
            filename = '/' + record[4]         # docker run has -v /data/kpf/masters:/masters
            checksum = record[5]
            infobits = record[6]

            print('cId = {}'.format(cId))
            print('filename = {}'.format(filename))
            print('checksum = {}'.format(checksum))


            # See if file exists.

            isExist = os.path.exists(filename)
            print('File existence = {}'.format(isExist))

            if isExist is True:
                print("File exists...")
            else:
                print("*** Error: File does not exist; quitting...")
                self.exit_code = 65
                return


            # Compute checksum and compare with database value.

            cksum = md5(filename)
            print('cksum = {}'.format(cksum))

            if  cksum == 68:
                self.exit_code = 68
                return

            if cksum == checksum:
                print("File checksum is correct ({})...".format(filename))
                self.cId = cId
                self.db_level = db_level
                self.db_cal_type = db_cal_type
                self.db_object = db_object
                self.infobits = infobits
                self.filename = filename
                self.exit_code = 0
            else:
                print("*** Error: File checksum is incorrect ({}); quitting...".format(filename))
                self.exit_code = 66
                return


    def get_nearest_master_file_after(self,obsdatetime,cal_file_level,contentbitmask,cal_type_pair,max_cal_file_age='1000 days'):

        '''
        Get nearest master file after for the specified set of input parameters.

        obsdatetime is an # ISO datetime string, generally from the DATE-MID FITS keyword.
        '''

        # Reinitialize.

        self.cId = None
        self.db_level = None
        self.db_cal_type = None
        self.db_object = None
        self.infobits = None
        self.filename = None
        self.exit_code = 0


        # Define query template.

        query_template =\
            "select * from getCalFileAfter(" +\
            "cast('OBSDATETIME' as timestamp)," +\
            "cast(LEVEL as smallint)," +\
            "cast('CALTYPE' as character varying(32))," +\
            "cast('OBJECT' as character varying(32))," +\
            "cast(CONTENTBITMASK as integer), " +\
            "cast('MAXFILEAGE' as interval)) as " +\
            "(cId integer," +\
            " level smallint," +\
            " caltype varchar(32)," +\
            " object varchar(32)," +\
            " filename varchar(255)," +\
            " checksum varchar(32)," +\
            " infobits integer," +\
            " startDate date);"


        # Query database for all cal_types.

        print('----> cal_file_level = {}'.format(cal_file_level))
        print('----> contentbitmask = {}'.format(contentbitmask))
        print('----> cal_type_pair = {}'.format(cal_type_pair))

        levelstr = str(cal_file_level)
        cal_type = cal_type_pair[0]
        object = cal_type_pair[1]

        rep = {"OBSDATETIME": obsdatetime,
               "LEVEL": levelstr,
               "CALTYPE": cal_type,
               "OBJECT": object,
               "MAXFILEAGE": max_cal_file_age}

        rep["CONTENTBITMASK"] = str(contentbitmask)

        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        query = pattern.sub(lambda m: rep[re.escape(m.group(0))], query_template)

        print('query = {}'.format(query))


        # Execute query.

        try:
            self.cur.execute(query)

        except (Exception, psycopg2.DatabaseError) as error:
            print('*** Error executing query ({}); skipping...'.format(query))
            self.exit_code = 67
            return


        record = self.cur.fetchone()

        if record is not None:
            cId = record[0]
            db_level = record[1]
            db_cal_type = record[2]
            db_object = record[3]
            filename = '/' + record[4]         # docker run has -v /data/kpf/masters:/masters
            checksum = record[5]
            infobits = record[6]

            print('cId = {}'.format(cId))
            print('filename = {}'.format(filename))
            print('checksum = {}'.format(checksum))


            # See if file exists.

            isExist = os.path.exists(filename)
            print('File existence = {}'.format(isExist))

            if isExist is True:
                print("File exists...")
            else:
                print("*** Error: File does not exist; quitting...")
                self.exit_code = 65
                return


            # Compute checksum and compare with database value.

            cksum = md5(filename)
            print('cksum = {}'.format(cksum))

            if  cksum == 68:
                self.exit_code = 68
                return

            if cksum == checksum:
                print("File checksum is correct ({})...".format(filename))
                self.cId = cId
                self.db_level = db_level
                self.db_cal_type = db_cal_type
                self.db_object = db_object
                self.infobits = infobits
                self.filename = filename
                self.exit_code = 0
            else:
                print("*** Error: File checksum is incorrect ({}); quitting...".format(filename))
                self.exit_code = 66
                return
