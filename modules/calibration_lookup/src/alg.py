
from datetime import datetime
import pandas as pd
import os
import re
import psycopg2

from database.modules.query_db_nearest_master_files.src.query_db_nearest_master_files_framework import md5
from keckdrpframework.models.arguments import Arguments
from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger

def query_database(date, cal_types, cal_file_levels, log):
        # Get database connection parameters from environment.
        # *** This code is duplicated in QueryDBNearestMasterFilesFramework and needs 
        # to be consolidated

        dbport = os.getenv('DBPORT')
        dbname = os.getenv('DBNAME')
        dbuser = os.getenv('DBUSER')
        dbpass = os.getenv('DBPASS')
        dbserver = os.getenv('DBSERVER')


        # Connect to database

        try:
            conn = psycopg2.connect(host=dbserver,database=dbname,port=dbport,user=dbuser,password=dbpass)
        except:
            log.warning("Could not connect to database...")
            return Arguments(64)

        # Open database cursor.

        cur = conn.cursor()


        # Select database version.

        q1 = 'SELECT version();'
        log.debug('q1 = {}'.format(q1))
        cur.execute(q1)
        db_version = cur.fetchone()
        log.debug('PostgreSQL database version = {}'.format(db_version))


        # Check database current_user.

        q2 = 'SELECT current_user;'
        log.debug('q2 = {}'.format(q2))
        cur.execute(q2)
        for record in cur:
            log.debug('record = {}'.format(record))


        # Define query template.

        query_template =\
            "select * from getCalFile(" +\
            "cast(OBSDATE as date)," +\
            "cast(LEVEL as smallint)," +\
            "cast('CALTYPE' as character varying(32))," +\
            "cast('OBJECT' as character varying(32))," +\
            "cast(CONTENTBITMASK as integer)) as " +\
            "(cId integer," +\
            " level smallint," +\
            " caltype varchar(32)," +\
            " object varchar(32)," +\
            " filename varchar(255)," +\
            " checksum varchar(32)," +\
            " infobits integer," +\
            " startDate date);"

        obsdate = "'" + date[0:4] + "-" + date[4:6] + "-" + date[6:8] + "'"


        # Query database for all cal_types.

        contentbitmask_list = [3]        # Mask values for GREEN, RED, and CA_HK together, and then for just GREEN and RED.
        
        nearest_master_files_list = []

        log.debug('----> self.cal_file_levels = {}'.format(cal_file_levels))
        log.debug('----> self.cal_types = {}'.format(cal_types))

        for contentbitmask in contentbitmask_list:
            for level,cal_type_pair in zip(cal_file_levels,cal_types):
                log.debug('level = {}'.format(level))
                levelstr = str(level)
                log.debug('cal_type_pair = {}'.format(cal_type_pair))
                cal_type = cal_type_pair[0]
                object = cal_type_pair[1]

                rep = {"OBSDATE": obsdate,
                       "LEVEL": levelstr,
                       "CALTYPE": cal_type,
                       "OBJECT": object}

                rep["CONTENTBITMASK"] = str(contentbitmask)

                rep = dict((re.escape(k), v) for k, v in rep.items()) 
                pattern = re.compile("|".join(rep.keys()))
                query = pattern.sub(lambda m: rep[re.escape(m.group(0))], query_template)

                log.debug('query = {}'.format(query))

                cur.execute(query)
                record = cur.fetchone()

                if record is not None:
                    cId = record[0]
                    db_level = record[1]
                    db_cal_type = record[2]
                    db_object = record[3]
                    filename = '/' + record[4]        # docker run has -v /data/kpf/masters:/masters
                    checksum = record[5]
                    infobits = record[6]
                    
                    log.debug('cId = {}'.format(cId))
                    log.debug('filename = {}'.format(filename))
                    log.debug('checksum = {}'.format(checksum))


                    # See if file exists.

                    isExist = os.path.exists(filename)
                    log.debug('File existence = {}'.format(isExist))


                    # Compute checksum and compare with database value.

                    cksum = md5(filename)
                    log.debug('cksum = {}'.format(cksum))

                    if cksum == checksum:
                        log.debug("File checksum is correct...")
                    else:
                        log.debug("*** Error: File checksum is incorrect; quitting...")
                        exitcode = 64

                    cal_file_record = [cId, db_level, db_cal_type, db_object, contentbitmask, infobits, filename]
                    nearest_master_files_list.append(cal_file_record)

                    query_db_nearest_master_files_exit_code = 0

                
        # Close database cursor and then connection.

        try:
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            log.error(error)
            query_db_nearest_master_files_exit_code = 1
        finally:
            if conn is not None:
                conn.close()
                log.debug('Database connection closed.')

        exit_list = [query_db_nearest_master_files_exit_code,nearest_master_files_list]
        
        return exit_list

def query_wls(datetime, cal_type, max_cal_file_age, log):
        """
        Returns [exitcode_for_before_query,[before_master_file_record],exitcode_for_after_query,[after_master_file_record]].

        """

        # Get database connection parameters from environment.
        dbport = os.getenv('DBPORT')
        dbname = os.getenv('DBNAME')
        dbuser = os.getenv('DBUSER')
        dbpass = os.getenv('DBPASS')
        dbserver = os.getenv('DBSERVER')
        
        # hard code some parameters for WLS lookup
        # cal_type = [['WLS','autocal-lfc-all'], ['WLS', 'autocal-thar-all']]
        cal_file_level = 1  # can assume WLS is in L1 format
        contentbitmask = 3
        # max_cal_file_age = '3 days'

        # Connect to database

        try:
            conn = psycopg2.connect(host=dbserver,database=dbname,port=dbport,user=dbuser,password=dbpass)
        except:
            log.warning("Could not connect to database...")
            return [64]


        # Open database cursor.

        cur = conn.cursor()


        # Select database version.

        q1 = 'SELECT version();'
        log.debug('q1 = {}'.format(q1))
        cur.execute(q1)
        db_version = cur.fetchone()
        log.debug('PostgreSQL database version = {}'.format(db_version))


        # Check database current_user.

        q2 = 'SELECT current_user;'
        log.debug('q2 = {}'.format(q2))
        cur.execute(q2)
        for record in cur:
            log.debug('record = {}'.format(record))


        # Define query templates for database stored functions defined in database/schema/kpfOpsProcs.sql

        query_template_before =\
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

        query_template_after =\
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


        # Populate query-template dictionaries with parameters.

        log.debug('----> self.cal_file_level = {}'.format(cal_file_level))
        log.debug('----> self.contentbitmask = {}'.format(contentbitmask))
        log.debug('----> self.cal_type = {}'.format(cal_type))

        contentbitmask = contentbitmask
        level = cal_file_level
        cal_type = cal_type

        log.debug('level = {}'.format(level))
        levelstr = str(level)
        log.debug('cal_type = {}'.format(cal_type))

        object_before = cal_type[0][1] + "-eve"
        object_after = object_before.replace('eve', 'morn')

        rep_before = {"OBSDATETIME": datetime,
                      "LEVEL": levelstr,
                      "CALTYPE": 'WLS',
                      "OBJECT": object_before,
                      "MAXFILEAGE": max_cal_file_age}

        rep_after = {"OBSDATETIME": datetime,
                     "LEVEL": levelstr,
                     "CALTYPE": 'WLS',
                     "OBJECT": object_after,
                     "MAXFILEAGE": max_cal_file_age}

        rep_before["CONTENTBITMASK"] = str(contentbitmask)
        rep_after["CONTENTBITMASK"] = str(contentbitmask)


        # Execute database queries.


        exit_code_before,results_before = run_query(cur,rep_before,query_template_before, contentbitmask, log)

        exit_code_after,results_after = run_query(cur,rep_after,query_template_after, contentbitmask, log)


        # Close database cursor and then connection.

        try:
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            log.error(error)
        finally:
            if conn is not None:
                conn.close()
                log.debug('Database connection closed.')

        exit_list = [exit_code_before,results_before,exit_code_after,results_after]

        return exit_list

    
def run_query(cur,rep,query_template, contentbitmask, log):

    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    query = pattern.sub(lambda m: rep[re.escape(m.group(0))], query_template)

    log.debug('query = {}'.format(query))

    cur.execute(query)
    record = cur.fetchone()

    exit_code = 1
    results_list = []

    if record is not None:

        exit_code = 0

        cId = record[0]
        db_level = record[1]
        db_cal_type = record[2]
        db_object = record[3]
        filename = '/' + record[4]        # docker run has -v /data/kpf/masters:/masters
        checksum = record[5]
        infobits = record[6]

        log.debug('cId = {}'.format(cId))
        log.debug('filename = {}'.format(filename))
        log.debug('checksum = {}'.format(checksum))


        # See if file exists.

        isExist = os.path.exists(filename)
        log.debug('File existence = {}'.format(isExist))


        # Compute checksum and compare with database value.

        cksum = md5(filename)
        log.debug('cksum = {}'.format(cksum))

        if cksum == checksum:
            log.debug("File checksum is correct...")
        else:
            log.debug("*** Error: File checksum is incorrect; quitting...")
            exit_code = 3

        results_list = [cId, db_level, db_cal_type, db_object, contentbitmask, infobits, filename]

    return exit_code,results_list

def extract_from_db_results(results, cal_type):
    if results[0] == 1:
        return ''
    elif cal_type.lower() == 'wls':
        return [results[1][6], results[3][6]]
    else:
        cal_list = results[1]
        for cal in cal_list:
            if cal_type.lower() == cal[2].lower():
                return cal[6]
                
        cals = []
        for i in enumerate(results):
            cal_list = results

class GetCalibrations:
    """This utility looks up the associated calibrations for a given datetime and
       returns a dictionary with all calibration types.

    """
    def __init__(self, datetime, default_config_path, logger=None):
        
        # Initialize DB class
        # self.db_lookup = QueryDBNearestMasterFilesFramework(self.action, self.context)

        #Input arguments
        self.datetime = datetime   # ISO datetime string
        self.config = ConfigClass(default_config_path)
        if logger == None:
            self.log = start_logger('GetCalibrations', default_config_path)
        else:
            self.log = logger

        self.caldate_files = eval(self.config['PARAM']['date_files'])
        self.lookup_map = eval(self.config['PARAM']['lookup_map'])
        self.db_cal_types = eval(self.config['PARAM']['db_cal_types'])
        self.db_cal_file_levels = eval(self.config['PARAM']['db_cal_file_levels'])
        self.wls_cal_types = eval(self.config['PARAM']['wls_cal_types'])
        self.max_age = eval(self.config['PARAM']['max_cal_age'])
        self.defaults = eval(self.config['PARAM']['defaults'])

    def lookup(self):
        dt = datetime.strptime(self.datetime, "%Y-%m-%dT%H:%M:%S.%f")
        date_str = datetime.strftime(dt, "%Y%m%d")

        output_cals = {}
        db_results = None
        for cal,lookup in self.lookup_map.items():
            if lookup == 'file':
                filename = self.caldate_files[cal]
                df = pd.read_csv(filename, header=0, skipinitialspace=True)
                for i, row in df.iterrows():
                    start = datetime.strptime(row['UT_start_date'], "%Y-%m-%d %H:%M:%S")
                    end = datetime.strptime(row['UT_end_date'], "%Y-%m-%d %H:%M:%S")
                    if start <= dt < end:
                        try:
                            output_cals[cal] = eval(row['CALPATH'])
                        except SyntaxError:
                            output_cals[cal] = row['CALPATH']
            elif lookup == 'database' and db_results == None:
                db_results = query_database(date_str, self.db_cal_types, self.db_cal_file_levels, self.log)
                if db_results[0] == 0:
                    output_cals[cal] = extract_from_db_results(db_results, cal)
                else:
                    output_cals[cal] = self.defaults[cal]
            elif lookup == 'database' and db_results != None:
                if db_results[0] == 0:
                    output_cals[cal] = extract_from_db_results(db_results, cal)
                else:
                    output_cals[cal] = self.defaults[cal]
            elif lookup == 'wls':
                wls_results = query_wls(self.datetime, self.wls_cal_types, self.max_age, self.log)
                if wls_results[0] == 0 and wls_results[2] == 0:
                    output_cals[cal] = extract_from_db_results(wls_results, cal)
                else:
                    output_cals[cal] = self.defaults[cal]

        return output_cals

