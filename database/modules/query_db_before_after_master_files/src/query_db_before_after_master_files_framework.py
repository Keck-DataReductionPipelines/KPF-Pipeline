import os
import configparser as cp
import psycopg2
import re
import hashlib
import ast

# Pipeline dependencies
from kpfpipe.logger import *
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments

import database.modules.utils.kpf_db as db


# Global read-only variables
DEFAULT_CFG_PATH = 'database/modules/query_db_before_after_master_files/configs/default.cfg'


class QueryDBBeforeAfterMasterFilesFramework(KPF0_Primitive):

    """
    Description:
        Queries the KPF pipeline-operations database for two different master files.
        The first query is for the nearest-in-time before, and the second query is for
        the nearest-in-time after.  Each query has the same inputs, except for OBJECT,
        which will generally be different before vs. after.

    Arguments:
        data_type (str): Type of data (e.g., KPF).
        obs_date_time (str): Timestamp of observation (e.g., '2024-01-28 04:15:17').
        cal_file_level (int): Level of master file to be queried from the database (0, 1, or 2).
                              Set to -1 to fall back on settings in default.cfg file.
        contentbitmask (int): Content bit mask to be ANDed with contentbits column of CalFiles database record;
                              contentbitmask = 3 means require at least GREEN and RED CCDs master file.
        object_before (str): OBJECT for the nearest-in-time before query (e.g., 'autocal-etalon-all-eve')
        object_after (str): OBJECT for the nearest-in-time after query (e.g., 'autocal-etalon-all-morn')
        max_cal_file_age (str): Maximum startdate age of master file relative to obs_date_time,
                                expressed as a database interval, such as '2 days'.

    Outputs:
        List containing exit code, before master filename, after master filename

    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.obs_date_time = self.action.args[1]
        self.cal_type = self.action.args[2]
        self.cal_file_level = self.action.args[3]
        self.contentbitmask = self.action.args[4]
        self.object_before = self.action.args[5]
        self.object_after = self.action.args[6]
        self.max_cal_file_age = self.action.args[7]

        try:
            self.module_config_path = context.config_path['query_db_before_after_master_files']
            print("--->",self.__class__.__name__,": self.module_config_path =",self.module_config_path)
        except:
            self.module_config_path = DEFAULT_CFG_PATH

        print("{} class: self.module_config_path = {}".format(self.__class__.__name__,self.module_config_path))

        print("Starting logger...")
        self.logger = start_logger(self.__class__.__name__, self.module_config_path)

        if self.logger is not None:
            print("--->self.logger is not None...")
        else:
            print("--->self.logger is None...")

        self.logger.info('Started {}'.format(self.__class__.__name__))
        self.logger.debug('module_config_path = {}'.format(self.module_config_path))

        module_config_obj = cp.ConfigParser()
        res = module_config_obj.read(self.module_config_path)
        if res == []:
            raise IOError('failed to read {}'.format(self.module_config_path))

        module_param_cfg = module_config_obj['PARAM']

        cal_file_level_cfg_str = module_param_cfg.get('cal_file_level')
        self.cal_file_level_cfg = ast.literal_eval(cal_file_level_cfg_str)
        contentbitmask_cfg_str = module_param_cfg.get('contentbitmask')
        self.contentbitmask_cfg = ast.literal_eval(contentbitmask_cfg_str)
        cal_type_cfg_str = module_param_cfg.get('cal_type')
        self.cal_type_cfg = ast.literal_eval(cal_type_cfg_str)
        self.max_cal_file_age_cfg = module_param_cfg.get('max_cal_file_age')

        self.logger.info('self.cal_file_level = {}'.format(self.cal_file_level))
        self.logger.info('self.contentbitmask = {}'.format(self.contentbitmask))
        self.logger.info('self.cal_type = {}'.format(self.cal_type))
        self.logger.info('self.max_cal_file_age = {}'.format(self.max_cal_file_age))

        self.logger.info('self.cal_file_level_cfg = {}'.format(self.cal_file_level_cfg))
        self.logger.info('self.contentbitmask_cfg = {}'.format(self.contentbitmask_cfg))
        self.logger.info('self.cal_type_cfg = {}'.format(self.cal_type_cfg))
        self.logger.info('self.max_cal_file_age_cfg = {}'.format(self.max_cal_file_age_cfg))

        self.logger.info('Type of self.cal_file_level_cfg = {}'.format(type(self.cal_file_level_cfg)))
        self.logger.info('Type of self.contentbitmask_cfg = {}'.format(type(self.contentbitmask_cfg)))
        self.logger.info('Type of self.cal_type_cfg = {}'.format(type(self.cal_type_cfg)))
        self.logger.info('Type of self.max_cal_file_age_cfg = {}'.format(type(self.max_cal_file_age_cfg)))

        if self.cal_file_level == -1:
            self.cal_file_level = self.cal_file_level_cfg
            self.contentbitmask = self.contentbitmask_cfg
            self.cal_type = self.cal_type_cfg
            self.max_cal_file_age = self.max_cal_file_age_cfg


    def run_query(self,cur,rep,query_template):

        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        query = pattern.sub(lambda m: rep[re.escape(m.group(0))], query_template)

        self.logger.info('query = {}'.format(query))

        cur.execute(query)
        record = cur.fetchone()

        exit_code = 1

        if record is not None:

            exit_code = 0

            cId = record[0]
            db_level = record[1]
            db_cal_type = record[2]
            db_object = record[3]
            filename = '/' + record[4]        # docker run has -v /data/kpf/masters:/masters
            checksum = record[5]
            infobits = record[6]

            self.logger.info('cId = {}'.format(cId))
            self.logger.info('filename = {}'.format(filename))
            self.logger.info('checksum = {}'.format(checksum))


            # See if file exists.

            isExist = os.path.exists(filename)
            self.logger.info('File existence = {}'.format(isExist))


            # Compute checksum and compare with database value.

            cksum = db.md5(filename)
            self.logger.info('cksum = {}'.format(cksum))

            if cksum == checksum:
                print("File checksum is correct...")
            else:
                print("*** Error: File checksum is incorrect; quitting...")
                exit_code = 3

            results_list = [cId, db_level, db_cal_type, db_object, self.contentbitmask, infobits, filename]

        return exit_code,results_list


    def _perform(self):

        """
        Returns [exitcode_for_before_query,[before_master_file_record],exitcode_for_after_query,[after_master_file_record]].

        """

        # Get database connection parameters from environment.

        dbport = os.getenv('DBPORT')
        dbname = os.getenv('DBNAME')
        dbuser = os.getenv('DBUSER')
        dbpass = os.getenv('DBPASS')
        dbserver = os.getenv('DBSERVER')


        # Connect to database

        try:
            conn = psycopg2.connect(host=dbserver,database=dbname,port=dbport,user=dbuser,password=dbpass)
        except:
            print("Could not connect to database...")
            self.logger.info('Could not connect to database...')
            return Arguments([64,])


        # Open database cursor.

        cur = conn.cursor()


        # Select database version.

        q1 = 'SELECT version();'
        self.logger.info('q1 = {}'.format(q1))
        cur.execute(q1)
        db_version = cur.fetchone()
        self.logger.info('PostgreSQL database version = {}'.format(db_version))


        # Check database current_user.

        q2 = 'SELECT current_user;'
        self.logger.info('q2 = {}'.format(q2))
        cur.execute(q2)
        for record in cur:
            self.logger.info('record = {}'.format(record))


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

        self.logger.info('----> self.cal_file_level = {}'.format(self.cal_file_level))
        self.logger.info('----> self.contentbitmask = {}'.format(self.contentbitmask))
        self.logger.info('----> self.cal_type = {}'.format(self.cal_type))

        contentbitmask = self.contentbitmask
        level = self.cal_file_level
        cal_type = self.cal_type

        self.logger.info('level = {}'.format(level))
        levelstr = str(level)
        self.logger.info('cal_type = {}'.format(cal_type))

        object_before = self.object_before
        object_after = self.object_after

        rep_before = {"OBSDATETIME": self.obs_date_time,
                      "LEVEL": levelstr,
                      "CALTYPE": cal_type,
                      "OBJECT": object_before,
                      "MAXFILEAGE": self.max_cal_file_age}

        rep_after = {"OBSDATETIME": self.obs_date_time,
                     "LEVEL": levelstr,
                     "CALTYPE": cal_type,
                     "OBJECT": object_after,
                     "MAXFILEAGE": self.max_cal_file_age}

        rep_before["CONTENTBITMASK"] = str(contentbitmask)
        rep_after["CONTENTBITMASK"] = str(contentbitmask)


        # Execute database queries.


        exit_code_before,results_before = self.run_query(cur,rep_before,query_template_before)

        exit_code_after,results_after = self.run_query(cur,rep_after,query_template_after)


        # Close database cursor and then connection.

        try:
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
                print('Database connection closed.')

        self.logger.info('Finished {}'.format(self.__class__.__name__))

        exit_list = [exit_code_before,results_before,exit_code_after,results_after]


        # Return [exitcode_for_before_query,[before_master_file_record],exitcode_for_after_query,[after_master_file_record]].

        return Arguments(exit_list)
