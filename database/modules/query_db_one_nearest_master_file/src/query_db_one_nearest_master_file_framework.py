import os
import configparser as cp
from datetime import datetime, timezone
import psycopg2
import re
import hashlib
import ast

# Pipeline dependencies
from kpfpipe.logger import *
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.pipelines.fits_primitives import to_fits
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'database/modules/query_db_one_nearest_master_file/configs/default.cfg'

def md5(fname):
    hash_md5 = hashlib.md5()

    try:
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        print("*** Error: Cannot open file =",fname,"; quitting...")
        exit(65)

class QueryDBOneNearestMasterFileFramework(KPF0_Primitive):

    """
    Description:
        Queries the KPF pipeline-operations database for the one nearest-in-time master file.
        Currently, only a master file made for data earlier than the observation date is returned.

    Arguments:
        data_type (str): Type of data (e.g., KPF).
        obs_date (str): Date of observations to be processed (e.g., 20230224).

    Outputs:
        List of metadata for one master file made from data earlier than the observation date.

    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.date_dir = self.action.args[1]
        self.cal_file_level = self.action.args[2]
        self.contentbitmask = self.action.args[3]
        self.cal_type_pair = self.action.args[4]

        try:
            self.module_config_path = context.config_path['query_db_one_nearest_master_file']
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
        cal_type_pair_cfg_str = module_param_cfg.get('cal_type_pair')
        self.cal_type_pair_cfg = ast.literal_eval(cal_type_pair_cfg_str)

        self.logger.info('self.cal_file_level = {}'.format(self.cal_file_level))
        self.logger.info('self.contentbitmask = {}'.format(self.contentbitmask))
        self.logger.info('self.cal_type_pair = {}'.format(self.cal_type_pair))

        self.logger.info('self.cal_file_level_cfg = {}'.format(self.cal_file_level_cfg))
        self.logger.info('self.contentbitmask_cfg = {}'.format(self.contentbitmask_cfg))
        self.logger.info('self.cal_type_pair_cfg = {}'.format(self.cal_type_pair_cfg))

        self.logger.info('Type of self.cal_file_level_cfg = {}'.format(type(self.cal_file_level_cfg)))
        self.logger.info('Type of self.contentbitmask_cfg = {}'.format(type(self.contentbitmask_cfg)))
        self.logger.info('Type of self.cal_type_pair_cfg = {}'.format(type(self.cal_type_pair_cfg)))

    def _perform(self):

        """
        Returns [exitcode, one_nearest_master_file_list].

        """

        query_db_one_nearest_master_file_exit_code = 1
        one_nearest_master_file_list = []

        # Define absolute path to master file.

        master_file_path = '/masters' + '/' + self.date_dir + '/' + '*.fits'
        self.logger.info('master_file_path = {}'.format(master_file_path))


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
            return Arguments(64)

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

        obsdate = "'" + self.date_dir[0:4] + "-" + self.date_dir[4:6] + "-" + self.date_dir[6:8] + "'"


        # Query database for all cal_types.

        self.logger.info('----> self.cal_file_level = {}'.format(self.cal_file_level))
        self.logger.info('----> self.contentbitmask = {}'.format(self.contentbitmask))
        self.logger.info('----> self.cal_type_pair = {}'.format(self.cal_type_pair))

        contentbitmask = self.contentbitmask
        level = self.cal_file_level
        cal_type_pair = self.cal_type_pair

        self.logger.info('level = {}'.format(level))
        levelstr = str(level)
        self.logger.info('cal_type_pair = {}'.format(cal_type_pair))
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

        self.logger.info('query = {}'.format(query))

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

            self.logger.info('cId = {}'.format(cId))
            self.logger.info('filename = {}'.format(filename))
            self.logger.info('checksum = {}'.format(checksum))


            # See if file exists.

            isExist = os.path.exists(filename)
            self.logger.info('File existence = {}'.format(isExist))


            # Compute checksum and compare with database value.

            cksum = md5(filename)
            self.logger.info('cksum = {}'.format(cksum))

            if cksum == checksum:
                print("File checksum is correct...")
            else:
                print("*** Error: File checksum is incorrect; quitting...")
                exitcode = 64

            one_nearest_master_file_list = [cId, db_level, db_cal_type, db_object, contentbitmask, infobits, filename]

            query_db_one_nearest_master_file_exit_code = 0


        # Close database cursor and then connection.

        try:
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            query_db_one_nearest_master_file_exit_code = 2
        finally:
            if conn is not None:
                conn.close()
                print('Database connection closed.')

        self.logger.info('Finished {}'.format(self.__class__.__name__))

        exit_list = [query_db_one_nearest_master_file_exit_code,one_nearest_master_file_list]

        return Arguments(exit_list)
