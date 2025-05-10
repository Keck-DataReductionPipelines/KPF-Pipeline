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

import database.modules.utils.kpf_db as db


# Global read-only variables
DEFAULT_CFG_PATH = 'database/modules/query_db_one_l0_file/configs/default.cfg'


class QueryDBOneL0FileFramework(KPF0_Primitive):

    """
    Description:
        Queries the KPF pipeline-operations database for the one L0 file for
        the specified image type, observation date, and contentbitmask.

    Arguments:
        data_type (str): Type of data (e.g., KPF).
        im_type (str): Type of image/exposure (e.g., Arclamp)
        obs_date (str): Date of observation to be queried for (e.g., 20230224).
        contentbitmask (int): Content bit mask to be ANDed with contentbits column of CalFiles database record;
                              contentbitmask = 3 means require at least GREEN and RED CCDs master file.
        my_param (int): Placeholder for parameter to be added in the future.

    Outputs:
        List of metadata for one arclamp file made earlier than the observation date.

    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.im_type = self.action.args[1]
        self.obs_date = self.action.args[2]
        self.contentbitmask = self.action.args[3]
        self.my_param = -1

        try:
            self.module_config_path = context.config_path['query_db_one_l0_file']
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

        my_param_cfg_str = module_param_cfg.get('my_param')
        self.my_param_cfg = ast.literal_eval(my_param_cfg_str)

        self.logger.info('self.im_type = {}'.format(self.im_type))
        self.logger.info('self.obs_date = {}'.format(self.obs_date))
        self.logger.info('self.contentbitmask = {}'.format(self.contentbitmask))
        self.logger.info('self.my_param = {}'.format(self.my_param))

        self.logger.info('self.my_param_cfg = {}'.format(self.my_param_cfg))

        self.logger.info('Type of self.my_param_cfg = {}'.format(type(self.my_param_cfg)))

        if self.my_param == -1:
            self.my_param = self.my_param_cfg


    def _perform(self):

        """
        Returns [exitcode, one_l0_file_list].

        """

        query_db_one_l0_file_exit_code = 1
        one_l0_file_list = []


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


        # Define query template.

        query_template =\
            "select rid,mjdobs,imtype,contentbits,object,infobits,filename,checksum from L0files " +\
            "where imtype = 'TEMPLATE_IMTYPE' " +\
            "and dateobs = 'TEMPLATE_DATEOBS' " +\
            "and cast((contentbits & TEMPLATE_CONTENTBITMASK) as integer) = TEMPLATE_CONTENTBITMASK " +\
            "limit 1;"


        # Query database for all cal_types.

        self.logger.info('----> self.im_type = {}'.format(self.im_type))
        self.logger.info('----> self.contentbitmask = {}'.format(self.contentbitmask))
        self.logger.info('----> self.obs_date = {}'.format(self.obs_date))

        im_type = self.im_type
        contentbitmask = self.contentbitmask
        obs_date = self.obs_date

        rep = {"TEMPLATE_DATEOBS": obs_date}

        rep["TEMPLATE_IMTYPE"] = im_type
        rep["TEMPLATE_CONTENTBITMASK"] = str(contentbitmask)

        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        query = pattern.sub(lambda m: rep[re.escape(m.group(0))], query_template)

        self.logger.info('query = {}'.format(query))

        cur.execute(query)
        record = cur.fetchone()

        if record is not None:
            rId = record[0]
            mjdobs = record[1]
            imtype = record[2]
            contentbits = record[3]
            object = record[4]
            infobits = record[5]
            filename = record[6]
            checksum = record[7]

            self.logger.info('rId = {}'.format(rId))
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
                exitcode = 64

            one_l0_file_list = [rId, mjdobs, imtype, contentbits, object, infobits, filename, checksum]

            query_db_one_l0_file_exit_code = 0


        # Close database cursor and then connection.

        try:
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            query_db_one_l0_file_exit_code = 2
        finally:
            if conn is not None:
                conn.close()
                print('Database connection closed.')

        self.logger.info('Finished {}'.format(self.__class__.__name__))

        exit_list = [query_db_one_l0_file_exit_code,one_l0_file_list]

        return Arguments(exit_list)
