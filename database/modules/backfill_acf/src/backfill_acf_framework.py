import os
from os.path import exists
import numpy as np
import numpy.ma as ma
import configparser as cp
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
DEFAULT_CFG_PATH = 'database/modules/backfill_acf/configs/default.cfg'

debug = 0


#
# Global methods
#

def md5(fname):
    hash_md5 = hashlib.md5()

    try:
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        self.logger.info('*** Error: Cannot open file ({}); quitting...'.format(fname))
        exit(65)


class BackfillAcfFramework(KPF0_Primitive):

    """
    Description:
        Glean required keywords GRACFFLN and RDACFFLN from FITS header for updating L0Files database table.

    Arguments:
        data_type (str): Type of data (e.g., KPF).
        l0_filename (str): Full path and filename of L0 FITS file within container.
        n_sigma (float): Number of sigmas for overscan-value outlier rejection (e.g., 3.0).
        rId (float): Primary database key of L0 FITS file in L0Files database record.


    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.l0_filename = self.action.args[1]
        self.rId = self.action.args[2]

        try:
            self.module_config_path = context.config_path['backfill_acf']
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

        backfill_repopulate_db_recs_cfg_str = module_param_cfg.get('backfill_repopulate_db_recs')
        self.backfill_repopulate_db_recs_cfg = ast.literal_eval(backfill_repopulate_db_recs_cfg_str)
        self.backfill_repopulate_db_query_template = module_param_cfg.get('backfill_repopulate_db_query_template')

        self.logger.info('self.data_type = {}'.format(self.data_type))
        self.logger.info('self.l0_filename = {}'.format(self.l0_filename))
        self.logger.info('self.rId = {}'.format(self.rId))

        self.logger.info('self.backfill_repopulate_db_recs_cfg = {}'.format(self.backfill_repopulate_db_recs_cfg))

        self.logger.info('Type of self.backfill_repopulate_db_recs_cfg = {}'.format(type(self.backfill_repopulate_db_recs_cfg)))

        self.logger.info('self.backfill_repopulate_db_query_template = {}'.format(self.backfill_repopulate_db_query_template))



    def computeBackfillAcfForSingleL0File(self,input_rid,input_filename,cur):

        backfill_acf_exit_code = 0


        # See if file exists.

        isExist = os.path.exists(input_filename)
        #self.logger.info('File existence = {}'.format(isExist))

        if isExist is False:
            self.logger.info('Input file does not exist...')
            backfill_acf_exit_code = 65
            return backfill_acf_exit_code


        # Read image data object from L0 FITS file.

        hdul_input = KPF0.from_fits(input_filename,self.data_type)


        try:
            gracffln = hdul_input.header['PRIMARY']['GRACFFLN']
        except KeyError:
            gracffln = "null"

        try:
            rdacffln = hdul_input.header['PRIMARY']['RDACFFLN']
        except KeyError:
            rdacffln = "null"


        # Define query template for database stored function that executes insert/update SQL statement.

        query_template =\
            "update L0Files " +\
            "set gracffln = cast(GRACFFLN as character varying), " +\
            "rdacffln = cast(RDACFFLN as character varying) " +\
            "where rid = cast(RID as integer);"


        # Substitute values into template for registering database record.

        rIdstr = str(input_rid)

        rep = {"RID": rIdstr}

        if gracffln == '':
            rep["GRACFFLN"] = 'null'
        elif gracffln != 'null':
            rep["GRACFFLN"] = "'" + gracffln + "'"

        if rdacffln == '':
            rep["RDACFFLN"] = 'null'
        elif rdacffln != 'null':
            rep["RDACFFLN"] = "'" + rdacffln + "'"

        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        query = pattern.sub(lambda m: rep[re.escape(m.group(0))], query_template)

        self.logger.info('query = {}'.format(query))


        # Execute query.

        try:
            cur.execute(query)

            try:
                for record in cur:
                    self.logger.info(record)
            except:
                    self.logger.info("Nothing returned from database stored function; continuing...")

        except (Exception, psycopg2.DatabaseError) as error:
            self.logger.info('*** Error inserting record ({}); skipping...'.format(error))
            backfill_acf_exit_code = 66


        return backfill_acf_exit_code


    def _perform(self):

        """
        Perform the following steps:
        1. Connect to pipeline-operations database
        2. Glean required keywords from FITS header for updating L0Files database table.
           a. if self.backfill_repopulate_db_recs_cfg == 0, do just for the L0 FITS file 
              specified in the recipe config file.  The L0 FITS file(s) must exist and MD5 checksum
              stored in datbase is not checked in this case.
           b. if self.backfill_repopulate_db_recs_cfg == 1, do for the L0 FITS files
              returned from the database query specified in the module-specific config file.
              In this case, the L0 FITS file(s) must exist and have MD5 checksum matching
              that stored in L0Files database record.
        3. Disconnect from database.


        Returns exitcode:
            0 = Normal
            2 = Exception raised closing database connection
           64 = Cannot connect to database
           65 = Input file does not exist
           66 = Could not insert database record
        """
            
        backfill_acf_exit_code = 0


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
            self.logger.info('Could not connect to database...')
            backfill_acf_exit_code = 64
            return Arguments(backfill_acf_exit_code)


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
            pass


        ###########################################################################
        # Perform calculations for database record(s).
        ###########################################################################


        if self.backfill_repopulate_db_recs_cfg == 0:


            # Do a single L0 FITS file.

            backfill_acf_exit_code = self.computeBackfillAcfForSingleL0File(self.rId,self.l0_filename,cur)


            # Commit transaction.

            conn.commit()

        else:


            # Use self.l0_filename to get the inside-container path

            filename_match = re.match(r"(.+?)(\d\d\d\d\d\d\d\d/)(KP.+)", self.l0_filename)

            try:
                filename_path_prefix_from_config_file = filename_match.group(1)
                filename_path_date = filename_match.group(2)
                filename_only = filename_match.group(3)

                print("-------------------> filename_path_prefix_from_config_file =",filename_path_prefix_from_config_file)
                print("-------------------> filename_path_date =",filename_path_date)
                print("-------------------> filename_only =",filename_only)

            except:
                print("-------------------> No filename match found")


            # Query for all frames, up to countmax.

            countmax = 1000000

            count = 0
            
            query_template = self.backfill_repopulate_db_query_template

            query = query_template

            print("query to get list of files = ",query)


            self.logger.info('query = {}'.format(query))


            # Execute query.

            try:
                cur.execute(query)

            except (Exception, psycopg2.DatabaseError) as error:
                self.logger.info('*** Error querying records ({}); skipping...'.format(error))
                backfill_acf_exit_code = 66


            rId_list = []
            filename_list = []
            checksum_list = []
                
            for record in cur:
                self.logger.info('record = {}'.format(record))

                if record is not None:
                    rId = record[0]
                    filename = record[1]
                    checksum = record[2]

                    rId_list.append(rId)
                    filename_list.append(filename)
                    checksum_list.append(checksum)
                    
            self.logger.info('Number of files returned by database query = {}'.format(len(rId_list)))

            for i in range(0,len(rId_list)):

                rId = rId_list[i]
                filename = filename_list[i]
                checksum = checksum_list[i]


                # Replace the outside-container path with the inside-container path.

                filename_match = re.match(r"(.+?)(\d\d\d\d\d\d\d\d/)(KP.+)", filename)

                try:
                    filename_path_prefix = filename_match.group(1)
                    filename_path_date = filename_match.group(2)
                    filename_only = filename_match.group(3)

                    print("-------------------> filename_path_prefix =",filename_path_prefix)
                    print("-------------------> filename_path_date =",filename_path_date)
                    print("-------------------> filename_only =",filename_only)

                    filename = filename_path_prefix_from_config_file + filename_path_date + filename_only
                    print("-------------------> filename =",filename)

                except:
                    print("-------------------> No filename match found")
                    continue

                
                # See if file exists.

                isExist = os.path.exists(filename)

                if debug == 1:
                    print('File,existence = {},{}'.format(filename,isExist))

                if isExist == False:
                    self.logger.info('*** Error: File does not exist ({}); skipping...'.format(filename))
                    continue


                # Compute checksum and compare with database value.

                cksum = md5(filename)

                if debug == 1:
                    print('cksum = {}'.format(cksum))

                if cksum == checksum:
                    if debug == 1:
                        print("File checksum is correct...")
                else:
                    self.logger.info('*** Error: File checksum is incorrect ({}); skipping...'.format(filename))
                    continue


 
                self.logger.info('rId,filename = {},{}'.format(rId,filename))


                # Compute read noise for a single L0 FITS file.

                backfill_acf_exit_code = self.computeBackfillAcfForSingleL0File(rId,filename,cur)
                self.logger.info('backfill_acf_exit_code returned from method self.computeBackfillAcfForSingleL0File = {}'.format(backfill_acf_exit_code))

                if backfill_acf_exit_code != 0:
                    break


                # Commit transaction.

                conn.commit()


                # Increment counter.
            
                count = count + 1

                print("count =",count)
                self.logger.info('count,countmax = {},{}'.format(count,countmax))

                if count >= countmax:
                    break;


        ###########################################################################
        ###########################################################################


        # Close database cursor and then connection.

        try:
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            self.logger.info('*** Error closing database connection ({}); skipping...'.format(error))
            backfill_acf_exit_code = 2
        finally:
            if conn is not None:
                conn.close()

        self.logger.info('Finished {}'.format(self.__class__.__name__))

        return Arguments(backfill_acf_exit_code)
