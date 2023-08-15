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
DEFAULT_CFG_PATH = 'modules/query_db_l0_file/configs/default.cfg'


class QueryDBL0FileFramework(KPF0_Primitive):

    """
    Description:
        Queries the KPF pipeline-operations database for a record from the L0Files database table.
        Updates the header of the corresponding 2D file with quality-control information.

    Arguments:
        data_type (str): Type of data (e.g., KPF).
        rId (int): Database primary key for L0Files table that points to desired record.
        fits_filename (str): Full path/filename of 2D file to update FITS header.
        verbose (int): Verbosity flag (0 = quiet, 1 = verbose).

    Outputs:
        List of metadata in database record.

    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.rId = self.action.args[1]
        self.fits_filename = self.action.args[2]
        self.verbose = self.action.args[3]

        if self.verbose != 1:
            self.verbose = 0

        try:
            self.module_config_path = context.config_path['query_db_l0_file']
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

        verbose_cfg_str = module_param_cfg.get('verbose')
        self.verbose_cfg = ast.literal_eval(verbose_cfg_str)

        self.logger.info('self.date_type = {}'.format(self.data_type))
        self.logger.info('self.rId = {}'.format(self.rId))
        self.logger.info('self.fits_filename = {}'.format(self.fits_filename))
        self.logger.info('self.verbose = {}'.format(self.verbose))

        self.logger.info('self.verbose_cfg = {}'.format(self.verbose_cfg))

        self.logger.info('Type of self.rId = {}'.format(type(self.rId)))
        self.logger.info('Type of self.verbose = {}'.format(type(self.verbose)))


    def _perform(self):

        """
        Returns [exit_code, db_record].

        """

        exit_code = 1


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


        # Check database current_user.

        q3 = 'SELECT bit, definition, max(created) FROM l0infobits GROUP BY bit, definition ORDER BY bit, definition;'
        self.logger.info('q3 = {}'.format(q3))
        cur.execute(q3)
        bits = []
        defs = []
        i = 0
        for record in cur:
            self.logger.info('record = {}'.format(record))
            bits.append(record[0])
            defs.append(record[1])

            if self.verbose == 1:
                self.logger.info('i,bit,def = {},{}'.format(i,bits[i],defs[i]))

            i = i + 1


        # Define query template for L0Files database table.

        query_template =\
            "SELECT * " +\
            "FROM L0Files " +\
            "WHERE rId = RID;"


        # Query L0Files database table for given primary key.

        rIdstr = str(self.rId)

        rep = {"RID": rIdstr}

        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        query = pattern.sub(lambda m: rep[re.escape(m.group(0))], query_template)

        self.logger.info('query = {}'.format(query))

        cur.execute(query)
        db_record = cur.fetchone()

        if db_record is not None:
            rId = db_record[0]
            dateobs = db_record[1]
            ut = db_record[2]
            datebeg = db_record[3]
            mjdobs = db_record[4]
            exptime = db_record[5]
            progname = db_record[6]
            imtype = db_record[7]
            sciobj = db_record[8]
            calobj = db_record[9]
            skyobj = db_record[10]
            object = db_record[11]
            contentbits = db_record[12]
            infobits = db_record[13]
            filename = db_record[14]
            checksum = db_record[15]
            status = db_record[16]
            created = db_record[17]
            targname = db_record[18]
            gaiaid = db_record[19]
            twomassid = db_record[20]
            ra = db_record[21]
            dec = db_record[22]
            medgreen1 = db_record[23]
            p16green1 = db_record[24]
            p84green1 = db_record[25]
            medgreen2 = db_record[26]
            p16green2 = db_record[27]
            p84green2 = db_record[28]
            medgreen3 = db_record[29]
            p16green3 = db_record[30]
            p84green3 = db_record[31]
            medgreen4 = db_record[32]
            p16green4 = db_record[33]
            p84green4 = db_record[34]
            medred1 = db_record[35]
            p16red1 = db_record[36]
            p84red1 = db_record[37]
            medred2 = db_record[38]
            p16red2 = db_record[39]
            p84red2 = db_record[40]
            medcahk = db_record[41]
            p16cahk = db_record[42]
            p84cahk = db_record[43]
            comment = db_record[44]

            
            if self.verbose == 1:
                self.logger.info('rId = {}'.format(rId))
                self.logger.info('dateobs = {}'.format(dateobs))
                self.logger.info('ut = {}'.format(ut))
                self.logger.info('infobits = {}'.format(infobits))
                self.logger.info('ra = {}'.format(ra))
                self.logger.info('dec = {}'.format(dec))
                self.logger.info('p84red2 = {}'.format(p84red2))

            exit_code = 0


        # Close database cursor and then connection.

        try:
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            exit_code = 2
        finally:
            if conn is not None:
                conn.close()
                print('Database connection closed.')

        self.logger.info('Finished {}'.format(self.__class__.__name__))


        # Update FITS header.

        fits_obj = KPF0.from_fits(self.fits_filename,self.data_type)
        fits_obj.header['PRIMARY']['DBRID'] = (rId,'DB raw image ID')
        fits_obj.header['PRIMARY']['L0QCBITS'] = (infobits,'L0 QC bitwise flags (see defs below)')

        try:
            del fits_obj.header['PRIMARY']['L0BIT13']

        except KeyError as err:
            pass

        n_bits = len(bits)
        for i in range(0, n_bits):
            numstr = str(bits[i])
            if bits[i] < 10:
                numstr = '0' + numstr
            keyword = "L0BIT" + numstr
            value = defs[i]
            fits_obj.header['PRIMARY'][keyword] = value
            
        fits_obj.to_fits(self.fits_filename)


        # Return with arguments.
        
        exit_list = [exit_code,db_record]

        return Arguments(exit_list)
