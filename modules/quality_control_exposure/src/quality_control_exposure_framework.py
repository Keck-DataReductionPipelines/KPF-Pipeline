import os
import numpy as np
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
DEFAULT_CFG_PATH = 'modules/quality_control_exposure/configs/default.cfg'

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

def sextodec(ra,dec):
    xstr = ra.split(":")
    hr = float(xstr[0])
    min = float(xstr[1])
    sec = float(xstr[2])
    rad = (sec + 60.0 * min + 3600.0 * abs(hr)) / 3600.0 * 360.0 / 24.0;
    if rad < 0.0:
        rad = - rad
    ystr = dec.split(":")
    deg = float(ystr[0])
    dmin = float(ystr[1])
    dsec = float(ystr[2])
    decd = (dsec + 60.0 * dmin + 3600.0 * abs(deg)) / 3600.0;
    if "-" in dec:
        decd = -decd;
    return (rad, decd)

class QualityControlExposureFramework(KPF0_Primitive):

    """
    Description:
        Analyzes an L0 FITS file.  Harvests info from primary FITS header and stores it in the database L0Files table.

    Arguments:
        data_type (str): Type of data (e.g., KPF).
        l0_filename (str): Full path and filename of L0 FITS file within container.
        actual_dir (str): Prefix of actual directory outside container that maps to /data (e.g., /data/kpf)


    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.l0_filename = self.action.args[1]
        self.lev0_ffi_exts = self.action.args[2]
        self.actual_dir = self.action.args[3]

        try:
            self.module_config_path = context.config_path['quality_control_exposure']
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

        product_level_cfg_str = module_param_cfg.get('product_level')
        self.product_level_cfg = ast.literal_eval(product_level_cfg_str)

        self.logger.info('self.data_type = {}'.format(self.data_type))
        self.logger.info('self.l0_filename = {}'.format(self.l0_filename))

        self.logger.info('self.product_level_cfg = {}'.format(self.product_level_cfg))

        self.logger.info('Type of self.product_level_cfg = {}'.format(type(self.product_level_cfg)))


    def _perform(self):

        """
        Returns exitcode:
            0 = Normal
            2 = Exception raised closing database connection
           64 = Cannot connect to database
           65 = Input file does not exist
           66 = Could not insert database record
        """

        quality_control_exposure_exit_code = 0
        status = 1
        comment = ""


        # See if file exists.

        isExist = os.path.exists(self.l0_filename)
        #self.logger.info('File existence = {}'.format(isExist))

        if isExist is False:
            self.logger.info('Input file does not exist...')
            quality_control_exposure_exit_code = 65
            return Arguments(quality_control_exposure_exit_code)


        # Parse date from filename.  Assume filename has the following form: KP.20230529.69419.77.fits

        filename_parts = self.l0_filename.split(".")
        filename_date_num = int(filename_parts[1])
        #self.logger.info('filename_date_num = {}'.format(filename_date_num))


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
            quality_control_exposure_exit_code = 64
            return Arguments(quality_control_exposure_exit_code)

        # Open database cursor.

        cur = conn.cursor()


        # Select database version.

        q1 = 'SELECT version();'
        #self.logger.info('q1 = {}'.format(q1))
        cur.execute(q1)
        db_version = cur.fetchone()
        #self.logger.info('PostgreSQL database version = {}'.format(db_version))


        # Check database current_user.

        q2 = 'SELECT current_user;'
        #self.logger.info('q2 = {}'.format(q2))
        cur.execute(q2)
        for record in cur:
            #self.logger.info('record = {}'.format(record))
            pass


        # Get parameters for infobits.

        p1_bits = []
        p2_bits = []

        for i in range(0, 14):
            bit = i

            q_bit = 'SELECT param1, param2 from l0infobits where bit = ' + str(bit) + ' order by created desc limit 1;'
            #self.logger.info('q_bit = {}'.format(q_bit))
            cur.execute(q_bit)
            record = cur.fetchone()

            if record is not None:
                p1_bits.append(record[0])
                p2_bits.append(record[1])


        # Read image data object from FITS file.

        l0_file = KPF0.from_fits(self.l0_filename,self.data_type)

        infobits = 0

        for ffi in self.lev0_ffi_exts:

            image = np.array(l0_file[ffi])

            image_shape = np.shape(image)
            len_image_shape = len(image_shape)
            image_size = np.size(image)
            image_type = type(image)

            #self.logger.info('ffi,image_shape,len_image_shape,image_size,image_type = {},{},{},{},{}'.\
            #                 format(ffi,image_shape,len_image_shape,image_size,image_type))

            if len_image_shape >= 2:
                median = np.nanpercentile(image, 50)
                p16 = np.nanpercentile(image, 16)
                p84 = np.nanpercentile(image, 84)

                if ffi == "GREEN_AMP1":
                    bit_num_dead = 0
                    bit_num_satur = 7
                elif ffi == "GREEN_AMP2":
                    bit_num_dead = 1
                    bit_num_satur = 8
                elif ffi == "GREEN_AMP3":
                    bit_num_dead = 2
                    bit_num_satur = 9
                elif ffi == "GREEN_AMP4":
                    bit_num_dead = 3
                    bit_num_satur = 10
                elif ffi == "RED_AMP1":
                    bit_num_dead = 4
                    bit_num_satur = 11
                elif ffi == "RED_AMP2":
                    bit_num_dead = 5
                    bit_num_satur = 12
                elif ffi == "CA_HK":
                    bit_num_dead = 6
                    bit_num_satur = 13

                p1_bit_dead = p1_bits[bit_num_dead]
                p2_bit_dead = p2_bits[bit_num_dead]
                p1_bit_satur = p1_bits[bit_num_satur]
                p2_bit_satur = p2_bits[bit_num_satur]

                dead_pix = np.where(image < p2_bit_dead,1,0)
                dead_count = np.sum(dead_pix, dtype=int)

                satur_pix = np.where(image > p2_bit_satur,1,0)
                satur_count = np.sum(satur_pix, dtype=int)

                det_count = image_shape[0] * image_shape[1]

                #self.logger.info('ffi,dead_count,satur_count,det_count = {},{},{},{}'.format(ffi,dead_count,satur_count,det_count))

                bit_val_dead = 2**bit_num_dead
                bit_val_satur = 2**bit_num_satur

                if dead_count > p1_bit_dead * det_count / 100.0:
                    infobits = infobits | bit_val_dead                         # Set bit _dead of infobits
                    #self.logger.info('Setting bit _dead: ffi,infobits = {},{}'.format(ffi,infobits))

                if satur_count > p1_bit_satur * det_count / 100.0:
                    infobits = infobits | bit_val_satur                         # Set bit _satur of infobits
                    #self.logger.info('Setting bit _satur: ffi,infobits = {},{}'.format(ffi,infobits))

            else:
                median = 'null'
                p16 = 'null'
                p84 = 'null'

            #self.logger.info('ffi,median = {},{}'.format(ffi,median))

            if ffi == "GREEN_AMP1":
                medgreen1 = median
                p16green1 = p16
                p84green1 = p84

            elif ffi == "GREEN_AMP2":
                medgreen2 = median
                p16green2 = p16
                p84green2 = p84

            elif ffi == "GREEN_AMP3":
                medgreen3 = median
                p16green3 = p16
                p84green3 = p84

            elif ffi == "GREEN_AMP4":
                medgreen4 = median
                p16green4 = p16
                p84green4 = p84

            if ffi == "RED_AMP1":
                medred1 = median
                p16red1 = p16
                p84red1 = p84

            elif ffi == "RED_AMP2":
                medred2 = median
                p16red2 = p16
                p84red2 = p84

            elif ffi == "CA_HK":
                medcahk = median
                p16cahk = p16
                p84cahk = p84


        date_obs = l0_file.header['PRIMARY']['DATE-OBS']

        date_obs_parts = date_obs.split("-")
        date_obs_num = int(date_obs_parts[0] + date_obs_parts[1] + date_obs_parts[2])
        #self.logger.info('date_obs_num = {}'.format(date_obs_num))

        if date_obs_num != filename_date_num:          # Ensure filename date matches DATE-OBS.
            status = 0
            comment = "Filename date does not match DATE-OBS."

        ut = l0_file.header['PRIMARY']['UT']
        date_beg = l0_file.header['PRIMARY']['DATE-BEG']

        try:
            mjd_obs_str = float(l0_file.header['PRIMARY']['MJD-OBS'])

            try:
                mjd_obs = float(mjd_obs_str)

            except ValueError:
                mjd_obs = -99

        except KeyError:
            mjd_obs = "NotFound"

        try:
            exptime_str = float(l0_file.header['PRIMARY']['EXPTIME'])

            try:
                exptime = float(exptime_str)

            except ValueError:
                exptime = -99

        except KeyError:
            mjd_obs = "NotFound"

        progname = l0_file.header['PRIMARY']['PROGNAME']
        imtype = l0_file.header['PRIMARY']['IMTYPE']
        sci_obj = l0_file.header['PRIMARY']['SCI-OBJ']
        cal_obj = l0_file.header['PRIMARY']['CAL-OBJ']
        sky_obj = l0_file.header['PRIMARY']['SKY-OBJ']
        object_keyval = l0_file.header['PRIMARY']['OBJECT']

        green_found = l0_file.header['PRIMARY']['GREEN']
        red_found = l0_file.header['PRIMARY']['RED']
        ca_hk_found = l0_file.header['PRIMARY']['CA_HK']

        if green_found == 'YES':
            green_contentbit = 1
        else:
            green_contentbit = 0

        if red_found == 'YES':
            red_contentbit = 2
        else:
            red_contentbit = 0

        if ca_hk_found == 'YES':
            ca_hk_contentbit = 4
        else:
            ca_hk_contentbit = 0

        contentbits = green_contentbit | red_contentbit | ca_hk_contentbit

        try:
            targname = l0_file.header['PRIMARY']['TARGNAME']

        except KeyError:
            targname = "NotFound"

        try:
            gaiaid = l0_file.header['PRIMARY']['GAIAID']

        except KeyError:
            gaiaid = "NotFound"

        try:
            twomassid = l0_file.header['PRIMARY']['2MASSID']

        except KeyError:
            twomassid = "NotFound"

        try:
            ra = l0_file.header['PRIMARY']['RA']
            dec = l0_file.header['PRIMARY']['DEC']
            (rad,decd) = sextodec(ra,dec)
        except KeyError:
            rad = None
            decd = None


        try:
            medgreen1_str = str(medgreen1)
            p16green1_str = str(p16green1)
            p84green1_str = str(p84green1)
        except KeyError:
            medgreen1_str = "null"
            p16green1_str = "null"
            p84green1_str = "null"

        try:
            medgreen2_str = str(medgreen2)
            p16green2_str = str(p16green2)
            p84green2_str = str(p84green2)
        except KeyError:
            medgreen2_str = "null"
            p16green2_str = "null"
            p84green2_str = "null"

        try:
            medgreen3_str = str(medgreen3)
            p16green3_str = str(p16green3)
            p84green3_str = str(p84green3)
        except KeyError:
            medgreen3_str = "null"
            p16green3_str = "null"
            p84green3_str = "null"

        try:
            medgreen4_str = str(medgreen4)
            p16green4_str = str(p16green4)
            p84green4_str = str(p84green4)
        except KeyError:
            medgreen4_str = "null"
            p16green4_str = "null"
            p84green4_str = "null"

        try:
            medred1_str = str(medred1)
            p16red1_str = str(p16red1)
            p84red1_str = str(p84red1)
        except KeyError:
            medred1_str = "null"
            p16red1_str = "null"
            p84red1_str = "null"

        try:
            medred2_str = str(medred2)
            p16red2_str = str(p16red2)
            p84red2_str = str(p84red2)
        except KeyError:
            medred2_str = "null"
            p16red2_str = "null"
            p84red2_str = "null"

        try:
            medcahk_str = str(medcahk)
            p16cahk_str = str(p16cahk)
            p84cahk_str = str(p84cahk)
        except KeyError:
            medcahk_str = "null"
            p16cahk_str = "null"
            p84cahk_str = "null"


        filename = self.l0_filename.replace("/data",self.actual_dir,1)
        #self.logger.info('filename = {}'.format(filename))

        cksum = md5(self.l0_filename)
        #self.logger.info('cksum = {}'.format(cksum))


        # Define query template for insert statement.

        columns = 'dateobs,ut,datebeg,mjdobs,exptime,progname,imtype,sciobj,calobj,' +\
                  'skyobj,"object",contentbits,infobits,filename,checksum,status,' +\
                  'targname,gaiaid,twomassid,ra,dec,medgreen1,p16green1,p84green1,' +\
                  'medgreen2,p16green2,p84green2,medgreen3,p16green3,p84green3,' +\
                  'medgreen4,p16green4,p84green4,medred1,p16red1,p84red1,' +\
                  'medred2,p16red2,p84red2,medcahk,p16cahk,p84cahk,comment'

        values = "cast('DATEOBS' as date)," +\
                 "cast('UT' as time without time zone)," +\
                 "cast('DATEBEG' as timestamp without time zone)," +\
                 "cast(MJDOBS as double precision)," +\
                 "cast(EXPTIME as real)," +\
                 "cast('PROGNAME' as character varying)," +\
                 "cast('IMTYPE' as character varying)," +\
                 "cast('SCIOBJ' as character varying)," +\
                 "cast('CALOBJ' as character varying)," +\
                 "cast('SKYOBJ' as character varying)," +\
                 "cast('OBJECT' as character varying)," +\
                 "cast(CONTENTBITS as integer)," +\
                 "cast(INFOBITS as bigint)," +\
                 "cast('FILENAME' as character varying)," +\
                 "cast('CHECKSUM' as character varying)," +\
                 "cast(STATUS as integer)," +\
                 "cast(TARGNAME as character varying)," +\
                 "cast(GAIAID as character varying)," +\
                 "cast(TWOMASSID as character varying)," +\
                 "cast(RAD as double precision)," +\
                 "cast(DECD as double precision)," +\
                 "cast(MEDGREEN1 as real)," +\
                 "cast(P16GREEN1 as real)," +\
                 "cast(P84GREEN1 as real)," +\
                 "cast(MEDGREEN2 as real)," +\
                 "cast(P16GREEN2 as real)," +\
                 "cast(P84GREEN2 as real)," +\
                 "cast(MEDGREEN3 as real)," +\
                 "cast(P16GREEN3 as real)," +\
                 "cast(P84GREEN3 as real)," +\
                 "cast(MEDGREEN4 as real)," +\
                 "cast(P16GREEN4 as real)," +\
                 "cast(P84GREEN4 as real)," +\
                 "cast(MEDRED1 as real)," +\
                 "cast(P16RED1 as real)," +\
                 "cast(P84RED1 as real)," +\
                 "cast(MEDRED2 as real)," +\
                 "cast(P16RED2 as real)," +\
                 "cast(P84RED2 as real)," +\
                 "cast(MEDCAHK as real)," +\
                 "cast(P16CAHK as real)," +\
                 "cast(P84CAHK as real)," +\
                 "cast(COMMENT as character varying)"


        query_insert_template =\
            "INSERT INTO L0Files (" + columns + ") " +\
            "VALUES (" + values + ") RETURNING rid;"


        # Substitute values into template for record insert.

        rep = {"DATEOBS": date_obs,
               "UT": ut,
               "DATEBEG": date_beg,
               "IMTYPE": imtype,
               "SCIOBJ": sci_obj,
               "CALOBJ": cal_obj,
               "SKYOBJ": sky_obj,
               "OBJECT": object_keyval}

        if progname == '':
            rep["PROGNAME"] = "blank"
        else:
            rep["PROGNAME"] = progname

        if mjd_obs == "NotFound":
            rep["MJDOBS"] = -99
        else:
            rep["MJDOBS"] = str(mjd_obs)

        if exptime == "NotFound":
            rep["EXPTIME"] = -99
        else:
            rep["EXPTIME"] = str(exptime)

        rep["CONTENTBITS"] = str(contentbits)
        rep["INFOBITS"] = str(infobits)
        rep["FILENAME"] = filename
        rep["CHECKSUM"] = cksum
        rep["STATUS"] = str(status)

        if targname == "NotFound":
            rep["TARGNAME"] = "null"
        else:
            rep["TARGNAME"] = "'" + targname + "'"

        if gaiaid == "NotFound":
            rep["GAIAID"] = "null"
        else:
            rep["GAIAID"] = "'" + gaiaid + "'"

        if twomassid == "NotFound":
            rep["TWOMASSID"] = "null"
        else:
            rep["TWOMASSID"] = "'" + twomassid + "'"

        if rad is None:
            rep["RAD"] = 'null'
        else:
            rep["RAD"] = str(rad)

        if decd is None:
            rep["DECD"] = 'null'
        else:
            rep["DECD"] = str(decd)

        rep["MEDGREEN1"] = medgreen1_str
        rep["P16GREEN1"] = p16green1_str
        rep["P84GREEN1"] = p84green1_str
        rep["MEDGREEN2"] = medgreen2_str
        rep["P16GREEN2"] = p16green2_str
        rep["P84GREEN2"] = p84green2_str
        rep["MEDGREEN3"] = medgreen3_str
        rep["P16GREEN3"] = p16green3_str
        rep["P84GREEN3"] = p84green3_str
        rep["MEDGREEN4"] = medgreen4_str
        rep["P16GREEN4"] = p16green4_str
        rep["P84GREEN4"] = p84green4_str

        rep["MEDRED1"] = medred1_str
        rep["P16RED1"] = p16red1_str
        rep["P84RED1"] = p84red1_str
        rep["MEDRED2"] = medred2_str
        rep["P16RED2"] = p16red2_str
        rep["P84RED2"] = p84red2_str

        rep["MEDCAHK"] = medcahk_str
        rep["P16CAHK"] = p16cahk_str
        rep["P84CAHK"] = p84cahk_str

        if comment == '':
            rep["COMMENT"] = 'null'
        else:
            rep["COMMENT"] = "'" + comment + "'"

        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        query = pattern.sub(lambda m: rep[re.escape(m.group(0))], query_insert_template)

        self.logger.info('query = {}'.format(query))

        try:
            cur.execute(query)
            rid = cur.fetchone()
            self.logger.info('PostgreSQL database L0Image ID: rid = {}'.format(rid))

        except (Exception, psycopg2.DatabaseError) as error:
            self.logger.info('*** Error inserting record ({}); skipping...'.format(error))
            quality_control_exposure_exit_code = 66


        # Commit transaction.

        conn.commit()


        # Close database cursor and then connection.

        try:
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            self.logger.info('*** Error closing database connection ({}); skipping...'.format(error))
            quality_control_exposure_exit_code = 2
        finally:
            if conn is not None:
                conn.close()

        self.logger.info('Finished {}'.format(self.__class__.__name__))

        return Arguments(quality_control_exposure_exit_code)
