import os
from os.path import exists
import numpy as np
import configparser as cp
import psycopg2
import ast

# Pipeline dependencies
from kpfpipe.logger import *
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.pipelines.fits_primitives import to_fits
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/var_exts/configs/default.cfg'

debug = 1


class VarExtsFramework(KPF0_Primitive):

    """
    Description:
        Input L0 filename and database primary key rId for the L0Files database table.
        Select the record from the ReadNoise database table, and square for the read-noise variances.
        Gather all the other variances, sum them all, and write the resulting total variance images
        to FITS extensions ['GREEN_VAR','RED_VAR'] in the associated 2D FITS file.

    Arguments:
        data_type (str): Type of data (e.g., KPF).
        l0_filename (str): Full path and filename of L0 FITS file within container.
        masterbias_path (str): Input master bias.
        masterdark_path (str): Input master dark.
        masterflat_path (str): Input master flat.
        rId (int): Primary database key of L0 FITS file in L0Files database record.
    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.l0_filename = self.action.args[1]
        self.masterbias_path = self.action.args[2]
        self.masterdark_path = self.action.args[3]
        self.masterflat_path = self.action.args[4]
        self.rId = self.action.args[5]

        try:
            self.module_config_path = context.config_path['var_exts']
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

        rn_flag_cfg_str = module_param_cfg.get('rn_flag')
        self.rn_flag_cfg = ast.literal_eval(rn_flag_cfg_str)

        self.logger.info('self.data_type = {}'.format(self.data_type))
        self.logger.info('self.l0_filename = {}'.format(self.l0_filename))
        self.logger.info('self.masterbias_path = {}'.format(self.masterbias_path))
        self.logger.info('self.masterdark_path = {}'.format(self.masterdark_path))
        self.logger.info('self.masterflat_path = {}'.format(self.masterflat_path))
        self.logger.info('self.rId = {}'.format(self.rId))

        self.logger.info('self.rn_flag_cfg = {}'.format(self.rn_flag_cfg))

        self.logger.info('Type of self.rn_flag_cfg = {}'.format(type(self.rn_flag_cfg)))




    def select_read_noise(self,input_rid):

        var_exts_exit_code = 0




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
            var_exts_exit_code = 64
            return Arguments(var_exts_exit_code)


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
        ###########################################################################


        # Execute query.

        query = "SELECT rngreen1,rngreen2,rngreen3,rngreen4,rnred1,rnred2,rnred3,rnred4 from ReadNoise where rId = " +\
            str(self.rId) + ";"

        self.logger.info('query = {}'.format(query))

        try:
            cur.execute(query)
            record = cur.fetchone()

            if record is not None:
                rngreen1 = record[0]
                rngreen2 = record[1]
                rngreen3 = record[2]
                rngreen4 = record[3]
                rnred1 = record[4]
                rnred2 = record[5]
                rnred3 = record[6]
                rnred4 = record[7]

                self.logger.info(record)
            else:
                self.logger.info("Database record not found; skipping...")
                var_exts_exit_code = 66
                return var_exts_exit_code

        except (Exception, psycopg2.DatabaseError) as error:
            self.logger.info('*** Error selecting record ({}); skipping...'.format(error))
            var_exts_exit_code = 67
            return var_exts_exit_code

        ###########################################################################
        ###########################################################################


        # Close database cursor and then connection.

        try:
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            self.logger.info('*** Error closing database connection ({}); skipping...'.format(error))
            var_exts_exit_code = 2
        finally:
            if conn is not None:
                conn.close()



        self.logger.info('rngreen1 = {}'.format(rngreen1))
        self.logger.info('rngreen2 = {}'.format(rngreen2))
        self.logger.info('rngreen3 = {}'.format(rngreen3))
        self.logger.info('rngreen4 = {}'.format(rngreen4))

        self.logger.info('rnred1 = {}'.format(rnred1))
        self.logger.info('rnred2 = {}'.format(rnred2))
        self.logger.info('rnred3 = {}'.format(rnred3))
        self.logger.info('rnred4 = {}'.format(rnred4))


        return var_exts_exit_code,rngreen1,rngreen2,rngreen3,rngreen4,rnred1,rnred2,rnred3,rnred4


    def assemble_read_noise_var_images(self,rngreen1,rngreen2,rngreen3,rngreen4,rnred1,rnred2,rnred3,rnred4):

        if rngreen3 is None:
            num_amps_green = 2
        else:
            num_amps_green = 4

        if rnred3 is None:
            num_amps_red = 2
        else:
            num_amps_red = 4



        # Read image data object from 2D FITS file.

        fits_filename = self.l0_filename
        fits_filename = fits_filename.replace('L0', '2D')
        fits_filename = fits_filename.replace('.fits', '_2D.fits')

        fits_filename_exists = exists(fits_filename)
        if not fits_filename_exists:
            self.logger.info('*** 2D file does not exist ({}); skipping...'.format(fits_filename))
            return

        hdul_input = KPF0.from_fits(fits_filename,self.data_type)
        exp_time = float(hdul_input.header['PRIMARY']['EXPTIME'])

        if debug == 1:
            print("exp_time = {}".format(exp_time))

        exts = ['GREEN_CCD','RED_CCD']
        rngreenvarimg = None
        rnredvarimg = None

        for ext in exts:

            try:
                naxis1 = hdul_input.header[ext]["NAXIS1"]
            except:
                continue

            try:
                naxis2 = hdul_input.header[ext]["NAXIS2"]
            except:
                continue

            if debug == 1:
                print("ext,naxis1,naxis2 = {},{},{}".\
                    format(ext,naxis1,naxis2))

            if 'GREEN' in ext:
                num_amps = num_amps_green
                rn1 = rngreen1
                rn2 = rngreen2
                rn3 = rngreen3
                rn4 = rngreen4
            else:
                num_amps = num_amps_red
                rn1 = rnred1
                rn2 = rnred2
                rn3 = rnred3
                rn4 = rnred4

            if num_amps == 2:
                ny = naxis2
                nx = int(naxis1 / 2)
                var1 = rn1 * rn1
                var2 = rn2 * rn2
                amp1 = np.full((ny,nx),var1,dtype=float)
                amp2 = np.full((ny,nx),var2,dtype=float)
                var_img = np.concatenate((amp1, amp2), axis=1)
            else:
                ny = int(naxis2 / 2)
                nx = int(naxis1 / 2)
                var1 = rn1 * rn1
                var2 = rn2 * rn2
                var3 = rn3 * rn3
                var4 = rn4 * rn4
                amp1 = np.full((ny,nx),var1,dtype=float)
                amp2 = np.full((ny,nx),var2,dtype=float)
                amp3 = np.full((ny,nx),var3,dtype=float)
                amp4 = np.full((ny,nx),var4,dtype=float)
                img_top = np.concatenate((amp1, amp2), axis=1)
                img_bot = np.concatenate((amp3, amp4), axis=1)
                var_img = np.concatenate((img_top, img_bot), axis=0)

            if 'GREEN' in ext:
                rngreenvarimg = var_img
            else:
                rnredvarimg = var_img

        return exp_time,rngreenvarimg,rnredvarimg


    def assemble_var_images(self, fits_filename):


        # Read image data object from master file.

        fits_filename_exists = exists(fits_filename)
        if not fits_filename_exists:
            self.logger.info('*** Master file does not exist ({}); skipping...'.format(fits_filename))
            return

        hdul_input = KPF0.from_fits(fits_filename,self.data_type)

        exts = ['GREEN_CCD_UNC','RED_CCD_UNC']
        greenvarimg = None
        redvarimg = None

        for ext in exts:

            try:
                unc_img = np.array(hdul_input[ext])
            except:
                continue

            var_img = unc_img * unc_img

            if 'GREEN' in ext:
                greenvarimg = var_img
            else:
                redvarimg = var_img

        return greenvarimg,redvarimg


    def assemble_ccd_images(self):

        # Read image data object from 2D FITS file.

        fits_filename = self.l0_filename
        fits_filename = fits_filename.replace('L0', '2D')
        fits_filename = fits_filename.replace('.fits', '_2D.fits')

        fits_filename_exists = exists(fits_filename)
        if not fits_filename_exists:
            self.logger.info('*** 2D file does not exist ({}); skipping...'.format(fits_filename))
            return

        hdul_input = KPF0.from_fits(fits_filename,self.data_type)

        exts = ['GREEN_CCD','RED_CCD']
        greenccdimg = None
        redccdimg = None

        for ext in exts:

            try:
                ccd_img = np.array(hdul_input[ext])
            except:
                continue

            ccd_img = np.where(ccd_img >= 0.0, ccd_img, 0.0)        # Ensure the photon noise is positive.

            if 'GREEN' in ext:
                greenccdimg = ccd_img
            else:
                redccdimg = ccd_img

        return greenccdimg,redccdimg


    def write_var_exts(self,greenvarimg,redvarimg):

        fits_filename = self.l0_filename
        fits_filename = fits_filename.replace('L0', '2D')
        fits_filename = fits_filename.replace('.fits', '_2D.fits')

        fits_filename_exists = exists(fits_filename)
        if not fits_filename_exists:
            self.logger.info('*** 2D File does not exist ({}); skipping...'.format(fits_filename))
            return

        fits_obj = KPF0.from_fits(fits_filename,self.data_type)

        exts = ['GREEN_VAR','RED_VAR']

        for ext in exts:

            if 'GREEN' in ext:
                if greenvarimg is None:
                    continue
                else:
                    img = np.array(greenvarimg)
            else:
                if redvarimg is None:
                    continue
                else:
                    img = np.array(redvarimg)

            img_shape = np.shape(img)
            self.logger.info('--->ext,img_shape = {},{}'.format(ext,img_shape))

            fits_obj[ext] = img.astype(np.float32)
            fits_obj.header[ext]['BUNIT'] = ('electrons squared','Units of variance')

        # Remove any AMP extensions (which are automatically re-added as empty extensions for L0 FITS objects).

        del_ext_list = ['GREEN_AMP1','GREEN_AMP2','GREEN_AMP3','GREEN_AMP4','RED_AMP1','RED_AMP2','RED_AMP3','RED_AMP4']
        for ext in del_ext_list:
            try:
                fits_obj.del_extension(ext)
            except:
                pass

        fits_obj.to_fits(fits_filename)

        return


    def _perform(self):

        """
        Perform the following steps:
        1. Connect to pipeline-operations database
        2. Perform calculations for record(s) in the .
           a. if self.rn_flag_cfg == 0, select record from ReadNoise database table
              for given rId.
           b. if self.rn_flag_cfg == 1, skip this step.
        3. Disconnect from database.

        Returns exitcode:
            0 = Normal
            2 = Exception raised closing database connection
           64 = Cannot connect to database
           65 = Input L0 file does not exist
           66 = Database record for rId not found
           67 = Could not select database record
           68 = Input master bias does not exist
           69 = Input master dark does not exist
           70 = Input master flat does not exist
        """

        var_exts_exit_code = 0

        # See if input L0 file exists.

        isExist = os.path.exists(self.l0_filename)
        self.logger.info('File existence = {}'.format(isExist))

        if isExist is False:
            self.logger.info('Input L0 file does not exist ({})...'.format(self.l0_filename))
            var_exts_exit_code = 65
            return var_exts_exit_code

        # See if input master files exist.

        isExist1 = os.path.exists(self.masterbias_path)
        self.logger.info('File existence = {}'.format(isExist1))

        if isExist1 is False:
            self.logger.info('Input master file does not exist ({})...'.format(self.masterbias_path))
            var_exts_exit_code = 68
            return var_exts_exit_code

        isExist2 = os.path.exists(self.masterdark_path)
        self.logger.info('File existence = {}'.format(isExist2))

        if isExist2 is False:
            self.logger.info('Input master file does not exist ({})...'.format(self.masterdark_path))
            var_exts_exit_code = 69
            return var_exts_exit_code

        isExist3 = os.path.exists(self.masterflat_path)
        self.logger.info('File existence = {}'.format(isExist3))

        if isExist3 is False:
            self.logger.info('Input master file does not exist ({})...'.format(self.masterflat_path))
            var_exts_exit_code = 70
            return var_exts_exit_code


        ###########################################################################
        # Perform calculation for read noise.
        ###########################################################################


        if self.rn_flag_cfg == 0:

            # Select read noise for a single L0 FITS file via database query.

            var_exts_exit_code,rngreen1,rngreen2,rngreen3,rngreen4,rnred1,rnred2,rnred3,rnred4 =\
                self.select_read_noise(self.rId)

        else:
            rngreen1 = 4.0
            rngreen2 = 4.0
            rngreen3 = 4.0
            rngreen4 = 4.0
            rnred1 = 4.0
            rnred2 = 4.0
            rnred3 = 4.0
            rnred4 = 4.0

        # Assemble CCD images.

        greenccdimg,redccdimg = self.assemble_ccd_images()

        # Assemble read-noise variance images.

        exp_time,rn_greenvarimg,rn_redvarimg = \
            self.assemble_read_noise_var_images(rngreen1,rngreen2,rngreen3,rngreen4,rnred1,rnred2,rnred3,rnred4)

        # Assemble master-file variance images.

        bias_greenvarimg,bias_redvarimg = self.assemble_var_images(self.masterbias_path)
        dark_greenvarimg,dark_redvarimg = self.assemble_var_images(self.masterdark_path)
        flat_greenvarimg,flat_redvarimg = self.assemble_var_images(self.masterflat_path)

        # Sum the variances for GREEN and RED chips, after converting all terms to electrons squared.
        # The terms in the following formulas are, respectively:
        # 1. Read-noise variance
        # 2. Master-bias variance
        # 3. Master-dark variance
        # 4. Master-flat variance
        # 5. Photon-noise variance

        # GREEN
        try:
            greenvarimg = rn_greenvarimg +\
                bias_greenvarimg +\
                dark_greenvarimg * exp_time +\
                flat_greenvarimg * greenccdimg +\
                greenccdimg
        except Exception as e:
            print("Exception raised [",e,"]; continuing...")
            greenvarimg = None

        # RED
        try:
            redvarimg = rn_redvarimg +\
                bias_redvarimg +\
                dark_redvarimg * exp_time +\
                flat_redvarimg * redccdimg +\
                redccdimg
        except Exception as e:
            print("Exception raised [",e,"]; continuing...")
            redvarimg = None

        # Write variance FITS-extensions.

        if (greenvarimg is not None) or (redvarimg is not None):

            self.write_var_exts(greenvarimg,redvarimg)

        self.logger.info('Finished {}'.format(self.__class__.__name__))

        return Arguments(var_exts_exit_code)
