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
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/var_exts/configs/default.cfg'

class VarExtsFramework(KPF0_Primitive):

    """
    Description:
        Input L0 filename and associated 2D kpf object.  Input master bias, dark, and flat filenames.
        Gather all the other variances, sum them all, and write the resulting total variance to the KPF object
        to FITS extensions ['GREEN_VAR','RED_VAR'] in the associated 2D FITS file.

    Arguments:
        data_type (str): Type of data (e.g., KPF).
        l0_filename (str): Full path and filename of L0 FITS file within container.
        masterbias_path (str): Input master bias.
        masterdark_path (str): Input master dark.
        masterflat_path (str): Input master flat.
        kpf_object_2d (KPF0): KPF0 object of the associated 2D FITS file.
    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.l0_filename = self.action.args[1]
        self.masterbias_path = self.action.args[2]
        self.masterdark_path = self.action.args[3]
        self.masterflat_path = self.action.args[4]
        self.kpf_object_2d = self.action.args[5]

        try:
            self.module_config_path = context.config_path['var_exts']
            print("--->",self.__class__.__name__,": self.module_config_path =",self.module_config_path)
        except:
            self.module_config_path = DEFAULT_CFG_PATH

        print("{} class: self.module_config_path = {}".format(self.__class__.__name__,self.module_config_path))

        self.logger = start_logger(self.__class__.__name__, self.module_config_path)
        self.logger.info('Started {}'.format(self.__class__.__name__))
        self.logger.debug('module_config_path = {}'.format(self.module_config_path))

        module_config_obj = cp.ConfigParser()
        res = module_config_obj.read(self.module_config_path)
        if res == []:
            raise IOError('failed to read {}'.format(self.module_config_path))

        self.logger.info('self.data_type = {}'.format(self.data_type))
        self.logger.info('self.l0_filename = {}'.format(self.l0_filename))
        self.logger.info('self.masterbias_path = {}'.format(self.masterbias_path))
        self.logger.info('self.masterdark_path = {}'.format(self.masterdark_path))
        self.logger.info('self.masterflat_path = {}'.format(self.masterflat_path))

    def assemble_var_images(self, fits_filename):

        # Read masters data object from disk.
        fits_filename_exists = exists(fits_filename)
        if not fits_filename_exists:
            self.logger.info('*** Master file does not exist ({}); skipping...'.format(fits_filename))
            return

        # Read in the master file. They have these _UNC extensions. Science frames do not.
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

    def make_variance_image(self, namps, img_shape, rn_value1, rn_value2, rn_value3=4.0, rn_value4=4.0):
        """
        Create a variance image for 2-amp (left/right halves) or 4-amp (quadrants)
        CCD layouts, using RN^2 values.

        Parameters
        ----------
        namps : int
            Number of amplifiers (2 or 4).
        img_shape : tuple
            Shape of the CCD image (ny, nx).
        rn_value1, rn_value2, rn_value3, rn_value4 : float
            Read noise values for each amplifier.

        Returns
        -------
        var_img : ndarray
            Variance image of shape `img_shape`, filled with RN^2 values
            according to the amplifier layout.
        """
        rn_values = [rn_value1, rn_value2, rn_value3, rn_value4]
        ny, nx = img_shape
        rn_values = [float(v)**2 for v in rn_values]  # square RN

        if namps == 2:
            half_x = nx // 2
            left  = np.full((ny, half_x), rn_values[0], dtype=float)
            right = np.full((ny, nx - half_x), rn_values[1], dtype=float)
            var_img = np.hstack((left, right))

        elif namps == 4:
            half_y, half_x = ny // 2, nx // 2
            tl = np.full((half_y, half_x), rn_values[0], dtype=float)
            tr = np.full((half_y, nx - half_x), rn_values[1], dtype=float)
            bl = np.full((ny - half_y, half_x), rn_values[2], dtype=float)
            br = np.full((ny - half_y, nx - half_x), rn_values[3], dtype=float)
            top = np.hstack((tl, tr))
            bot = np.hstack((bl, br))
            var_img = np.vstack((top, bot))
        else:
            raise ValueError("rn_values must contain exactly 2 or 4 elements")

        return var_img

    def _perform(self):

        """
        Perform the following steps:

        Returns exitcode:
            0 = Normal
            2 = Exception raised closing database connection
           64 = Cannot connect to database
           65 = Input L0 file does not exist
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

        # Replace the call to the database that holds the readnoise, with a read of the header
        # The read noise is calculated in analyze_l0.py and stored in the header in diagnostics.py

        rngreen1 = self.kpf_object_2d.header['PRIMARY'].get('RNNGGR1',4.0)
        rngreen2 = self.kpf_object_2d.header['PRIMARY'].get('RNNGGR2',4.0)
        rnred1 = self.kpf_object_2d.header['PRIMARY'].get('RNNGRD1',4.0)
        rnred2 = self.kpf_object_2d.header['PRIMARY'].get('RNNGRD2',4.0)

        rngreen3 = self.kpf_object_2d.header['PRIMARY'].get('RNNGGR3',4.0)
        rngreen4 = self.kpf_object_2d.header['PRIMARY'].get('RNNGGR4',4.0)
        rnred3 = self.kpf_object_2d.header['PRIMARY'].get('RNNGRD3',4.0)
        rnred4 = self.kpf_object_2d.header['PRIMARY'].get('RNNGRD4',4.0)
        self.logger.info('READING readnoise from header: rngreen1 = {}'.format(rngreen1))

        # Assemble CCD science images.
        greenccdimg = np.array(self.kpf_object_2d['GREEN_CCD'])
        redccdimg = np.array(self.kpf_object_2d['RED_CCD'])
        exp_time = float(self.kpf_object_2d.header['PRIMARY']['EXPTIME'])

        green_ccd_shape = greenccdimg.shape
        red_ccd_shape = redccdimg.shape
        
        # Determine number of amplifiers based on whether rngreen3/rnred3 exist
        n_green_amps = 4 if rngreen3 is not None else 2
        n_red_amps = 4 if rnred3 is not None else 2
        
        rn_greenvarimg = self.make_variance_image(n_green_amps, green_ccd_shape,
                                            rngreen1, rngreen2, rn_value3=rngreen3, rn_value4=rngreen4)

        rn_redvarimg   = self.make_variance_image(n_red_amps, red_ccd_shape,
                                            rnred1, rnred2, rn_value3=rnred3, rn_value4=rnred4)

        # READ IN MASTERS.
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
        self.kpf_object_2d['GREEN_VAR'] = greenvarimg.astype(np.float32)
        self.kpf_object_2d.header['GREEN_VAR']['BUNIT'] = ('electrons^2',)
        self.kpf_object_2d['RED_VAR'] = redvarimg.astype(np.float32)
        self.kpf_object_2d.header['RED_VAR']['BUNIT'] = ('electrons^2',)
        fits_obj_out = self.kpf_object_2d

        if (greenvarimg is not None) or (redvarimg is not None):
            self.logger.info("Final Length of variance images greenvarimg,redvarimg = {},{}".format(len(greenvarimg),len(redvarimg)))
        else:
            self.logger.info('No variance images to write; skipping...')

        self.logger.info('Finished {}'.format(self.__class__.__name__))

        return Arguments([var_exts_exit_code,fits_obj_out])